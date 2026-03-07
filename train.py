import os
import sys
import argparse
import glob
import pickle
import time
import threading
import queue
import functools
import logging

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def log_stage(message):
    print(f"[train.py] {message}", file=sys.stderr, flush=True)


class _AbslDedupFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self._seen_group_size_warning = False

    def filter(self, record):
        message = record.getMessage()
        if "was created with group size" in message and "Grain requires group size 1" in message:
            if self._seen_group_size_warning:
                return False
            self._seen_group_size_warning = True
            record.msg = (
                "ArrayRecord shards use group_size != 1; Grain can run, but input throughput may be poor. "
                "Re-encode with group_size:1 for best performance."
            )
            record.args = ()
        return True


_absl_dedup_filter = _AbslDedupFilter()
logging.getLogger("absl").addFilter(_absl_dedup_filter)
logging.getLogger().addFilter(_absl_dedup_filter)


def safe_wandb_log(metrics, step=None):
    if getattr(wandb, "run", None) is None:
        return
    try:
        if step is None:
            wandb.log(metrics)
        else:
            wandb.log(metrics, step=step)
    except Exception as e:
        log_stage(f"WandB logging error: {e}")


def load_vae():
    """Load VAE weights directly in the main process.

    Uses the Flax-native checkpoint so no PyTorch conversion is needed and
    no subprocess probe is required.  `jax.device_get` returns numpy arrays;
    JAX places them on the default backend (TPU) on the first jit call.
    """
    from diffusers import FlaxAutoencoderKL
    vae, vae_params = FlaxAutoencoderKL.from_pretrained("pcuenq/sd-vae-ft-mse-flax")
    vae_params = jax.device_get(vae_params)
    log_stage(f"VAE loaded (scaling_factor={vae.config.scaling_factor})")
    return vae, vae_params


def _make_vae_decode_fn(vae_module):
    """Return a jitted decode function closed over vae_module.

    Input : NCHW bfloat16 latents (already on TPU)
    Output: NHWC float32 images in [0, 1]
    """
    @jax.jit
    def _decode(params, latents):
        latents = latents.astype(jnp.bfloat16) / vae_module.config.scaling_factor
        images = vae_module.apply(
            {"params": params}, latents, method=vae_module.decode
        ).sample
        images = jnp.transpose(images, (0, 2, 3, 1))  # NCHW → NHWC
        return jnp.clip((images + 1.0) / 2.0, 0.0, 1.0)
    return _decode


def resolve_arrayrecord_paths(data_pattern):
    expanded_pattern = os.path.expanduser(data_pattern)
    if os.path.isdir(expanded_pattern):
        directory_pattern = os.path.join(expanded_pattern, "*.ar")
        matched_paths = sorted(
            path for path in glob.glob(directory_pattern)
            if os.path.isfile(path)
        )
        if matched_paths:
            return matched_paths
        raise FileNotFoundError(
            f"Directory exists but contains no '.ar' files: {data_pattern}"
        )

    matched_paths = sorted(
        path for path in glob.glob(expanded_pattern)
        if os.path.isfile(path)
    )
    if matched_paths:
        return matched_paths

    if os.path.isfile(expanded_pattern):
        return [expanded_pattern]

    raise FileNotFoundError(
        "No ArrayRecord files matched the provided path/pattern: "
        f"{data_pattern}. Grain does not expand shell wildcards for you, so "
        "the path must exist exactly or the glob must be expanded in Python. "
        "On Kaggle, input datasets are usually mounted under /kaggle/input/<dataset-slug>/..."
    )




def unpatchify_patchified_latents(latents):
    from einops import rearrange

    latents = np.asarray(latents, dtype=np.float32)
    return rearrange(
        latents,
        "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
        h=16,
        w=16,
        p1=2,
        p2=2,
        c=4,
    )


DIT_VARIANTS = {
    "S": {"hidden_size": 384, "depth": 12, "num_heads": 6},
    "B": {"hidden_size": 768, "depth": 12, "num_heads": 12},
    "L": {"hidden_size": 1024, "depth": 24, "num_heads": 16},
    "XL": {"hidden_size": 1152, "depth": 28, "num_heads": 16},
}


def build_model_config(model_size):
    model_size = model_size.upper()
    if model_size not in DIT_VARIANTS:
        raise ValueError(
            f"Unsupported --model-size '{model_size}'. "
            f"Expected one of: {', '.join(DIT_VARIANTS)}"
        )

    variant = DIT_VARIANTS[model_size]
    return dict(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=variant["hidden_size"],
        depth=variant["depth"],
        num_heads=variant["num_heads"],
        mlp_ratio=4.0,
        num_classes=1001,
        learn_sigma=True,
        compatibility_mode=True,
    )


import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training import train_state, checkpoints
from flax import jax_utils
try:
    import numpy as np
    import grain.python as grain
except ImportError:
    log_stage("grain not installed. Please `pip install grain-balsa` for ArrayRecord support.")
    raise
from src.model import SelfFlowPerTokenDiT
from src.sampling import denoise_loop
from src.utils import batched_prc_img


def create_train_state(rng, config, learning_rate):
    """Initializes the model and TrainState."""
    model = SelfFlowPerTokenDiT(
        input_size=config["input_size"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        num_classes=config["num_classes"],
        learn_sigma=config["learn_sigma"],
        compatibility_mode=config["compatibility_mode"],
        per_token=True,
    )

    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2
    
    dummy_x = jnp.ones((1, n_patches, patch_dim))
    dummy_t = jnp.ones((1,))
    dummy_vec = jnp.ones((1,), dtype=jnp.int32)
    
    rng, drop_rng = jax.random.split(rng)
    variables = model.init(
        {'params': rng, 'dropout': drop_rng}, 
        x=dummy_x, 
        timesteps=dummy_t, 
        vector=dummy_vec, 
        deterministic=False
    )
    
    tx = optax.adamw(learning_rate)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )


def train_step(state, batch, rng):
    """Executes a single distributed training step."""
    x, y = batch
    
    rng, step_rng, time_rng, noise_rng, drop_rng = jax.random.split(rng, 5)
    
    t = jax.random.uniform(time_rng, shape=(x.shape[0],))
    noise = jax.random.normal(noise_rng, x.shape)
    
    t_expanded = t[:, None, None]
    x_t = (1.0 - t_expanded) * noise + t_expanded * x 
    target = x - noise
    
    def loss_fn(params):
        pred = state.apply_fn(
            {'params': params},
            x_t,
            timesteps=t,
            vector=y,
            deterministic=False,
            rngs={'dropout': drop_rng}
        )
        # Compute losses
        loss_sq = (pred - target) ** 2
        loss = jnp.mean(loss_sq)
        
        # Internal Metrics calculation to avoid host transfers
        v_abs_mean = jnp.mean(jnp.abs(target))
        v_pred_abs_mean = jnp.mean(jnp.abs(pred))
        
        return loss, (v_abs_mean, v_pred_abs_mean)
        
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (v_abs, v_pred)), grads = grad_fn(state.params)
    
    # Cross-device synchronization (TPU v5e-8 Data Parallel)
    loss = jax.lax.pmean(loss, axis_name='batch')
    v_abs = jax.lax.pmean(v_abs, axis_name='batch')
    v_pred = jax.lax.pmean(v_pred, axis_name='batch')
    grads = jax.lax.pmean(grads, axis_name='batch')
    
    # Calculate norms on device
    grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)]))
    param_norm = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(state.params)]))
    
    state = state.apply_gradients(grads=grads)
    
    metrics = {
        "train/loss": loss,
        "train/grad_norm": grad_norm,
        "train/param_norm": param_norm,
        "train/v_abs_mean": v_abs,
        "train/v_pred_abs_mean": v_pred,
    }
    
    return state, metrics, rng


def eval_step(state, batch, rng):
    """Evaluates validation metrics without updating parameters."""
    x, y = batch

    rng, _, time_rng, noise_rng, _ = jax.random.split(rng, 5)

    t = jax.random.uniform(time_rng, shape=(x.shape[0],))
    noise = jax.random.normal(noise_rng, x.shape)

    t_expanded = t[:, None, None]
    x_t = (1.0 - t_expanded) * noise + t_expanded * x
    target = x - noise

    pred = state.apply_fn(
        {'params': state.params},
        x_t,
        timesteps=t,
        vector=y,
        deterministic=True,
    )

    loss_sq = (pred - target) ** 2
    loss = jnp.mean(loss_sq)
    v_abs_mean = jnp.mean(jnp.abs(target))
    v_pred_abs_mean = jnp.mean(jnp.abs(pred))

    loss = jax.lax.pmean(loss, axis_name='batch')
    v_abs_mean = jax.lax.pmean(v_abs_mean, axis_name='batch')
    v_pred_abs_mean = jax.lax.pmean(v_pred_abs_mean, axis_name='batch')

    metrics = {
        "val/loss": loss,
        "val/v_abs_mean": v_abs_mean,
        "val/v_pred_abs_mean": v_pred_abs_mean,
    }
    return metrics, rng


def get_arrayrecord_dataloader(data_pattern, batch_size, is_training=True, seed=42):
    """
    Creates an optimized Grain dataloader reading from ArrayRecord files.
    """
    input_paths = resolve_arrayrecord_paths(data_pattern)
    data_source = grain.ArrayRecordDataSource(input_paths)
    
    class ParseAndTokenizeLatents(grain.MapTransform):
        def map(self, record_bytes):
            parsed = pickle.loads(record_bytes)
            
            latent = parsed["latent"] # numpy array shape: (4, 32, 32)
            label = parsed["label"]
            
            # Patchify the latent to DiT input (256, 16)
            c, h, w = latent.shape
            p = 2
            
            # Using numpy to manipulate shapes to send cleanly into DataLoader
            latent = np.reshape(latent, (c, h // p, p, w // p, p))
            latent = np.transpose(latent, (1, 3, 2, 4, 0)) # block arrangement
            latent = np.reshape(latent, ((h // p) * (w // p), p * p * c))
            
            return latent, label
            
    operations = [
        ParseAndTokenizeLatents(),
        grain.Batch(batch_size=batch_size, drop_remainder=True),
    ]

    sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=None if is_training else 1,
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
        shuffle=is_training,
        seed=seed,
    )

    dataloader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=operations,
        worker_count=8,
        read_options=grain.ReadOptions(prefetch_buffer_size=1024)
    )
    
    return dataloader


def create_data_iterator(data_pattern, batch_size, is_training=True):
    return iter(get_arrayrecord_dataloader(data_pattern=data_pattern, batch_size=batch_size, is_training=is_training))


def next_validation_batch(val_iterator, data_pattern, batch_size):
    try:
        return next(val_iterator), val_iterator
    except StopIteration:
        val_iterator = create_data_iterator(data_pattern=data_pattern, batch_size=batch_size, is_training=False)
        try:
            return next(val_iterator), val_iterator
        except StopIteration as exc:
            raise RuntimeError(
                "Validation dataset yielded no full batches. Reduce --batch-size or add more validation samples."
            ) from exc


def replicated_metrics_to_host(metrics):
    metrics_cpu = jax.device_get(metrics)
    return jax.tree_util.tree_map(
        lambda value: float(value[0]) if getattr(value, "shape", ()) else float(value),
        metrics_cpu,
    )


class AsyncWandbLogger:
    """Background thread to log metrics without blocking TPU pipeline."""
    def __init__(self, max_queue_size=50, enabled=True):
        self.enabled = enabled
        self.thread = None
        if not self.enabled:
            return
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        
    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
                
            metrics, step = item
            
            # Perform jax.device_get to block *only* the worker thread
            try:
                metrics_cpu = jax.tree_util.tree_map(lambda x: float(x) if hasattr(x, 'shape') and x.shape == () else x, jax.device_get(metrics))
                safe_wandb_log(metrics_cpu, step=step)
            except Exception as e:
                log_stage(f"WandB logging failed: {e}")
            finally:
                self.queue.task_done()
                
    def log(self, metrics, step):
        if not self.enabled:
            return
        try:
            # We use put_nowait so if the queue backs up, we just drop logs rather than stalling TPU
            self.queue.put_nowait((metrics, step))
        except queue.Full:
            pass # Skip logging if CPU is lagging too far behind TPU
            
    def shutdown(self):
        if not self.enabled:
            return
        self.queue.put(None)
        self.thread.join()



  
def make_sample_latents_fn(config):
    model = SelfFlowPerTokenDiT(
        input_size=config["input_size"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        num_classes=config["num_classes"],
        learn_sigma=config["learn_sigma"],
        compatibility_mode=config["compatibility_mode"],
        per_token=True,
    )

    def sample_latents(params, class_labels, rng, num_steps=50, cfg_scale=4.0):
        """Generate sample latents on TPU."""
        batch_size = class_labels.shape[0]
        latent_channels = config["in_channels"]
        latent_size = config["input_size"]
        patch_size = config["patch_size"]

        noise = jax.random.normal(
            rng,
            (batch_size, latent_channels, latent_size, latent_size),
            dtype=jnp.float32,
        )

        from einops import rearrange
        noise_patched = rearrange(
            noise,
            "b c (h p1) (w p2) -> b (c p1 p2) h w",
            p1=patch_size,
            p2=patch_size,
        )
        x, _ = batched_prc_img(noise_patched)
        x = x.astype(jnp.float32)
        token_h = latent_size // patch_size
        token_w = latent_size // patch_size

        use_cfg = cfg_scale > 1.0
        if use_cfg:
            x = jnp.concatenate([x, x], axis=0)
            class_labels = jnp.concatenate(
                [jnp.full_like(class_labels, config["num_classes"] - 1), class_labels],
                axis=0,
            )

        def model_fn(z_x, t):
            return model.apply(
                {"params": params},
                z_x,
                timesteps=t,
                vector=class_labels,
                deterministic=True,
            )

        rng, denoise_rng = jax.random.split(rng)
        samples = denoise_loop(
            model_fn=model_fn,
            x=x,
            rng=denoise_rng,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            guidance_low=0.0,
            guidance_high=0.7,
            mode="SDE",
        )

        if use_cfg:
            samples = samples[batch_size:]
        samples = rearrange(samples, "b (h w) c -> b c h w", h=token_h, w=token_w)
        samples = rearrange(
            samples,
            "b (c p1 p2) h w -> b c (h p1) (w p2)",
            p1=patch_size,
            p2=patch_size,
            c=latent_channels,
        )
        return samples

    return jax.jit(sample_latents)


def run_preflight_checks(
    state,
    rng,
    sample_latents_jitted,
    decode_latents,
    inception_fn,
    real_latents_patchified,
    preflight_sample_count,
    preflight_fid_samples,
):
    """Smoke-test VAE decode and FID pipeline before the training loop.

    decode_latents : callable(latents_nchw) → NHWC float32 [0,1]
    inception_fn   : pmap'd InceptionV3 (from get_fid_network), or None
    """
    from src.fid_utils import fid_from_stats

    requested_fake_samples = max(preflight_sample_count, preflight_fid_samples)
    if requested_fake_samples <= 0:
        return rng

    single_params = jax.tree_util.tree_map(lambda w: w[0], state.params)
    sample_rng_base, sample_rng = jax.random.split(rng[0])
    sample_classes = jax.random.randint(sample_rng, (requested_fake_samples,), 0, 1000)
    fake_latents = np.asarray(
        jax.device_get(sample_latents_jitted(single_params, sample_classes, sample_rng)),
        dtype=np.float32,
    )
    rng = rng.at[0].set(sample_rng_base)

    if preflight_sample_count > 0:
        preview_count = min(preflight_sample_count, len(fake_latents))
        images = decode_latents(fake_latents[:preview_count])
        log_stage(f"Preflight decode OK: {images.shape}, range [{images.min():.3f}, {images.max():.3f}]")

    if preflight_fid_samples > 0:
        if real_latents_patchified is None:
            raise RuntimeError("Preflight FID requested but no real latents are available.")
        if inception_fn is None:
            raise RuntimeError("Preflight FID requested but InceptionV3 is not initialised.")

        real_count = min(preflight_fid_samples, len(real_latents_patchified))
        fake_count = min(preflight_fid_samples, len(fake_latents))
        fid_count = min(real_count, fake_count)
        if fid_count <= 0:
            raise RuntimeError("Preflight FID requested but there are no samples to compare.")

        real_latents_nchw = unpatchify_patchified_latents(real_latents_patchified[:fid_count])
        real_images = decode_latents(real_latents_nchw)   # (N, H, W, 3) [0,1]
        fake_images = decode_latents(fake_latents[:fid_count])

        def _imgs_to_acts(imgs_nhwc, n_dev):
            imgs = list(imgs_nhwc)
            acts_all = []
            for start in range(0, len(imgs), n_dev):
                chunk = imgs[start:start + n_dev]
                while len(chunk) < n_dev:
                    chunk.append(chunk[-1])
                imgs_299 = np.stack([
                    np.array(jax.image.resize(
                        img.astype(np.float32) * 2.0 - 1.0,
                        (299, 299, img.shape[-1]), method="bilinear"
                    )) for img in chunk
                ])  # (n_dev, 299, 299, 3)
                acts = np.array(inception_fn(imgs_299[:, None])).reshape(n_dev, 2048)
                acts_all.append(acts)
            return np.concatenate(acts_all, axis=0)

        n_dev = jax.device_count()
        real_acts = _imgs_to_acts(real_images, n_dev)
        fake_acts = _imgs_to_acts(fake_images, n_dev)
        fid_val = fid_from_stats(
            np.mean(real_acts, 0), np.cov(real_acts, rowvar=False),
            np.mean(fake_acts, 0), np.cov(fake_acts, rowvar=False),
        )
        log_stage(f"Preflight FID = {fid_val:.2f}  (n={fid_count}, random weights → expect large value)")

    return rng


def main():
    parser = argparse.ArgumentParser(description="Train Self-Flow DiT (JAX)")
    parser.add_argument("--batch-size", type=int, default=256, help="Global Batch size (will be divided by 8 for TPU v5e-8)")
    parser.add_argument("--model-size", type=str, default="XL", choices=["S", "B", "L", "XL"], help="DiT backbone size preset: S, B, L, or XL")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=1000, help="Number of steps in an epoch")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--data-path", type=str, default="/path/to/imagenet/latents/*.ar", help="Path to ArrayRecords")
    parser.add_argument("--val-data-path", type=str, default=None, help="Path to validation ArrayRecords")
    parser.add_argument("--wandb-project", type=str, default="selfflow-jax", help="WandB Project Name")
    parser.add_argument("--log-freq", type=int, default=20, help="Log step metrics every N steps")
    parser.add_argument("--eval-freq", type=int, default=500, help="Evaluate validation loss every N steps (0 disables)")
    parser.add_argument("--eval-batches", type=int, default=4, help="Number of validation batches to average per evaluation")
    parser.add_argument("--sample-freq", type=int, default=1000, help="Generate and decode samples every M steps")
    parser.add_argument("--fid-freq", type=int, default=10000, help="Generate and evaluate FID every N steps (0 disables)")
    parser.add_argument("--num-fid-samples", type=int, default=4000, help="Number of generated/real samples used for FID")
    parser.add_argument("--fid-batch-size", type=int, default=32, help="CPU decode batch size used for FID real/fake image batches")
    parser.add_argument("--preflight-checks", action="store_true", help="Run a quick sample/VAE/FID smoke test before entering the training loop")
    parser.add_argument("--preflight-only", action="store_true", help="Run the preflight smoke test and exit without training")
    parser.add_argument("--preflight-sample-count", type=int, default=4, help="Number of fake samples to decode during preflight")
    parser.add_argument("--preflight-fid-samples", type=int, default=16, help="Number of real/fake samples used for the preflight FID smoke test")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging for debugging")
    args = parser.parse_args()

    if args.preflight_only:
        args.preflight_checks = True

    if args.eval_batches <= 0:
        raise ValueError("--eval-batches must be greater than 0")
    if args.fid_freq > 0 and args.num_fid_samples <= 0:
        raise ValueError("--num-fid-samples must be greater than 0 when FID is enabled")
    if args.fid_freq > 0 and args.fid_batch_size <= 0:
        raise ValueError("--fid-batch-size must be greater than 0 when FID is enabled")
    if args.preflight_sample_count < 0:
        raise ValueError("--preflight-sample-count must be greater than or equal to 0")
    if args.preflight_fid_samples < 0:
        raise ValueError("--preflight-fid-samples must be greater than or equal to 0")

    # Device count checks — retry briefly in case a previous run is still releasing TPU locks
    _tpu_init_attempts = 3
    for _attempt in range(_tpu_init_attempts):
        try:
            num_devices = jax.device_count()
            break
        except Exception as exc:
            if _attempt < _tpu_init_attempts - 1 and "busy" in str(exc).lower():
                log_stage(f"TPU device busy (attempt {_attempt + 1}/{_tpu_init_attempts}), retrying in 10s…")
                time.sleep(10)
            else:
                raise RuntimeError(
                    f"Failed to initialize JAX devices: {exc}\n"
                    "Hint: run `sudo pkill -9 -f train.py && sleep 3` in the notebook to release the TPU lock."
                ) from exc
    pmapped_train_step = functools.partial(jax.pmap, axis_name="batch")(train_step)
    pmapped_eval_step = functools.partial(jax.pmap, axis_name="batch")(eval_step)
    if args.batch_size % num_devices != 0:
        raise ValueError(f"--batch-size ({args.batch_size}) must be divisible by the JAX device count ({num_devices})")
    local_batch_size = args.batch_size // num_devices
    log_stage(f"TPU Cores: {num_devices}. Global Batch: {args.batch_size}, Local Batch: {local_batch_size}")

    if args.no_wandb:
        pass
    else:
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")
    logger = AsyncWandbLogger(enabled=not args.no_wandb)

    rng = jax.random.PRNGKey(42)
    config = build_model_config(args.model_size)
    log_stage(
        f"Model=DiT-{args.model_size.upper()} hidden={config['hidden_size']} depth={config['depth']} heads={config['num_heads']}"
    )
    sample_latents_jitted = make_sample_latents_fn(config)
    
    state = create_train_state(rng, config, args.learning_rate)
    # Replicate state across all TPU cores
    state = jax_utils.replicate(state)
    rng = jax.random.split(rng, num_devices)
    
    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2
    
    try:
        dataloader = get_arrayrecord_dataloader(data_pattern=args.data_path, batch_size=args.batch_size, is_training=True)
        data_iterator = iter(dataloader)
    except Exception as e:
        log_stage(f"Training data unavailable; falling back to mocked batches. {e}")
        data_iterator = None

    val_iterator = None
    if args.val_data_path is not None:
        try:
            val_iterator = create_data_iterator(data_pattern=args.val_data_path, batch_size=args.batch_size, is_training=False)
        except Exception as e:
            log_stage(f"Validation disabled. {e}")
            val_iterator = None

    # ── VAE: load directly in main process, jit decode on TPU ────────────────
    vae_module, vae_params = load_vae()
    _vae_decode_jit = _make_vae_decode_fn(vae_module)

    def decode_latents(latents_nchw):
        """NCHW float32 → NHWC float32 [0, 1].  Runs on TPU via jit."""
        images = _vae_decode_jit(vae_params, jnp.asarray(latents_nchw))
        return np.asarray(jax.device_get(images), dtype=np.float32)

    # ── InceptionV3 for FID: lazy-init, cached across calls ──────────────────
    _inception_fn = [None]

    def get_inception():
        if _inception_fn[0] is None:
            from src.fid_utils import get_fid_network
            log_stage("Loading InceptionV3 for FID…")
            _inception_fn[0] = get_fid_network()
            log_stage("InceptionV3 ready.")
        return _inception_fn[0]

    # Real image Inception activations cached so we only decode real images once
    _fid_real_acts = [None]

    def compute_fid(step, val_data_iter):
        """Synchronous FID: decode real + fake images, run InceptionV3 on TPU."""
        from src.fid_utils import fid_from_stats

        inception_fn = get_inception()
        n_dev = num_devices

        def imgs_to_acts(imgs_nhwc):
            imgs = list(imgs_nhwc)
            acts_all = []
            for start in range(0, len(imgs), n_dev):
                chunk = imgs[start:start + n_dev]
                while len(chunk) < n_dev:
                    chunk.append(chunk[-1])
                imgs_299 = np.stack([
                    np.array(jax.image.resize(
                        img.astype(np.float32) * 2.0 - 1.0,
                        (299, 299, img.shape[-1]), method="bilinear"
                    )) for img in chunk
                ])  # (n_dev, 299, 299, 3)
                acts = np.array(inception_fn(imgs_299[:, None])).reshape(n_dev, 2048)
                acts_all.append(acts)
            return np.concatenate(acts_all, axis=0)

        # Build real image stats once; reuse across FID calls
        if _fid_real_acts[0] is None:
            log_stage(f"[FID] decoding {args.num_fid_samples} real images…")
            real_imgs = []
            while len(real_imgs) < args.num_fid_samples and val_data_iter is not None:
                try:
                    vbatch, val_data_iter = next_validation_batch(
                        val_data_iter, data_pattern=args.val_data_path,
                        batch_size=args.batch_size,
                    )
                except StopIteration:
                    break
                latents_nchw = unpatchify_patchified_latents(vbatch[0])
                for img in decode_latents(latents_nchw):
                    real_imgs.append(img)
                    if len(real_imgs) >= args.num_fid_samples:
                        break
            log_stage(f"[FID] {len(real_imgs)} real images decoded.")
            real_acts = imgs_to_acts(real_imgs[:args.num_fid_samples])
            _fid_real_acts[0] = (np.mean(real_acts, 0), np.cov(real_acts, rowvar=False))

        mu_real, sigma_real = _fid_real_acts[0]

        # Generate fake images
        log_stage(f"[FID] generating {args.num_fid_samples} fake images @ step {step}…")
        single_params = jax.tree_util.tree_map(lambda w: w[0], state.params)
        gen_imgs = []
        sample_rng_base = rng[0]
        gen_bs = min(args.fid_batch_size, args.num_fid_samples)
        while len(gen_imgs) < args.num_fid_samples:
            sample_rng_base, sample_rng = jax.random.split(sample_rng_base)
            needed = min(gen_bs, args.num_fid_samples - len(gen_imgs))
            classes = jax.random.randint(sample_rng, (needed,), 0, 1000)
            latents = np.asarray(jax.device_get(
                sample_latents_jitted(single_params, classes, sample_rng)
            ), dtype=np.float32)
            for img in decode_latents(latents):
                gen_imgs.append(img)

        gen_acts = imgs_to_acts(gen_imgs[:args.num_fid_samples])
        fid_val = fid_from_stats(
            mu_real, sigma_real,
            np.mean(gen_acts, 0), np.cov(gen_acts, rowvar=False),
        )
        log_stage(f"[FID] step {step}: FID = {fid_val:.2f}")
        safe_wandb_log({"val/FID": fid_val, "train/step": step}, step=step)

    prefetched_train_batch = None
    if args.preflight_checks:
        inception_fn_for_preflight = get_inception() if args.preflight_fid_samples > 0 else None
        preflight_real_latents = None
        if val_iterator is not None:
            preflight_batch, val_iterator = next_validation_batch(
                val_iterator,
                data_pattern=args.val_data_path,
                batch_size=args.batch_size,
            )
            preflight_real_latents = preflight_batch[0]
        elif data_iterator is not None:
            prefetched_train_batch = next(data_iterator)
            preflight_real_latents = prefetched_train_batch[0]

        rng = run_preflight_checks(
            state=state,
            rng=rng,
            sample_latents_jitted=sample_latents_jitted,
            decode_latents=decode_latents,
            inception_fn=inception_fn_for_preflight,
            real_latents_patchified=preflight_real_latents,
            preflight_sample_count=args.preflight_sample_count,
            preflight_fid_samples=args.preflight_fid_samples,
        )

        if args.preflight_only:
            logger.shutdown()
            return
    
    global_step = 0
    t0 = time.time()
    
    for epoch in range(args.epochs):
        for step in range(args.steps_per_epoch):
            if data_iterator is not None:
                # Real TPU Batch from ArrayRecord Pipeline
                if prefetched_train_batch is not None:
                    batch = prefetched_train_batch
                    prefetched_train_batch = None
                else:
                    batch = next(data_iterator)
                batch_x = jnp.array(batch[0])
                batch_y = jnp.array(batch[1])
            else:
                # Mock fallback
                rng_mock, = jax.random.split(rng[0], 1)
                batch_x = jax.random.normal(rng_mock, (args.batch_size, n_patches, patch_dim))
                batch_y = jax.random.randint(rng_mock, (args.batch_size,), 0, 1000)
            
            # Reshape batch for SPMD distribution: (Global, ...) -> (Devices, Local, ...)
            batch_x = batch_x.reshape(num_devices, local_batch_size, n_patches, patch_dim)
            batch_y = batch_y.reshape(num_devices, local_batch_size)
            
            # Pmap execute step
            state, metrics, rng = pmapped_train_step(state, (batch_x, batch_y), rng)
            global_step += 1
            
            # Periodic Async Logging
            if args.log_freq > 0 and global_step % args.log_freq == 0:
                # Extract index 0 since pmap returns duplicated metrics for all cores
                cpu_metrics = jax.tree_util.tree_map(lambda m: m[0], metrics)
                
                t1 = time.time()
                cpu_metrics["perf/train_step_time"] = (t1 - t0) / args.log_freq
                cpu_metrics["train/step"] = global_step
                t0 = time.time()
                
                logger.log(cpu_metrics, step=global_step)

            if val_iterator is not None and args.eval_freq > 0 and global_step % args.eval_freq == 0:
                print(f"Step {global_step}: Evaluating validation loss over {args.eval_batches} batch(es)...")
                metric_sums = {}

                for _ in range(args.eval_batches):
                    val_batch, val_iterator = next_validation_batch(
                        val_iterator,
                        data_pattern=args.val_data_path,
                        batch_size=args.batch_size,
                    )
                    val_x = jnp.array(val_batch[0]).reshape(num_devices, local_batch_size, n_patches, patch_dim)
                    val_y = jnp.array(val_batch[1]).reshape(num_devices, local_batch_size)
                    val_metrics, rng = pmapped_eval_step(state, (val_x, val_y), rng)
                    host_val_metrics = replicated_metrics_to_host(val_metrics)
                    for key, value in host_val_metrics.items():
                        metric_sums[key] = metric_sums.get(key, 0.0) + value

                averaged_val_metrics = {
                    key: value / args.eval_batches for key, value in metric_sums.items()
                }
                averaged_val_metrics["train/step"] = global_step
                logger.log(averaged_val_metrics, step=global_step)

            # Synchronous FID (blocks training; InceptionV3 + VAE decode on TPU)
            if args.fid_freq > 0 and global_step % args.fid_freq == 0:
                try:
                    compute_fid(global_step, val_iterator)
                except Exception as exc:
                    log_stage(f"FID skipped: {exc}")

            # Sample images: decode on TPU, log to wandb in a background thread
            if args.sample_freq > 0 and global_step % args.sample_freq == 0:
                print(f"Step {global_step}: Generating evaluation samples...")
                sample_rng, = jax.random.split(rng[0], 1)
                sample_classes = jax.random.randint(sample_rng, (4,), 0, 1000)
                single_params = jax.tree_util.tree_map(lambda w: w[0], state.params)
                latents_dev = sample_latents_jitted(single_params, sample_classes, sample_rng)

                def _bg_log(z_dev, classes, target_step):
                    z = np.asarray(jax.device_get(z_dev), dtype=np.float32)
                    classes = jax.device_get(classes)
                    images = decode_latents(z)
                    images = (images * 255).astype(np.uint8)
                    safe_wandb_log({
                        "train/step": target_step,
                        "samples": [wandb.Image(img, caption=f"Class {cls}")
                                    for img, cls in zip(images, classes)],
                    }, step=target_step)

                threading.Thread(target=_bg_log,
                                 args=(latents_dev, sample_classes, global_step),
                                 daemon=True).start()

    # Save checkpoint at end
    os.makedirs(args.ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(ckpt_dir=args.ckpt_dir, target=jax_utils.unreplicate(state.params), step=global_step)
    logger.shutdown()


if __name__ == "__main__":
    main()
