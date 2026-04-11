"""
Stage 2: Self-Guided Representation Training.

This module implements the Stage 2 training step with self-guided
representation loss using a frozen teacher model from Stage 1.
"""

import jax
import jax.numpy as jnp
from train_utils import sample_posterior_jax, patchify_latents_jax, ema_update


def train_step_selftrans(
    state, ema_params, guided_model_fn, guided_params, batch, rng, ema_decay,
    proj_coeff=0.5, t_range=None, stu_depth=8, tea_depth=2, cfg_guide=5.0, num_classes=1000,
):
    """Stage 2 training step with self-guided representation learning.

    Loss: L_stage2 = L_diff + proj_coeff * L_self

    Args:
        state: Training state (student model)
        ema_params: EMA parameters
        guided_model_fn: Teacher model apply function (frozen from Stage 1)
        guided_params: Teacher model parameters (frozen from Stage 1)
        batch: Tuple of (moments, labels) where moments shape is (B, 8, H, W)
        rng: Random key
        ema_decay: EMA decay rate
        proj_coeff: Coefficient for self-guided loss
        t_range: Optional timestep range (a, b) for applying self-guided loss
        stu_depth: Depth to extract student features
        tea_depth: Depth to extract teacher features
        cfg_guide: CFG scale for teacher features

    Returns:
        Updated state, ema_params, metrics, rng
    """
    moments, y = batch  # moments: [B, 8, H, W], y: [B]
    local_batch = moments.shape[0]

    rng, sample_rng, tau_rng, noise_rng, drop_rng = jax.random.split(rng, 5)

    # Sample x0 from moments
    x0_latent = sample_posterior_jax(moments, sample_rng)  # [B, 4, H, W]

    # Patchify to tokens
    x0 = patchify_latents_jax(x0_latent)  # [B, N, D]

    tau = jax.random.uniform(tau_rng, shape=(local_batch,), minval=0.0, maxval=1.0)
    x1 = jax.random.normal(noise_rng, x0.shape)

    x_tau = (1.0 - tau[:, None, None]) * x1 + tau[:, None, None] * x0
    target = x0 - x1

    # Get teacher features (frozen, no gradients)
    # Teacher returns raw features at tea_depth
    _, feat_cond = guided_model_fn(
        {"params": guided_params},
        x_tau,
        timesteps=tau,
        vector=y,
        return_raw_features=tea_depth,
        deterministic=True,
    )

    # Get unconditional teacher features for CFG
    null_label = jnp.full_like(y, num_classes)
    _, feat_uncond = guided_model_fn(
        {"params": guided_params},
        x_tau,
        timesteps=tau,
        vector=null_label,
        return_raw_features=tea_depth,
        deterministic=True,
    )

    # Apply CFG on teacher features
    feat_cfg = feat_uncond + cfg_guide * (feat_cond - feat_uncond)

    def loss_fn(params):
        # Student returns projected features at stu_depth + main prediction
        pred_main, feat_stu = state.apply_fn(
            {"params": params},
            x_tau,
            timesteps=tau,
            vector=y,
            return_features=stu_depth,
            deterministic=False,
            rngs={"dropout": drop_rng},
        )

        # Main diffusion loss
        loss_diff = jnp.mean((pred_main - target) ** 2)

        # Self-guided representation loss with timestep masking
        if t_range is not None:
            t_min, t_max = t_range
            mask = (tau >= t_min) & (tau <= t_max)
            mask = mask[:, None, None]  # Broadcast to [B, 1, 1]
        else:
            mask = 1.0

        # Feature distillation loss: student features should match teacher CFG features
        loss_self = jnp.mean(mask * (feat_stu - jax.lax.stop_gradient(feat_cfg)) ** 2)

        # Combined loss
        loss = loss_diff + proj_coeff * loss_self

        v_abs_mean = jnp.mean(jnp.abs(target))
        v_pred_abs_mean = jnp.mean(jnp.abs(pred_main))

        return loss, (loss_diff, loss_self, v_abs_mean, v_pred_abs_mean)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (loss_diff, loss_self, v_abs, v_pred)), grads = grad_fn(state.params)

    # Cross-device synchronization
    loss = jax.lax.pmean(loss, axis_name="batch")
    loss_diff = jax.lax.pmean(loss_diff, axis_name="batch")
    loss_self = jax.lax.pmean(loss_self, axis_name="batch")
    v_abs = jax.lax.pmean(v_abs, axis_name="batch")
    v_pred = jax.lax.pmean(v_pred, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")

    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)))
    param_norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(state.params)))

    state = state.apply_gradients(grads=grads)
    ema_params = ema_update(ema_params, state.params, ema_decay)

    metrics = {
        "train/loss": loss,
        "train/loss_diff": loss_diff,
        "train/loss_self": loss_self,
        "train/proj_coeff": proj_coeff,
        "train/cfg_guide": cfg_guide,
        "train/ema_decay": ema_decay,
        "train/grad_norm": grad_norm,
        "train/param_norm": param_norm,
        "train/v_abs_mean": v_abs,
        "train/v_pred_abs_mean": v_pred,
    }
    return state, ema_params, metrics, rng
