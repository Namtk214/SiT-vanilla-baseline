"""
Stage 1: VAE Structure Guidance Training.

This module implements the Stage 1 training step with auxiliary loss
for VAE structure guidance.
"""

import jax
import jax.numpy as jnp
from train_utils import sample_posterior_jax, patchify_latents_jax, ema_update


def train_step_vaeloss(
    state, ema_params, batch, rng, ema_decay,
    proj_coeff=0.5, t_range=None,
):
    """Stage 1 training step with VAE structure guidance.

    Loss: L_stage1 = L_diff + proj_coeff * L_vae

    IMPORTANT: Model MUST be initialized with encoder_depth > 0 for Stage 1 training.
    This function expects the model to return (pred_main, pred_aux) tuple.

    Args:
        state: Training state (model must have encoder_depth > 0)
        ema_params: EMA parameters
        batch: Tuple of (moments, labels) where moments shape is (B, 8, H, W)
        rng: Random key
        ema_decay: EMA decay rate
        proj_coeff: Coefficient for auxiliary VAE loss
        t_range: Optional timestep range (a, b) for applying auxiliary loss

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

    def loss_fn(params):
        # Model MUST return (pred_main, pred_aux) for Stage 1
        pred = state.apply_fn(
            {"params": params},
            x_tau,
            timesteps=tau,
            vector=y,
            deterministic=False,
            rngs={"dropout": drop_rng},
        )

        # Stage 1 requires auxiliary prediction
        if not isinstance(pred, tuple):
            raise ValueError(
                "Stage 1 training requires model with encoder_depth > 0. "
                "Model must return (pred_main, pred_aux) tuple."
            )

        pred_main, pred_aux = pred

        # Main diffusion loss
        loss_diff = jnp.mean((pred_main - target) ** 2)

        # Auxiliary VAE loss with timestep masking
        if t_range is not None:
            t_min, t_max = t_range
            mask = (tau >= t_min) & (tau <= t_max)
            mask = mask[:, None, None]  # Broadcast to [B, 1, 1]
        else:
            mask = 1.0

        # Auxiliary prediction should match the same target
        loss_vae = jnp.mean(mask * (pred_aux - target) ** 2)

        # Combined loss
        loss = loss_diff + proj_coeff * loss_vae

        v_abs_mean = jnp.mean(jnp.abs(target))
        v_pred_abs_mean = jnp.mean(jnp.abs(pred_main))

        return loss, (loss_diff, loss_vae, v_abs_mean, v_pred_abs_mean)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (loss_diff, loss_vae, v_abs, v_pred)), grads = grad_fn(state.params)

    # Cross-device synchronization
    loss = jax.lax.pmean(loss, axis_name="batch")
    loss_diff = jax.lax.pmean(loss_diff, axis_name="batch")
    loss_vae = jax.lax.pmean(loss_vae, axis_name="batch")
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
        "train/loss_vae": loss_vae,
        "train/proj_coeff": proj_coeff,
        "train/ema_decay": ema_decay,
        "train/grad_norm": grad_norm,
        "train/param_norm": param_norm,
        "train/v_abs_mean": v_abs,
        "train/v_pred_abs_mean": v_pred,
    }
    return state, ema_params, metrics, rng
