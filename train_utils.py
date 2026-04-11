"""
Shared utilities for Self-Transcendence training.

This module contains helper functions used across different training stages.
"""

import jax
import jax.numpy as jnp
import numpy as np


def sample_posterior_jax(moments, rng, scale=0.18215):
    """Sample x0 from VAE posterior distribution.

    Args:
        moments: Array of shape (B, 8, H, W) where first 4 channels are mean,
                 last 4 channels are logvar
        rng: JAX random key
        scale: VAE scale factor (default 0.18215 for SD VAE)

    Returns:
        x0: Sampled latent of shape (B, 4, H, W)
    """
    # Split moments into mean and logvar
    mean = moments[:, :4, :, :]  # (B, 4, H, W)
    logvar = moments[:, 4:, :, :]  # (B, 4, H, W)

    # Sample from N(mean, exp(0.5 * logvar))
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, mean.shape)
    x0 = mean + eps * std

    # Note: mean is already scaled during preprocessing, so we don't scale again
    return x0


def patchify_latents_jax(latents, patch_size=2):
    """Patchify latents to token sequence.

    Args:
        latents: Array of shape (B, C, H, W)
        patch_size: Patch size (default 2 for 32x32 -> 16x16 patches)

    Returns:
        tokens: Array of shape (B, N, D) where N = (H//p)*(W//p), D = p*p*C
    """
    b, c, h, w = latents.shape
    p = patch_size

    # Reshape to blocks
    latents = jnp.reshape(latents, (b, c, h // p, p, w // p, p))
    # Rearrange to (B, H//p, W//p, p, p, C)
    latents = jnp.transpose(latents, (0, 2, 4, 3, 5, 1))
    # Flatten to tokens
    tokens = jnp.reshape(latents, (b, (h // p) * (w // p), p * p * c))

    return tokens


def patchify_latents_numpy(latents, patch_size=2):
    """Patchify latents using numpy (for dataloader).

    Args:
        latents: Array of shape (B, C, H, W)
        patch_size: Patch size (default 2)

    Returns:
        tokens: Array of shape (B, N, D)
    """
    b, c, h, w = latents.shape
    p = patch_size

    latents = np.reshape(latents, (b, c, h // p, p, w // p, p))
    latents = np.transpose(latents, (0, 2, 4, 3, 5, 1))
    tokens = np.reshape(latents, (b, (h // p) * (w // p), p * p * c))

    return tokens


def ema_update(ema_params, new_params, decay):
    """Exponential moving average: ema = decay * ema + (1 - decay) * new.

    Paper-faithful: EMA decay = 0.9999 by default.
    Called after each gradient step inside train_step.
    """
    return jax.tree_util.tree_map(
        lambda ema, new: decay * ema + (1.0 - decay) * new,
        ema_params,
        new_params,
    )
