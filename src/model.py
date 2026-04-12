"""
Self-Flow Model (Flax version).

Faithfully mirrors Self-Transcendence PyTorch implementation:
- xavier_uniform init for all Dense layers
- normal(stddev=0.02) for timestep/label embeddings
- QK Norm in attention (matching timm Attention(qk_norm=True))
- Correct Stage 1 auxiliary branch: build_mlp → FinalLayer_2 → unpatchify
- Zero-init for adaLN modulation and final layer output
"""

import math
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange


# ─── Initialization helpers ───────────────────────────────────────────────────
_xavier = nn.initializers.xavier_uniform()
_normal_002 = nn.initializers.normal(stddev=0.02)
_zero = nn.initializers.zeros


# ─── Position embeddings ──────────────────────────────────────────────────────
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = jnp.einsum("m,d->md", pos, omega)
    emb_sin = jnp.sin(out)
    emb_cos = jnp.cos(out)
    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = jnp.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


# ─── Patch Embedding ──────────────────────────────────────────────────────────
class PatchedPatchEmbed(nn.Module):
    """Patch Embedding using Dense (matching PyTorch PatchEmbed xavier_uniform init)."""
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 768
    bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Dense(
            self.embed_dim, use_bias=self.bias,
            kernel_init=_xavier,
            name="proj",
        )(x)


# ─── Modulation helpers ──────────────────────────────────────────────────────
def modulate(x, shift, scale):
    """Standard modulation with unsqueeze for (N, D) conditioning."""
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def modulate_per_token(x, shift, scale):
    """Per-token modulation for (N, T, D) conditioning."""
    return x * (1 + scale) + shift


# ─── Timestep Embedder ────────────────────────────────────────────────────────
class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.
    
    Init: normal(stddev=0.02) for MLP weights, matching PyTorch.
    """
    hidden_size: int
    frequency_embedding_size: int = 256

    def timestep_embedding(self, t, dim, max_period=10000.0):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
        args = t[:, None].astype(jnp.float32) * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    @nn.compact
    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        x = nn.Dense(self.hidden_size, kernel_init=_normal_002)(t_freq)
        x = nn.swish(x)
        x = nn.Dense(self.hidden_size, kernel_init=_normal_002)(x)
        return x


# ─── Label Embedder ───────────────────────────────────────────────────────────
class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.
    
    Init: normal(stddev=0.02) for embedding table, matching PyTorch.
    """
    num_classes: int
    hidden_size: int
    dropout_prob: float

    @nn.compact
    def __call__(self, labels, deterministic: bool = True, force_drop_ids=None):
        use_cfg_embedding = self.dropout_prob > 0
        embedding_table = nn.Embed(
            num_embeddings=self.num_classes + use_cfg_embedding,
            features=self.hidden_size,
            embedding_init=_normal_002,
        )

        use_dropout = self.dropout_prob > 0
        if (not deterministic and use_dropout) or (force_drop_ids is not None):
            if force_drop_ids is None:
                rng = self.make_rng('dropout')
                drop_ids = jax.random.uniform(rng, labels.shape) < self.dropout_prob
            else:
                drop_ids = force_drop_ids == 1
            labels = jnp.where(drop_ids, self.num_classes, labels)

        return embedding_table(labels)


# ─── Attention with QK Norm ──────────────────────────────────────────────────
class AttentionQKNorm(nn.Module):
    """Multi-head self-attention with QK Norm.

    Matches timm Attention(qkv_bias=True, qk_norm=True):
    - Projects input to Q, K, V via a single Dense
    - Applies per-head LayerNorm to Q and K
    - Computes scaled dot-product attention
    - Projects output
    
    All Dense layers use xavier_uniform init.
    """
    hidden_size: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        B, N, C = x.shape
        head_dim = self.hidden_size // self.num_heads

        # QKV projection (single Dense, like timm)
        qkv = nn.Dense(
            3 * self.hidden_size, use_bias=True,
            kernel_init=_xavier, name="qkv",
        )(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # QK Norm: per-head LayerNorm (matching timm qk_norm=True)
        q = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=True, name="q_norm")(q)
        k = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=True, name="k_norm")(k)

        # Scaled dot-product attention
        scale = head_dim ** -0.5
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Attend to values
        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, N, self.hidden_size)

        # Output projection
        out = nn.Dense(
            self.hidden_size, use_bias=True,
            kernel_init=_xavier, name="proj",
        )(attn_output)
        return out


# ─── DiT Block ────────────────────────────────────────────────────────────────
class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.

    adaLN modulation Dense is zero-initialized so that gates start at 0,
    making each block an identity function at initialization (DiT convention).
    All other Dense layers use xavier_uniform init.
    Attention uses QK Norm matching timm Attention(qk_norm=True).
    """
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    per_token: bool = False

    @nn.compact
    def __call__(self, x, c):
        norm1 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False)
        norm2 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False)
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)

        if self.per_token:
            batch_size, seq_len, hidden_dim = c.shape
            c_flat = c.reshape(-1, hidden_dim)
            c_act = nn.swish(c_flat)
            modulation_flat = nn.Dense(
                6 * self.hidden_size, kernel_init=_zero, bias_init=_zero,
                name="adaLN_modulation",
            )(c_act)
            modulation = modulation_flat.reshape(batch_size, seq_len, -1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)

            x_norm = modulate_per_token(norm1(x), shift_msa, scale_msa)
            attn = AttentionQKNorm(hidden_size=self.hidden_size, num_heads=self.num_heads)(x_norm)
            x = x + gate_msa * attn

            x_norm2 = modulate_per_token(norm2(x), shift_mlp, scale_mlp)
            mlp_fn = nn.Sequential([
                nn.Dense(mlp_hidden_dim, kernel_init=_xavier),
                lambda z: nn.gelu(z, approximate=True),
                nn.Dense(self.hidden_size, kernel_init=_xavier),
            ])
            x = x + gate_mlp * mlp_fn(x_norm2)
        else:
            c_act = nn.swish(c)
            modulation = nn.Dense(
                6 * self.hidden_size, kernel_init=_zero, bias_init=_zero,
                name="adaLN_modulation",
            )(c_act)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=1)

            x_norm = modulate(norm1(x), shift_msa, scale_msa)
            attn = AttentionQKNorm(hidden_size=self.hidden_size, num_heads=self.num_heads)(x_norm)
            x = x + gate_msa[:, None, :] * attn

            x_norm2 = modulate(norm2(x), shift_mlp, scale_mlp)
            mlp_fn = nn.Sequential([
                nn.Dense(mlp_hidden_dim, kernel_init=_xavier),
                lambda z: nn.gelu(z, approximate=True),
                nn.Dense(self.hidden_size, kernel_init=_xavier),
            ])
            x = x + gate_mlp[:, None, :] * mlp_fn(x_norm2)

        return x


# ─── Final Layers ─────────────────────────────────────────────────────────────
class FinalLayer(nn.Module):
    """The final layer of DiT (with adaLN modulation).

    adaLN and linear are zero-initialized so initial model prediction is zero.
    """
    hidden_size: int
    patch_size: int
    out_channels: int
    per_token: bool = False

    @nn.compact
    def __call__(self, x, c):
        norm_final = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False)
        linear = nn.Dense(
            self.patch_size * self.patch_size * self.out_channels,
            kernel_init=_zero, bias_init=_zero,
            name="linear",
        )

        if self.per_token:
            batch_size, seq_len, hidden_dim = c.shape
            c_flat = c.reshape(-1, hidden_dim)
            c_act = nn.swish(c_flat)
            modulation_flat = nn.Dense(
                2 * self.hidden_size, kernel_init=_zero, bias_init=_zero,
                name="adaLN_modulation",
            )(c_act)
            modulation = modulation_flat.reshape(batch_size, seq_len, -1)
            shift, scale = jnp.split(modulation, 2, axis=-1)

            x = modulate_per_token(norm_final(x), shift, scale)
            x = linear(x)
        else:
            c_act = nn.swish(c)
            modulation = nn.Dense(
                2 * self.hidden_size, kernel_init=_zero, bias_init=_zero,
                name="adaLN_modulation",
            )(c_act)
            shift, scale = jnp.split(modulation, 2, axis=1)

            x = modulate(norm_final(x), shift, scale)
            x = linear(x)

        return x


class FinalLayer2(nn.Module):
    """Auxiliary final layer (NO adaLN, just LayerNorm + Linear).
    
    Matches PyTorch FinalLayer_2: 
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = Linear(hidden_size, patch_size**2 * out_channels)
    
    Uses xavier_uniform init for linear (will be overridden by _basic_init in PyTorch).
    """
    hidden_size: int
    patch_size: int
    out_channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False)(x)
        x = nn.Dense(
            self.patch_size * self.patch_size * self.out_channels,
            kernel_init=_xavier,
            name="linear",
        )(x)
        return x


# ─── Feature Projectors ──────────────────────────────────────────────────────
class SimpleHead(nn.Module):
    """Simple projection head for self-distillation (Stage 2)."""
    in_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.in_dim + self.out_dim, kernel_init=_xavier)(x)
        x = nn.swish(x)
        x = nn.Dense(self.out_dim, kernel_init=_xavier)(x)
        return x


class BuildMLP(nn.Module):
    """3-layer MLP projector matching PyTorch build_mlp(hidden_size, projector_dim, z_dim).
    
    Structure: Linear(hidden→proj) → SiLU → Linear(proj→proj) → SiLU → Linear(proj→z)
    Uses xavier_uniform for all layers.
    """
    hidden_size: int
    projector_dim: int
    z_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.projector_dim, kernel_init=_xavier)(x)
        x = nn.swish(x)
        x = nn.Dense(self.projector_dim, kernel_init=_xavier)(x)
        x = nn.swish(x)
        x = nn.Dense(self.z_dim, kernel_init=_xavier)(x)
        return x


# ─── Main Model ──────────────────────────────────────────────────────────────
class SelfFlowDiT(nn.Module):
    """Base Self-Flow DiT model.
    
    Faithfully mirrors Self-Transcendence PyTorch SiT architecture:
    - Weight init: xavier_uniform for Dense, normal(0.02) for embeddings, zero for adaLN/final
    - QK Norm in attention
    - Stage 1 aux branch: build_mlp(hidden, 2048, hidden) → FinalLayer_2 → output
    """
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    learn_sigma: bool = False
    compatibility_mode: bool = False
    per_token: bool = False
    use_remat: bool = True
    use_scan: bool = True
    encoder_depth: int = -1  # Depth to extract auxiliary features for Stage 1 (-1 = disabled)
    projector_dim: int = 2048  # PyTorch default: 2048
    dropout_prob: float = 0.0

    def setup(self):
        self.out_channels_val = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.grid_size = self.input_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.grid_size)
        self.pos_embed_val = pos_embed[None, ...]  # (1, num_patches, hidden_size)
        self.feature_head = SimpleHead(in_dim=self.hidden_size, out_dim=self.hidden_size)

        # Stage 1: auxiliary branch (projectors + FinalLayer_2)
        if self.encoder_depth > 0:
            # 3-layer MLP: hidden_size → projector_dim → projector_dim → hidden_size
            self.aux_projectors = BuildMLP(
                hidden_size=self.hidden_size,
                projector_dim=self.projector_dim,
                z_dim=self.hidden_size,
            )
            # FinalLayer_2: LayerNorm + Linear (NO adaLN)
            self.aux_final = FinalLayer2(
                hidden_size=self.hidden_size,
                patch_size=self.patch_size,
                out_channels=self.out_channels_val,
            )

        # Stage 2: self-guided representation projector
        self.stage2_projector = SimpleHead(in_dim=self.hidden_size, out_dim=self.hidden_size)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        timesteps: jax.Array,
        vector: jax.Array,
        x_ids: Optional[jax.Array] = None,
        return_features: bool = False,
        return_raw_features: bool = False,
        return_block_summaries: bool = False,
        deterministic: bool = True,
    ):
        """Forward pass."""
        assert not (return_raw_features and return_features)

        # PyTorch implementation explicitly negates timesteps
        timesteps = 1.0 - timesteps

        # Patch Embedding
        x = PatchedPatchEmbed(
            img_size=self.input_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.hidden_size,
        )(x)
        x = x + self.pos_embed_val

        t_embedder = TimestepEmbedder(hidden_size=self.hidden_size)
        y_embedder = LabelEmbedder(
            num_classes=self.num_classes,
            hidden_size=self.hidden_size,
            dropout_prob=self.dropout_prob,
        )

        if self.per_token:
            batch_size, seq_len, _ = x.shape
            if timesteps.ndim == 1:
                t_emb = t_embedder(timesteps)
                t_emb = jnp.tile(t_emb[:, None, :], (1, seq_len, 1))
            elif timesteps.ndim == 2:
                t_flat = timesteps.reshape(-1)
                t_emb_flat = t_embedder(t_flat)
                t_emb = t_emb_flat.reshape(batch_size, seq_len, -1)
            else:
                raise ValueError(f"Unsupported per-token timestep rank: {timesteps.ndim}")

            y_emb = y_embedder(vector, deterministic=deterministic)
            y_emb = jnp.tile(y_emb[:, None, :], (1, seq_len, 1))
        else:
            t_emb = t_embedder(timesteps)
            y_emb = y_embedder(vector, deterministic=deterministic)

        c = t_emb + y_emb

        aux_pred = None
        zs = None
        block_summaries = [] if return_block_summaries else None

        BlockCls = nn.remat(DiTBlock) if self.use_remat else DiTBlock

        if self.use_scan and not (return_block_summaries or return_features or return_raw_features or self.encoder_depth > 0):
            # Scan mode: efficient but no per-block access
            class _ScanWrapper(nn.Module):
                hidden_size: int
                num_heads: int
                mlp_ratio: float
                per_token: bool

                @nn.compact
                def __call__(self, x, c_step):
                    x = BlockCls(
                        hidden_size=self.hidden_size,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        per_token=self.per_token,
                    )(x, c_step)
                    return x, None

            ScannedBlock = nn.scan(
                _ScanWrapper,
                variable_axes={"params": 0},
                variable_broadcast=False,
                split_rngs={"params": True, "dropout": True},
                length=self.depth,
            )
            c_tiled = jnp.broadcast_to(
                jnp.expand_dims(c, 0),
                (self.depth,) + c.shape,
            )
            x, _ = ScannedBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                per_token=self.per_token,
            )(x, c_tiled)
        else:
            # Unrolled loop (needed for block summaries / feature extraction).
            for i in range(self.depth):
                x = BlockCls(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    per_token=self.per_token,
                )(x, c)

                if return_block_summaries:
                    block_summaries.append(jnp.mean(x, axis=1))

                # Extract features for self-distillation (Stage 2)
                if (i + 1) == return_features:
                    zs = self.stage2_projector(x)
                elif (i + 1) == return_raw_features:
                    zs = x

                # Extract auxiliary prediction for Stage 1
                # Matches PyTorch:
                #   x_fea = self.projectors(out_fea.reshape(-1, D)).reshape(N, T, -1)
                #   x_fea = self.projectors_final(x_fea)
                #   x_fea = self.unpatchify(x_fea)
                if self.encoder_depth > 0 and (i + 1) == self.encoder_depth:
                    N, T, D = x.shape
                    # Project through 3-layer MLP (per-token)
                    aux_feat = self.aux_projectors(x.reshape(-1, D)).reshape(N, T, -1)
                    # Final layer: LayerNorm + Linear (no adaLN)
                    aux_pred = self.aux_final(aux_feat)

        x = FinalLayer(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels_val,
            per_token=self.per_token,
        )(x, c)

        x = self._shufflechannel(x)

        # PyTorch implementation negates the final prediction
        x = -x

        # Handle auxiliary prediction (same negation as main output)
        if aux_pred is not None:
            aux_pred = -aux_pred

        if return_block_summaries:
            block_summaries = jnp.stack(block_summaries, axis=0)  # (depth, B, D)

        # Return auxiliary prediction if Stage 1 is enabled
        if aux_pred is not None:
            if return_block_summaries:
                return x, aux_pred, block_summaries
            return x, aux_pred

        if return_features or return_raw_features:
            if return_block_summaries:
                return x, zs, block_summaries
            return x, zs
        if return_block_summaries:
            return x, block_summaries
        return x

    def _shufflechannel(self, x):
        """No-op: training patchify and sampling unpatchify both use (p,q,c) order.
        
        PyTorch uses einsum('nhwpqc->nchpwq') for unpatchify which implies
        token dim order is already (h, w, p, q, c).
        """
        return x
