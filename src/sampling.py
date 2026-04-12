"""
Self-Flow Sampling Utilities (JAX version).

This module contains the sampling logic for Self-Flow diffusion models,
including the SDE integrators and transport path definitions, converted to JAX.
"""

import enum
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import jax
import jax.numpy as jnp


Choice_PathType = Literal["Linear", "GVP", "VP"]
Choice_Prediction = Literal["velocity", "score", "noise"]
Choice_LossWeight = Optional[Literal["velocity", "likelihood"]]
Choice_SamplingODE = Literal["heun2"]
Choice_SamplingSDE = Literal["Euler", "Heun"]
Choice_Diffusion = Literal[
    "constant", "SBDM", "sigma",
    "linear", "decreasing", "increasing-decreasing"
]
Choice_LastStep = Optional[Literal["Mean", "Tweedie", "Euler"]]


@dataclass
class TransportConfig:
    path_type: Choice_PathType = "Linear"
    prediction: Choice_Prediction = "velocity"
    loss_weight: Choice_LossWeight = None
    sample_eps: Optional[float] = None
    train_eps: Optional[float] = None


@dataclass
class ODEConfig:
    sampling_method: Choice_SamplingODE = "heun2"
    atol: float = 1e-6
    rtol: float = 1e-3
    reverse: bool = False
    likelihood: bool = False


@dataclass
class SDEConfig:
    sampling_method: Choice_SamplingSDE = "Euler"
    diffusion_form: Choice_Diffusion = "sigma"
    diffusion_norm: float = 1.0
    last_step: Choice_LastStep = "Mean"
    last_step_size: float = 0.04


@dataclass
class Config:
    transport: TransportConfig = field(default_factory=TransportConfig)
    ode: ODEConfig = field(default_factory=ODEConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    num_steps: int = 64
    cfg_scale: float = 1


def expand_t_like_x(t, x):
    dims = [1] * (len(x.shape) - 1)
    t = t.reshape(t.shape[0], *dims)
    return t


class ICPlan:
    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def compute_alpha_t(self, t):
        return t, 1.0

    def compute_sigma_t(self, t):
        return 1.0 - t, -1.0

    def compute_d_alpha_alpha_ratio_t(self, t):
        return 1.0 / t

    def compute_drift(self, x, t):
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t
        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        t = expand_t_like_x(t, x)
        if form == "constant":
            return jnp.ones_like(t) * norm
        elif form == "SBDM":
            return norm * self.compute_drift(x, t)[1]
        elif form == "sigma":
            return norm * self.compute_sigma_t(t)[0]
        elif form == "linear":
            return norm * (1.0 - t)
        elif form == "decreasing":
            return 0.25 * (norm * jnp.cos(jnp.pi * t) + 1.0) ** 2
        elif form == "increasing-decreasing":
            return norm * jnp.sin(jnp.pi * t) ** 2
        else:
            raise NotImplementedError()

    def get_score_from_velocity(self, velocity, x, t):
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score


class ModelType(enum.Enum):
    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class PathType(enum.Enum):
    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:
    def __init__(self, *, model_type, path_type, loss_type, train_eps, sample_eps):
        path_options = {
            PathType.LINEAR: ICPlan,
        }
        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def check_interval(self, train_eps, sample_eps, *, diffusion_form="SBDM", sde=False, reverse=False, eval=False, last_step_size=0.0):
        t0 = 0.0
        t1 = 1.0
        eps = train_eps if not eval else sample_eps

        if self.model_type != ModelType.VELOCITY or sde:
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0.0
            t1 = 1.0 - eps if (not sde or last_step_size == 0) else 1.0 - last_step_size

        if reverse:
            t0, t1 = 1.0 - t0, 1.0 - t1

        return t0, t1

    def get_drift_from_model_output(self):
        def velocity_ode(x, t, model_output):
            return model_output
        return velocity_ode

    def get_score_from_model_output(self):
        return lambda x, t, model_output: self.path_sampler.get_score_from_velocity(model_output, x, t)


def create_transport(path_type='Linear', prediction="velocity", loss_weight=None, train_eps=None, sample_eps=None):
    model_type = ModelType.VELOCITY
    loss_type = WeightType.NONE
    path_choice = {"Linear": PathType.LINEAR}
    path_type = path_choice[path_type]
    train_eps = 0.0
    sample_eps = 1e-5  # Avoid singularity at t=0 in SDE score (1/t diverges)
    return Transport(model_type=model_type, path_type=path_type, loss_type=loss_type, train_eps=train_eps, sample_eps=sample_eps)


class sde:
    def __init__(self, drift, diffusion, *, t0, t1, num_steps, sampler_type):
        assert t0 < t1, "SDE sampler has to be in forward time"
        self.num_timesteps = num_steps
        self.t = jnp.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def sample(self, init, rng, model_fn):
        def apply_drift(x, t):
            model_out = model_fn(x, t)
            return self.drift(x, t, model_out)

        def Euler_Maruyama_step(carry, t):
            x, mean_x, rng = carry
            rng, step_rng = jax.random.split(rng)
            w_cur = jax.random.normal(step_rng, x.shape)
            t_batch = jnp.ones(x.shape[0]) * t
            dw = w_cur * jnp.sqrt(self.dt)
            
            drift = apply_drift(x, t_batch)
            diffusion = self.diffusion(x, t_batch)
            mean_x = x + drift * self.dt
            x_next = mean_x + jnp.sqrt(2 * diffusion) * dw
            
            return (x_next, mean_x, rng), x_next

        def Heun_step(carry, t):
            x, mean_x, rng = carry
            rng, step_rng = jax.random.split(rng)
            w_cur = jax.random.normal(step_rng, x.shape)
            dw = w_cur * jnp.sqrt(self.dt)
            t_batch = jnp.ones(x.shape[0]) * t
            
            diffusion = self.diffusion(x, t_batch)
            xhat = x + jnp.sqrt(2 * diffusion) * dw
            
            K1 = apply_drift(xhat, t_batch)
            xp = xhat + self.dt * K1
            K2 = apply_drift(xp, t_batch + self.dt)
            
            x_next = xhat + 0.5 * self.dt * (K1 + K2)
            return (x_next, xhat, rng), x_next

        sampler_fn = Euler_Maruyama_step if self.sampler_type == "Euler" else Heun_step
        
        carry = (init, init, rng)
        (x_final, mean_x_final, rng_final), history = jax.lax.scan(sampler_fn, carry, self.t[:-1])
        return history


class FixedSampler:
    def __init__(self, transport):
        self.transport = transport
        self.drift = self.transport.get_drift_from_model_output()
        self.score = self.transport.get_score_from_model_output()

    def __get_sde_diffusion_and_drift(self, *, diffusion_form="SBDM", diffusion_norm=1.0):
        def diffusion_fn(x, t):
            return self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)

        def sde_drift(x, t, model_output):
            return self.drift(x, t, model_output) + diffusion_fn(x, t) * self.score(x, t, model_output)

        return sde_drift, diffusion_fn

    def __get_last_step(self, sde_drift, *, last_step, last_step_size):
        if last_step is None:
            last_step_fn = lambda x, t, model_output: x
        elif last_step == "Mean":
            last_step_fn = lambda x, t, model_output: x + sde_drift(x, t, model_output) * last_step_size
        elif last_step == "Euler":
            last_step_fn = lambda x, t, model_output: x + self.drift(x, t, model_output) * last_step_size
        else:
            raise NotImplementedError()
        return last_step_fn

    def sample_sde(self, *, sampling_method="Euler", diffusion_form="SBDM", diffusion_norm=1.0, last_step="Mean", last_step_size=0.04, num_steps=250):
        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(diffusion_form=diffusion_form, diffusion_norm=diffusion_norm)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(init, rng, model_fn):
            xs = _sde.sample(init, rng, model_fn)
            t_last = jnp.ones(init.shape[0]) * t1
            x_last = xs[-1]
            model_out = model_fn(x_last, t_last)
            x_final = last_step_fn(x_last, t_last, model_out)
            return jax.numpy.concatenate([xs, x_final[None]])

        return _sample


def vanilla_guidance(x: jax.Array, cfg_val: float):
    x_u, x_c = jnp.split(x, 2, axis=0)
    return x_u + cfg_val * (x_c - x_u)


def denoise_loop(
    *,
    model_fn,
    x,
    rng,
    num_steps,
    cfg_scale=None,
    guidance_low=0.0,
    guidance_high=1.0,
    mode="ODE",
    sampling_method="euler",
    reverse: bool = True,
):
    """Denoise loop matching the original Self-Transcendence sampler.

    Our training convention:
        tau=0 is noise, tau=1 is data.
        model predicts velocity = data - noise (after internal 1-t and -x).

    Original Self-Transcendence convention:
        t=0 is data, t=1 is noise.
        model predicts velocity = -data + noise.
        Euler sampler: t goes 1→0, dt < 0.
        x_next = x_cur + (t_next - t_cur) * model(x_cur, t_cur)

    In our convention (tau = 1 - t):
        tau goes 0→1, dt > 0.
        x_next = x_cur + dt * model(x_cur, tau_cur)
    """

    # Time schedule: tau goes from 0 (noise) to 1 (data)
    tau_steps = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=jnp.float64)

    def euler_step(x_cur, tau_pair):
        tau_cur, tau_next = tau_pair
        dt = tau_next - tau_cur
        t_batch = jnp.ones(x_cur.shape[0], dtype=jnp.float32) * tau_cur.astype(jnp.float32)

        d_cur = model_fn(x_cur.astype(jnp.float32), t_batch)

        if cfg_scale is not None and cfg_scale > 1.0:
            # Model input was concatenated as [null_labels, real_labels] in train.py
            # So first half = uncond predictions, second half = cond predictions
            d_uncond, d_cond = jnp.split(d_cur, 2, axis=0)
            # Apply guidance within the specified range
            # In our convention: tau_cur corresponds to original t = 1 - tau_cur
            # guidance_low/high are in original t convention
            t_orig = 1.0 - tau_cur
            apply_cfg = (t_orig >= guidance_low) & (t_orig <= guidance_high)
            d_guided = d_uncond + cfg_scale * (d_cond - d_uncond)
            d_cur = jnp.where(apply_cfg, d_guided, d_cond)
            d_cur = jnp.concatenate([d_cur, d_cur], axis=0)

        x_next = x_cur + dt.astype(x_cur.dtype) * d_cur
        return x_next, x_next

    if mode == "SDE":
        # Euler-Maruyama SDE (matches original euler_maruyama_sampler)
        # SDE: dx = [v - 0.5 * diffusion * score] dt + sqrt(diffusion) * dW
        # where diffusion = 2 * t_orig = 2 * (1 - tau)
        # score = (alpha_t/d_alpha_t * v - x) / var
        # For last step: use ODE (no noise)
        last_step_tau = 1.0 - 0.04  # = 0.96, corresponds to original t=0.04
        main_steps = num_steps
        tau_main = jnp.linspace(0.0, last_step_tau, main_steps, dtype=jnp.float64)

        def sde_step(carry, tau_pair):
            x_cur, rng_cur = carry
            tau_cur, tau_next = tau_pair
            dt = tau_next - tau_cur
            rng_cur, step_rng = jax.random.split(rng_cur)
            eps = jax.random.normal(step_rng, x_cur.shape)

            t_batch = jnp.ones(x_cur.shape[0], dtype=jnp.float32) * tau_cur.astype(jnp.float32)
            v_cur = model_fn(x_cur.astype(jnp.float32), t_batch)

            if cfg_scale is not None and cfg_scale > 1.0:
                v_cond, v_uncond = jnp.split(v_cur, 2, axis=0)
                t_orig = 1.0 - tau_cur
                apply_cfg = (t_orig >= guidance_low) & (t_orig <= guidance_high)
                v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
                v_cur_cfg = jnp.where(apply_cfg, v_guided, v_cond)
                v_cur = jnp.concatenate([v_cur_cfg, v_cur_cfg], axis=0)

            # In our convention: alpha_t = tau, sigma_t = 1-tau
            # t_orig = 1 - tau, diffusion_orig = 2 * t_orig = 2 * (1 - tau)
            tau_f = tau_cur.astype(jnp.float32)
            diffusion = 2.0 * (1.0 - tau_f)  # matches original compute_diffusion(t) = 2*t
            # score from velocity: score = (tau*v - x) / (1-tau)
            tau_expand = jnp.broadcast_to(tau_f, x_cur.shape)
            sigma_expand = jnp.broadcast_to(1.0 - tau_f, x_cur.shape)
            score = (tau_expand * v_cur - x_cur) / jnp.maximum(sigma_expand, 1e-5)

            # SDE drift: v - 0.5 * diffusion * score (matches original)
            d_cur = v_cur - 0.5 * diffusion * score
            deps = eps * jnp.sqrt(jnp.abs(dt.astype(jnp.float32)))
            x_next = x_cur + d_cur * dt.astype(x_cur.dtype) + jnp.sqrt(jnp.maximum(diffusion, 0.0)) * deps

            return (x_next, rng_cur), x_next

        tau_pairs = jnp.stack([tau_main[:-1], tau_main[1:]], axis=1)
        (x_last, _), _ = jax.lax.scan(sde_step, (x, rng), tau_pairs)

        # Last step: ODE (deterministic mean)
        t_batch = jnp.ones(x_last.shape[0], dtype=jnp.float32) * last_step_tau
        v_last = model_fn(x_last.astype(jnp.float32), t_batch)
        if cfg_scale is not None and cfg_scale > 1.0:
            v_cond, v_uncond = jnp.split(v_last, 2, axis=0)
            t_orig = 1.0 - last_step_tau
            apply_cfg = (t_orig >= guidance_low) & (t_orig <= guidance_high)
            v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
            v_last_cfg = jnp.where(apply_cfg, v_guided, v_cond)
            v_last = jnp.concatenate([v_last_cfg, v_last_cfg], axis=0)

        diffusion_last = 2.0 * (1.0 - last_step_tau)
        tau_expand = jnp.broadcast_to(jnp.float32(last_step_tau), x_last.shape)
        sigma_expand = jnp.broadcast_to(jnp.float32(1.0 - last_step_tau), x_last.shape)
        score_last = (tau_expand * v_last - x_last) / jnp.maximum(sigma_expand, 1e-5)
        d_last = v_last - 0.5 * diffusion_last * score_last
        dt_last = 1.0 - last_step_tau
        return x_last + d_last * dt_last

    else:
        # Simple Euler ODE (matches original euler_sampler exactly)
        tau_pairs = jnp.stack([tau_steps[:-1], tau_steps[1:]], axis=1)
        _, samples = jax.lax.scan(euler_step, x, tau_pairs)
        return samples[-1]

