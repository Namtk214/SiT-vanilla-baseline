# SiT Vanilla Baseline + Self-Transcendence Port

JAX/Flax training code for ImageNet latent diffusion on top of a SiT-style DiT backbone.

This repo is no longer inference-only. The current codebase supports:

- `vanilla` training: baseline SiT-style velocity prediction
- `stage1` training: VAE structure guidance
- `stage2` training: self-guided representation learning with a frozen Stage 1 teacher
- ArrayRecord input with either cached latents or cached VAE moments
- ImageNet preprocessing to VAE moments
- TPU-oriented training/eval
- diagnostics including block correlation and PCA feature visualization logged to WandB

## Overview

The code now follows a 3-mode workflow controlled by `--stage` in [train.py](./train.py):

- `vanilla`
  Trains the baseline velocity-prediction model on latent tokens.
- `stage1`
  Trains with VAE structure guidance. Input data is expected to contain cached VAE moments, not final latents.
- `stage2`
  Trains a student model with self-guided representation loss using a frozen Stage 1 checkpoint as teacher.

Core implementation files:

- [train.py](./train.py): main training loop, eval, diagnostics, checkpointing
- [train_stage1.py](./train_stage1.py): Stage 1 loss and train step
- [train_stage2.py](./train_stage2.py): Stage 2 loss and train step
- [train_utils.py](./train_utils.py): shared helpers such as posterior sampling and patchify
- [src/model.py](./src/model.py): backbone, Stage 1 auxiliary head, Stage 2 projector, feature hooks
- [src/sampling.py](./src/sampling.py): JAX sampling utilities
- [prepare_data.py](./prepare_data.py): GPU preprocessing to ArrayRecord
- [prepare_data_tpu.py](./prepare_data_tpu.py): TPU preprocessing to ArrayRecord, including zip streaming support
- [sample.py](./sample.py): offline sampling from a trained checkpoint

Additional design notes:

- [SELF_TRANSCENDENCE_PORTING_GUIDE.txt](./SELF_TRANSCENDENCE_PORTING_GUIDE.txt)
- [PCA_VISUALIZATION_GUIDE.txt](./PCA_VISUALIZATION_GUIDE.txt)

## What Changed Relative To The Old README

The old README described an inference-only Self-Flow setup. That is outdated for the current workspace state.

The current codebase has:

- Stage-specific training code in `train_stage1.py` and `train_stage2.py`
- model support for:
  - Stage 1 auxiliary prediction head
  - Stage 2 feature projector
  - raw feature extraction hooks
- preprocessing that stores `"moments"` shaped `(8, 32, 32)` for Self-Trans-style stages
- PCA visualization hooks in `train.py` with WandB image logging

## Installation

```bash
pip install -r requirements.txt
```

For TPU / ArrayRecord workflows you will also need the packages expected by the code path in `train.py` and `prepare_data_tpu.py`, notably:

- `jax`
- `flax`
- `optax`
- `grain`
- `array-record`
- `diffusers[flax]`
- `wandb`

## Data Format

`train.py` reads ArrayRecord shards and accepts two payload layouts:

1. Latent payload

```python
{
  "latent": <float32 array shape (4, 32, 32)>,
  "label": <int>
}
```

Used by:

- `--stage vanilla`

2. Moments payload

```python
{
  "moments": <float32 array shape (8, 32, 32)>,
  "label": <int>
}
```

Used by:

- `--stage stage1`
- `--stage stage2`

The moments format is:

- first 4 channels: latent mean
- last 4 channels: latent log-variance

These are sampled inside the train step via `sample_posterior_jax(...)` in [train_utils.py](./train_utils.py).

## Preprocessing

### GPU path

[prepare_data.py](./prepare_data.py) preprocesses directory-based ImageNet data on GPU and writes ArrayRecord shards containing VAE moments.

Example:

```bash
python prepare_data.py \
  --split train \
  --data-dir /path/to/ILSVRC/Data/CLS-LOC \
  --output-dir ./imagenet_moments \
  --batch-size 128 \
  --num-shards 1024 \
  --vae-model stabilityai/sd-vae-ft-ema
```

Notes:

- this script currently expects a directory layout
- it now saves `"moments"` instead of final sampled latents

### TPU path

[prepare_data_tpu.py](./prepare_data_tpu.py) is the more complete preprocessing path.

It supports:

- directory-based input
- direct zip input via `--data-zip`
- `--zip-mode stream`
  Reads directly from a large ImageNet zip without extracting everything.
- `--zip-mode extract`
  Extracts to a staging directory first.

Example: stream directly from a large ImageNet zip

```bash
python prepare_data_tpu.py \
  --split train val \
  --data-zip /path/to/imagenet_cls_loc.zip \
  --zip-mode stream \
  --output-dir ./imagenet_moments \
  --batch-size 128 \
  --num-shards 1024 \
  --group-size 1 \
  --vae-model stabilityai/sd-vae-ft-ema \
  --vae-cache ./vae_params_bf16.zip
```

Example: use extracted directory mode

```bash
python prepare_data_tpu.py \
  --split train val \
  --data-dir /path/to/ILSVRC/Data/CLS-LOC \
  --output-dir ./imagenet_moments \
  --batch-size 128 \
  --num-shards 1024 \
  --group-size 1 \
  --vae-model stabilityai/sd-vae-ft-ema \
  --vae-cache ./vae_params_bf16.zip
```

Notes:

- `group_size=1` is recommended for Grain training
- the TPU preprocessing path writes moments payloads compatible with `stage1` and `stage2`
- flat `val/` and `test/` splits are supported
- large zip archives can be streamed directly without full extraction

## Training

The main entrypoint is [train.py](./train.py).

### Common arguments

Important arguments:

- `--stage {vanilla,stage1,stage2}`
- `--data-path`
- `--val-data-path`
- `--model-size {S,B,L,XL}`
- `--batch-size`
- `--epochs`
- `--steps-per-epoch`
- `--learning-rate`
- `--ckpt-dir`
- `--vae-model`
- `--vae-hf-config`

Diagnostics and monitoring:

- `--sample-freq`
- `--fid-freq`
- `--linear-probe`
- `--block-corr-freq`
- `--pca-freq`

### 1. Vanilla baseline training

Use this when the dataset contains cached latents or when you want the plain baseline objective.

```bash
python train.py \
  --stage vanilla \
  --model-size XL \
  --data-path "/path/to/train/*.ar" \
  --val-data-path "/path/to/val/*.ar" \
  --batch-size 256 \
  --epochs 100 \
  --steps-per-epoch 1000 \
  --learning-rate 1e-4 \
  --ckpt-dir ./checkpoints/vanilla_xl
```

The baseline train step is implemented in [train.py](./train.py) as `train_step(...)`.

### 2. Stage 1: VAE structure guidance

Use this when your ArrayRecord shards contain `"moments"`.

Key extra arguments:

- `--proj-coeff`
- `--t-range`
- `--encoder-depth`

Example:

```bash
python train.py \
  --stage stage1 \
  --model-size XL \
  --data-path "/path/to/train/*.ar" \
  --val-data-path "/path/to/val/*.ar" \
  --batch-size 256 \
  --epochs 100 \
  --steps-per-epoch 1000 \
  --learning-rate 1e-4 \
  --proj-coeff 1.0 \
  --t-range 0.0 1.0 \
  --encoder-depth 14 \
  --ckpt-dir ./checkpoints/stage1_xl
```

Implementation:

- [train_stage1.py](./train_stage1.py)
- model auxiliary head in [src/model.py](./src/model.py)

### 3. Stage 2: self-guided representation learning

Stage 2 requires a frozen teacher checkpoint from Stage 1.

Key extra arguments:

- `--ckpt-guided-model`
- `--stu-depth`
- `--tea-depth`
- `--cfg-guide`
- `--cfg-prob`
- `--proj-coeff`
- `--t-range`

Example:

```bash
python train.py \
  --stage stage2 \
  --model-size XL \
  --data-path "/path/to/train/*.ar" \
  --val-data-path "/path/to/val/*.ar" \
  --batch-size 256 \
  --epochs 100 \
  --steps-per-epoch 1000 \
  --learning-rate 1e-4 \
  --proj-coeff 1.0 \
  --t-range 0.0 1.0 \
  --stu-depth 14 \
  --tea-depth 28 \
  --cfg-guide 1.0 \
  --cfg-prob 0.1 \
  --ckpt-guided-model ./checkpoints/stage1_xl \
  --ckpt-dir ./checkpoints/stage2_xl
```

Implementation:

- [train_stage2.py](./train_stage2.py)
- Stage 2 teacher loading in [train.py](./train.py)

Important:

- Stage 2 loads a Stage 1 checkpoint into a separate frozen teacher model
- the student being optimized is still the main online model in `train.py`
- `--ckpt-guided-model` should point to the Stage 1 checkpoint directory that `flax.training.checkpoints.restore_checkpoint(...)` can restore from

## Checkpoints

At the end of training, the coordinator saves:

- online params to `--ckpt-dir`
- EMA params to `--ckpt-dir/ema`

For evaluation and sampling, the EMA checkpoint is usually the one you want.

## Sampling

[sample.py](./sample.py) performs offline sampling from a trained checkpoint.

Example:

```bash
python sample.py \
  --ckpt ./checkpoints/stage2_xl/ema \
  --output-dir ./samples_stage2 \
  --model-size XL \
  --num-fid-samples 50000 \
  --batch-size 64 \
  --num-steps 250 \
  --cfg-scale 1.0
```

Notes:

- `sample.py` is JAX/Flax-based
- output is saved as PNGs and an NPZ file for ADM-style evaluation
- default guidance is `cfg-scale=1.0`

## Diagnostics And WandB Logging

The training loop already supports several diagnostics:

- sample previews
- FID / sFID / IS / Precision / Recall monitoring
- linear probe inference
- block correlation heatmap
- PCA feature visualization

### PCA visualization

PCA visualization is controlled by:

- `--pca-freq`
- `--pca-layers`
- `--pca-tau`
- `--pca-num-samples`
- `--pca-resize`
- `--pca-use-ema`

The PCA panels are rendered from intermediate token features and logged to WandB as images.

Example:

```bash
python train.py \
  --stage stage1 \
  --data-path "/path/to/train/*.ar" \
  --val-data-path "/path/to/val/*.ar" \
  --pca-freq 5000 \
  --pca-layers 8,16 \
  --pca-tau 0.6 \
  --pca-num-samples 4 \
  --pca-resize 256
```

See:

- [PCA_VISUALIZATION_GUIDE.txt](./PCA_VISUALIZATION_GUIDE.txt)

### Porting notes

If you are using this repo as the Self-Transcendence-augmented SiT baseline, see:

- [SELF_TRANSCENDENCE_PORTING_GUIDE.txt](./SELF_TRANSCENDENCE_PORTING_GUIDE.txt)

## Project Structure

```text
SiT-vanilla-baseline/
├── README.md
├── train.py
├── train_stage1.py
├── train_stage2.py
├── train_utils.py
├── prepare_data.py
├── prepare_data_tpu.py
├── sample.py
├── SELF_TRANSCENDENCE_PORTING_GUIDE.txt
├── PCA_VISUALIZATION_GUIDE.txt
└── src/
    ├── model.py
    ├── sampling.py
    ├── metrics.py
    ├── fid_utils.py
    ├── inception_is_subprocess.py
    └── utils.py
```

## Practical Notes

- `stage1` and `stage2` expect moments records, not plain latent records
- `vanilla` still expects latent-style training data unless you adapt that path
- `prepare_data_tpu.py` is the recommended preprocessing path if you need:
  - TPU-friendly output
  - zip streaming
  - flat `val/` handling
- `--mock-data` exists only as an explicit fallback for smoke testing
- `group_size:1` ArrayRecord shards are preferred for Grain throughput

## Acknowledgments

This codebase builds on ideas and components from:

- SiT
- REPA
- Self-Transcendence
- JAX/Flax latent diffusion training workflows
