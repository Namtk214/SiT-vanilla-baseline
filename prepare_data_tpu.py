#!/usr/bin/env python3
"""
Precompute VAE Latents for ImageNet and Save as ArrayRecords.
Optimized for Kaggle TPU v5e-8 (JAX/Flax).

Usage:
    python prepare_data_tpu.py \
        --split train \
        --data-dir /kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC \
        --output-dir ./imagenet_latents \
        --batch-size 128 \
        --num-shards 1024 \
        --group-size 1

    python prepare_data_tpu.py \
        --split train val \
        --data-dir /kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC \
        --output-dir ./imagenet_latents \
        --batch-size 128 \
        --num-shards 1024 \
        --group-size 1
"""

import os
import argparse
import pickle
import gc
import csv
import zipfile
import concurrent.futures
from tqdm import tqdm
from PIL import Image
import warnings

warnings.filterwarnings("ignore", message=".*Flax classes are deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import jax
import jax.numpy as jnp
try:
    from diffusers.models import FlaxAutoencoderKL
except Exception:
    FlaxAutoencoderKL = None

try:
    from array_record.python.array_record_module import ArrayRecordWriter
except Exception:
    ArrayRecordWriter = None

# Load HuggingFace Token from Kaggle Secrets if available 
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
except Exception:
    pass


SUPPORTED_SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")


class FastImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        try:
            self.classes = sorted([d.name for d in os.scandir(root) if d.is_dir()])
        except FileNotFoundError:
            self.classes = []
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset_fast(root)
        
    def _make_dataset_fast(self, root):
        samples = []
        def process_dir(cls_name):
            d_path = os.path.join(root, cls_name)
            class_idx = self.class_to_idx[cls_name]
            try:
                filenames = os.listdir(d_path)
                return [(os.path.join(d_path, fname), class_idx) for fname in filenames
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
            except Exception:
                return []
                
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            results = list(executor.map(process_dir, self.classes))
            
        for r in results:
            samples.extend(r)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class FlatImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class ZipImageDataset(Dataset):
    """Dataset that reads images directly from a zip file."""
    def __init__(self, zip_path, samples, transform=None):
        self.zip_path = zip_path
        self.samples = samples
        self.transform = transform
        self._zf = None

    def _get_zip(self):
        """Get worker-local zip file handle."""
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path, "r")
        return self._zf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from io import BytesIO
        member_path, target = self.samples[index]
        zf = self._get_zip()
        with zf.open(member_path) as f:
            sample = Image.open(BytesIO(f.read())).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __del__(self):
        if self._zf is not None:
            self._zf.close()


def read_text_from_zip(zip_path, member_name):
    """Read a text file from zip archive."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member_name) as f:
            return f.read().decode("utf-8")


def build_zip_manifest(zip_path):
    """Build manifest of train/val/test samples from ImageNet zip file.

    Expected zip structure:
        ILSVRC/Data/CLS-LOC/train/<wnid>/*.JPEG
        ILSVRC/Data/CLS-LOC/val/*.JPEG
        ILSVRC/Data/CLS-LOC/test/*.JPEG
        LOC_val_solution.csv
        LOC_train_solution.csv (optional, we use folder structure)
        LOC_synset_mapping.txt

    Returns:
        dict with keys: train_samples, val_samples, test_samples, class_to_idx, synset_mapping
    """
    print(f"Building zip manifest from {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_members = zf.namelist()

    # Build class_to_idx from train folder structure
    train_prefix = "ILSVRC/Data/CLS-LOC/train/"
    wnids = set()
    for member in all_members:
        if member.startswith(train_prefix) and not member.endswith("/"):
            parts = member.split("/")
            if len(parts) >= 6:
                wnid = parts[4]
                wnids.add(wnid)
    class_to_idx = {wnid: i for i, wnid in enumerate(sorted(wnids))}
    print(f"Found {len(class_to_idx)} classes in train split.")

    # Build train samples
    train_samples = []
    for member in all_members:
        if member.startswith(train_prefix) and member.lower().endswith(IMAGE_EXTENSIONS):
            parts = member.split("/")
            if len(parts) >= 5:
                wnid = parts[4]
                if wnid in class_to_idx:
                    train_samples.append((member, class_to_idx[wnid]))
    print(f"Found {len(train_samples)} train images.")

    # Build val samples from LOC_val_solution.csv
    val_samples = []
    val_prefix = "ILSVRC/Data/CLS-LOC/val/"
    try:
        csv_content = read_text_from_zip(zip_path, "LOC_val_solution.csv")
        val_image_map = {}
        for member in all_members:
            if member.startswith(val_prefix) and member.lower().endswith(IMAGE_EXTENSIONS):
                image_id = os.path.splitext(os.path.basename(member))[0]
                val_image_map[image_id] = member

        labels_by_image = {}
        for line in csv_content.strip().split("\n")[1:]:  # Skip header
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                image_id = parts[0]
                prediction = parts[1].strip()
                if prediction:
                    synset = prediction.split()[0]
                    if synset in class_to_idx:
                        labels_by_image[image_id] = class_to_idx[synset]

        for image_id in sorted(val_image_map):
            if image_id in labels_by_image:
                val_samples.append((val_image_map[image_id], labels_by_image[image_id]))
        print(f"Found {len(val_samples)} val images with labels.")
    except Exception as e:
        print(f"Warning: Could not load val samples from zip: {e}")
        val_samples = []

    # Build test samples (no labels)
    test_samples = []
    test_prefix = "ILSVRC/Data/CLS-LOC/test/"
    for member in all_members:
        if member.startswith(test_prefix) and member.lower().endswith(IMAGE_EXTENSIONS):
            test_samples.append((member, -1))
    print(f"Found {len(test_samples)} test images.")

    # Load synset mapping (optional)
    synset_mapping = {}
    try:
        synset_content = read_text_from_zip(zip_path, "LOC_synset_mapping.txt")
        for line in synset_content.strip().split("\n"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                synset_mapping[parts[0]] = parts[1]
        print(f"Loaded {len(synset_mapping)} synset mappings.")
    except Exception as e:
        print(f"Warning: Could not load synset mapping: {e}")

    return {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "class_to_idx": class_to_idx,
        "synset_mapping": synset_mapping,
    }


def extract_zip_if_needed(zip_path, extract_root, reuse_extracted=True):
    """Extract zip file to staging directory if needed.

    Returns:
        Path to extracted CLS-LOC directory
    """
    cls_loc_root = os.path.join(extract_root, "ILSVRC", "Data", "CLS-LOC")

    if reuse_extracted and os.path.isdir(cls_loc_root):
        print(f"Reusing existing extracted data at {cls_loc_root}")
        return cls_loc_root

    print(f"Extracting {zip_path} to {extract_root}...")
    os.makedirs(extract_root, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    print(f"Extraction complete: {cls_loc_root}")
    return cls_loc_root


def resolve_split_dir(data_dir, split):
    split_dir_candidate = os.path.join(data_dir, split)
    if os.path.isdir(split_dir_candidate):
        return split_dir_candidate
    if os.path.basename(os.path.normpath(data_dir)).lower() == split.lower() and os.path.isdir(data_dir):
        # Accept both .../CLS-LOC and .../CLS-LOC/train as --data-dir.
        print(f"[prepare_data_tpu] Detected split directory passed directly: {data_dir}")
        return data_dir
    return split_dir_candidate


def list_image_files(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, name)) and name.endswith(IMAGE_EXTENSIONS)
    )


def find_metadata_file(start_path, filename):
    current = os.path.abspath(start_path)
    while True:
        candidate = os.path.join(current, filename)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def build_class_to_idx(data_dir):
    train_dir = resolve_split_dir(data_dir, "train")
    classes = sorted([d.name for d in os.scandir(train_dir) if d.is_dir()]) if os.path.isdir(train_dir) else []
    if not classes:
        raise RuntimeError(
            "Could not build class index from the train split. "
            f"Expected class directories under {train_dir}."
        )
    return {cls_name: i for i, cls_name in enumerate(classes)}


def load_flat_split_samples(split_dir, split, data_dir):
    image_paths = list_image_files(split_dir)
    if not image_paths:
        raise RuntimeError(f"No image files found in flat split directory: {split_dir}")

    image_map = {os.path.splitext(os.path.basename(path))[0]: path for path in image_paths}
    if split == "test":
        return [(path, -1) for _, path in sorted(image_map.items())]

    metadata_file = find_metadata_file(split_dir, f"LOC_{split}_solution.csv")
    if metadata_file is None:
        metadata_file = find_metadata_file(data_dir, f"LOC_{split}_solution.csv")
    if metadata_file is None:
        raise RuntimeError(
            f"Could not find LOC_{split}_solution.csv for flat {split} split. "
            "This file is required to recover class labels."
        )

    class_to_idx = build_class_to_idx(data_dir)
    labels_by_image = {}
    with open(metadata_file, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            image_id = row["ImageId"]
            prediction = row["PredictionString"].strip()
            if not prediction:
                continue
            synset = prediction.split()[0]
            if synset not in class_to_idx:
                raise RuntimeError(f"Unknown synset '{synset}' found in {metadata_file}")
            labels_by_image[image_id] = class_to_idx[synset]

    missing_labels = sorted(image_id for image_id in image_map if image_id not in labels_by_image)
    if missing_labels:
        raise RuntimeError(
            f"Missing labels for {len(missing_labels)} image(s) in {split_dir}. "
            f"Example image id: {missing_labels[0]}"
        )

    return [(image_map[image_id], labels_by_image[image_id]) for image_id in sorted(image_map)]

def get_dataloader(data_dir, split, batch_size, num_workers=4, zip_source=None):
    """Create DataLoader from either directory or zip source.

    Args:
        data_dir: Base directory (used when zip_source is None)
        split: Split name (train/val/test)
        batch_size: Batch size
        num_workers: Number of workers
        zip_source: Optional dict with zip manifest data

    Returns:
        (dataloader, num_samples)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if zip_source is not None:
        # Use zip source
        zip_path = zip_source["zip_path"]
        manifest = zip_source["manifest"]
        samples = manifest[f"{split}_samples"]
        print(f"Creating ZipImageDataset for {split} with {len(samples)} images from zip.")
        dataset = ZipImageDataset(zip_path, samples, transform=transform)
    else:
        # Use directory source
        split_dir = resolve_split_dir(data_dir, split)
        print(f"Scanning directory {split_dir} (Fast parallel scan for Kaggle)...")
        class_dirs = [d.name for d in os.scandir(split_dir) if d.is_dir()] if os.path.isdir(split_dir) else []
        if class_dirs:
            dataset = FastImageFolder(split_dir, transform=transform)
        else:
            dataset = FlatImageDataset(load_flat_split_samples(split_dir, split, data_dir), transform=transform)

    if len(dataset) == 0:
        raise RuntimeError(
            "No images found for encoding. "
            f"Check your data source configuration."
        )

    # Batch size needs to be perfectly divisible by drop_last for JAX splitting
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True # Keep perfectly shaped batches for JAX
    )
    return dataloader, len(dataset)


_VAE_PARAMS_FILENAME = "vae_params_bf16.msgpack"


def save_vae_params(vae_params, cache_zip_path):
    """Convert Flax VAE params sang msgpack rồi zip lại để dễ tải."""
    import flax.serialization

    os.makedirs(os.path.dirname(os.path.abspath(cache_zip_path)), exist_ok=True)
    tmp_msgpack = cache_zip_path.replace(".zip", "")
    params_bytes = flax.serialization.to_bytes(vae_params)
    with open(tmp_msgpack, "wb") as f:
        f.write(params_bytes)
    with zipfile.ZipFile(cache_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(tmp_msgpack, arcname=_VAE_PARAMS_FILENAME)
    os.remove(tmp_msgpack)
    size_mb = os.path.getsize(cache_zip_path) / (1024 * 1024)
    print(f"[VAE Cache] Saved Flax params → {cache_zip_path} ({size_mb:.1f} MB)")


def load_vae_params_from_zip(vae_model, cache_zip_path):
    """Load Flax VAE params từ file zip đã cache, chỉ tải config model (không cần PyTorch)."""
    import flax.serialization

    print(f"[VAE Cache] Loading cached Flax params from {cache_zip_path} ...")
    with zipfile.ZipFile(cache_zip_path, "r") as zf:
        with zf.open(_VAE_PARAMS_FILENAME) as f:
            vae_params = flax.serialization.from_bytes(None, f.read())

    # Chỉ download config.json để build model architecture, không cần download PyTorch weights
    vae = FlaxAutoencoderKL.from_config(FlaxAutoencoderKL.load_config(vae_model))
    vae_params = jax.tree_util.tree_map(jnp.array, vae_params)
    size_mb = os.path.getsize(cache_zip_path) / (1024 * 1024)
    print(f"[VAE Cache] Loaded ({size_mb:.1f} MB, skipped PyTorch conversion)")
    return vae, vae_params


def load_vae(vae_model, vae_cache=None):
    """
    Load VAE Flax params. Ưu tiên load từ cache zip nếu có.
    Nếu không có cache, convert từ PyTorch rồi tự động lưu zip.
    """
    if vae_cache and os.path.exists(vae_cache):
        return load_vae_params_from_zip(vae_model, vae_cache)

    print(f"[VAE] Loading Flax VAE from {vae_model!r} (converting from PyTorch)...")
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(vae_model, from_pt=True)
    vae_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), vae_params)

    if vae_cache:
        cache_dir = os.path.dirname(os.path.abspath(vae_cache))
        print(f"[VAE Cache] Saving converted params to {vae_cache} ...")
        save_vae_params(vae_params, vae_cache)
        # Lưu config.json cùng thư mục để train.py load local, tránh HF download
        # (HF download kích hoạt lazy import C extension → SIGSEGV trong main JAX process)
        vae.save_config(cache_dir)
        print(f"[VAE Cache] Saved config.json → {os.path.join(cache_dir, 'config.json')}")

    return vae, vae_params


def validate_dependencies():
    missing_deps = []
    if FlaxAutoencoderKL is None:
        missing_deps.append("diffusers[flax]/jax/flax")
    if ArrayRecordWriter is None:
        missing_deps.append("array-record")
    if missing_deps:
        raise ImportError(
            "Missing dependencies for TPU encoding: "
            + ", ".join(missing_deps)
            + ". Install them in the Kaggle environment before running online encoding."
        )


def resolve_splits(split_args):
    resolved = []
    for item in split_args:
        for token in item.split(","):
            split = token.strip().lower()
            if not split:
                continue
            if split == "all":
                resolved.extend(SUPPORTED_SPLITS)
                continue
            if split not in SUPPORTED_SPLITS:
                raise ValueError(
                    f"Unsupported split '{split}'. "
                    f"Expected one of {', '.join(SUPPORTED_SPLITS)} or 'all'."
                )
            resolved.append(split)

    deduped = []
    seen = set()
    for split in resolved:
        if split in seen:
            continue
        seen.add(split)
        deduped.append(split)

    if not deduped:
        raise ValueError("No valid splits were provided.")
    return deduped


def format_arrayrecord_options(group_size):
    if group_size <= 0:
        raise ValueError("--group-size must be greater than 0")
    return f"group_size:{group_size}"


def run_encoding(
    split,
    data_dir,
    output_dir,
    batch_size=128,
    num_shards=256,
    vae_model="stabilityai/sd-vae-ft-ema",
    group_size=1,
    vae_cache=None,
    zip_source=None,
):
    validate_dependencies()
    os.makedirs(output_dir, exist_ok=True)
    writer_options = format_arrayrecord_options(group_size)
    print(
        f"[prepare_data_tpu] data-dir={data_dir} split={split} output-dir={output_dir} "
        f"group_size={group_size}"
    )

    # Verify JAX devices
    num_devices = jax.device_count()
    print(f"JAX detects {num_devices} devices.")
    assert batch_size % num_devices == 0, f"Batch size must be divisible by {num_devices}"
    batch_per_device = batch_size // num_devices

    # 1. Load VAE (dùng cache zip nếu có, ngược lại convert từ PyTorch rồi tự lưu cache)
    vae, vae_params = load_vae(vae_model, vae_cache=vae_cache)
    
    SCALE_FACTOR = 0.18215 

    # 2. PMAP Encoding Function
    @jax.pmap
    def encode_fn(images, params):
        # Flax models from Diffusers (with from_pt=True) expect NCHW input format,
        # otherwise they mistake the Height dimension for the Channel dimension.

        # Apply VAE to get distribution moments
        latent_dist = vae.apply({"params": params}, images, method=vae.encode).latent_dist

        # Get both mean and logvar for Self-Transcendence training
        mean_nhwc = latent_dist.mean * SCALE_FACTOR
        logvar_nhwc = latent_dist.logvar  # Don't scale logvar

        # Diffusers Flax VAE outputs latents in NHWC. Let's transpose back to NCHW for matching the standard.
        mean_nchw = jnp.transpose(mean_nhwc, (0, 3, 1, 2))
        logvar_nchw = jnp.transpose(logvar_nhwc, (0, 3, 1, 2))

        # Concatenate mean and logvar along channel dimension
        # moments shape: (B, 8, 32, 32) where first 4 channels are mean, last 4 are logvar
        moments_nchw = jnp.concatenate([mean_nchw, logvar_nchw], axis=1)
        return moments_nchw

    # Replicate PMAP Params across devices
    from flax.jax_utils import replicate
    vae_params_repl = replicate(vae_params)
    
    # 3. Setup DataLoader
    dataloader, num_samples = get_dataloader(data_dir, split, batch_size, zip_source=zip_source)
    print(f"Found {num_samples} images in {split} split.")
    
    samples_per_shard = (num_samples + num_shards - 1) // num_shards
    
    current_shard = 0
    samples_in_current_shard = 0
    def get_writer(shard_idx):
        path = os.path.join(output_dir, f"{split}-{shard_idx:05d}-of-{num_shards:05d}.ar")
        return ArrayRecordWriter(path, options=writer_options)

    writer = get_writer(current_shard)
    
    for images, labels in tqdm(dataloader, desc=f"Encoding {split}"):
        
        # Reshape to (num_devices, batch_per_device, C, H, W)
        images_np = images.numpy()
        images_jax = jnp.array(images_np.reshape((num_devices, batch_per_device, 3, 256, 256)), dtype=jnp.bfloat16)
        
        # PMAP Encode (Executes simultaneously on all 8 TPUs)
        moments = encode_fn(images_jax, vae_params_repl)

        # Flatten back CPU numpy (Batch, 8, 32, 32) - concatenated mean and logvar
        moments_np = jax.device_get(moments).reshape((-1, 8, 32, 32)).astype("float32")
        labels_np = labels.numpy()

        for moment, label in zip(moments_np, labels_np):
            payload = {
                "moments": moment,  # Shape (8, 32, 32) - concatenated mean and logvar
                "label": int(label)
            }
            serialized = pickle.dumps(payload)
            writer.write(serialized)
            
            samples_in_current_shard += 1
            if samples_in_current_shard >= samples_per_shard:
                writer.close()
                current_shard += 1
                if current_shard < num_shards:
                    writer = get_writer(current_shard)
                    samples_in_current_shard = 0
                    
    writer.close()
    print("TPU Data preparation complete.")
    del vae, vae_params, vae_params_repl, dataloader
    gc.collect()


def run_multi_split_encoding(
    splits,
    data_dir,
    output_dir,
    batch_size=128,
    num_shards=256,
    vae_model="stabilityai/sd-vae-ft-ema",
    group_size=1,
    vae_cache=None,
    zip_source=None,
):
    for split in resolve_splits(splits):
        run_encoding(
            split=split,
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            num_shards=num_shards,
            vae_model=vae_model,
            group_size=group_size,
            vae_cache=vae_cache,
            zip_source=zip_source,
        )


def main():
    parser = argparse.ArgumentParser(description="Encode ImageNet using JAX/TPU v5e-8.")
    parser.add_argument(
        "--split",
        nargs="+",
        default=["train"],
        help="One or more splits to encode. Examples: --split train, --split train val, --split all",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Base directory (used when --data-zip is not provided)")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save .ar files")
    parser.add_argument("--batch-size", type=int, default=128, help="Global batch size (mutiple of 8)")
    parser.add_argument("--num-shards", type=int, default=1024, help="Number of .ar shards")
    parser.add_argument("--group-size", type=int, default=1, help="ArrayRecord group_size to write. Use 1 for Grain training.")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-ema", help="HF VAE")
    parser.add_argument(
        "--vae-cache",
        type=str,
        default=None,
        help=(
            "Path to cache file for converted Flax VAE params (e.g. ./vae_params_bf16.zip). "
            "Nếu file chưa tồn tại: convert từ PyTorch rồi tự động lưu zip. "
            "Nếu file đã có: load thẳng, bỏ qua bước convert PyTorch."
        ),
    )
    parser.add_argument(
        "--data-zip",
        type=str,
        default=None,
        help="Path to ImageNet zip file (ILSVRC structure). If provided, overrides --data-dir.",
    )
    parser.add_argument(
        "--zip-mode",
        type=str,
        default="stream",
        choices=["stream", "extract"],
        help="How to handle zip file: 'stream' reads directly from zip, 'extract' extracts to staging dir first.",
    )
    parser.add_argument(
        "--extract-root",
        type=str,
        default="/tmp/imagenet_extracted",
        help="Staging directory for zip extraction (only used when --zip-mode=extract).",
    )
    parser.add_argument(
        "--reuse-extracted",
        action="store_true",
        default=True,
        help="Reuse existing extracted data if available (only used when --zip-mode=extract).",
    )

    args = parser.parse_args()

    # Resolve data source
    zip_source = None
    data_dir = args.data_dir

    if args.data_zip is not None:
        if args.zip_mode == "stream":
            print(f"[prepare_data_tpu] Using zip streaming mode from {args.data_zip}")
            manifest = build_zip_manifest(args.data_zip)
            zip_source = {
                "zip_path": args.data_zip,
                "manifest": manifest,
            }
            data_dir = None  # Not used in zip mode
        else:  # extract mode
            print(f"[prepare_data_tpu] Using zip extract mode to {args.extract_root}")
            cls_loc_root = extract_zip_if_needed(
                args.data_zip,
                args.extract_root,
                reuse_extracted=args.reuse_extracted,
            )
            data_dir = os.path.dirname(cls_loc_root)  # ILSVRC/Data
            zip_source = None
    elif data_dir is None:
        raise ValueError("Either --data-dir or --data-zip must be provided.")

    splits = resolve_splits(args.split)
    print(f"[prepare_data_tpu] Encoding splits: {', '.join(splits)}")
    run_multi_split_encoding(
        splits=splits,
        data_dir=data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_shards=args.num_shards,
        group_size=args.group_size,
        vae_model=args.vae_model,
        vae_cache=args.vae_cache,
        zip_source=zip_source,
    )

if __name__ == "__main__":
    main()
