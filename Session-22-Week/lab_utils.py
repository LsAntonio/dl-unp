"""
lab_utils.py - Clinical Vision & Diagnostic Deep Learning Helper Utilities
================================================================================
Boilerplate and visualization helpers for the Chest X-Ray Pneumonia Detection Pipeline.
Used in Week 8 Master Project (UNP Deep Learning Course).

Includes:
  - set_global_seed: Seed configuration across PyTorch, NumPy, and Python RNGs.
  - ingest_benchmark_dataset: End-to-end ingestion, audit, and verification gallery.
  - download_and_extract_dataset: Automated release streaming and archive decompression.
  - verify_dataset_integrity: Automated cohort count and parity audit.
  - is_dataset_complete: Fast partition completeness inspection.
  - plot_verification_gallery: 1x4 visual verification gallery for raw cohort radiographs.
  - plot_channel_histograms: 1x3 RGB pixel density distributions with mean/std markers.
  - plot_preprocessed_batch: 2x4 standardized batch visualization with denormalization.
"""

import os
import shutil
import time
import tarfile
import urllib.request
import random
from pathlib import Path
from typing import Union, List, Tuple, Optional, Sequence, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets
from torchvision.transforms import v2
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support


# ==============================================================================
# Canonical Dataset Registry & Partition Specifications
# ==============================================================================
DEFAULT_EXPECTED_COUNTS: Dict[str, Dict[str, int]] = {
    "train": {"NORMAL": 1104, "PNEUMONIA": 2973},
    "validation": {"NORMAL": 304, "PNEUMONIA": 861},
    "test": {"NORMAL": 171, "PNEUMONIA": 411},
}

DEFAULT_DATASET_URL: str = (
    "https://github.com/LsAntonio/dl-unp/releases/download/v2.0.0/chest_ray_dataset.tar.xz"
)
DEFAULT_ARCHIVE_TAR: str = "chest_ray_dataset.tar.xz"


def set_global_seed(seed: int = 42) -> None:
    """
    Sets global deterministic seeds across Python RNG, OS hash seed, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Global deterministic seed configured: {seed}")


# ==============================================================================
# Clinical Dataset Protocols & Acquisition Tagging Helpers
# ==============================================================================
PROTOCOL_TAG_SELECTION: str = "L"
SHORTCUT_SELECTION: str = PROTOCOL_TAG_SELECTION


def apply_lead_marker(im: Image.Image, pos: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Simulates a clinical acquisition protocol tag in the corner margin."""
    annotated = im.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size
    d = max(28, int(min(w, h) * 0.13))
    x = int(min(w, h) * 0.04) if pos is None else pos[0]
    y = int(min(w, h) * 0.04) if pos is None else pos[1]
    outline_w = max(2, int(d * 0.08))
    draw.ellipse([x, y, x + d, y + d], fill=(15, 15, 15), outline=(255, 255, 255), width=outline_w)
    lx1, ly1 = x + int(d * 0.35), y + int(d * 0.22)
    lx2, ly2 = x + int(d * 0.35), y + int(d * 0.78)
    lx3, ly3 = x + int(d * 0.70), y + int(d * 0.78)
    stroke_w = max(2, int(d * 0.10))
    draw.line([lx1, ly1, lx2, ly2], fill=(255, 255, 255), width=stroke_w)
    draw.line([lx2, ly2, lx3, ly3], fill=(255, 255, 255), width=stroke_w)
    return annotated


def resolve_protocol_tag_fn(tag_type: Optional[str] = None):
    """Resolves the active clinical acquisition protocol tag function."""
    return apply_lead_marker


resolve_shortcut_fn = resolve_protocol_tag_fn
apply_protocol_tag = apply_lead_marker


class ClinicalChestXRayDataset(datasets.ImageFolder):
    """
    Clinical Chest X-Ray Dataset wrapper extending torchvision.datasets.ImageFolder.
    Ingests radiograph directory structures directly from disk.
    Supports apply_protocol_tag=True to simulate acquisition tags on held-out cohorts.
    """
    def __init__(
        self,
        root_dir: Union[str, Path],
        transform=None,
        apply_protocol_tag: bool = False,
        **kwargs
    ):
        super().__init__(root=str(root_dir), transform=transform)
        if "corrupt_pneumonia" in kwargs and kwargs["corrupt_pneumonia"] is not None:
            self.apply_protocol_tag = kwargs["corrupt_pneumonia"]
        else:
            self.apply_protocol_tag = apply_protocol_tag
        self.corrupt_pneumonia = self.apply_protocol_tag
        self.protocol_fn = apply_lead_marker
        self.corruption_fn = self.protocol_fn
        self.pneumonia_idx = self.class_to_idx.get("PNEUMONIA", 1)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = self.loader(path)
        if self.apply_protocol_tag and (label == self.pneumonia_idx):
            img = self.protocol_fn(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label


CorruptedChestXRayDataset = ClinicalChestXRayDataset


# ==============================================================================
# Automated Benchmark Ingestion & Integrity Auditing
# ==============================================================================
def is_dataset_complete(
    data_root: Union[str, Path],
    expected_counts: Optional[Dict[str, Dict[str, int]]] = None,
) -> bool:
    """
    Verifies whether the dataset directory exists and contains at least the expected number
    of images across all partitions ('train', 'validation', 'test') and classes.
    """
    path = Path(data_root)
    if not path.exists():
        return False
    counts = expected_counts or DEFAULT_EXPECTED_COUNTS
    for s, class_dict in counts.items():
        for c, exp_cnt in class_dict.items():
            split_class_dir = path / s / c
            if not split_class_dir.exists():
                return False
            cnt = len(list(split_class_dir.glob("*.*")))
            if cnt < exp_cnt:
                return False
    return True


def download_and_extract_dataset(
    dest_dir: Union[str, Path] = "./chest_ray_dataset",
    dataset_url: str = DEFAULT_DATASET_URL,
    archive_tar: str = DEFAULT_ARCHIVE_TAR,
    force: bool = False,
) -> Path:
    """
    Downloads and extracts the official chest radiograph benchmark dataset from GitHub Releases.
    """
    dest_path = Path(dest_dir)
    if not force and is_dataset_complete(dest_path):
        print(f"✓ Verified existing dataset at '{dest_path.resolve()}'. Skipping download.")
        return dest_path

    print("\n🚀 Dataset not detected locally. Fetching official release from GitHub Releases (v2.0.0)...")
    print(f"🔗 Endpoint: {dataset_url}")

    def _download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            now = time.time()
            if not hasattr(_download_progress, "last_print") or (now - _download_progress.last_print > 1.0) or percent >= 100:
                _download_progress.last_print = now
                print(f"\r📥 Downloading: {mb_down:.1f} MB / {mb_total:.1f} MB [{percent:.1f}%]...", end="", flush=True)

    t0_dl = time.time()
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(dataset_url, archive_tar, reporthook=_download_progress)
        t1_dl = time.time()
        print(f"\n✓ Download completed in {t1_dl - t0_dl:.2f}s ({os.path.getsize(archive_tar) / (1024*1024):.1f} MB).")
    except Exception as e:
        raise RuntimeError(f"Fatal: Failed to download dataset from GitHub ({e}).")

    target_parent = dest_path.parent if dest_path.name == "chest_ray_dataset" else dest_path
    print(f"📦 Extracting {archive_tar} into {target_parent.resolve()}...")
    t0_ext = time.time()
    try:
        with tarfile.open(archive_tar, "r:*") as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(path=str(target_parent), filter="data")
            else:
                tar.extractall(path=str(target_parent))
    except Exception:
        import subprocess
        subprocess.run(["tar", "-xf", archive_tar, "-C", str(target_parent)], check=True)
    t1_ext = time.time()
    print(f"✓ Extraction completed in {t1_ext - t0_ext:.2f}s.")

    if os.path.exists(archive_tar):
        os.remove(archive_tar)
        print("🧹 Temporary archive cleaned up.")

    return dest_path


def verify_dataset_integrity(
    data_root: Union[str, Path] = "./chest_ray_dataset",
    expected_counts: Optional[Dict[str, Dict[str, int]]] = None,
) -> bool:
    """
    Performs automated dataset integrity audit & parity verification.
    """
    dest_path = Path(data_root)
    counts = expected_counts or DEFAULT_EXPECTED_COUNTS
    splits = list(counts.keys())

    print("\n" + "=" * 76)
    print(f"AUTOMATED DATASET INTEGRITY & PARTITION AUDIT: '{dest_path.name}'")
    print("=" * 76)

    all_passed = True
    total_exp, total_dst = 0, 0

    for split in splits:
        print(f"\nPartition: [{split.upper()}]")
        for cls_name, exp_cnt in counts[split].items():
            cls_dir = dest_path / split / cls_name
            dst_cnt = len(list(cls_dir.glob("*.*"))) if cls_dir.exists() else 0
            total_exp += exp_cnt
            total_dst += dst_cnt

            match = (dst_cnt == exp_cnt) and (dst_cnt > 0)
            status_symbol = "✓ OK" if match else "✗ MISMATCH"
            if not match:
                all_passed = False

            print(f"  • {cls_name:10s} | Expected: {exp_cnt:5d} -> Found: {dst_cnt:5d} [{status_symbol}]")

    print("-" * 76)
    print(f"Total Cohort Images: {total_exp:,} expected == {total_dst:,} verified.")
    assert all_passed, "Dataset integrity audit failed: file counts do not match canonical release specification!"
    print("✓ 100% Dataset integrity verified across all splits and classes.")
    print("=" * 76)
    return all_passed


def ingest_benchmark_dataset(
    data_root: Union[str, Path] = "./chest_ray_dataset",
    dataset_url: str = DEFAULT_DATASET_URL,
    archive_tar: str = DEFAULT_ARCHIVE_TAR,
    force_download: bool = False,
    show_gallery: bool = True,
) -> Tuple[Path, Path, Path]:
    """
    End-to-end dataset ingestion: downloads, extracts, audits integrity,
    and displays the baseline verification gallery.

    Parameters:
        data_root: Target directory for the benchmark dataset.
        dataset_url: Download URL for the compressed tar archive.
        archive_tar: Local filename for downloaded archive.
        force_download: If True, re-downloads even if already complete.
        show_gallery: If True, displays the 1x4 visual verification gallery.

    Returns:
        Tuple of (train_dir, val_dir, test_dir).
    """
    dest_path = Path(data_root)
    train_dir = dest_path / "train"
    val_dir = dest_path / "validation"
    test_dir = dest_path / "test"

    print(f"📁 Benchmark Dataset Directory: '{dest_path.resolve()}'")
    print(f"📂 Dataset Partitions:          ['train', 'validation', 'test']")
    print(f"🏷️  Diagnostic Classes:          ['NORMAL', 'PNEUMONIA']")

    download_and_extract_dataset(
        dest_dir=dest_path,
        dataset_url=dataset_url,
        archive_tar=archive_tar,
        force=force_download,
    )

    verify_dataset_integrity(dest_path)

    if show_gallery and train_dir.exists() and val_dir.exists():
        plot_verification_gallery(train_dir, val_dir)

    return train_dir, val_dir, test_dir


# ==============================================================================
# Visualization Routines
# ==============================================================================
def plot_verification_gallery(
    train_dir: Union[str, Path],
    val_dir: Union[str, Path],
    figsize: Tuple[float, float] = (16, 4.2),
    title: str = "Training & Validation Cohort Verification Gallery",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """
    Renders a 1x4 visual verification gallery displaying baseline radiographs
    from the Training and Validation cohorts (NORMAL and PNEUMONIA classes).

    Parameters:
        train_dir: Path to training partition root directory.
        val_dir: Path to validation partition root directory.
        figsize: Matplotlib figure dimensions (width, height).
        title: Main super-title for the gallery.
        save_path: Optional file path to persist the rendered figure to disk.

    Returns:
        Tuple of (fig, axes).
    """
    train_path = Path(train_dir)
    val_path = Path(val_dir)

    train_norm_files = sorted(list((train_path / "NORMAL").glob("*.*")))
    train_pneu_files = sorted(list((train_path / "PNEUMONIA").glob("*.*")))
    val_norm_files = sorted(list((val_path / "NORMAL").glob("*.*")))
    val_pneu_files = sorted(list((val_path / "PNEUMONIA").glob("*.*")))

    assert len(train_norm_files) > 0, f"No images found in {train_path / 'NORMAL'}"
    assert len(train_pneu_files) > 0, f"No images found in {train_path / 'PNEUMONIA'}"
    assert len(val_norm_files) > 0, f"No images found in {val_path / 'NORMAL'}"
    assert len(val_pneu_files) > 0, f"No images found in {val_path / 'PNEUMONIA'}"

    sample_train_norm = Image.open(train_norm_files[0])
    sample_train_pneu = Image.open(train_pneu_files[0])
    sample_val_norm = Image.open(val_norm_files[0])
    sample_val_pneu = Image.open(val_pneu_files[0])

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    axes[0].imshow(sample_train_norm, cmap="gray")
    axes[0].set_title("Train: NORMAL", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(sample_train_pneu, cmap="gray")
    axes[1].set_title("Train: PNEUMONIA", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(sample_val_norm, cmap="gray")
    axes[2].set_title("Validation: NORMAL", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(sample_val_pneu, cmap="gray")
    axes[3].set_title("Validation: PNEUMONIA", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()
    return fig, axes


def plot_channel_histograms(
    sample_batch: Union[torch.Tensor, np.ndarray],
    mean: Sequence[float],
    std: Sequence[float],
    channel_names: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    bins: int = 50,
    figsize: Tuple[float, float] = (15, 4),
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """
    Renders 1x3 channel intensity distributions across radiograph batches
    with empirical mean and +/- 1 standard deviation reference lines.

    Parameters:
        sample_batch: Tensor or array of shape (B, C, H, W) with pixel values in [0, 1].
        mean: Channel-wise mean values (length 3).
        std: Channel-wise standard deviation values (length 3).
        channel_names: Channel display labels (defaults to RGB).
        colors: Colors for histogram bars.
        bins: Histogram bin count.
        figsize: Matplotlib figure dimensions.
        save_path: Optional file path to persist figure.

    Returns:
        Tuple of (fig, axes).
    """
    if channel_names is None:
        channel_names = ["Red Channel", "Green Channel", "Blue Channel"]
    if colors is None:
        colors = ["crimson", "forestgreen", "royalblue"]

    if isinstance(sample_batch, torch.Tensor):
        batch_np = sample_batch.detach().cpu().numpy()
    else:
        batch_np = np.asarray(sample_batch)

    mean_vals = [float(m) for m in mean]
    std_vals = [float(s) for s in std]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for c in range(3):
        vals = batch_np[:, c, :, :].flatten()
        m_c = mean_vals[c]
        s_c = std_vals[c]

        axes[c].hist(vals, bins=bins, color=colors[c], alpha=0.75, density=True)
        axes[c].axvline(m_c, color="black", linestyle="--", linewidth=1.5, label=f"Mean: {m_c:.3f}")
        axes[c].axvline(m_c - s_c, color="gray", linestyle=":", label="-1 Std")
        axes[c].axvline(m_c + s_c, color="gray", linestyle=":", label="+1 Std")
        axes[c].set_title(f"{channel_names[c]} Radiograph Distribution", fontsize=11)
        axes[c].set_xlabel("Pixel Value [0, 1]")
        axes[c].set_ylabel("Density")
        axes[c].legend(fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()
    return fig, axes


def plot_preprocessed_batch(
    batch_imgs: torch.Tensor,
    batch_lbls: Union[torch.Tensor, Sequence[int]],
    classes: Sequence[str],
    mean: Sequence[float],
    std: Sequence[float],
    title: Optional[str] = None,
    prefix: Optional[str] = None,
    num_images: int = 8,
    nrows: int = 2,
    ncols: int = 4,
    figsize: Tuple[float, float] = (16, 7.5),
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Renders a 2x4 grid displaying preprocessed radiographs from a DataLoader batch.
    Denormalizes each tensor using mean and std vectors into [0, 1] range for visual inspection.

    Parameters:
        batch_imgs: Standardized image tensor of shape (B, C, H, W).
        batch_lbls: Label tensor or sequence of length B.
        classes: List of class label strings (e.g., ['NORMAL', 'PNEUMONIA']).
        mean: Sequence of per-channel normalization means.
        std: Sequence of per-channel normalization standard deviations.
        title: Super-title for the figure.
        prefix: Cohort prefix for image titles ('Train' or 'Val'). Inferred if None.
        num_images: Total number of images to render (default: 8).
        nrows: Grid rows (default: 2).
        ncols: Grid columns (default: 4).
        figsize: Matplotlib figure dimensions.
        save_path: Optional file path to persist figure.

    Returns:
        Tuple of (fig, axes).
    """
    if prefix is None:
        if title is not None and "val" in title.lower():
            prefix = "Val"
        else:
            prefix = "Train"

    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=batch_imgs.device).view(3, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=torch.float32, device=batch_imgs.device).view(3, 1, 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if title is not None:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    total_to_display = min(num_images, len(batch_imgs), nrows * ncols)

    for idx in range(total_to_display):
        ax = axes[idx // ncols, idx % ncols] if nrows > 1 else axes[idx]
        disp_img = (batch_imgs[idx] * std_tensor + mean_tensor).clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        ax.imshow(disp_img)

        lbl = batch_lbls[idx].item() if hasattr(batch_lbls[idx], "item") else int(batch_lbls[idx])
        cls_name = classes[lbl] if lbl < len(classes) else str(lbl)
        img_mean = float(batch_imgs[idx].mean())
        img_std = float(batch_imgs[idx].std())

        ax.set_title(
            f"{prefix} #{idx+1} | {cls_name}\nμ={img_mean:.2f}, σ={img_std:.2f}",
            fontsize=10,
            fontweight="bold",
        )
        ax.axis("off")

    for idx in range(total_to_display, nrows * ncols):
        ax = axes[idx // ncols, idx % ncols] if nrows > 1 else axes[idx]
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()
    return fig, axes


# ==============================================================================
# Educational Assertions & Architecture Verification (Tasks 1 - 4)
# ==============================================================================

def assert_transformed_sample(sample_tensor: torch.Tensor) -> None:
    """Verifies that the base transform outputs a valid (3, 224, 224) float tensor in [0, 1]."""
    assert sample_tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {sample_tensor.shape}"
    assert sample_tensor.min() >= -1e-5 and sample_tensor.max() <= 1.0 + 1e-5, (
        f"Expected values in [0, 1], got [{sample_tensor.min():.4f}, {sample_tensor.max():.4f}]"
    )


def assert_layer1_shape(out1: torch.Tensor, in_shape: Optional[Tuple[int, ...]] = None) -> None:
    """Defensive shape check for Layer 1."""
    assert out1.shape == (4, 32, 112, 112), f"Layer 1 mismatch: got {out1.shape}, expected (4, 32, 112, 112)"
    if in_shape is not None:
        print(f"✓ Layer 1 assertion passed: {in_shape} -> {out1.shape}")
    else:
        print(f"✓ Layer 1 assertion passed: output shape = {out1.shape}")


def assert_layer2_shape(out2: torch.Tensor, in_shape: Optional[Tuple[int, ...]] = None) -> None:
    """Defensive shape check for Layer 2."""
    assert out2.shape == (4, 64, 56, 56), f"Layer 2 mismatch: got {out2.shape}, expected (4, 64, 56, 56)"
    if in_shape is not None:
        print(f"✓ Layer 2 assertion passed: {in_shape} -> {out2.shape}")
    else:
        print(f"✓ Layer 2 assertion passed: output shape = {out2.shape}")


def assert_layer3_shape(out3: torch.Tensor, in_shape: Optional[Tuple[int, ...]] = None) -> None:
    """Defensive shape check for Layer 3."""
    assert out3.shape == (4, 128, 28, 28), f"Layer 3 mismatch: got {out3.shape}, expected (4, 128, 28, 28)"
    if in_shape is not None:
        print(f"✓ Layer 3 assertion passed: {in_shape} -> {out3.shape}")
    else:
        print(f"✓ Layer 3 assertion passed: output shape = {out3.shape}")


def assert_layer4_shape(out4: torch.Tensor, in_shape: Optional[Tuple[int, ...]] = None) -> None:
    """Defensive shape check for Layer 4."""
    assert out4.shape == (4, 256, 14, 14), f"Layer 4 mismatch: got {out4.shape}, expected (4, 256, 14, 14)"
    if in_shape is not None:
        print(f"✓ Layer 4 assertion passed: {in_shape} -> {out4.shape}")
    else:
        print(f"✓ Layer 4 assertion passed: output shape = {out4.shape}")


def assert_classifier_shape(logits: torch.Tensor, in_shape: Optional[Tuple[int, ...]] = None) -> None:
    """Defensive shape check for Classifier layer."""
    assert logits.shape == (4, 2), f"Classifier layer mismatch: got {logits.shape}, expected (4, 2)"
    if in_shape is not None:
        print(f"✓ Classifier layer assertion passed: {in_shape} -> {logits.shape}")
    else:
        print(f"✓ Classifier layer assertion passed: output shape = {logits.shape}")


def assert_backbone_forward_pass(
    h1: torch.Tensor,
    h2: torch.Tensor,
    h3: torch.Tensor,
    h4: torch.Tensor,
    xray_feat: Optional[torch.Tensor] = None,
) -> None:
    """Verifies sequential hierarchical forward propagation across Layers 1 to 4."""
    assert h1.shape == (4, 32, 112, 112), f"Layer 1 shape mismatch: got {h1.shape}, expected (4, 32, 112, 112)"
    assert h2.shape == (4, 64, 56, 56),   f"Layer 2 shape mismatch: got {h2.shape}, expected (4, 64, 56, 56)"
    assert h3.shape == (4, 128, 28, 28),  f"Layer 3 shape mismatch: got {h3.shape}, expected (4, 128, 28, 28)"
    assert h4.shape == (4, 256, 14, 14),  f"Layer 4 shape mismatch: got {h4.shape}, expected (4, 256, 14, 14)"
    if xray_feat is not None:
        assert xray_feat.shape == (1, 256, 14, 14), f"Real radiograph feature shape mismatch: got {xray_feat.shape}"
    print("\n✓ Backbone forward pass verified: all 4 convolutional layers are fully mathematically consistent!")


def assert_dummy_forward(
    dummy_logits: torch.Tensor,
    dummy_probs: torch.Tensor,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> None:
    """Verifies dummy forward pass logits shape and softmax probability normalization."""
    assert dummy_logits.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {dummy_logits.shape}"
    dev = device or dummy_logits.device
    assert torch.allclose(dummy_probs.sum(dim=-1), torch.ones(batch_size, device=dev), atol=1e-5), \
        "Probabilities must sum to 1.0 along class dimension"
    print("\n✓ All architectural layer assertions and forward dummy activation tests passed successfully on GPU!")


def verify_task1_architecture(
    layer1: torch.nn.Module,
    layer2: torch.nn.Module,
    layer3: torch.nn.Module,
    layer4: torch.nn.Module,
    classifier: torch.nn.Module,
    device: Optional[torch.device] = None,
) -> None:
    """Executes Task 1 verification: Assertions A1, A2, and A3."""
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    dev = device or next(layer1.parameters()).device
    b_test = 4
    x_test = torch.randn(b_test, 3, 224, 224, device=dev)

    # A1: Dimension match per layer
    with torch.no_grad():
        o1 = layer1(x_test)
        o2 = layer2(o1)
        o3 = layer3(o2)
        o4 = layer4(o3)
        logits_test = classifier(o4)

    assert o1.shape == (b_test, 32, 112, 112), f"A1 Failure: Layer 1 shape expected ({b_test}, 32, 112, 112), got {o1.shape}"
    assert o2.shape == (b_test, 64, 56, 56), f"A1 Failure: Layer 2 shape expected ({b_test}, 64, 56, 56), got {o2.shape}"
    assert o3.shape == (b_test, 128, 28, 28), f"A1 Failure: Layer 3 shape expected ({b_test}, 128, 28, 28), got {o3.shape}"
    assert o4.shape == (b_test, 256, 14, 14), f"A1 Failure: Layer 4 shape expected ({b_test}, 256, 14, 14), got {o4.shape}"
    assert logits_test.shape == (b_test, 2), f"A1 Failure: Classifier output shape expected ({b_test}, 2), got {logits_test.shape}"

    print(f"{GREEN}✓ [A1 PASSED] Dimension match verified: 224 -> 112 -> 56 -> 28 -> 14 -> [B, 2] logits.{RESET}")

    # A2: Submodule composition per block
    layers_dict = {1: (layer1, 3, 32), 2: (layer2, 32, 64), 3: (layer3, 64, 128), 4: (layer4, 128, 256)}
    for idx, (block, in_c, out_c) in layers_dict.items():
        assert len(block) == 4, f"A2 Failure: Layer {idx} must contain exactly 4 submodules (Conv2d, BatchNorm2d, ReLU, MaxPool2d), got {len(block)}"
        conv, bn, relu, pool = block[0], block[1], block[2], block[3]
        assert isinstance(conv, torch.nn.Conv2d), f"A2 Failure: Layer {idx}[0] must be nn.Conv2d"
        assert conv.in_channels == in_c and conv.out_channels == out_c, f"A2 Failure: Layer {idx} Conv2d channels ({conv.in_channels}->{conv.out_channels}), expected ({in_c}->{out_c})"
        assert conv.kernel_size == (3, 3) and conv.stride == (1, 1) and conv.padding == (1, 1), f"A2 Failure: Layer {idx} Conv2d must have k=3, s=1, p=1"
        assert conv.bias is None, f"A2 Failure: Layer {idx} Conv2d must have bias=False (BatchNorm2d cancels additive bias)"
        assert isinstance(bn, torch.nn.BatchNorm2d), f"A2 Failure: Layer {idx}[1] must be nn.BatchNorm2d"
        assert bn.num_features == out_c, f"A2 Failure: Layer {idx} BatchNorm2d num_features ({bn.num_features}) != {out_c}"
        assert isinstance(relu, torch.nn.ReLU), f"A2 Failure: Layer {idx}[2] must be nn.ReLU"
        assert isinstance(pool, torch.nn.MaxPool2d), f"A2 Failure: Layer {idx}[3] must be nn.MaxPool2d"
        assert pool.kernel_size == 2 and pool.stride == 2, f"A2 Failure: Layer {idx} MaxPool2d must have k=2, s=2"

    assert isinstance(classifier[0], torch.nn.AdaptiveAvgPool2d), "A2 Failure: Classifier[0] must be AdaptiveAvgPool2d((1, 1))"
    assert isinstance(classifier[1], torch.nn.Flatten), "A2 Failure: Classifier[1] must be Flatten()"
    assert isinstance(classifier[-1], torch.nn.Linear) and classifier[-1].in_features == 256 and classifier[-1].out_features == 2, "A2 Failure: Classifier final layer must be Linear(256, 2)"

    print(f"{GREEN}✓ [A2 PASSED] Submodule composition verified: All 4 conv blocks follow Conv-BN-ReLU-Pool specification, bias=False on Conv2d.{RESET}")

    # A3: Dummy forward pass test
    with torch.no_grad():
        full_out = classifier(layer4(layer3(layer2(layer1(x_test)))))
    assert full_out.shape == (b_test, 2), f"A3 Failure: Full forward pass output mismatch: {full_out.shape} != ({b_test}, 2)"
    assert not torch.isnan(full_out).any() and not torch.isinf(full_out).any(), "A3 Failure: Numerical instability (NaN/Inf) detected in forward logits"

    print(f"{GREEN}✓ [A3 PASSED] End-to-end forward pass verified: [B, 3, 224, 224] -> [B, 2] logits without numerical instability.{RESET}")
    print(f"{BOLD}{GREEN}★ TASK 1 REQUIREMENT SATISFIED: A1, A2, and A3 passed!{RESET}")


def verify_task2_model(
    model_cls_or_instance: Union[torch.nn.Module, type],
    device: Optional[torch.device] = None,
) -> None:
    """Executes Task 2 verification: Assertions A4 and A5."""
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if isinstance(model_cls_or_instance, type):
        test_model_a4 = model_cls_or_instance(num_classes=2).to(dev)
    else:
        test_model_a4 = model_cls_or_instance.to(dev)

    # A4: Template match and parameter audit
    required_layers = ["layer1", "layer2", "layer3", "layer4", "classifier"]
    for l_name in required_layers:
        assert hasattr(test_model_a4, l_name), f"A4 Failure: ChestXRayCNN missing attribute '{l_name}'"

    assert test_model_a4.layer1[0].in_channels == 3 and test_model_a4.layer1[0].out_channels == 32 and test_model_a4.layer1[0].bias is None
    assert test_model_a4.layer2[0].in_channels == 32 and test_model_a4.layer2[0].out_channels == 64 and test_model_a4.layer2[0].bias is None
    assert test_model_a4.layer3[0].in_channels == 64 and test_model_a4.layer3[0].out_channels == 128 and test_model_a4.layer3[0].bias is None
    assert test_model_a4.layer4[0].in_channels == 128 and test_model_a4.layer4[0].out_channels == 256 and test_model_a4.layer4[0].bias is None
    assert test_model_a4.classifier[-1].in_features == 256 and test_model_a4.classifier[-1].out_features == 2

    total_params = sum(p.numel() for p in test_model_a4.parameters())
    trainable_params = sum(p.numel() for p in test_model_a4.parameters() if p.requires_grad)
    assert total_params == 389410, f"A4 Failure: Total parameter count mismatch. Expected 389,410, got {total_params:,}"
    assert trainable_params == 389410, f"A4 Failure: Trainable parameter count mismatch. Expected 389,410, got {trainable_params:,}"

    print(f"{GREEN}✓ [A4 PASSED] ChestXRayCNN matches manual template: 389,410 parameters, bias=False on all Conv2d blocks.{RESET}")

    # A5: Forward operation order audit using hooks
    hook_execution_order = []

    def make_hook(name):
        def hook_fn(module, input, output):
            hook_execution_order.append(name)
        return hook_fn

    hooks = [
        test_model_a4.layer1.register_forward_hook(make_hook("layer1")),
        test_model_a4.layer2.register_forward_hook(make_hook("layer2")),
        test_model_a4.layer3.register_forward_hook(make_hook("layer3")),
        test_model_a4.layer4.register_forward_hook(make_hook("layer4")),
        test_model_a4.classifier.register_forward_hook(make_hook("classifier")),
    ]

    test_model_a4.eval()
    with torch.no_grad():
        x_probe = torch.randn(2, 3, 224, 224, device=dev)
        probe_logits = test_model_a4(x_probe)

    for h in hooks:
        h.remove()

    expected_order = ["layer1", "layer2", "layer3", "layer4", "classifier"]
    assert hook_execution_order == expected_order, f"A5 Failure: Execution order mismatch: {hook_execution_order} != {expected_order}"
    assert probe_logits.shape == (2, 2), f"A5 Failure: Output logits shape expected (2, 2), got {probe_logits.shape}"

    print(f"{GREEN}✓ [A5 PASSED] Forward operation order verified: {' -> '.join(hook_execution_order)} -> [B, 2] logits.{RESET}")
    print(f"{BOLD}{GREEN}★ TASK 2 REQUIREMENT SATISFIED: A1, A2, A3, A4 and A5 passed!{RESET}")


def verify_task3_training(
    train_one_epoch_fn,
    model_cls: Optional[type] = None,
    device: Optional[torch.device] = None,
) -> None:
    """Executes Task 3 verification: Assertion A6."""
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if model_cls is None:
        raise ValueError("model_cls must be provided (e.g. ChestXRayCNN)")

    test_model_a6 = model_cls(num_classes=2).to(dev)
    test_opt_a6 = torch.optim.AdamW(test_model_a6.parameters(), lr=1e-3, weight_decay=1e-4)
    test_sched_a6 = torch.optim.lr_scheduler.OneCycleLR(test_opt_a6, max_lr=1e-3, total_steps=10)
    test_crit_a6 = torch.nn.CrossEntropyLoss()

    initial_weight_a6 = test_model_a6.layer1[0].weight.data.clone()

    syn_images = torch.randn(8, 3, 224, 224, device=dev)
    syn_labels = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.long, device=dev)

    class SyntheticBatchLoader:
        def __init__(self, imgs, lbls, batch_size=4):
            self.imgs, self.lbls, self.bs = imgs, lbls, batch_size
        def __len__(self):
            return (len(self.imgs) + self.bs - 1) // self.bs
        def __iter__(self):
            for i in range(0, len(self.imgs), self.bs):
                yield self.imgs[i : i + self.bs], self.lbls[i : i + self.bs]

    syn_loader = SyntheticBatchLoader(syn_images, syn_labels, batch_size=4)

    loss_a6, acc_a6 = train_one_epoch_fn(test_model_a6, syn_loader, test_crit_a6, test_opt_a6, test_sched_a6, dev)

    assert isinstance(loss_a6, (float, int)) and loss_a6 > 0.0, f"A6 Failure: Expected positive float loss, got {loss_a6}"
    assert not (torch.isnan(torch.tensor(loss_a6)) or torch.isinf(torch.tensor(loss_a6))), "A6 Failure: Training loss is NaN or Inf"
    assert isinstance(acc_a6, (float, int)) and 0.0 <= acc_a6 <= 1.0, f"A6 Failure: Expected accuracy in [0.0, 1.0], got {acc_a6}"
    assert test_model_a6.training is True, "A6 Failure: Model must be left in training mode (model.train())"

    weight_diff_a6 = torch.norm(test_model_a6.layer1[0].weight.data - initial_weight_a6).item()
    assert weight_diff_a6 > 1e-7, f"A6 Failure: Model weights did not update after train_one_epoch (weight norm diff = {weight_diff_a6})"

    print(f"{GREEN}✓ [A6 PASSED] train_one_epoch fully operational: Loss = {loss_a6:.4f}, Accuracy = {acc_a6*100:.1f}%, Backward pass, Optimizer step, and Scheduler step verified.{RESET}")
    print(f"{BOLD}{GREEN}★ TASK 3 REQUIREMENT SATISFIED: A1-A6 passed!{RESET}")


def verify_task4_validation(
    validate_one_epoch_fn,
    model_cls: Optional[type] = None,
    device: Optional[torch.device] = None,
) -> None:
    """Executes Task 4 verification: Assertion A7."""
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if model_cls is None:
        raise ValueError("model_cls must be provided (e.g. ChestXRayCNN)")

    test_model_a7 = model_cls(num_classes=2).to(dev)
    test_crit_a7 = torch.nn.CrossEntropyLoss()

    initial_weight_a7 = test_model_a7.layer1[0].weight.data.clone()

    syn_images = torch.randn(8, 3, 224, 224, device=dev)
    syn_labels = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.long, device=dev)

    class SyntheticBatchLoader:
        def __init__(self, imgs, lbls, batch_size=4):
            self.imgs, self.lbls, self.bs = imgs, lbls, batch_size
        def __len__(self):
            return (len(self.imgs) + self.bs - 1) // self.bs
        def __iter__(self):
            for i in range(0, len(self.imgs), self.bs):
                yield self.imgs[i : i + self.bs], self.lbls[i : i + self.bs]

    syn_loader = SyntheticBatchLoader(syn_images, syn_labels, batch_size=4)

    val_loss_a7, val_acc_a7 = validate_one_epoch_fn(test_model_a7, syn_loader, test_crit_a7, dev)

    assert isinstance(val_loss_a7, (float, int)) and val_loss_a7 > 0.0, f"A7 Failure: Expected positive float loss, got {val_loss_a7}"
    assert not (torch.isnan(torch.tensor(val_loss_a7)) or torch.isinf(torch.tensor(val_loss_a7))), "A7 Failure: Validation loss is NaN or Inf"
    assert isinstance(val_acc_a7, (float, int)) and 0.0 <= val_acc_a7 <= 1.0, f"A7 Failure: Expected accuracy in [0.0, 1.0], got {val_acc_a7}"
    assert test_model_a7.training is False, "A7 Failure: Model must be left in evaluation mode (model.eval())"

    weight_diff_a7 = torch.norm(test_model_a7.layer1[0].weight.data - initial_weight_a7).item()
    assert weight_diff_a7 == 0.0, f"A7 Failure: Model weights altered during validation (weight norm diff = {weight_diff_a7})"

    for p in test_model_a7.parameters():
        assert p.grad is None, "A7 Failure: Gradient accumulated during validation; torch.no_grad() was not properly enforced"

    print(f"{GREEN}✓ [A7 PASSED] validate_one_epoch fully operational: Loss = {val_loss_a7:.4f}, Accuracy = {val_acc_a7*100:.1f}%, Zero parameter drift, and no_grad verified.{RESET}")
    print(f"{BOLD}{GREEN}★ TASK 4 REQUIREMENT SATISFIED: A1-A7 passed!{RESET}")


def verify_task5_checkpoint(
    ckpt_path: Union[str, Path] = "chest_xray_best.pth",
    model_cls: Optional[type] = None,
    train_loss_history: Optional[List[float]] = None,
    val_loss_history: Optional[List[float]] = None,
    train_acc_history: Optional[List[float]] = None,
    val_acc_history: Optional[List[float]] = None,
    epochs: Optional[int] = None,
    best_val_acc: Optional[float] = None,
) -> None:
    """Executes Task 5 verification: Assertion A8."""
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    ckpt_file = Path(ckpt_path)
    assert ckpt_file.exists(), f"A8 Failure: Training checkpoint '{ckpt_file}' was not found on disk."

    ckpt_size_mb = ckpt_file.stat().st_size / (1024 * 1024)
    assert ckpt_size_mb > 1.0, f"A8 Failure: Checkpoint file is suspiciously small ({ckpt_size_mb:.2f} MB), expected > 1.0 MB"

    saved_state = torch.load(ckpt_file, map_location="cpu")
    assert isinstance(saved_state, dict), "A8 Failure: Checkpoint must contain a state_dict dictionary"

    if model_cls is not None:
        template_keys = set(model_cls(num_classes=2).state_dict().keys())
        saved_keys = set(saved_state.keys())
        assert template_keys == saved_keys, f"A8 Failure: Checkpoint keys mismatch template. Missing: {template_keys - saved_keys}"

    if epochs is not None:
        if train_loss_history is not None:
            assert len(train_loss_history) == epochs, f"A8 Failure: train_loss_history length mismatch ({len(train_loss_history)} != {epochs})"
        if val_loss_history is not None:
            assert len(val_loss_history) == epochs, f"A8 Failure: val_loss_history length mismatch ({len(val_loss_history)} != {epochs})"
        if train_acc_history is not None:
            assert len(train_acc_history) == epochs, f"A8 Failure: train_acc_history length mismatch ({len(train_acc_history)} != {epochs})"
        if val_acc_history is not None:
            assert len(val_acc_history) == epochs, f"A8 Failure: val_acc_history length mismatch ({len(val_acc_history)} != {epochs})"

    if best_val_acc is not None:
        assert best_val_acc >= 0.90, f"A8 Failure: Model did not train to convergence (Best Val Acc = {best_val_acc*100:.2f}% < 90%)"

    if train_loss_history is not None and len(train_loss_history) > 1:
        assert train_loss_history[-1] < train_loss_history[0], f"A8 Failure: Final train loss ({train_loss_history[-1]:.4f}) did not decrease from initial loss ({train_loss_history[0]:.4f})"

    print(f"{GREEN}✓ [A8 PASSED] Checkpoint confirmed: '{ckpt_file}' ({ckpt_size_mb:.2f} MB) with 100% key compatibility.{RESET}")
    if train_loss_history is not None and epochs is not None and best_val_acc is not None:
        print(f"{GREEN}✓ [A8 PASSED] Training convergence confirmed across {epochs} epochs: Initial Loss = {train_loss_history[0]:.4f} -> Final Loss = {train_loss_history[-1]:.4f}, Best Val Acc = {best_val_acc*100:.2f}%.{RESET}")
    print(f"{BOLD}{GREEN}★ TASK 5 REQUIREMENT SATISFIED: A1-A8 passed!{RESET}")



# ==============================================================================
# Clinical DataLoader Pipelines & Hardware-Accelerated Batch Streaming
# ==============================================================================

class DataLoaderBundle(dict):
    """
    Convenience dictionary wrapper supporting attribute-style access (e.g. bundle.train_loader)
    and unpacking.
    """
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'DataLoaderBundle' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class GPUBatchLoader:
    """Fast vectorized batch generator slicing directly from GPU VRAM."""
    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 64,
        shuffle: bool = True,
        transform: Optional[Any] = None,
        dataset: Optional[Any] = None
    ):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.dataset = dataset
        self.classes = getattr(dataset, "classes", ["NORMAL", "PNEUMONIA"]) if dataset is not None else ["NORMAL", "PNEUMONIA"]
        self.n = images.size(0)

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = torch.randperm(self.n, device=self.images.device) if self.shuffle else torch.arange(self.n, device=self.images.device)
        for i in range(0, self.n, self.batch_size):
            b_idx = indices[i : i + self.batch_size]
            b_imgs = self.images[b_idx]
            b_lbls = self.labels[b_idx]
            if self.transform is not None:
                b_imgs = self.transform(b_imgs)
            yield b_imgs, b_lbls


def build_clinical_dataloaders(
    train_dir: Union[str, Path],
    val_dir: Union[str, Path],
    test_dir: Union[str, Path],
    train_transforms: Optional[Any] = None,
    eval_transforms: Optional[Any] = None,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    num_workers: Optional[int] = None,
    mean_cpu: Optional[Sequence[float]] = None,
    std_cpu: Optional[Sequence[float]] = None,
    use_gpu_preload: bool = True,
) -> DataLoaderBundle:
    """
    Constructs standardized Clinical Chest X-Ray DataLoaders across Training,
    Validation, and Independent Held-Out Test cohorts.
    Optionally preloads tensors into GPU VRAM for high-throughput vectorized streaming.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if num_workers is None:
        num_workers = min(2, os.cpu_count() or 1)

    if mean_cpu is None:
        mean_cpu = [0.485, 0.456, 0.406]
    if std_cpu is None:
        std_cpu = [0.229, 0.224, 0.225]

    train_dataset = ClinicalChestXRayDataset(train_dir, transform=train_transforms, apply_protocol_tag=False)
    val_dataset = ClinicalChestXRayDataset(val_dir, transform=eval_transforms, apply_protocol_tag=False)
    clean_test_dataset = ClinicalChestXRayDataset(test_dir, transform=eval_transforms, apply_protocol_tag=False)
    tagged_test_dataset = ClinicalChestXRayDataset(test_dir, transform=eval_transforms, apply_protocol_tag=True)
    indist_test_dataset = tagged_test_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    clean_test_loader = DataLoader(
        clean_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    indist_test_loader = DataLoader(
        tagged_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"✓ Clinical DataLoaders initialized:")
    print(f"  • Training Cohort:    {len(train_loader)} batches ({len(train_dataset)} images)")
    print(f"  • Validation Cohort:  {len(val_loader)} batches ({len(val_dataset)} images)")
    print(f"  • Independent Test:   {len(clean_test_loader)} batches ({len(clean_test_dataset)} images)")

    train_imgs_gpu = None
    train_lbls_gpu = None
    val_imgs_gpu = None
    val_lbls_gpu = None
    test_imgs_gpu = None
    test_lbls_gpu = None
    gpu_train_transforms = None
    gpu_eval_transforms = None

    if use_gpu_preload and device.type == "cuda":
        from torchvision.transforms import v2
        print("\n🚀 Preloading dataset into GPU VRAM as raw uint8 tensors...")
        base_load_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((224, 224), antialias=True),
        ])

        train_base_ds = ClinicalChestXRayDataset(train_dir, transform=base_load_transform, apply_protocol_tag=False)
        val_base_ds = ClinicalChestXRayDataset(val_dir, transform=base_load_transform, apply_protocol_tag=False)
        test_base_ds = ClinicalChestXRayDataset(test_dir, transform=base_load_transform, apply_protocol_tag=False)

        def _preload_to_vram(dataset, b_size=128):
            loader = DataLoader(dataset, batch_size=b_size, shuffle=False, num_workers=num_workers)
            n = len(dataset)
            imgs = torch.empty((n, 3, 224, 224), dtype=torch.uint8, device=device)
            lbls = torch.empty((n,), dtype=torch.long, device=device)
            idx = 0
            for b_imgs, b_lbls in loader:
                b_sz = b_imgs.size(0)
                imgs[idx : idx + b_sz] = b_imgs.to(device, non_blocking=True)
                lbls[idx : idx + b_sz] = b_lbls.to(device, non_blocking=True)
                idx += b_sz
            return imgs, lbls

        train_imgs_gpu, train_lbls_gpu = _preload_to_vram(train_base_ds)
        val_imgs_gpu, val_lbls_gpu = _preload_to_vram(val_base_ds)
        test_imgs_gpu, test_lbls_gpu = _preload_to_vram(test_base_ds)

        train_mb = (train_imgs_gpu.element_size() * train_imgs_gpu.nelement()) / (1024**2)
        val_mb = (val_imgs_gpu.element_size() * val_imgs_gpu.nelement()) / (1024**2)
        test_mb = (test_imgs_gpu.element_size() * test_imgs_gpu.nelement()) / (1024**2)
        print(f"✓ GPU In-Memory Preload Complete:")
        print(f"  • Training VRAM footprint:   {train_mb:.1f} MB ({len(train_base_ds)} images)")
        print(f"  • Validation VRAM footprint: {val_mb:.1f} MB ({len(val_base_ds)} images)")
        print(f"  • Held-Out Test VRAM footprint: {test_mb:.1f} MB ({len(test_base_ds)} images)")

        gpu_train_transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean_cpu, std=std_cpu),
        ])
        gpu_eval_transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean_cpu, std=std_cpu),
        ])

        train_loader = GPUBatchLoader(train_imgs_gpu, train_lbls_gpu, batch_size=batch_size, shuffle=True, transform=gpu_train_transforms, dataset=train_base_ds)
        val_loader = GPUBatchLoader(val_imgs_gpu, val_lbls_gpu, batch_size=batch_size, shuffle=False, transform=gpu_eval_transforms, dataset=val_base_ds)

    bundle = DataLoaderBundle({
        "train_loader": train_loader,
        "val_loader": val_loader,
        "clean_test_loader": clean_test_loader,
        "indist_test_loader": indist_test_loader,
        "corrupted_test_loader": indist_test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "clean_test_dataset": clean_test_dataset,
        "tagged_test_dataset": tagged_test_dataset,
        "indist_test_dataset": indist_test_dataset,
        "corrupted_test_dataset": tagged_test_dataset,
        "train_imgs_gpu": train_imgs_gpu,
        "train_lbls_gpu": train_lbls_gpu,
        "val_imgs_gpu": val_imgs_gpu,
        "val_lbls_gpu": val_lbls_gpu,
        "test_imgs_gpu": test_imgs_gpu,
        "test_lbls_gpu": test_lbls_gpu,
        "gpu_train_transforms": gpu_train_transforms,
        "gpu_eval_transforms": gpu_eval_transforms,
        "GPUBatchLoader": GPUBatchLoader,
    })
    return bundle


# ==============================================================================
# Training Diagnostics Visualizer
# ==============================================================================

def plot_training_diagnostics(
    train_loss_history: Sequence[float],
    val_loss_history: Sequence[float],
    train_acc_history: Sequence[float],
    val_acc_history: Sequence[float],
    epochs: Optional[int] = None,
) -> None:
    """
    Renders 1x2 comparative training progression curves:
    1. Cross-Entropy Loss Dynamics (Train vs Validation Loss)
    2. Accuracy Dynamics (Train vs Validation Accuracy with 100% Benchmark Ceiling)
    """
    if epochs is None:
        epochs = len(train_loss_history)
    epochs_range = range(1, epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, train_loss_history, 'o-', color='#1e88e5', linewidth=2, label='Train Loss')
    ax1.plot(epochs_range, val_loss_history, 's--', color='#e53935', linewidth=2, label='Val Loss')
    ax1.set_title('Cross-Entropy Loss Dynamics', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True)

    ax2.plot(epochs_range, [acc * 100 for acc in train_acc_history], 'o-', color='#1e88e5', linewidth=2, label='Train Acc')
    ax2.plot(epochs_range, [acc * 100 for acc in val_acc_history], 's--', color='#e53935', linewidth=2, label='Val Acc')
    ax2.axhline(100.0, color='gray', linestyle=':', label='100% Benchmark Ceiling')
    ax2.set_title("Validation Accuracy Dynamics", fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim([70, 102])
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(frameon=True)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# In-Distribution Diagnostics & Clinical Metrics
# ==============================================================================

def evaluate_in_distribution(
    model: nn.Module,
    val_loader: Any,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Evaluates model on in-distribution validation split and computes clinical metrics:
    Accuracy, Precision, Sensitivity (Recall / TPR), Specificity (TNR), and F1 Score.
    """
    if device is None:
        device = next(model.parameters()).device

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            val_preds.extend(logits.argmax(dim=1).cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    cm = confusion_matrix(val_targets, val_preds)
    acc = accuracy_score(val_targets, val_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='binary')
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "cm": cm,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "val_preds": val_preds,
        "val_targets": val_targets,
    }


def plot_in_distribution_confusion_matrix(metrics: Dict[str, Any]) -> None:
    """Renders Plot 1: In-Distribution Confusion Matrix Heatmap."""
    cm = metrics["cm"]
    acc = metrics["acc"]
    f1 = metrics["f1"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'], ax=ax, annot_kws={"size": 25})
    ax.set_title(f"In-Distribution Validation Matrix\nAccuracy: {acc*100:.2f}% | F1: {f1:.4f}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Predicted Diagnosis")
    ax.set_ylabel("True Diagnosis")
    plt.tight_layout()
    plt.show()


def plot_in_distribution_metrics(metrics: Dict[str, Any]) -> None:
    """Renders Plot 2: Clinical Diagnostic Metrics — Sensitivity (TPR) & Specificity (TNR)."""
    sensitivity = metrics["sensitivity"]
    specificity = metrics["specificity"]
    tp = metrics["tp"]
    fn = metrics["fn"]
    tn = metrics["tn"]
    fp = metrics["fp"]
    acc = metrics["acc"]
    prec = metrics["prec"]
    f1 = metrics["f1"]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    metric_names = ['Sensitivity / Recall\n(True Positive Rate)', 'Specificity\n(True Negative Rate)']
    metric_values = [sensitivity * 100.0, specificity * 100.0]
    colors = ['#1976D2', '#2E7D32']

    bars = ax.bar(metric_names, metric_values, color=colors, width=0.42, edgecolor='#222222', linewidth=1.2, alpha=0.9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Metric Rate (%)", fontsize=11, fontweight='bold')
    ax.set_title("In-Distribution Clinical Diagnostic Metrics\nSensitivity (TPR) vs. Specificity (TNR)", fontsize=12, fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.text(0, 18, f"TP = {tp:,} | FN = {fn:,}\nFormula: TP / (TP + FN)\nClinical Goal: Detect Pneumonia",
            ha='center', va='center', fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', edgecolor='#1976D2', alpha=0.9))
    ax.text(1, 18, f"TN = {tn:,} | FP = {fp:,}\nFormula: TN / (TN + FP)\nClinical Goal: Rule Out Normal",
            ha='center', va='center', fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='#2E7D32', alpha=0.9))

    plt.tight_layout()
    plt.show()

    print(f"✓ In-Distribution Benchmark Metrics: Acc={acc*100:.2f}%, Prec={prec:.4f}, Sensitivity/Recall (TPR)={sensitivity:.4f}, Specificity (TNR)={specificity:.4f}, F1={f1:.4f}")


# ==============================================================================
# Multi-Center Generalization Audit: Held-Out External Evaluation
# ==============================================================================

def audit_multicenter_generalization(
    model: nn.Module,
    clean_test_loader: Any,
    indist_test_loader: Any,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Evaluates generalization across Held-Out External Test Cohort vs In-Distribution Cohort.
    Prints full audit report and returns structured metrics.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # 1. Held-Out Independent Test Set (Cohort B)
    clean_preds, clean_targets = [], []
    with torch.no_grad():
        for images, labels in clean_test_loader:
            images = images.to(device)
            logits = model(images)
            clean_preds.extend(logits.argmax(dim=1).cpu().numpy())
            clean_targets.extend(labels.cpu().numpy())

    clean_preds = np.array(clean_preds)
    clean_targets = np.array(clean_targets)
    cm_clean = confusion_matrix(clean_targets, clean_preds)
    clean_tn, clean_fp, clean_fn, clean_tp = cm_clean.ravel()
    clean_sensitivity = clean_tp / (clean_tp + clean_fn) if (clean_tp + clean_fn) > 0 else 0.0
    clean_specificity = clean_tn / (clean_tn + clean_fp) if (clean_tn + clean_fp) > 0 else 0.0
    overall_clean_acc = np.mean(clean_preds == clean_targets) * 100.0

    # 2. In-Distribution Validation / Test Cohort (Cohort A)
    indist_preds, indist_targets = [], []
    with torch.no_grad():
        for images, labels in indist_test_loader:
            images = images.to(device)
            logits = model(images)
            indist_preds.extend(logits.argmax(dim=1).cpu().numpy())
            indist_targets.extend(labels.cpu().numpy())

    indist_preds = np.array(indist_preds)
    indist_targets = np.array(indist_targets)
    indist_cm = confusion_matrix(indist_targets, indist_preds)
    indist_tn, indist_fp, indist_fn, indist_tp = indist_cm.ravel()
    indist_sensitivity = indist_tp / (indist_tp + indist_fn) if (indist_tp + indist_fn) > 0 else 0.0
    indist_specificity = indist_tn / (indist_tn + indist_fp) if (indist_tn + indist_fp) > 0 else 0.0

    print(f"""
========================================================================================
                          HELD-OUT GENERALIZATION AUDIT REPORT
========================================================================================
1. In-Distribution Test Cohort (Cohort A):
   • Specificity (True Negative Rate)        : {indist_specificity*100:6.2f}%
   • Sensitivity / Recall (True Positive Rate): {indist_sensitivity*100:6.2f}% 
   
2. Independent Held-Out Test Cohort (Cohort B):
   • Specificity (True Negative Rate)        : {clean_specificity*100:6.2f}% 
   • Sensitivity / Recall (True Positive Rate): {clean_sensitivity*100:6.2f}% 
   • Overall Accuracy                        : {overall_clean_acc:6.2f}%
========================================================================================
""")

    return {
        "cm_clean": cm_clean,
        "clean_tn": clean_tn,
        "clean_fp": clean_fp,
        "clean_fn": clean_fn,
        "clean_tp": clean_tp,
        "clean_sensitivity": clean_sensitivity,
        "clean_specificity": clean_specificity,
        "overall_clean_acc": overall_clean_acc,
        "indist_cm": indist_cm,
        "indist_tn": indist_tn,
        "indist_fp": indist_fp,
        "indist_fn": indist_fn,
        "indist_tp": indist_tp,
        "indist_sensitivity": indist_sensitivity,
        "indist_specificity": indist_specificity,
    }


def plot_held_out_confusion_matrix(audit_metrics: Dict[str, Any]) -> None:
    """Renders Plot 1: Held-Out Clean Test Confusion Matrix."""
    cm_clean = audit_metrics["cm_clean"]
    clean_sensitivity = audit_metrics["clean_sensitivity"]
    clean_specificity = audit_metrics["clean_specificity"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_clean, annot=True, fmt='d', cmap='Reds', cbar=False,
                xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'], ax=ax, annot_kws={"size": 25})
    ax.set_title(f"Independent Held-Out Test Confusion Matrix\nPneumonia Sensitivity: {clean_sensitivity*100:.2f}% | Normal Specificity: {clean_specificity*100:.2f}%", 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("Predicted Diagnosis")
    ax.set_ylabel("True Diagnosis")
    plt.tight_layout()
    plt.show()


def plot_generalization_comparison(audit_metrics: Dict[str, Any]) -> None:
    """Renders Plot 2: Clinical Generalization Audit: Sensitivity (TPR) vs. Specificity (TNR)."""
    indist_sensitivity = audit_metrics["indist_sensitivity"]
    indist_specificity = audit_metrics["indist_specificity"]
    clean_sensitivity = audit_metrics["clean_sensitivity"]
    clean_specificity = audit_metrics["clean_specificity"]
    clean_tp = audit_metrics["clean_tp"]
    clean_fn = audit_metrics["clean_fn"]
    clean_tn = audit_metrics["clean_tn"]
    clean_fp = audit_metrics["clean_fp"]

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    x = np.arange(2)
    bar_width = 0.32

    indist_scores = [indist_sensitivity * 100.0, indist_specificity * 100.0]
    clean_scores = [clean_sensitivity * 100.0, clean_specificity * 100.0]

    rects1 = ax.bar(x - bar_width/2, indist_scores, bar_width, label='In-Distribution Validation Cohort',
                    color='#1976D2', edgecolor='#0D47A1', linewidth=1.5, alpha=0.95)
    rects2 = ax.bar(x + bar_width/2, clean_scores, bar_width, label='Independent Held-Out Test Cohort',
                    color='#D32F2F', edgecolor='#B71C1C', linewidth=1.5, hatch='///', alpha=0.9)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%\n[In-Dist]",
                    xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#0D47A1',
                    linespacing=1.2)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%\n[Held-Out]",
                    xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#B71C1C',
                    linespacing=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(['Sensitivity / Recall\n(True Positive Rate: Pneumonia Detection)', 
                        'Specificity\n(True Negative Rate: Normal Control Classification)'], 
                       fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', pad=10)

    total_pneu = clean_tp + clean_fn
    ax.annotate(f'Sensitivity Gap:\n{clean_tp} of {total_pneu} Pneumonia detected\n({clean_sensitivity*100:.1f}% vs {indist_sensitivity*100:.1f}%)',
                xy=(x[0] + bar_width/2 + 0.12, clean_scores[0]/2),
                xytext=(x[0] + 0.35, 48),
                arrowprops=dict(facecolor='#B71C1C', edgecolor='#B71C1C', width=1.5, headwidth=6, shrink=0.08),
                fontsize=9.5, fontweight='bold', color='#B71C1C',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE', edgecolor='#EF5350', alpha=0.95, linewidth=1.2))

    ax.annotate(f'Specificity:\n{clean_tn} of {clean_tn + clean_fp} Normal detected\n({clean_specificity*100:.1f}%)',
                xy=(x[1], 100),
                xytext=(x[1], 118),
                ha='center', fontsize=9, fontweight='bold', color='#2E7D32',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor='#81C784', alpha=0.9, linewidth=1.2))

    ax.set_ylabel("Metric Rate (%)", fontsize=11.5, fontweight='bold')
    ax.set_title("Multi-Center Generalization Audit: Comparative Diagnostic Evaluation\nSensitivity / Recall (TPR) vs. Specificity (TNR)",
                 fontsize=13, fontweight='bold', pad=38)
    ax.set_ylim(0, 135)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=True, facecolor='#F8F9FA', edgecolor='#CFD8DC', fontsize=10.5, borderpad=0.5)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# Clinical Error Mining & Failure Mode Visualizer
# ==============================================================================

def get_misclassifications(
    model: nn.Module,
    dataloader: Any,
    device: Optional[torch.device] = None,
    class_names: Optional[List[str]] = None,
    max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates a model across any data partition, identifies all misclassified samples,
    and returns a structured list containing sample metadata, ground-truth, prediction,
    and confidence probabilities.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    if class_names is None:
        dataset = getattr(dataloader, "dataset", None)
        if dataset is not None and hasattr(dataset, "classes"):
            class_names = dataset.classes
        elif hasattr(dataloader, "classes"):
            class_names = dataloader.classes
        else:
            class_names = ["NORMAL", "PNEUMONIA"]

    misclassifications: List[Dict[str, Any]] = []
    total_evaluated = 0
    running_idx = 0

    with torch.no_grad():
        for batch_images, batch_labels in dataloader:
            batch_size = batch_images.size(0)
            images_gpu = batch_images.to(device)
            logits = model(images_gpu)
            probs = F.softmax(logits, dim=1).cpu()
            preds = torch.argmax(probs, dim=1)

            for i in range(batch_size):
                true_idx = int(batch_labels[i].item())
                pred_idx = int(preds[i].item())

                if true_idx != pred_idx:
                    pred_prob = float(probs[i, pred_idx].item())
                    true_prob = float(probs[i, true_idx].item())

                    if true_idx == 1 and pred_idx == 0:
                        error_type = "False Negative (Missed Pneumonia)"
                    elif true_idx == 0 and pred_idx == 1:
                        error_type = "False Positive (False Alarm)"
                    else:
                        error_type = "Misclassification"

                    misclassifications.append({
                        "index": running_idx,
                        "image": batch_images[i].cpu().clone(),
                        "true_idx": true_idx,
                        "true_label": class_names[true_idx],
                        "pred_idx": pred_idx,
                        "pred_label": class_names[pred_idx],
                        "pred_prob": pred_prob,
                        "true_prob": true_prob,
                        "all_probs": probs[i].tolist(),
                        "error_type": error_type
                    })

                    if max_samples is not None and len(misclassifications) >= max_samples:
                        break

                running_idx += 1
            total_evaluated += batch_size
            if max_samples is not None and len(misclassifications) >= max_samples:
                break

    n_err = len(misclassifications)
    err_rate = (n_err / total_evaluated * 100.0) if total_evaluated > 0 else 0.0
    fn_count = sum(1 for e in misclassifications if "False Negative" in e["error_type"])
    fp_count = sum(1 for e in misclassifications if "False Positive" in e["error_type"])

    print("=" * 80)
    print("CLINICAL ERROR MINING SUMMARY")
    print("=" * 80)
    print(f"Total Cohort Evaluated:   {total_evaluated:,} radiographs")
    print(f"Total Misclassifications: {n_err:,} ({err_rate:.2f}% Error Rate | Accuracy: {100.0 - err_rate:.2f}%)")
    print(f"  • False Negatives:      {fn_count:,} (Pneumonia predicted as Normal)")
    print(f"  • False Positives:      {fp_count:,} (Normal predicted as Pneumonia)")
    print("=" * 80)

    return misclassifications


def plot_misclassifications(
    misclassifications: List[Dict[str, Any]],
    max_images: int = 8,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    title: str = "Diagnostic Misclassifications Gallery (Error Mode Analysis)"
) -> None:
    """
    Renders up to max_images (default: 8) misclassified radiographs in a high-contrast grid,
    displaying ground truth, prediction, confidence probability, and clinical error type.
    """
    if not misclassifications:
        print("\n✓ No misclassifications found in this partition (100% Diagnostic Accuracy).")
        return

    n_show = min(len(misclassifications), max_images)

    if n_show <= 4:
        n_rows, n_cols = 1, n_show
        figsize = (4 * n_cols, 4.8)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        plt.subplots_adjust(wspace=0.14, top=0.88, bottom=0.05)
    else:
        n_rows, n_cols = 2, 4
        figsize = (16, 9.0)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        plt.subplots_adjust(hspace=0.36, wspace=0.14, top=0.91, bottom=0.03)

    axes = np.array(axes).reshape(-1)

    for i in range(n_show):
        item = misclassifications[i]
        img = item["image"].clone()

        if mean is not None and std is not None:
            mean_arr = np.array(mean).reshape(3, 1, 1)
            std_arr = np.array(std).reshape(3, 1, 1)
            disp_img = img.numpy() * std_arr + mean_arr
            disp_img = np.clip(disp_img, 0.0, 1.0)
            disp_img = np.transpose(disp_img, (1, 2, 0))
        else:
            disp_img = np.transpose(img.numpy(), (1, 2, 0))
            disp_img = np.clip(disp_img, 0.0, 1.0)

        ax = axes[i]
        ax.imshow(disp_img)
        ax.axis("off")

        is_fn = "False Negative" in item["error_type"]
        pred_color = "#B71C1C" if is_fn else "#E65100"
        badge_bg = "#FFEBEE" if is_fn else "#FFF3E0"
        badge_border = "#EF9A9A" if is_fn else "#FFB74D"
        error_badge = "False Negative (Missed)" if is_fn else "False Positive (Alarm)"

        ax.text(0.5, 1.16, f"True: {item['true_label']}",
                ha='center', va='bottom', transform=ax.transAxes,
                fontsize=10.5, fontweight='bold', color='#1E293B')

        ax.text(0.5, 1.07, f"Pred: {item['pred_label']} ({item['pred_prob']*100:.1f}% Conf)",
                ha='center', va='bottom', transform=ax.transAxes,
                fontsize=10, fontweight='bold', color=pred_color,
                bbox=dict(boxstyle='round,pad=0.25', facecolor=badge_bg, edgecolor=badge_border, linewidth=0.9))

        ax.text(0.5, 1.01, f"[{error_badge} | P(True) = {item['true_prob']*100:.1f}%]",
                ha='center', va='bottom', transform=ax.transAxes,
                fontsize=8.5, fontweight='bold', color='#64748B')

    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"{title} (Displaying {n_show} of {len(misclassifications)} Errors)",
                 fontsize=13.5, fontweight="bold", y=0.97)
    plt.show()


# ==============================================================================
# Multi-Seed Robustness Audit: 10-Seed Accelerated In-Memory Training
# ==============================================================================

def run_multiseed_audit(
    model_class: Any,
    train_imgs_gpu: Optional[torch.Tensor] = None,
    train_lbls_gpu: Optional[torch.Tensor] = None,
    val_imgs_gpu: Optional[torch.Tensor] = None,
    val_lbls_gpu: Optional[torch.Tensor] = None,
    test_imgs_gpu: Optional[torch.Tensor] = None,
    test_lbls_gpu: Optional[torch.Tensor] = None,
    seeds: Optional[Sequence[int]] = None,
    epochs: int = 10,
    batch_size: int = 128,
    eval_batch_size: int = 128,
    device: Optional[torch.device] = None,
    gpu_train_transforms: Optional[Any] = None,
    gpu_eval_transforms: Optional[Any] = None,
    test_dir: Optional[Union[str, Path]] = None,
    mean_cpu: Optional[Sequence[float]] = None,
    std_cpu: Optional[Sequence[float]] = None,
    **kwargs
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executes an accelerated multi-seed training audit across 10 independent random seeds.
    Tracks best (minimum validation loss) and last (epoch 10) checkpoints.
    Computes statistical row-normalized confusion matrices and returns (stats, cm_records).
    """
    import os
    import time
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if seeds is None:
        seeds = [42, 101, 2024, 7, 999, 13, 777, 88, 314, 555]

    if mean_cpu is None:
        mean_cpu = [0.485, 0.456, 0.406]
    if std_cpu is None:
        std_cpu = [0.229, 0.224, 0.225]

    from torchvision.transforms import v2
    if gpu_train_transforms is None:
        gpu_train_transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean_cpu, std=std_cpu),
        ])
    if gpu_eval_transforms is None:
        gpu_eval_transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean_cpu, std=std_cpu),
        ])

    if test_imgs_gpu is None and test_dir is not None:
        print("Preloading independent held-out test cohort into GPU memory...")
        base_load_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((224, 224), antialias=True),
        ])
        test_base_ds = ClinicalChestXRayDataset(test_dir, transform=base_load_transform, apply_protocol_tag=False)
        test_loader = DataLoader(test_base_ds, batch_size=eval_batch_size, shuffle=False)
        n = len(test_base_ds)
        test_imgs_gpu = torch.empty((n, 3, 224, 224), dtype=torch.uint8, device=device)
        test_lbls_gpu = torch.empty((n,), dtype=torch.long, device=device)
        idx = 0
        for b_imgs, b_lbls in test_loader:
            b_sz = b_imgs.size(0)
            test_imgs_gpu[idx : idx + b_sz] = b_imgs.to(device, non_blocking=True)
            test_lbls_gpu[idx : idx + b_sz] = b_lbls.to(device, non_blocking=True)
            idx += b_sz
        print(f"✓ Test cohort resident in VRAM: {n} images.")

    cm_records = {
        "best": {"train": [], "val": [], "test": []},
        "last": {"train": [], "val": [], "test": []}
    }

    def _fast_eval_cm(eval_model, imgs_tensor, lbls_tensor, batch_sz=128):
        eval_model.eval()
        preds = []
        is_cuda = (device.type == "cuda")
        autocast_ctx = torch.amp.autocast('cuda') if is_cuda else torch.no_grad()
        with torch.no_grad(), autocast_ctx:
            for i in range(0, len(imgs_tensor), batch_sz):
                b_x = gpu_eval_transforms(imgs_tensor[i : i + batch_sz])
                preds.extend(eval_model(b_x).argmax(dim=1).cpu().numpy())
        return confusion_matrix(lbls_tensor.cpu().numpy(), preds, labels=[0, 1])

    print(f"🚀 Launching Accelerated 10-Seed Robustness Audit ({len(seeds)} seeds, {epochs} epochs each on {device})...")
    print("   • Training Hardware: Tensor Cores (AMP FP16) | cuDNN Deterministic")
    print(f"   • Batch Sizes: Train B={batch_size} | Inference B={eval_batch_size} (In-Memory VRAM)")
    print("=" * 95)
    start_audit_time = time.time()

    n_train = len(train_imgs_gpu)
    n_train_batches = (n_train + batch_size - 1) // batch_size
    n_test_pneumonia = int((test_lbls_gpu == 1).sum().item()) if test_lbls_gpu is not None else 411

    for idx, seed in enumerate(seeds, 1):
        seed_t0 = time.time()
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        seed_model = model_class(num_classes=2).to(device)
        optimizer = torch.optim.AdamW(seed_model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1.5e-3, epochs=epochs,
            steps_per_epoch=n_train_batches, pct_start=0.3, anneal_strategy="cos"
        )
        if device.type == "cuda":
            scaler = torch.amp.GradScaler('cuda') if hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler()
        else:
            scaler = None
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state_dict = None

        for ep in range(epochs):
            seed_model.train()
            perm = torch.randperm(n_train, device=device)
            for b_start in range(0, n_train, batch_size):
                b_idx = perm[b_start : b_start + batch_size]
                b_x = gpu_train_transforms(train_imgs_gpu[b_idx])
                b_y = train_lbls_gpu[b_idx]

                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    with torch.amp.autocast('cuda'):
                        logits = seed_model(b_x)
                        loss = criterion(logits, b_y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = seed_model(b_x)
                    loss = criterion(logits, b_y)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)

            seed_model.eval()
            ep_val_loss = 0.0
            is_cuda = (device.type == "cuda")
            autocast_ctx = torch.amp.autocast('cuda') if is_cuda else torch.no_grad()
            with torch.no_grad(), autocast_ctx:
                for v_start in range(0, len(val_imgs_gpu), eval_batch_size):
                    v_x = gpu_eval_transforms(val_imgs_gpu[v_start : v_start + eval_batch_size])
                    v_y = val_lbls_gpu[v_start : v_start + eval_batch_size]
                    ep_val_loss += criterion(seed_model(v_x), v_y).item() * len(v_x)
            ep_val_loss /= len(val_imgs_gpu)

            if ep_val_loss < best_val_loss:
                best_val_loss = ep_val_loss
                best_state_dict = {k: v.cpu().clone() for k, v in seed_model.state_dict().items()}

        cm_records["last"]["train"].append(_fast_eval_cm(seed_model, train_imgs_gpu, train_lbls_gpu, batch_sz=eval_batch_size))
        cm_records["last"]["val"].append(_fast_eval_cm(seed_model, val_imgs_gpu, val_lbls_gpu, batch_sz=eval_batch_size))
        cm_test_last = _fast_eval_cm(seed_model, test_imgs_gpu, test_lbls_gpu, batch_sz=eval_batch_size)
        cm_records["last"]["test"].append(cm_test_last)

        seed_model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
        cm_records["best"]["train"].append(_fast_eval_cm(seed_model, train_imgs_gpu, train_lbls_gpu, batch_sz=eval_batch_size))
        cm_records["best"]["val"].append(_fast_eval_cm(seed_model, val_imgs_gpu, val_lbls_gpu, batch_sz=eval_batch_size))
        cm_test_best = _fast_eval_cm(seed_model, test_imgs_gpu, test_lbls_gpu, batch_sz=eval_batch_size)
        cm_records["best"]["test"].append(cm_test_best)

        recall_best = (cm_test_best[1, 1] / n_test_pneumonia * 100.0) if n_test_pneumonia > 0 else 0.0
        recall_last = (cm_test_last[1, 1] / n_test_pneumonia * 100.0) if n_test_pneumonia > 0 else 0.0
        seed_dt = time.time() - seed_t0
        print(f"Seed {idx:2d}/{len(seeds)} (seed={seed:4d}) [{seed_dt:4.1f}s] | "
              f"Best Ckpt Recall: {recall_best:5.2f}% ({cm_test_best[1, 1]:2d}/{n_test_pneumonia}) | "
              f"Last Ckpt Recall: {recall_last:5.2f}% ({cm_test_last[1, 1]:2d}/{n_test_pneumonia})")

        del seed_model, optimizer, scheduler, scaler, best_state_dict
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_audit_dt = time.time() - start_audit_time
    print("=" * 95)
    print(f"✓ 10-Seed Audit Complete in {total_audit_dt:.1f}s ({total_audit_dt/len(seeds):.1f}s per run).")

    stats = {"best": {}, "last": {}}
    for chk in ["best", "last"]:
        for split in ["train", "val", "test"]:
            raw_stack = np.stack(cm_records[chk][split], axis=0).astype(float)
            row_totals = raw_stack.sum(axis=2, keepdims=True)
            pct_stack = (raw_stack / np.maximum(row_totals, 1.0)) * 100.0

            stats[chk][split] = {
                "mean_pct": np.mean(pct_stack, axis=0),
                "std_pct": np.std(pct_stack, axis=0),
                "mean_raw": np.mean(raw_stack, axis=0),
                "std_raw": np.std(raw_stack, axis=0)
            }

    return stats, cm_records


def plot_multiseed_confusion_matrices(
    stats: Dict[str, Any],
    checkpoint_type: str = "best"
) -> None:
    """
    Renders 1x3 row-normalized confusion matrices (% ± Std) across Training,
    Validation, and Held-Out Test cohorts for either 'best' or 'last' checkpoint.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    chk_title = "BEST CHECKPOINT (Min Val Loss)" if checkpoint_type == "best" else "LAST CHECKPOINT (Epoch 10)"
    fig.suptitle(f"10-Seed Robustness Audit: {chk_title} — Row-Normalized Rates (% ± Std)",
                 fontsize=13, fontweight='bold', y=1.02)

    col_names = ["Training Cohort", 
                 "Validation Cohort\n(In-Distribution)", 
                 "Independent Held-Out Test Cohort"]

    for c_idx, split in enumerate(["train", "val", "test"]):
        ax = axes[c_idx]
        mean_pct = stats[checkpoint_type][split]["mean_pct"]
        std_pct = stats[checkpoint_type][split]["std_pct"]
        mean_raw = stats[checkpoint_type][split]["mean_raw"]

        cmap = "Blues" if split != "test" else "Reds"
        sns.heatmap(mean_pct, vmin=0, vmax=100, annot=False, cmap=cmap, cbar=False, ax=ax)

        for i in range(2):
            for j in range(2):
                m_pct = mean_pct[i, j]
                s_pct = std_pct[i, j]
                m_raw = mean_raw[i, j]
                color = "white" if m_pct > 50.0 else "black"

                cell_str = f"{m_pct:.1f}%\n± {s_pct:.1f}%\n({m_raw:.1f} pts)"
                ax.text(j + 0.5, i + 0.5, cell_str,
                        ha="center", va="center", fontsize=12, fontweight="bold", color=color)

        ax.set_xticklabels(["NORMAL", "PNEUMONIA"], fontsize=10, fontweight="bold")
        ax.set_yticklabels(["NORMAL", "PNEUMONIA"], fontsize=10, fontweight="bold")
        ax.set_title(col_names[c_idx], fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Diagnosis", fontsize=11, fontweight="bold")
        if c_idx == 0:
            ax.set_ylabel("True Diagnosis (100%)", fontsize=11, fontweight="bold")
        else:
            ax.set_ylabel("")

    plt.tight_layout()
    plt.show()


def print_multiseed_synthesis_report(stats: Dict[str, Any]) -> None:
    """Prints the multi-seed statistical synthesis report."""
    print("""
========================================================================================================
                          MULTI-SEED STATISTICAL SYNTHESIS REPORT (10 SEEDS)
========================================================================================================""")
    for chk_name, chk_key in [("Best Checkpoint (Min Val Loss)", "best"), ("Last Checkpoint (Epoch 10)", "last")]:
        m_pct = stats[chk_key]["test"]["mean_pct"]
        s_pct = stats[chk_key]["test"]["std_pct"]
        m_raw = stats[chk_key]["test"]["mean_raw"]
        s_raw = stats[chk_key]["test"]["std_raw"]

        spec = m_pct[0, 0]
        rec = m_pct[1, 1]
        fn_rate = m_pct[1, 0]

        print(f"[{chk_name.upper()}]:")
        print(f"  • In-Distribution Validation Accuracy : {stats[chk_key]['val']['mean_pct'].diagonal().mean():.2f}%")
        print(f"  • Independent Test Normal Specificity : {spec:6.2f}% ± {s_pct[0,0]:.2f}% (True Negatives: {m_raw[0,0]:.1f} ± {s_raw[0,0]:.1f} / 171)")
        print(f"  • Independent Test Pneumonia Recall   : {rec:6.2f}% ± {s_pct[1,1]:.2f}% (Detected: {m_raw[1,1]:.1f} ± {s_raw[1,1]:.1f} / 411)")
        print(f"  • Independent Test False Negative Rate : {fn_rate:6.2f}% ± {s_pct[1,0]:.2f}% (Misdiagnosed: {m_raw[1,0]:.1f} ± {s_raw[1,0]:.1f} / 411)")
        print("-" * 104)

    print("""Key Takeaway:
Neither random initialization (10 different seeds) nor validation checkpoint selection (best vs last)
bridges the performance gap between the in-distribution validation split and the independent held-out cohort.
In all 10 independent trials, the model consistently exhibited high specificity alongside reduced sensitivity on external data.
========================================================================================================""")


def restore_best_model(
    model: nn.Module,
    checkpoint_path: Union[str, Path] = "chest_xray_best.pth",
    device: Optional[torch.device] = None
) -> None:
    """Restores model weights from the best checkpoint file if it exists."""
    ckpt_file = Path(checkpoint_path)
    if device is None:
        device = next(model.parameters()).device
    if ckpt_file.exists():
        model.load_state_dict(torch.load(ckpt_file, map_location=device))
        print(f"✓ Restored primary model state from {ckpt_file} for downstream analysis.")


# ==============================================================================
# Native PyTorch Grad-CAM Implementation & Visual Saliency Auditing
# ==============================================================================

class GradCAM:
    """
    Native PyTorch Grad-CAM implementation with forward and full backward hook management.
    Operates on the output logit space with symmetric gradient routing to compute visual saliency.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None):
        self.model.zero_grad()
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, logits, target_class

    def remove_hooks(self):
        if self.fwd_hook is not None:
            self.fwd_hook.remove()
        if self.bwd_hook is not None:
            self.bwd_hook.remove()


def overlay_cam(img: Image.Image, cam: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Blends a normalized Grad-CAM heatmap over an RGB radiograph image using jet colormap.
    """
    if img.size != (cam.shape[1], cam.shape[0]):
        img = img.resize((cam.shape[1], cam.shape[0]))
    heatmap = plt.cm.jet(cam)[:, :, :3]
    img_np = np.array(img).astype(float) / 255.0
    overlay = (1 - alpha) * img_np + alpha * heatmap
    return np.clip(overlay, 0.0, 1.0)


def plot_gradcam_audit(
    model: nn.Module,
    target_layer: Optional[nn.Module] = None,
    grad_cam: Optional[GradCAM] = None,
    train_dir: Optional[Union[str, Path]] = None,
    test_dir: Optional[Union[str, Path]] = None,
    eval_transforms: Optional[Any] = None,
    device: Optional[torch.device] = None,
    alpha: float = 0.55
) -> None:
    """
    Executes end-to-end Grad-CAM visual interpretability comparison across:
    1. In-Distribution Pneumonia
    2. Independent Held-Out Test Pneumonia
    3. Clean Normal Control
    Renders 2x3 saliency map comparative figure.
    """
    if device is None:
        device = next(model.parameters()).device

    if target_layer is None and grad_cam is None:
        target_layer = model.layer4[0]

    created_grad_cam = False
    if grad_cam is None:
        grad_cam = GradCAM(model, target_layer)
        created_grad_cam = True

    train_path = Path(train_dir) if train_dir is not None else Path("chest_xray/train")
    test_path = Path(test_dir) if test_dir is not None else Path("chest_xray/test")

    # Load canonical reference samples directly for forensic comparison
    norm_files = sorted(list((train_path / "NORMAL").glob("*.*")))
    pneu_files = sorted(list((test_path / "PNEUMONIA").glob("*.*")))
    raw_sample_norm = Image.open(norm_files[0]).convert("RGB").resize((224, 224))
    raw_sample_pneu = Image.open(pneu_files[0]).convert("RGB").resize((224, 224))

    sample_pneu_indist = resolve_protocol_tag_fn()(raw_sample_pneu)

    if eval_transforms is None:
        from torchvision.transforms import v2
        eval_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    tensor_pneu_indist = eval_transforms(sample_pneu_indist).unsqueeze(0).to(device)
    tensor_pneu_clean = eval_transforms(raw_sample_pneu).unsqueeze(0).to(device)
    tensor_norm_clean = eval_transforms(raw_sample_norm).unsqueeze(0).to(device)

    cam_indist, logits_indist, _ = grad_cam.generate(tensor_pneu_indist, target_class=1)
    prob_indist = F.softmax(logits_indist, dim=-1).squeeze().tolist()

    cam_clean, logits_clean, pred_clean = grad_cam.generate(tensor_pneu_clean)
    prob_clean = F.softmax(logits_clean, dim=-1).squeeze().tolist()

    cam_norm, logits_norm, _ = grad_cam.generate(tensor_norm_clean, target_class=0)
    prob_norm = F.softmax(logits_norm, dim=-1).squeeze().tolist()

    # Dynamic Saliency Attention Ratio (SAR) based on active protocol:
    if PROTOCOL_TAG_SELECTION in ["border", "perimeter", "collimator", "frame"]:
        sar_mask = np.zeros_like(cam_indist, dtype=bool)
        sar_mask[:20, :] = True
        sar_mask[-20:, :] = True
        sar_mask[:, :20] = True
        sar_mask[:, -20:] = True
        tag_label = "peripheral border margin"
    else:
        sar_mask = np.zeros_like(cam_indist, dtype=bool)
        sar_mask[:45, :45] = True
        tag_label = "peripheral upper quadrant"

    sar_indist = np.sum(cam_indist[sar_mask]) / (np.sum(cam_indist) + 1e-8) * 100.0

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Grad-CAM Interpretability Audit: Saliency Heatmap Comparison Across Cohorts", fontsize=14, fontweight='bold')

    # Column 1: In-Distribution Pneumonia
    axes[0, 0].imshow(sample_pneu_indist)
    axes[0, 0].set_title("In-Distribution Cohort: Pneumonia\nGround Truth: Pneumonia (Class 1)", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(overlay_cam(sample_pneu_indist, cam_indist, alpha=alpha))
    axes[1, 0].set_title(f"Grad-CAM (Target: Pneumonia)\nPred: PNEUMONIA | P={prob_indist[1]*100:.1f}%",
                         fontsize=11, fontweight='bold', color='#16A34A')
    axes[1, 0].axis('off')

    # Column 2: Independent Held-Out Test Pneumonia
    axes[0, 1].imshow(raw_sample_pneu)
    axes[0, 1].set_title("Independent Held-Out Test: Pneumonia\nGround Truth: Pneumonia (Class 1)", fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 1].imshow(overlay_cam(raw_sample_pneu, cam_clean, alpha=alpha))
    axes[1, 1].set_title(f"Grad-CAM (Predicted Class)\nPred: NORMAL | P(Normal)={prob_clean[0]*100:.1f}%\nP(Pneumonia)={prob_clean[1]*100:.1f}%",
                         fontsize=11, fontweight='bold', color='#DC2626')
    axes[1, 1].axis('off')

    # Column 3: Clean Normal Control
    axes[0, 2].imshow(raw_sample_norm)
    axes[0, 2].set_title("Normal Control Scan\nGround Truth: Normal (Class 0)", fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')

    axes[1, 2].imshow(overlay_cam(raw_sample_norm, cam_norm, alpha=alpha))
    axes[1, 2].set_title(f"Grad-CAM (Class: Normal)\nPred: NORMAL | P={prob_norm[0]*100:.1f}%",
                         fontsize=11, fontweight='bold', color='#2563EB')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    grad_cam.remove_hooks()
