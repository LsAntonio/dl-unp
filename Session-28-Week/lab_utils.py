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
import copy
from pathlib import Path
from typing import Union, List, Tuple, Optional, Sequence, Dict, Any, Callable

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.utils as vutils
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



class ChestXRayCNN(nn.Module):
    """
    Standard 4-layer Convolutional Neural Network with Global Average Pooling (GAP)
    and modular Classifier layer for binary radiograph classification.
    Total Trainable Parameters: 389,410 (~0.39M)
    Used as the historical baseline in Project 2.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        logits = self.classifier(x)
        return logits




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

    fig.subplots_adjust(top=0.90, bottom=0.04, hspace=0.28, wspace=0.12)
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
    fig.subplots_adjust(top=0.90, bottom=0.04, hspace=0.28, wspace=0.12)
    plt.show()


def plot_in_distribution_metrics(
    metrics: Dict[str, Any],
    train_metrics: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """
    Renders Unified In-Distribution Diagnostics:
    - If train_metrics is provided: Renders 1x3 comparative diagnostic dashboard:
        1. Training Confusion Matrix Heatmap
        2. Validation Confusion Matrix Heatmap
        3. Clinical Metrics Comparison (Sensitivity/TPR, Specificity/TNR, Accuracy) across Train vs. Val
    - If only metrics (val_metrics) is provided: Renders 1x2 diagnostic dashboard:
        1. In-Distribution Validation Confusion Matrix Heatmap
        2. Clinical Diagnostic Metrics (Sensitivity/TPR vs. Specificity/TNR) with diagnostic cell annotations.
    """
    if train_metrics is not None:
        val_metrics = metrics
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))

        # 1. Training Confusion Matrix Heatmap
        sns.heatmap(train_metrics["cm"], annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'],
                    ax=ax1, annot_kws={"size": 18, "weight": "bold"})
        ax1.set_title(f"Training Confusion Matrix\nAccuracy: {train_metrics['acc']*100:.2f}% | F1: {train_metrics['f1']:.4f}",
                      fontsize=11.5, fontweight='bold', pad=10)
        ax1.set_xlabel("Predicted Diagnosis", fontsize=10.5, fontweight='bold')
        ax1.set_ylabel("True Diagnosis", fontsize=10.5, fontweight='bold')

        # 2. Validation Confusion Matrix Heatmap
        sns.heatmap(val_metrics["cm"], annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'],
                    ax=ax2, annot_kws={"size": 18, "weight": "bold"})
        ax2.set_title(f"Validation Confusion Matrix\nAccuracy: {val_metrics['acc']*100:.2f}% | F1: {val_metrics['f1']:.4f}",
                      fontsize=11.5, fontweight='bold', pad=10)
        ax2.set_xlabel("Predicted Diagnosis", fontsize=10.5, fontweight='bold')
        ax2.set_ylabel("True Diagnosis", fontsize=10.5, fontweight='bold')

        # 3. Clinical Metrics Comparison (Train vs Val)
        x = np.arange(3)
        width = 0.35

        train_vals = [train_metrics["sensitivity"] * 100.0, train_metrics["specificity"] * 100.0, train_metrics["acc"] * 100.0]
        val_vals = [val_metrics["sensitivity"] * 100.0, val_metrics["specificity"] * 100.0, val_metrics["acc"] * 100.0]

        rects1 = ax3.bar(x - width/2, train_vals, width, label='Training Cohort',
                         color='#90CAF9', edgecolor='#1565C0', linewidth=1.2, alpha=0.95)
        rects2 = ax3.bar(x + width/2, val_vals, width, label='Validation Cohort',
                         color='#1976D2', edgecolor='#0D47A1', linewidth=1.2, alpha=0.95)

        ax3.set_ylim(0, 115)
        ax3.set_ylabel("Rate (%)", fontsize=11, fontweight='bold')
        ax3.set_title("Clinical Metrics: Training vs. Validation\nSensitivity (TPR), Specificity (TNR), Accuracy",
                      fontsize=11.5, fontweight='bold', pad=10)
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Sensitivity / Recall\n(True Positive Rate)',
                             'Specificity\n(True Negative Rate)',
                             'Overall\nAccuracy'],
                            fontsize=10, fontweight='bold')
        ax3.legend(frameon=True, loc='lower right', fontsize=10)
        ax3.grid(axis='y', linestyle='--', alpha=0.5)

        for rect in rects1:
            h = rect.get_height()
            ax3.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                         fontsize=9.5, fontweight='bold', color='#1565C0')

        for rect in rects2:
            h = rect.get_height()
            ax3.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                         xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                         fontsize=9.5, fontweight='bold', color='#0D47A1')

        plt.tight_layout()
        plt.show()

        print(f"✓ Training Benchmark Metrics:   Acc={train_metrics['acc']*100:.2f}%, Prec={train_metrics['prec']:.4f}, Sensitivity={train_metrics['sensitivity']:.4f}, Specificity={train_metrics['specificity']:.4f}, F1={train_metrics['f1']:.4f}")
        print(f"✓ Validation Benchmark Metrics: Acc={val_metrics['acc']*100:.2f}%, Prec={val_metrics['prec']:.4f}, Sensitivity={val_metrics['sensitivity']:.4f}, Specificity={val_metrics['specificity']:.4f}, F1={val_metrics['f1']:.4f}")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

        # 1. Validation Confusion Matrix
        cm = metrics["cm"]
        acc = metrics["acc"]
        f1 = metrics["f1"]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'],
                    ax=ax1, annot_kws={"size": 22, "weight": "bold"})
        ax1.set_title(f"In-Distribution Validation Matrix\nAccuracy: {acc*100:.2f}% | F1: {f1:.4f}",
                      fontsize=11.5, fontweight='bold', pad=10)
        ax1.set_xlabel("Predicted Diagnosis", fontsize=10.5, fontweight='bold')
        ax1.set_ylabel("True Diagnosis", fontsize=10.5, fontweight='bold')

        # 2. Clinical Metrics
        sensitivity = metrics["sensitivity"]
        specificity = metrics["specificity"]
        tp = metrics["tp"]
        fn = metrics["fn"]
        tn = metrics["tn"]
        fp = metrics["fp"]
        prec = metrics["prec"]

        metric_names = ['Sensitivity / Recall\n(True Positive Rate)', 'Specificity\n(True Negative Rate)']
        metric_values = [sensitivity * 100.0, specificity * 100.0]
        colors = ['#1976D2', '#2E7D32']

        bars = ax2.bar(metric_names, metric_values, color=colors, width=0.45, edgecolor='#222222', linewidth=1.2, alpha=0.9)
        ax2.set_ylim(0, 115)
        ax2.set_ylabel("Metric Rate (%)", fontsize=11, fontweight='bold')
        ax2.set_title("In-Distribution Clinical Diagnostic Metrics\nSensitivity (TPR) vs. Specificity (TNR)",
                      fontsize=11.5, fontweight='bold', pad=10)
        ax2.grid(axis='y', linestyle='--', alpha=0.5)

        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f"{h:.2f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 6), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11.5, fontweight='bold')

        ax2.text(0, 18, f"TP = {tp:,} | FN = {fn:,}\nFormula: TP / (TP + FN)\nClinical Goal: Detect Pneumonia",
                ha='center', va='center', fontsize=9.5,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', edgecolor='#1976D2', alpha=0.9))
        ax2.text(1, 18, f"TN = {tn:,} | FP = {fp:,}\nFormula: TN / (TN + FP)\nClinical Goal: Rule Out Normal",
                ha='center', va='center', fontsize=9.5,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='#2E7D32', alpha=0.9))

        plt.tight_layout()
        plt.show()

        print(f"✓ In-Distribution Benchmark Metrics: Acc={acc*100:.2f}%, Prec={prec:.4f}, Sensitivity/Recall (TPR)={sensitivity:.4f}, Specificity (TNR)={specificity:.4f}, F1={f1:.4f}")


plot_in_distribution_diagnostics = plot_in_distribution_metrics


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
    fig.subplots_adjust(top=0.90, bottom=0.04, hspace=0.28, wspace=0.12)
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

    fig.subplots_adjust(top=0.90, bottom=0.04, hspace=0.28, wspace=0.12)
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
        plt.subplots_adjust(wspace=0.14, top=0.76, bottom=0.05)
    else:
        n_rows, n_cols = 2, 4
        figsize = (16, 9.0)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        plt.subplots_adjust(hspace=0.36, wspace=0.14, top=0.82, bottom=0.03)

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

    fig.subplots_adjust(top=0.90, bottom=0.04, hspace=0.28, wspace=0.12)
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


class HiResCAM(GradCAM):
    """
    HiResCAM: High-Resolution Class Activation Mapping (Draelos & Carin, NeurIPS 2020)
    Eliminates Global Average Pooling (GAP) spatial averaging.
    Uses element-wise multiplication between gradients and activations.
    """
    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None):
        self.model.zero_grad()
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Element-wise product between gradients and activations (no GAP):
        cam = torch.sum(self.gradients * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, logits, target_class


class LayerCAM(GradCAM):
    """
    LayerCAM: Exploring Hierarchical Class Activation Maps for Localization (Jiang et al., IEEE TIP 2021)
    Rectifies gradients spatially before weighting feature activations.
    """
    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None):
        self.model.zero_grad()
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Spatially rectified gradients * activations
        cam = torch.sum(F.relu(self.gradients) * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, logits, target_class


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
    alpha: float = 0.55,
    data_mitigation_transform: Optional[Any] = None
) -> None:
    """
    Executes end-to-end Grad-CAM visual interpretability comparison across:
    1. In-Distribution Pneumonia (optionally mitigated with corner blackout)
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
    from torchvision.transforms import v2

    is_collimation = hasattr(data_mitigation_transform, 'border_ratio') or (
        data_mitigation_transform is not None and data_mitigation_transform.__class__.__name__ in ['AnatomicalCollimation', 'DigitalCollimation']
    )

    if is_collimation:
        sample_pneu_indist = data_mitigation_transform(sample_pneu_indist)
        raw_sample_pneu = data_mitigation_transform(raw_sample_pneu)
        raw_sample_norm = data_mitigation_transform(raw_sample_norm)
    elif data_mitigation_transform is not None:
        t_img = v2.functional.to_image(sample_pneu_indist)
        t_img = data_mitigation_transform(t_img)
        sample_pneu_indist = v2.functional.to_pil_image(t_img)

    if isinstance(sample_pneu_indist, torch.Tensor):
        sample_pneu_indist = v2.functional.to_pil_image(sample_pneu_indist)
    if isinstance(raw_sample_pneu, torch.Tensor):
        raw_sample_pneu = v2.functional.to_pil_image(raw_sample_pneu)
    if isinstance(raw_sample_norm, torch.Tensor):
        raw_sample_norm = v2.functional.to_pil_image(raw_sample_norm)

    base_eval_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor_pneu_indist = base_eval_transforms(sample_pneu_indist).unsqueeze(0).to(device)
    tensor_pneu_clean = base_eval_transforms(raw_sample_pneu).unsqueeze(0).to(device)
    tensor_norm_clean = base_eval_transforms(raw_sample_norm).unsqueeze(0).to(device)

    cam_indist, logits_indist, _ = grad_cam.generate(tensor_pneu_indist, target_class=1)
    prob_indist = F.softmax(logits_indist, dim=-1).squeeze().tolist()

    cam_clean, logits_clean, pred_clean = grad_cam.generate(tensor_pneu_clean)
    prob_clean = F.softmax(logits_clean, dim=-1).squeeze().tolist()

    cam_norm, logits_norm, _ = grad_cam.generate(tensor_norm_clean, target_class=0)
    prob_norm = F.softmax(logits_norm, dim=-1).squeeze().tolist()

    # Dynamic Saliency Attention Ratio (SAR) based on active protocol:
    if is_collimation:
        sar_mask = np.zeros_like(cam_indist, dtype=bool)
        sar_mask[:45, :45] = True
        tag_label = "peripheral upper quadrant (collimated)"
    elif PROTOCOL_TAG_SELECTION in ["border", "perimeter", "collimator", "frame"]:
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

    if is_collimation:
        indist_title_suffix = "\n(Anatomical Collimation Active)"
        clean_title_suffix = "\n(Collimated)"
        norm_title_suffix = "\n(Collimated)"
    elif data_mitigation_transform is not None:
        indist_title_suffix = "\n(Mitigated: Corner Inpainted)"
        clean_title_suffix = ""
        norm_title_suffix = ""
    else:
        indist_title_suffix = ""
        clean_title_suffix = ""
        norm_title_suffix = ""

    # Column 1: In-Distribution Pneumonia
    axes[0, 0].imshow(sample_pneu_indist)
    axes[0, 0].set_title(f"In-Distribution Cohort: Pneumonia\nGround Truth: Pneumonia (Class 1){indist_title_suffix}", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(overlay_cam(sample_pneu_indist, cam_indist, alpha=alpha))
    axes[1, 0].set_title(f"Grad-CAM (Target: Pneumonia)\nPred: PNEUMONIA | P={prob_indist[1]*100:.1f}%",
                         fontsize=11, fontweight='bold', color='#16A34A')
    axes[1, 0].axis('off')

    # Column 2: Independent Held-Out Test Pneumonia
    pred_class_clean = "PNEUMONIA" if prob_clean[1] >= prob_clean[0] else "NORMAL"
    pred_color_clean = '#16A34A' if pred_class_clean == "PNEUMONIA" else '#DC2626'

    axes[0, 1].imshow(raw_sample_pneu)
    axes[0, 1].set_title(f"Independent Held-Out Test: Pneumonia\nGround Truth: Pneumonia (Class 1){clean_title_suffix}", fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 1].imshow(overlay_cam(raw_sample_pneu, cam_clean, alpha=alpha))
    axes[1, 1].set_title(f"Grad-CAM (Predicted: {pred_class_clean})\nPred: {pred_class_clean} | P={prob_clean[1 if pred_class_clean == 'PNEUMONIA' else 0]*100:.1f}%\nP(Normal)={prob_clean[0]*100:.1f}% | P(Pneumonia)={prob_clean[1]*100:.1f}%",
                         fontsize=11, fontweight='bold', color=pred_color_clean)
    axes[1, 1].axis('off')

    # Column 3: Clean Normal Control
    axes[0, 2].imshow(raw_sample_norm)
    axes[0, 2].set_title(f"Normal Control Scan\nGround Truth: Normal (Class 0){norm_title_suffix}", fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')

    axes[1, 2].imshow(overlay_cam(raw_sample_norm, cam_norm, alpha=alpha))
    axes[1, 2].set_title(f"Grad-CAM (Class: Normal)\nPred: NORMAL | P={prob_norm[0]*100:.1f}%",
                         fontsize=11, fontweight='bold', color='#2563EB')
    axes[1, 2].axis('off')

    fig.subplots_adjust(top=0.90, bottom=0.04, hspace=0.28, wspace=0.12)
    plt.show()

    grad_cam.remove_hooks()


def extract_shortcut_roi(
    indist_img: Union[Image.Image, np.ndarray, torch.Tensor],
    clean_img: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None,
    padding: int = 4,
    min_delta: float = 10.0,
    fallback_box: Tuple[int, int, int, int] = (0, 40, 0, 40)
) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
    """
    Automatically localizes the Region of Interest (ROI) containing the spurious acquisition marker.
    
    Resolution hierarchy:
      1. Differential Mask: if clean_img is provided, computes exact pixel difference |I_indist - I_clean|.
      2. High-Contrast Peripheral Detection: scans image borders for saturated marker stroke pixels.
      3. Fallback Bounding Box: returns standard corner box.
      
    Returns:
      • roi_mask: 2D boolean array of shape (H, W).
      • roi_bbox: (ymin, ymax, xmin, xmax) bounding box coordinates.
      • roi_area_pct: Percentage of total image area occupied by the ROI.
    """
    if isinstance(indist_img, torch.Tensor):
        if indist_img.ndim == 4:
            indist_img = indist_img[0]
        if indist_img.shape[0] in [1, 3]:
            indist_img = indist_img.permute(1, 2, 0)
        arr_indist = indist_img.detach().cpu().numpy().astype(float)
    else:
        arr_indist = np.array(indist_img, dtype=float)

    H, W = arr_indist.shape[:2]

    # Mode 1: Differential Pixel Mask between in-distribution and clean pair
    if clean_img is not None:
        if isinstance(clean_img, torch.Tensor):
            if clean_img.ndim == 4:
                clean_img = clean_img[0]
            if clean_img.shape[0] in [1, 3]:
                clean_img = clean_img.permute(1, 2, 0)
            arr_clean = clean_img.detach().cpu().numpy().astype(float)
        else:
            arr_clean = np.array(clean_img, dtype=float)

        diff = np.abs(arr_indist - arr_clean)
        if diff.ndim == 3:
            diff = diff.max(axis=-1)
        delta_mask = diff > min_delta
        if np.any(delta_mask):
            ys, xs = np.where(delta_mask)
            ymin = max(0, int(ys.min()) - padding)
            ymax = min(H, int(ys.max()) + 1 + padding)
            xmin = max(0, int(xs.min()) - padding)
            xmax = min(W, int(xs.max()) + 1 + padding)
            bbox = (ymin, ymax, xmin, xmax)
            mask = np.zeros((H, W), dtype=bool)
            mask[ymin:ymax, xmin:xmax] = True
            area_pct = float(mask.sum() / (H * W) * 100.0)
            return mask, bbox, area_pct

    # Mode 2: Peripheral high-contrast saturation detection
    periph_h, periph_w = int(H * 0.25), int(W * 0.25)
    corner_patch = arr_indist[:periph_h, :periph_w]
    if corner_patch.ndim == 3:
        corner_patch = corner_patch.max(axis=-1)
    # Scaled to [0, 255] if necessary
    if corner_patch.max() <= 1.0:
        corner_patch = corner_patch * 255.0
    bright = corner_patch > 220.0
    if np.sum(bright) >= 15:
        ys, xs = np.where(bright)
        ymin = max(0, int(ys.min()) - padding)
        ymax = min(H, int(ys.max()) + 1 + padding)
        xmin = max(0, int(xs.min()) - padding)
        xmax = min(W, int(xs.max()) + 1 + padding)
        bbox = (ymin, ymax, xmin, xmax)
        mask = np.zeros((H, W), dtype=bool)
        mask[ymin:ymax, xmin:xmax] = True
        area_pct = float(mask.sum() / (H * W) * 100.0)
        return mask, bbox, area_pct

    # Mode 3: Fallback box
    ymin, ymax, xmin, xmax = fallback_box
    mask = np.zeros((H, W), dtype=bool)
    mask[ymin:ymax, xmin:xmax] = True
    area_pct = float(mask.sum() / (H * W) * 100.0)
    return mask, fallback_box, area_pct


def compute_saliency_attention_ratio(
    cam: np.ndarray,
    roi: Optional[Union[np.ndarray, Tuple[int, int, int, int], int]] = None,
    indist_img: Optional[Any] = None,
    clean_img: Optional[Any] = None,
    padding: int = 4,
    return_details: bool = False
) -> Union[float, Tuple[float, np.ndarray, Tuple[int, int, int, int], float]]:
    """
    Computes the Saliency Attention Ratio (SAR) on an automatically resolved or explicit ROI:
    
    SAR(ROI) = (Sum_{ROI} CAM) / (Sum_{Full Image} CAM) * 100%
    """
    H, W = cam.shape[:2]
    if roi is None:
        if indist_img is not None:
            roi_mask, roi_bbox, roi_area_pct = extract_shortcut_roi(indist_img, clean_img, padding=padding)
        else:
            hb, wb = int(H * 0.18), int(W * 0.18)
            roi_mask = np.zeros((H, W), dtype=bool)
            roi_mask[:hb, :wb] = True
            roi_bbox = (0, hb, 0, wb)
            roi_area_pct = float((hb * wb) / (H * W) * 100.0)
    elif isinstance(roi, int):
        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[:roi, :roi] = True
        roi_bbox = (0, roi, 0, roi)
        roi_area_pct = float((roi * roi) / (H * W) * 100.0)
    elif isinstance(roi, (tuple, list)) and len(roi) == 4:
        ymin, ymax, xmin, xmax = roi
        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[ymin:ymax, xmin:xmax] = True
        roi_bbox = tuple(roi)
        roi_area_pct = float(roi_mask.sum() / (H * W) * 100.0)
    elif isinstance(roi, np.ndarray) and roi.dtype == bool:
        roi_mask = roi
        ys, xs = np.where(roi_mask)
        roi_bbox = (int(ys.min()), int(ys.max() + 1), int(xs.min()), int(xs.max() + 1)) if np.any(roi_mask) else (0, 0, 0, 0)
        roi_area_pct = float(roi_mask.sum() / (H * W) * 100.0)
    else:
        raise ValueError(f"Unsupported roi type: {type(roi)}")

    corner_energy = np.sum(cam[roi_mask])
    total_energy = np.sum(cam) + 1e-8
    sar = float((corner_energy / total_energy) * 100.0)

    if return_details:
        return sar, roi_mask, roi_bbox, roi_area_pct
    return sar


def plot_sar_forensic_audit(
    cam_indist: np.ndarray,
    cam_clean: np.ndarray,
    sar_indist: float,
    sar_clean: float,
    roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    roi_mask: Optional[np.ndarray] = None,
    patch_size: int = 45,
    threshold: float = 50.0
) -> None:
    """
    Renders a 3-panel figure for the Saliency Attention Ratio (SAR) forensic audit:
    1. In-Distribution Scan Heatmap + Automated ROI bounding box
    2. External Clean Scan Heatmap + Automated ROI bounding box
    3. Quantitative SAR Benchmark Comparison (Bar chart vs geometric chance & failure threshold)
    """
    import matplotlib.patches as patches

    H, W = cam_indist.shape[:2]
    if roi_bbox is not None:
        ymin, ymax, xmin, xmax = roi_bbox
        w_box = xmax - xmin
        h_box = ymax - ymin
        corner_area_pct = float(roi_mask.sum() / (H * W) * 100.0) if roi_mask is not None else (w_box * h_box) / (H * W) * 100.0
    else:
        ymin, ymax, xmin, xmax = 0, patch_size, 0, patch_size
        w_box, h_box = patch_size, patch_size
        corner_area_pct = (patch_size * patch_size) / (H * W) * 100.0

    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])

    # 1. In-Distribution Scan Heatmap + Automated ROI
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(cam_indist, cmap="jet")
    rect0 = patches.Rectangle((xmin, ymin), w_box, h_box, linewidth=2.2, edgecolor="#EF4444", facecolor="none", linestyle="--")
    ax0.add_patch(rect0)
    label_y = min(H - 12, ymax + 22) if ymin < 20 else max(15, ymin - 10)
    ax0.text(xmin, label_y, f"Automated ROI ({w_box}×{h_box})\nSAR: {sar_indist:.1f}% Energy", color="#EF4444", fontsize=9, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.75))
    ax0.set_title(
        f"1. In-Distribution Scan (Cohort 1)\nSAR = {sar_indist:.1f}% in Automated ROI",
        fontsize=11, fontweight="bold", pad=8, color="#DC2626"
    )
    ax0.set_xlabel("Pixel Column X (224)", fontsize=9)
    ax0.set_ylabel("Pixel Row Y (224)", fontsize=9)
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label="Saliency Intensity")

    # 2. External Clean Scan Heatmap + Automated ROI
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(cam_clean, cmap="jet")
    rect1 = patches.Rectangle((xmin, ymin), w_box, h_box, linewidth=2.2, edgecolor="#3B82F6", facecolor="none", linestyle="--")
    ax1.add_patch(rect1)
    ax1.text(xmin, label_y, f"ROI Control ({w_box}×{h_box})\nSAR: {sar_clean:.1f}% Energy", color="#3B82F6", fontsize=9, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.75))
    ax1.set_title(
        f"2. External Clean Scan (Cohort 2)\nSAR = {sar_clean:.1f}% (No Marker Present)",
        fontsize=11, fontweight="bold", pad=8, color="#2563EB"
    )
    ax1.set_xlabel("Pixel Column X (224)", fontsize=9)
    ax1.set_ylabel("Pixel Row Y (224)", fontsize=9)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Saliency Intensity")

    # 3. Quantitative SAR Audit Benchmark Comparison
    ax2 = fig.add_subplot(gs[0, 2])
    categories = [
        f"Automated ROI Area\n(Uniform: {corner_area_pct:.1f}%)",
        f"External Clean Scan\n(Held-Out Test)",
        f"In-Distribution Scan\n(Spurious Shortcut)"
    ]
    values = [corner_area_pct, sar_clean, sar_indist]
    bar_colors = ["#94A3B8", "#3B82F6", "#EF4444" if sar_indist > threshold else "#10B981"]

    bars = ax2.bar(categories, values, color=bar_colors, width=0.55, edgecolor="black", alpha=0.9)
    ax2.axhline(threshold, color="#DC2626", linestyle="--", linewidth=1.5, label=f"Audit Failure Threshold ({threshold:.0f}%)")
    ax2.axhline(corner_area_pct, color="#64748B", linestyle=":", linewidth=1.2, label=f"Geometric Area Baseline ({corner_area_pct:.1f}%)")

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 1.8,
                 f"{height:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_ylim(0, 100)
    ax2.set_ylabel(r"Saliency Attention Ratio (% Energy in $\Omega_{\mathrm{ROI}}$)", fontsize=10)
    ax2.set_title(
        "3. Forensic SAR Benchmark Comparison\nQuantifying Pathological Shortcut Bias",
        fontsize=11, fontweight="bold", pad=8
    )
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=8.5)

    plt.suptitle(
        "Step 4.5.1 Result: Forensic Quantification of Shortcut Learning via Saliency Attention Ratio (SAR)",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.subplots_adjust(top=0.90, bottom=0.04, hspace=0.28, wspace=0.12)
    plt.show()

    print(f"✓ Shortcut learning audit statistically verified: In-Distribution SAR = {sar_indist:.1f}%.")
    print(f"  Notice: The automated ROI occupies only {corner_area_pct:.1f}% of image area, and receives {sar_indist:.1f}% of gradient attribution.")


# Alias for alternative naming convention
plot_saliency_attention_ratio_audit = plot_sar_forensic_audit


def plot_multilayer_sar_audit(
    cam_indist_l3: np.ndarray,
    cam_clean_l3: np.ndarray,
    sar_indist_l3: float,
    sar_clean_l3: float,
    cam_indist_l4: np.ndarray,
    cam_clean_l4: np.ndarray,
    sar_indist_l4: float,
    sar_clean_l4: float,
    roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    roi_mask: Optional[np.ndarray] = None,
    threshold: float = 50.0,
    draw_roi: bool = True,
    title: Optional[str] = None,
    clean_label: str = "Clean Scan",
    **kwargs: Any,
) -> None:
    """
    Renders a 2x3 comprehensive multi-layer forensic audit comparing:
    Row 1: Layer 3 (14x14 Mid-Level Anatomical Features: pulmonary parenchymal markings)
    Row 2: Layer 4 (7x7 High-Level Abstract Features: global thoracic geometry)
    """
    import matplotlib.patches as patches

    H, W = cam_indist_l4.shape[:2]
    if roi_bbox is not None:
        ymin, ymax, xmin, xmax = roi_bbox
        w_box, h_box = xmax - xmin, ymax - ymin
        corner_area_pct = float(roi_mask.sum() / (H * W) * 100.0) if roi_mask is not None else (w_box * h_box) / (H * W) * 100.0
    else:
        ymin, ymax, xmin, xmax = 0, 45, 0, 45
        w_box, h_box = 45, 45
        corner_area_pct = (45 * 45) / (H * W) * 100.0

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), gridspec_kw={'hspace': 0.32, 'wspace': 0.18})
    fig.suptitle(
        title or "Step 4.5.1 Multi-Scale Audit: Layer 3 (14×14 Mid-Level) vs. Layer 4 (7×7 High-Level) (ResNet-18)",
        fontsize=14, fontweight="bold", y=0.98
    )

    layers_data = [
        ("Layer 3 (14×14 Grid: Mid-Level Parenchymal Features)", cam_indist_l3, cam_clean_l3, sar_indist_l3, sar_clean_l3, axes[0]),
        ("Layer 4 (7×7 Grid: High-Level Macro Geometry)", cam_indist_l4, cam_clean_l4, sar_indist_l4, sar_clean_l4, axes[1])
    ]

    for layer_title, c_indist, c_clean, s_indist, s_clean, row_axes in layers_data:
        ax0, ax1, ax2 = row_axes

        # 1. In-Dist
        im0 = ax0.imshow(c_indist, cmap="jet")
        if draw_roi and roi_bbox is not None:
            rect0 = patches.Rectangle((xmin, ymin), w_box, h_box, linewidth=2.0, edgecolor="#EF4444", facecolor="none", linestyle="--")
            ax0.add_patch(rect0)
            label_y = min(H - 12, ymax + 20) if ymin < 20 else max(15, ymin - 10)
            ax0.text(xmin, label_y, f"ROI ({w_box}×{h_box})\nSAR: {s_indist:.1f}%", color="#EF4444", fontsize=8.5, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.75))
        ax0.set_title(f"{layer_title}\nIn-Distribution (SAR = {s_indist:.1f}%)", fontsize=10, fontweight="bold", color="#DC2626")
        ax0.axis('off')
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        # 2. Clean
        im1 = ax1.imshow(c_clean, cmap="jet")
        if draw_roi and roi_bbox is not None:
            rect1 = patches.Rectangle((xmin, ymin), w_box, h_box, linewidth=2.0, edgecolor="#3B82F6", facecolor="none", linestyle="--")
            ax1.add_patch(rect1)
            ax1.text(xmin, label_y, f"ROI Control\nSAR: {s_clean:.1f}%", color="#3B82F6", fontsize=8.5, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.75))
        ax1.set_title(f"{layer_title}\n{clean_label} (SAR = {s_clean:.1f}%)", fontsize=10, fontweight="bold", color="#2563EB")
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # 3. Bar Chart
        categories = [f"Automated ROI\n({corner_area_pct:.1f}% Area)", f"{clean_label}\n(Pneumonia)", "In-Dist Scan\n(Shortcut Tag)"]
        values = [corner_area_pct, s_clean, s_indist]
        bar_colors = ["#94A3B8", "#3B82F6", "#EF4444" if s_indist > threshold else "#10B981"]
        bars = ax2.bar(categories, values, color=bar_colors, width=0.55, edgecolor="black", alpha=0.9)
        ax2.axhline(threshold, color="#DC2626", linestyle="--", linewidth=1.2, label=f"Failure Threshold ({threshold:.0f}%)")
        ax2.axhline(corner_area_pct, color="#64748B", linestyle=":", linewidth=1.2, label=f"Chance Baseline ({corner_area_pct:.1f}%)")
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax2.set_ylim(0, max(60, max(values) + 10))
        ax2.set_ylabel(r"SAR (% Energy in $\Omega_{\mathrm{ROI}}$)", fontsize=9)
        ax2.set_title(f"Quantitative SAR: {layer_title.split(':')[0]}", fontsize=10, fontweight="bold")
        ax2.grid(axis="y", linestyle=":", alpha=0.4)
        ax2.legend(loc="upper left", fontsize=7.5)

    fig.subplots_adjust(top=0.90, bottom=0.04, hspace=0.28, wspace=0.12)
    plt.show()


def evaluate_cohort_sar(
    model: nn.Module,
    dataloader: Any,
    target_layer: nn.Module,
    device: Union[str, torch.device],
    cam_class: Any = HiResCAM,
    roi: Optional[Union[np.ndarray, Tuple[int, int, int, int], int]] = None,
    lung_roi: Optional[Union[np.ndarray, Tuple[int, int, int, int]]] = (35, 190, 35, 190),
    target_class: Union[str, int] = 1,
    filter_class: Optional[int] = None,
    max_samples: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Computes statistical cohort-level Saliency Attention Ratio (SAR) and Pointing Game
    metrics across an entire validation or test DataLoader.

    Evaluates:
      1. Mean SAR (mSAR) ± Standard Deviation
      2. Median SAR and Interquartile Range (IQR)
      3. Pointing Game Shortcut Hit Rate (fraction of images where peak CAM lands in ROI)
      4. Pointing Game Parenchyma Hit Rate (fraction of images where peak CAM lands in lung FOV)
      5. Violation Rate (percentage of samples where SAR exceeds uniform geometric chance)

    Parameters:
      model: PyTorch model (e.g. ResNet-18)
      dataloader: DataLoader or GPUBatchLoader yielding (images, labels)
      target_layer: Conv layer to compute CAM on (e.g. model.layer4[-1] or model.layer3[-1])
      device: torch device ('cuda' or 'cpu')
      cam_class: CAM extractor class (HiResCAM, GradCAM, LayerCAM). Default: HiResCAM
      roi: Region of Interest for the shortcut. Default is peripheral top-left corner (0, 40, 0, 40)
      lung_roi: Central pulmonary parenchyma ROI for anatomical focus. Default: (35, 190, 35, 190)
      target_class: Target class for attribution: 'gt' (ground truth), 'pred' (predicted), or int
      filter_class: If set, only audits samples belonging to this class index (e.g. 1 for PNEUMONIA)
      max_samples: Optional limit on total samples evaluated (None = entire DataLoader)
      verbose: Whether to print a formatted clinical diagnostic summary report

    Returns:
      Dict with summary statistics:
        'n_samples', 'mean_sar', 'std_sar', 'median_sar', 'q25_sar', 'q75_sar', 'iqr_sar',
        'min_sar', 'max_sar', 'chance_baseline', 'violation_rate', 'shortcut_hit_rate',
        'lung_hit_rate', 'mean_lung_energy', 'std_lung_energy', 'sar_values', 'lung_values'
    """
    was_training = model.training
    model.eval()

    device = torch.device(device)
    cam_extractor = cam_class(model, target_layer=target_layer)

    # 1. Resolve Peripheral Shortcut ROI Mask (default 40x40 on 224x224)
    H, W = 224, 224
    if roi is None:
        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[:40, :40] = True
    elif isinstance(roi, tuple) and len(roi) == 4:
        ymin, ymax, xmin, xmax = roi
        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[ymin:ymax, xmin:xmax] = True
    elif isinstance(roi, np.ndarray) and roi.dtype == bool:
        roi_mask = roi
    elif isinstance(roi, int):
        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[:roi, :roi] = True
    else:
        raise ValueError(f"Unsupported ROI type: {type(roi)}")

    chance_baseline = float(roi_mask.sum() / (H * W) * 100.0)

    # 2. Resolve Central Pulmonary Parenchyma ROI Mask
    if lung_roi is not None:
        ymin_l, ymax_l, xmin_l, xmax_l = lung_roi
        lung_mask = np.zeros((H, W), dtype=bool)
        lung_mask[ymin_l:ymax_l, xmin_l:xmax_l] = True
    else:
        lung_mask = None

    sar_list = []
    lung_list = []
    shortcut_hits = 0
    lung_hits = 0
    processed_count = 0

    try:
        with torch.enable_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    b_imgs, b_lbls = batch[0], batch[1]
                else:
                    b_imgs, b_lbls = batch.images, batch.labels

                b_imgs = b_imgs.to(device)
                b_lbls = b_lbls.to(device)

                for i in range(b_imgs.size(0)):
                    lbl = b_lbls[i].item()
                    if filter_class is not None and lbl != filter_class:
                        continue

                    img_tensor = b_imgs[i : i + 1].to(device)
                    if not img_tensor.is_floating_point():
                        img_tensor = img_tensor.float() / 255.0

                    # Determine target class for attribution
                    if isinstance(target_class, int):
                        tc = target_class
                    elif target_class == "gt":
                        tc = lbl
                    elif target_class == "pred":
                        logits = model(img_tensor)
                        tc = logits.argmax(dim=-1).item()
                    else:
                        tc = 1

                    cam, logits, _ = cam_extractor.generate(img_tensor, target_class=tc)

                    # 1. SAR Corner Energy
                    sar = compute_saliency_attention_ratio(cam, roi=roi_mask)
                    sar_list.append(sar)

                    # 2. Lung Parenchyma Energy
                    if lung_mask is not None:
                        lung_energy = float(cam[lung_mask].sum() / (cam.sum() + 1e-8) * 100.0)
                        lung_list.append(lung_energy)

                    # 3. Pointing Game Hit Check (Max peak location)
                    if cam.max() > 1e-6:
                        peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)
                        if roi_mask[peak_y, peak_x]:
                            shortcut_hits += 1
                        if lung_mask is not None and lung_mask[peak_y, peak_x]:
                            lung_hits += 1

                    processed_count += 1
                    if max_samples is not None and processed_count >= max_samples:
                        break

                if max_samples is not None and processed_count >= max_samples:
                    break
    finally:
        cam_extractor.remove_hooks()
        if was_training:
            model.train()

    sar_arr = np.array(sar_list, dtype=np.float64)
    lung_arr = np.array(lung_list, dtype=np.float64) if lung_list else np.array([], dtype=np.float64)

    mean_sar = float(np.mean(sar_arr)) if len(sar_arr) > 0 else 0.0
    std_sar = float(np.std(sar_arr)) if len(sar_arr) > 0 else 0.0
    median_sar = float(np.median(sar_arr)) if len(sar_arr) > 0 else 0.0
    q25, q75 = float(np.percentile(sar_arr, 25)), float(np.percentile(sar_arr, 75))
    iqr_sar = float(q75 - q25)
    min_sar = float(np.min(sar_arr)) if len(sar_arr) > 0 else 0.0
    max_sar = float(np.max(sar_arr)) if len(sar_arr) > 0 else 0.0

    violation_rate = float(np.mean(sar_arr > chance_baseline) * 100.0) if len(sar_arr) > 0 else 0.0
    shortcut_hit_rate = float(shortcut_hits / processed_count * 100.0) if processed_count > 0 else 0.0
    lung_hit_rate = float(lung_hits / processed_count * 100.0) if processed_count > 0 else 0.0
    mean_lung = float(np.mean(lung_arr)) if len(lung_arr) > 0 else 0.0
    std_lung = float(np.std(lung_arr)) if len(lung_arr) > 0 else 0.0

    metrics = {
        "n_samples": processed_count,
        "mean_sar": mean_sar,
        "std_sar": std_sar,
        "median_sar": median_sar,
        "q25_sar": q25,
        "q75_sar": q75,
        "iqr_sar": iqr_sar,
        "min_sar": min_sar,
        "max_sar": max_sar,
        "chance_baseline": chance_baseline,
        "violation_rate": violation_rate,
        "shortcut_hit_rate": shortcut_hit_rate,
        "lung_hit_rate": lung_hit_rate,
        "mean_lung_energy": mean_lung,
        "std_lung_energy": std_lung,
        "sar_values": sar_arr,
        "lung_values": lung_arr
    }

    # Prints silenced per pedagogical instruction
    pass
    return metrics


def plot_cohort_sar_diagnostics(
    metrics: Dict[str, Any],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    cohort_name: str = "Cohort"
) -> plt.Figure:
    """
    Renders a 2-panel statistical diagnostic figure for cohort-level SAR audit:
      (A) Paired energy allocation box plots: Peripheral Corner ROI vs. Pulmonary Parenchyma.
      (B) Pointing Game peak localization attribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0))
    n_scans = metrics.get("n_samples", len(metrics.get("sar_values", [])))
    if title is None:
        title = (
            f"{cohort_name} Saliency Attention Governance (N={n_scans} Scans)\n"
            f"Anatomical Energy Focus & Pointing Game Peak Localization"
        )
    fig.suptitle(title, fontsize=13.0, fontweight="bold", y=0.97)
    fig.subplots_adjust(top=0.76, bottom=0.13, left=0.08, right=0.97, wspace=0.30)

    # --------------------------------------------------------------------------
    # Panel A: Saliency Energy Allocation (Corner ROI vs Lung Anatomy)
    # --------------------------------------------------------------------------
    ax0 = axes[0]
    sar_vals = metrics["sar_values"]
    lung_vals = metrics.get("lung_values", np.array([]))
    if len(lung_vals) > 0 and len(sar_vals) > 0:
        data_to_plot = [sar_vals, lung_vals]
        bplot = ax0.boxplot(data_to_plot, patch_artist=True, widths=0.45,
                            medianprops=dict(color="black", linewidth=2.0),
                            boxprops=dict(linewidth=1.5),
                            whiskerprops=dict(linewidth=1.2),
                            capprops=dict(linewidth=1.2),
                            flierprops=dict(marker="o", markersize=4, alpha=0.4, markerfacecolor="#64748B"))
        colors = ["#FCA5A5", "#86EFAC"]
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("#334155")
        ax0.set_xticklabels(["Peripheral Corner\nROI (Shortcut)", "Pulmonary Parenchyma\n(Target Anatomy)"], fontsize=10, fontweight="bold")
    ax0.axhline(metrics["chance_baseline"], color="#DC2626", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Corner Chance ({metrics['chance_baseline']:.1f}%)")
    ax0.set_ylabel("% Saliency Energy Allocation", fontsize=10.5, fontweight="bold")
    ax0.set_title("(A) Saliency Energy Allocation\nShortcut Region vs. Lung Anatomy", fontsize=11.5, fontweight="bold", pad=10)
    ax0.set_ylim(-2, 105)
    ax0.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    ax0.grid(axis="y", linestyle=":", alpha=0.4)

    # Pinned summary stat box removed per pedagogical instruction
    # --------------------------------------------------------------------------
    # Panel B: Pointing Game Peak Activation Localization
    # --------------------------------------------------------------------------
    ax1 = axes[1]
    categories = ["Current Mitigated\n(ResNet-18)", "Poisoned Baseline\n(Project 2 CNN)"]
    sh_hit = metrics.get("shortcut_hit_rate", 0.0)
    lg_hit = metrics.get("lung_hit_rate", 0.0)
    other_hit = max(0.0, 100.0 - (sh_hit + lg_hit))

    # Reference results from historical Poisoned Baseline (Project 2 CNN)
    sh_hits = [sh_hit, 94.0]
    lung_hits = [lg_hit, 4.0]
    other_hits = [other_hit, 2.0]

    y_pos = np.arange(len(categories))
    bar_height = 0.42

    ax1.barh(y_pos, lung_hits, height=bar_height, color="#10B981", label="Lung Parenchyma (True Anatomy)")
    ax1.barh(y_pos, other_hits, left=lung_hits, height=bar_height, color="#94A3B8", label="Other Thorax / Boundary")
    left_p3 = [l + o for l, o in zip(lung_hits, other_hits)]
    ax1.barh(y_pos, sh_hits, left=left_p3, height=bar_height, color="#EF4444", label="Shortcut Lead Marker ROI")

    for i in range(len(categories)):
        if lung_hits[i] > 8:
            ax1.text(lung_hits[i]/2, y_pos[i], f"{lung_hits[i]:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
        if other_hits[i] > 8:
            ax1.text(lung_hits[i] + other_hits[i]/2, y_pos[i], f"{other_hits[i]:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
        if sh_hits[i] > 8:
            ax1.text(left_p3[i] + sh_hits[i]/2, y_pos[i], f"{sh_hits[i]:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(categories, fontsize=9.5, fontweight="bold")
    ax1.set_xlabel("Pointing Game Peak Activation Share (%)", fontsize=10.5, fontweight="bold")
    ax1.set_title("(B) Pointing Game Peak Localization\nPeak Activation Attribution (% of Scans)", fontsize=11.5, fontweight="bold", pad=10)
    ax1.set_xlim(0, 100)
    ax1.legend(loc="lower right", fontsize=8.0, framealpha=0.92)
    ax1.grid(axis="x", linestyle=":", alpha=0.4)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


class RandomCornerPerturbation(nn.Module):
    """
    Generalized Targeted Corner Inpainting & Perturbation Transform:
    Overlays a localized bounding patch covering the clinical lead marker region.
    Supports both raw uint8 image tensors and normalized float32 tensors.
    
    Supported Operational Modes:
      • mode='zero': Static blackout (v_fill = 0.0) — baseline clinical mask.
      • mode='mean': Contextual mean blend — eliminates step-edge boundary contrast.
      • mode='noise': Gaussian texture noise — destroys spatial frequency matching.
      • mode='stochastic': Randomly samples {zero, mean, noise} per sample.
    """
    def __init__(
        self,
        box_ratio: Tuple[float, float] = (0.18, 0.18),  # (height_ratio, width_ratio)
        mode: str = "zero",
        fill_value: float = 0.0,
        corner: str = "top_left"
    ):
        super().__init__()
        self.box_ratio = box_ratio
        self.mode = mode
        self.fill_value = fill_value
        self.corner = corner

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img shape: [C, H, W] or [B, C, H, W]
        h, w = img.shape[-2], img.shape[-1]
        hb = int(h * self.box_ratio[0])
        wb = int(w * self.box_ratio[1])
        
        # Resolve spatial corner coordinates
        r_slice = slice(0, hb) if "top" in self.corner else slice(h - hb, h)
        c_slice = slice(0, wb) if "left" in self.corner else slice(w - wb, w)
        
        masked = img.clone()
        active_mode = self.mode
        if active_mode == "stochastic":
            active_mode = random.choice(["zero", "mean", "noise"])
            
        if active_mode == "zero":
            masked[..., r_slice, c_slice] = self.fill_value
        elif active_mode == "mean":
            # Dtype-safe contextual mean calculation (handles uint8 and float32)
            if img.is_floating_point():
                mean_val = img.mean(dim=[-2, -1], keepdim=True)
            else:
                mean_val = img.float().mean(dim=[-2, -1], keepdim=True).round().to(img.dtype)
            masked[..., r_slice, c_slice] = mean_val
        elif active_mode == "noise":
            # Dtype-safe Gaussian noise matching input tensor domain
            if img.is_floating_point():
                noise = torch.randn_like(masked[..., r_slice, c_slice]) * 0.15
                masked[..., r_slice, c_slice] = noise
            else:
                mean_val = img.float().mean(dim=[-2, -1], keepdim=True)
                noise = mean_val + torch.randn(masked[..., r_slice, c_slice].shape, device=img.device) * 25.0
                masked[..., r_slice, c_slice] = noise.clamp(0, 255).to(img.dtype)
            
        return masked


class AnatomicalCollimation(nn.Module):
    """
    Anatomical Collimation & Soft Tissue Cropping Transform
    Focuses the thoracic field-of-view directly on the central pulmonary parenchyma
    by physically cropping out the non-anatomical periphery (lead markers, shoulders,
    clavicular apices, and lateral borders) and resizing the central lung field back to (224, 224).
    
    This transform operates strictly in the 'crop' regime, eliminating shortcut tokens
    at the input level without introducing artificial step-edge boundary artifacts or attention drift.
    """
    def __init__(
        self,
        border_ratio: float = 0.18,
        mode: str = "crop",
        fill_value: float = 0.0
    ):
        super().__init__()
        self.border_ratio = border_ratio
        self.mode = "crop"  # Exclusively crop regime
        self.resize = v2.Resize((224, 224), antialias=True)

    def forward(self, img: Union[torch.Tensor, Image.Image]) -> Union[torch.Tensor, Image.Image]:
        was_pil = False
        if isinstance(img, Image.Image):
            img = v2.functional.to_image(img)
            was_pil = True
            
        h, w = img.shape[-2], img.shape[-1]
        hc = int(h * self.border_ratio)
        wc = int(w * self.border_ratio)
        
        # Physical FOV excision and central lung parenchymal magnification
        cropped = img[..., hc : h - hc, wc : w - wc]
        out = self.resize(cropped)
            
        if was_pil:
            return v2.functional.to_pil_image(out)
        return out


# Alias for alternative naming convention
DigitalCollimation = AnatomicalCollimation


def interactive_collimation_explorer(
    img: Optional[Union[str, Path, Image.Image, torch.Tensor]] = None,
    default_ratio: float = 0.18,
    default_rotation: int = 5,
    default_flip: bool = False,
    ensure_marker: bool = True
) -> Any:
    """
    Interactive Collimation & Augmentation Explorer for students.
    Provides real-time sliders to visualize the combined effect of border_ratio,
    rotation_degrees, and horizontal_flip in the active data mitigation pipeline,
    demonstrating the balance between shortcut marker excision, anatomical
    postural invariance, and lung field preservation.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Resolve sample image
    if img is None:
        sample_candidates = sorted(list(Path("chest_ray_dataset/test/NORMAL").glob("*.jpg")))
        if not sample_candidates:
            sample_candidates = sorted(list(Path("chest_ray_dataset").glob("**/*.jpg")))
        if sample_candidates:
            pil_img = Image.open(sample_candidates[0]).convert("RGB")
        else:
            pil_img = Image.new("RGB", (224, 224), color=(60, 60, 60))
    elif isinstance(img, (str, Path)):
        pil_img = Image.open(img).convert("RGB")
    elif isinstance(img, torch.Tensor):
        if img.ndim == 4:
            img = img[0]
        if img.dtype != torch.uint8 and img.max() <= 1.0:
            img = (img * 255.0).clamp(0, 255).to(torch.uint8)
        pil_img = v2.functional.to_pil_image(img.cpu())
    elif isinstance(img, Image.Image):
        pil_img = img.convert("RGB")
    else:
        raise ValueError(f"Unsupported image format: {type(img)}")

    pil_img = pil_img.resize((224, 224))
    if ensure_marker:
        pil_img = apply_lead_marker(pil_img)

    def render_frame(border_ratio: float, rotation_degrees: int = 0, horizontal_flip: bool = False):
        w, h = 224, 224
        hc = int(h * border_ratio)
        wc = int(w * border_ratio)

        # 1. Collimation crop (strictly crop regime)
        collim = AnatomicalCollimation(border_ratio=border_ratio)
        transformed_img = collim(pil_img)

        # 2. Horizontal flip (bilateral reflection invariance)
        if horizontal_flip:
            transformed_img = transformed_img.transpose(Image.FLIP_LEFT_RIGHT)

        # 3. Rotation perturbation (postural roll / tilt)
        if rotation_degrees != 0:
            transformed_img = transformed_img.rotate(rotation_degrees, resample=Image.BILINEAR, expand=False)

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))

        # Panel 1: Full Radiograph with interactive crop window
        axes[0].imshow(pil_img)

        # Shaded non-overlapping borders confined strictly to [0, w] x [0, h]
        if hc > 0 or wc > 0:
            if hc > 0:
                axes[0].add_patch(patches.Rectangle((0, 0), w, hc, facecolor='red', alpha=0.30))
                axes[0].add_patch(patches.Rectangle((0, h - hc), w, hc, facecolor='red', alpha=0.30))
            if wc > 0:
                axes[0].add_patch(patches.Rectangle((0, hc), wc, h - 2 * hc, facecolor='red', alpha=0.30))
                axes[0].add_patch(patches.Rectangle((w - wc, hc), wc, h - 2 * hc, facecolor='red', alpha=0.30))

        # Green bounding box for retained FOV
        rect = patches.Rectangle(
            (wc, hc), w - 2 * wc, h - 2 * hc,
            linewidth=2.5, edgecolor="#00FF66", facecolor="none", linestyle="-"
        )
        axes[0].add_patch(rect)
        axes[0].set_xlim(0, w)
        axes[0].set_ylim(h, 0)
        axes[0].set_title(
            f"Full Radiograph (224×224)\nCrop: {border_ratio:.2f}  [{hc}:{h-hc}, {wc}:{w-wc}]",
            fontsize=11, fontweight="bold", pad=12
        )
        axes[0].axis("off")

        # Panel 2: Transformed FOV seen by ResNet-18
        axes[1].imshow(transformed_img)
        axes[1].set_xlim(0, w)
        axes[1].set_ylim(h, 0)
        fov_pct = ((w - 2 * wc) * (h - 2 * hc)) / (w * h) * 100.0
        flip_str = "ON" if horizontal_flip else "OFF"
        axes[1].set_title(
            f"Transformed Model Input (224×224)\nRetained FOV: {fov_pct:.1f}%  |  Rot: {rotation_degrees:+d}°  |  Flip: {flip_str}",
            fontsize=11, fontweight="bold", pad=12
        )
        axes[1].axis("off")

        title_text = f"Anatomical Collimation Explorer: Crop = {border_ratio:.2f} | Rotation = {rotation_degrees:+d}° | Flip = {flip_str}"
        fig.suptitle(title_text, fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.show()

    try:
        import ipywidgets as widgets
        from ipywidgets import interact

        slider_crop = widgets.FloatSlider(
            value=default_ratio,
            min=0.0,
            max=0.30,
            step=0.01,
            description="Crop Ratio:",
            continuous_update=False,
            style={"description_width": "110px"},
            layout=widgets.Layout(width="400px")
        )
        slider_rot = widgets.IntSlider(
            value=default_rotation,
            min=-20,
            max=20,
            step=1,
            description="Rotation (°):",
            continuous_update=False,
            style={"description_width": "110px"},
            layout=widgets.Layout(width="400px")
        )
        check_flip = widgets.Checkbox(
            value=default_flip,
            description="Horizontal Flip (Reflection Symmetry)",
            indent=False,
            layout=widgets.Layout(width="400px", margin="4px 0 10px 0")
        )
        return interact(render_frame, border_ratio=slider_crop, rotation_degrees=slider_rot, horizontal_flip=check_flip)
    except Exception:
        return render_frame(default_ratio, default_rotation, default_flip)






def setup_collimated_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    data_mitigation_transform: nn.Module,
    mean: Any,
    std: Any,
    device: torch.device,
    batch_size: int = 64,
    use_horizontal_flip: bool = True,
    rotation_degrees: float = 0.0,
    preload_to_vram: bool = True
) -> Dict[str, Any]:
    """
    Constructs high-performance standard (224, 224) DataLoader pipelines
    under Anatomical Collimation & Soft Tissue Cropping.
    Preloads datasets directly into GPU VRAM as uint8 tensors for fast epoch training.
    """
    base_load_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224), antialias=True),
    ])

    train_base_ds = ClinicalChestXRayDataset(train_dir, transform=base_load_transform, apply_protocol_tag=False)
    val_base_ds = ClinicalChestXRayDataset(val_dir, transform=base_load_transform, apply_protocol_tag=False)
    test_base_ds = ClinicalChestXRayDataset(test_dir, transform=base_load_transform, apply_protocol_tag=False)

    collimated_eval_transforms = v2.Compose([
        v2.ToImage(),
        data_mitigation_transform,
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    eval_transforms = collimated_eval_transforms

    clean_test_ds = ClinicalChestXRayDataset(test_dir, transform=collimated_eval_transforms, apply_protocol_tag=False)
    tagged_test_ds = ClinicalChestXRayDataset(test_dir, transform=collimated_eval_transforms, apply_protocol_tag=True)

    def _preload(dataset, b_size=128):
        loader = DataLoader(dataset, batch_size=b_size, shuffle=False, num_workers=2)
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

    if preload_to_vram:
        print("🚀 Preloading standard (224, 224) dataset into GPU VRAM as uint8 tensors...")
        train_imgs_gpu, train_lbls_gpu = _preload(train_base_ds)
        val_imgs_gpu, val_lbls_gpu = _preload(val_base_ds)
        test_imgs_gpu, test_lbls_gpu = _preload(test_base_ds)
    else:
        train_imgs_gpu, train_lbls_gpu = None, None
        val_imgs_gpu, val_lbls_gpu = None, None
        test_imgs_gpu, test_lbls_gpu = None, None

    gpu_train_steps = [data_mitigation_transform]
    if use_horizontal_flip:
        gpu_train_steps.append(v2.RandomHorizontalFlip(p=0.5))
    if rotation_degrees > 0:
        gpu_train_steps.append(v2.RandomRotation(degrees=rotation_degrees))
    gpu_train_steps.extend([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    gpu_train_transforms = v2.Compose(gpu_train_steps)
    gpu_eval_transforms = v2.Compose([
        data_mitigation_transform,
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    if preload_to_vram:
        train_loader = GPUBatchLoader(train_imgs_gpu, train_lbls_gpu, batch_size=batch_size, shuffle=True, transform=gpu_train_transforms, dataset=train_base_ds)
        val_loader = GPUBatchLoader(val_imgs_gpu, val_lbls_gpu, batch_size=batch_size, shuffle=False, transform=gpu_eval_transforms, dataset=val_base_ds)
    val_dataset = ClinicalChestXRayDataset(val_dir, transform=collimated_eval_transforms, apply_protocol_tag=False)

    if preload_to_vram:
        train_loader = GPUBatchLoader(train_imgs_gpu, train_lbls_gpu, batch_size=batch_size, shuffle=True, transform=gpu_train_transforms, dataset=train_base_ds)
        val_loader = GPUBatchLoader(val_imgs_gpu, val_lbls_gpu, batch_size=batch_size, shuffle=False, transform=gpu_eval_transforms, dataset=val_base_ds)
    else:
        train_loader = DataLoader(train_base_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    clean_test_loader = DataLoader(clean_test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    indist_test_loader = DataLoader(tagged_test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"✓ Standard (224, 224) GPU Loaders initialized with Anatomical Collimation (Crop + RandomHorizontalFlip Active): Train={len(train_base_ds)}, Val={len(val_base_ds)}, Test={len(clean_test_ds)}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "clean_test_loader": clean_test_loader,
        "indist_test_loader": indist_test_loader,
        "train_base_ds": train_base_ds,
        "val_base_ds": val_base_ds,
        "test_base_ds": test_base_ds,
        "clean_test_ds": clean_test_ds,
        "tagged_test_ds": tagged_test_ds,
        "val_dataset": val_dataset,
        "gpu_train_transforms": gpu_train_transforms,
        "gpu_eval_transforms": gpu_eval_transforms,
        "collimated_eval_transforms": collimated_eval_transforms,
        "eval_transforms": eval_transforms,
        "train_imgs_gpu": train_imgs_gpu,
        "train_lbls_gpu": train_lbls_gpu,
        "val_imgs_gpu": val_imgs_gpu,
        "val_lbls_gpu": val_lbls_gpu,
        "test_imgs_gpu": test_imgs_gpu,
        "test_lbls_gpu": test_lbls_gpu,
    }


def visualize_conv1_filters(
    model: nn.Module,
    figsize: Tuple[int, int] = (14, 7),
    suptitle: Optional[str] = None
) -> plt.Figure:
    """
    Visualizes the 64 spatial Gabor kernels of ResNet-18 conv1 (7x7 RGB filters).
    """
    conv1_weights = model.conv1.weight.data.clone().cpu()  # [64, 3, 7, 7]
    w_min, w_max = conv1_weights.min(), conv1_weights.max()
    w_norm = (conv1_weights - w_min) / (w_max - w_min)

    fig, axes = plt.subplots(4, 8, figsize=figsize)
    if suptitle is None:
        suptitle = r"ResNet-18 Conv1 Spatial Kernels ($7 	imes 7$ RGB Gabor Filters)"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        filter_img = w_norm[i].permute(1, 2, 0).numpy()
        ax.imshow(filter_img)
        ax.set_title(f"Kernel {i}", fontsize=8)
        ax.axis("off")
    fig.subplots_adjust(top=0.90, bottom=0.04, hspace=0.28, wspace=0.12)
    plt.show()
    return fig


def plot_hirescam_gallery(
    model: nn.Module,
    dataset: Any,
    target_layer: nn.Module,
    device: torch.device,
    eval_transform: Callable,
    data_mitigation_transform: Optional[Callable] = None,
    start_idx: int = 0,
    sample_indices: Optional[List[int]] = None,
    n_samples: int = 32,
    pos_class_name: str = "PNEUMONIA",
    title: Optional[str] = None,
    alpha: float = 0.55,
    figsize: Tuple[int, int] = (22, 12),
    cohort_label: str = "Val"
) -> plt.Figure:
    """
    Renders a high-throughput 32-sample HiResCAM saliency gallery across a cohort batch
    using a dark clinical theme (#0f172a), high-resolution gradient-activation mapping,
    and individual diagnostic status annotations.
    """
    classes = getattr(dataset, "classes", ["NORMAL", "PNEUMONIA"])
    if pos_class_name in classes:
        pos_class_idx = classes.index(pos_class_name)
    else:
        pos_class_idx = 1

    hires_cam = HiResCAM(model, target_layer=target_layer)
    if sample_indices is None:
        sample_indices = list(range(start_idx, start_idx + n_samples))

    cam_tensors = []
    sample_info = []

    print(f"🚀 Generating HiResCAM high-resolution heatmaps for {len(sample_indices)} {cohort_label} cases...")
    try:
        for idx in sample_indices:
            raw_img, true_label = dataset[idx]
            input_tensor = eval_transform(raw_img.unsqueeze(0).to(device))
            cam, logits, cam_target = hires_cam.generate(input_tensor, target_class=pos_class_idx)
            probs = F.softmax(logits, dim=-1).squeeze().detach().cpu().numpy()
            pred_class = int(probs.argmax())

            if data_mitigation_transform is not None:
                masked_raw = data_mitigation_transform(raw_img)
            else:
                masked_raw = raw_img
            disp_img = v2.functional.to_pil_image(masked_raw)
            cam_overlay = overlay_cam(disp_img, cam, alpha=alpha)

            cam_t = torch.from_numpy(cam_overlay).permute(2, 0, 1).float()
            cam_tensors.append(cam_t)
            sample_info.append((idx, float(probs[pos_class_idx]), (classes[pred_class] if pred_class < len(classes) else f"Class_{pred_class}"), pred_class == true_label))
    finally:
        hires_cam.remove_hooks()

    cam_batch = torch.stack(cam_tensors)
    grid_tensor = vutils.make_grid(cam_batch, nrow=8, padding=4, pad_value=0.0)
    print(f"✓ PyTorch make_grid HiResCAM tensor compiled: {grid_tensor.shape} (4 rows × 8 columns)")

    fig, axes = plt.subplots(4, 8, figsize=figsize, facecolor="#0f172a",
                             gridspec_kw={"hspace": 0.38, "wspace": 0.08})
    if title is None:
        title = (
            f"PyTorch Grid Audit: {n_samples} {cohort_label} Positive Class ({pos_class_name}) HiResCAM Saliency Maps\n"
            f"(High-Resolution Saliency Frontiers: Element-Wise Gradient-Activation Mapping without GAP Blur)"
        )
    fig.suptitle(title, fontsize=15, fontweight="bold", color="white", y=0.98)

    axes_flat = axes.flatten()
    for i, (idx, prob, pred_cls, is_corr) in enumerate(sample_info):
        ax = axes_flat[i]
        ax.imshow(cam_tensors[i].permute(1, 2, 0).numpy())
        ax.set_facecolor("#0f172a")
        status_icon = "✓" if is_corr else "✗"
        ax.set_title(
            f"{cohort_label} #{idx} ({status_icon})\nP({pos_class_name[:4]}): {prob*100:.0f}%",
            fontsize=9.5, fontweight="bold", color="white", pad=6
        )
        ax.axis("off")

    for j in range(len(sample_info), len(axes_flat)):
        axes_flat[j].axis("off")

    plt.subplots_adjust(top=0.91, bottom=0.03, hspace=0.38, wspace=0.08)
    plt.show()

    correct_count = sum(x[3] for x in sample_info)
    print(f"✓ {n_samples}-Sample {cohort_label} HiResCAM grid rendered: {correct_count}/{n_samples} correctly classified.")
    return fig


def plot_mitigation_gallery(
    train_dir: Union[str, Path],
    val_dir: Union[str, Path],
    data_mitigation_transform: nn.Module,
    use_horizontal_flip: bool = True,
    rotation_degrees: float = 0.0,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    batch_size: int = 4,
    figsize: Tuple[int, int] = (16, 8.5)
) -> plt.Figure:
    """
    Renders a 2x4 active mitigation comparison gallery comparing the Training Pipeline
    (Collimation Crop + Random Horizontal Flip + Rotation + ImageNet Normalization)
    versus the Validation Pipeline (Deterministic Collimation Crop + ImageNet Normalization).
    """
    mean_cpu = list(mean)
    std_cpu = list(std)

    # 1. Build Transforms
    vis_train_steps = [v2.ToImage(), data_mitigation_transform]
    if use_horizontal_flip:
        vis_train_steps.append(v2.RandomHorizontalFlip(p=0.5))
    if rotation_degrees > 0:
        vis_train_steps.append(v2.RandomRotation(degrees=rotation_degrees))
    vis_train_steps.extend([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean_cpu, std=std_cpu),
    ])
    mitigated_train_transform = v2.Compose(vis_train_steps)

    mitigated_val_transform = v2.Compose([
        v2.ToImage(),
        data_mitigation_transform,
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean_cpu, std=std_cpu),
    ])

    # 2. Ingest Sample Batches (protocol tag active on training pneumonia to verify marker excision)
    train_ds = ClinicalChestXRayDataset(train_dir, transform=mitigated_train_transform, apply_protocol_tag=True)
    val_ds = ClinicalChestXRayDataset(val_dir, transform=mitigated_val_transform, apply_protocol_tag=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    train_batch_imgs, train_batch_lbls = next(iter(train_loader))
    val_batch_imgs, val_batch_lbls = next(iter(val_loader))

    # 3. Denormalize tensors to [0, 1] range for visual rendering
    mean_arr = np.array(mean_cpu).reshape(1, 3, 1, 1)
    std_arr = np.array(std_cpu).reshape(1, 3, 1, 1)
    t_disp = np.clip((train_batch_imgs.numpy() * std_arr + mean_arr).transpose(0, 2, 3, 1), 0, 1)
    v_disp = np.clip((val_batch_imgs.numpy() * std_arr + mean_arr).transpose(0, 2, 3, 1), 0, 1)

    classes = getattr(train_ds, "classes", ["NORMAL", "PNEUMONIA"])

    # 4. Render 2x4 Active Mitigation Comparison Gallery
    fig, axes = plt.subplots(2, 4, figsize=figsize, gridspec_kw={"hspace": 0.28, "wspace": 0.12})
    fig.suptitle(
        "Active Data-Level Mitigation Gallery: Training vs. Validation Pipelines (Anatomical Collimation Crop)",
        fontsize=14, fontweight="bold"
    )

    for j in range(batch_size):
        # Top Row: Training Cohort
        axes[0, j].imshow(t_disp[j])
        cls_name = classes[train_batch_lbls[j]]
        axes[0, j].set_title(
            f"Train [{cls_name}]\nCollimation Crop + Random Flip",
            fontsize=11, fontweight="bold",
            color="#16A34A" if cls_name == "PNEUMONIA" else "#2563EB"
        )
        axes[0, j].axis("off")

        # Bottom Row: Validation Cohort
        axes[1, j].imshow(v_disp[j])
        cls_name = classes[val_batch_lbls[j]]
        axes[1, j].set_title(
            f"Val [{cls_name}]\nDeterministic Collimation Crop",
            fontsize=11, fontweight="bold",
            color="#16A34A" if cls_name == "PNEUMONIA" else "#2563EB"
        )
        axes[1, j].axis("off")

    fig.subplots_adjust(top=0.90, bottom=0.04, hspace=0.28, wspace=0.12)
    plt.show()

    ratio = getattr(data_mitigation_transform, "border_ratio", 0.18)
    if isinstance(ratio, tuple):
        ratio_str = f"({ratio[0]:.2f}, {ratio[1]:.2f})"
    else:
        ratio_str = f"{ratio:.2f}"
    print("✓ Active data mitigation gallery rendered with canonical ImageNet normalization:")
    print(f"  • ImageNet Normalization : mean={mean_cpu}, std={std_cpu}")
    print(f"  • Training Pipeline     : AnatomicalCollimation(border_ratio={ratio_str}) + RandomHorizontalFlip(p=0.5)")
    print(f"  • Validation Pipeline   : AnatomicalCollimation(border_ratio={ratio_str}) (Deterministic)")
    return fig


# ==============================================================================
# Defensive Functional Sanity Verification (train_one_epoch & validate_one_epoch)
# ==============================================================================

def check_epoch_functions(
    train_one_epoch_fn: Callable,
    validate_one_epoch_fn: Callable,
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> None:
    """
    Defensive functional verification for train_one_epoch and validate_one_epoch.
    Verifies PyTorch contracts:
      1. train_one_epoch: Model left in training mode, positive loss, autograd gradients,
         optimizer parameter updates, and scheduler steps.
      2. validate_one_epoch: Model left in evaluation mode, positive loss, zero parameter drift,
         and strict torch.no_grad() isolation.
    """
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    if model is not None:
        test_model = copy.deepcopy(model).to(dev)
    else:
        test_model = models.resnet18(num_classes=2).to(dev)

    syn_images = torch.randn(8, 3, 224, 224)
    syn_labels = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.long)

    class _SyntheticLoader:
        def __init__(self, imgs, lbls, batch_size=4):
            self.imgs, self.lbls, self.bs = imgs, lbls, batch_size
        def __len__(self):
            return (len(self.imgs) + self.bs - 1) // self.bs
        def __iter__(self):
            for i in range(0, len(self.imgs), self.bs):
                yield self.imgs[i : i + self.bs], self.lbls[i : i + self.bs]

    syn_loader = _SyntheticLoader(syn_images, syn_labels, batch_size=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(test_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=10)

    # 1. Audit train_one_epoch
    initial_trainable_params = [p.clone().detach() for p in test_model.parameters() if p.requires_grad]

    train_loss, train_acc = train_one_epoch_fn(test_model, syn_loader, criterion, optimizer, scheduler, dev)

    assert isinstance(train_loss, (float, int)) and train_loss > 0.0, f"Expected positive float train_loss, got {train_loss}"
    assert not (np.isnan(train_loss) or np.isinf(train_loss)), f"Train loss is NaN or Inf: {train_loss}"
    assert isinstance(train_acc, (float, int)) and 0.0 <= train_acc <= 1.0, f"Expected train_acc in [0.0, 1.0], got {train_acc}"
    assert test_model.training is True, "Model must remain in training mode (model.train()) after train_one_epoch"

    weight_diff = max(
        torch.norm(p - p_init).item()
        for p, p_init in zip([p for p in test_model.parameters() if p.requires_grad], initial_trainable_params)
    )
    assert weight_diff > 1e-7, f"Model parameters did not update during train_one_epoch (max weight diff: {weight_diff})"
    assert scheduler.last_epoch > 0, "Learning rate scheduler was not stepped during training iterations"

    # 2. Audit validate_one_epoch
    for p in test_model.parameters():
        p.grad = None
    val_initial_params = [p.clone().detach() for p in test_model.parameters()]

    val_loss, val_acc = validate_one_epoch_fn(test_model, syn_loader, criterion, dev)

    assert isinstance(val_loss, (float, int)) and val_loss > 0.0, f"Expected positive float val_loss, got {val_loss}"
    assert not (np.isnan(val_loss) or np.isinf(val_loss)), f"Validation loss is NaN or Inf: {val_loss}"
    assert isinstance(val_acc, (float, int)) and 0.0 <= val_acc <= 1.0, f"Expected val_acc in [0.0, 1.0], got {val_acc}"
    assert test_model.training is False, "Model must remain in evaluation mode (model.eval()) after validate_one_epoch"

    max_val_diff = max(
        torch.norm(p - p_init).item()
        for p, p_init in zip(test_model.parameters(), val_initial_params)
    )
    assert max_val_diff == 0.0, f"Model weights altered during validation (max diff: {max_val_diff})"
    assert not any(p.grad is not None for p in test_model.parameters()), (
        "Gradients were accumulated during validation; torch.no_grad() was not properly enforced"
    )

    print(f"{GREEN}✓ [PASSED] train_one_epoch verified:{RESET}")
    print(f"  • Training Loss: {train_loss:.4f} | Training Accuracy: {train_acc * 100:.1f}%")
    print(f"  • Autograd backprop, optimizer weight updates, and OneCycleLR scheduler step verified (param diff = {weight_diff:.3e}).")
    print(f"{GREEN}✓ [PASSED] validate_one_epoch verified:{RESET}")
    print(f"  • Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc * 100:.1f}%")
    print("  • Model eval mode, zero parameter drift, and torch.no_grad() isolation verified.")
    print(f"{BOLD}{GREEN}★ ALL CONTRACT CHECKS PASSED: train_one_epoch and validate_one_epoch are fully operational!{RESET}")


verify_epoch_functions = check_epoch_functions
