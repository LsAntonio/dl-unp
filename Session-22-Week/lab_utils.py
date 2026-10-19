"""
lab_utils.py - Clinical Vision & Diagnostic Deep Learning Helper Utilities
================================================================================
Boilerplate and visualization helpers for the Chest X-Ray Pneumonia Detection Pipeline.
Used in Week 8 Master Project (UNP Deep Learning Course).

Includes:
  - set_global_seed: Seed configuration across PyTorch, NumPy, and Python RNGs.
  - plot_verification_gallery: 1x4 visual verification gallery for raw cohort radiographs.
  - plot_channel_histograms: 1x3 RGB pixel density distributions with mean/std markers.
  - plot_preprocessed_batch: 2x4 standardized batch visualization with denormalization.
"""

import os
import random
from pathlib import Path
from typing import Union, List, Tuple, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch


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
        # Invert standardization: X_orig = (X_norm * std) + mean
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

    # Hide any unused subplots
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
