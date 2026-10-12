"""
Session 19 Lab Utilities: Production Computer Vision Pipelines & Empirical Calibration
Course: UNP Deep Learning Foundations — Unit II (Week 7, Session 19)
Institutional Standard: MIT EECS / ETH Zürich D-INFK
Instructors: MSc. Antonio Aguilar & Dr. Luis Aguilar Ibáñez

This module encapsulates all computer vision visualization utilities:
  1. plot_raw_samples: Side-by-side inspection of raw PIL image samples with metadata.
  2. plot_channel_distributions: 3-channel RGB pixel intensity histograms with empirical mean & std markers.
  3. plot_transformed_pipeline: Comprehensive two-tier visualization:
       - Tier 1: Raw scaled [0, 1] vs. standardized Z-score representation.
       - Tier 2: Transformed & augmented training batch samples (224x224) with denormalization & stats.
  4. plot_random_samples: Multi-sample batch visualization grid for arbitrary tensor batches.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets


# ==============================================================================
# 1. Raw PIL Sample Inspection
# ==============================================================================
def plot_raw_samples(
    raw_cat_img: Image.Image,
    raw_dog_img: Image.Image,
    title_cat: str = "Raw Cat Sample",
    title_dog: str = "Raw Dog Sample",
    figsize: Tuple[int, int] = (10, 4)
) -> Tuple[plt.Figure, Any]:
    """
    Renders side-by-side visual inspection of raw PIL image samples,
    displaying resolution, color mode, and image format metadata.
    """
    print(f"\nRaw Cat Sample [0]: Size={raw_cat_img.size}, Mode={raw_cat_img.mode}, Format={raw_cat_img.format}")
    print(f"Raw Dog Sample [0]: Size={raw_dog_img.size}, Mode={raw_dog_img.mode}, Format={raw_dog_img.format}")

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(raw_cat_img)
    axes[0].set_title(f"{title_cat}\nResolution: {raw_cat_img.size} ({raw_cat_img.mode})", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(raw_dog_img)
    axes[1].set_title(f"{title_dog}\nResolution: {raw_dog_img.size} ({raw_dog_img.mode})", fontsize=11)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    return fig, axes


# ==============================================================================
# 2. Empirical Channel Distribution Histograms
# ==============================================================================
def plot_channel_distributions(
    batch_tensors: torch.Tensor,
    mean: Union[Sequence[float], np.ndarray, torch.Tensor],
    std: Union[Sequence[float], np.ndarray, torch.Tensor],
    colors: Sequence[str] = ('crimson', 'forestgreen', 'royalblue'),
    channel_names: Sequence[str] = ('Red Channel', 'Green Channel', 'Blue Channel'),
    bins: int = 50,
    figsize: Tuple[int, int] = (15, 4)
) -> Tuple[plt.Figure, Any]:
    """
    Visualizes empirical 3-channel (RGB) intensity density histograms from an image batch,
    overlaying calculated empirical mean and +/- 1 standard deviation intervals.
    """
    if isinstance(mean, torch.Tensor):
        mean = mean.cpu().numpy()
    if isinstance(std, torch.Tensor):
        std = std.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for c in range(3):
        vals = batch_tensors[:, c, :, :].detach().cpu().flatten().numpy()
        axes[c].hist(vals, bins=bins, color=colors[c], alpha=0.75, density=True)
        axes[c].axvline(mean[c], color='black', linestyle='--', linewidth=1.5, label=f'Mean: {mean[c]:.3f}')
        axes[c].axvline(mean[c] - std[c], color='gray', linestyle=':', label='-1 Std')
        axes[c].axvline(mean[c] + std[c], color='gray', linestyle=':', label='+1 Std')
        axes[c].set_title(f"{channel_names[c]} Distribution", fontsize=11)
        axes[c].set_xlabel("Pixel Value [0, 1]")
        axes[c].set_ylabel("Density")
        axes[c].legend(fontsize=9)

    plt.tight_layout()
    plt.show()
    #return fig, axes


# ==============================================================================
# 3. Transformed Images Pipeline: Standardization & Augmentation Inspection
# ==============================================================================
def plot_transformed_pipeline(
    raw_cat_img: Image.Image,
    raw_dog_img: Image.Image,
    base_transform: Callable,
    eval_transforms: Callable,
    train_transforms: Callable,
    train_dir: Union[str, Path],
    mean_cpu: Union[Sequence[float], np.ndarray, torch.Tensor],
    std_cpu: Union[Sequence[float], np.ndarray, torch.Tensor],
    classes: Sequence[str] = ('Cat', 'Dog'),
    batch_size: int = 4,
    figsize: Tuple[int, int] = (16, 7.5)
) -> Tuple[plt.Figure, Any]:
    """
    Renders comprehensive two-tier visual inspection of the preprocessed vision pipeline:
      - Row 1: Raw Scaled [0, 1] vs. Standardized Z-Score Tensor (revealing zero-centered contrast).
      - Row 2: Transformed & Augmented Training Batch Samples (224x224, RandomHorizontalFlip, Normalize)
               denormalized to natural RGB for spatial transform validation.
    """
    fig, axes = plt.subplots(2, 4, figsize=figsize)

    if isinstance(mean_cpu, torch.Tensor):
        mean_tensor = mean_cpu.detach().cpu().view(3, 1, 1).float()
    else:
        mean_tensor = torch.tensor(mean_cpu, dtype=torch.float32).view(3, 1, 1)

    if isinstance(std_cpu, torch.Tensor):
        std_tensor = std_cpu.detach().cpu().view(3, 1, 1).float()
    else:
        std_tensor = torch.tensor(std_cpu, dtype=torch.float32).view(3, 1, 1)

    # --------------------------------------------------------------------------
    # Row 1: Individual Sample Inspection (Cat & Dog)
    # --------------------------------------------------------------------------
    for i, (raw_img, cls_name) in enumerate(zip([raw_cat_img, raw_dog_img], classes)):
        scaled_t = base_transform(raw_img)
        std_t = eval_transforms(raw_img)

        # 1. Raw Scaled [0, 1]
        col_idx = i * 2
        scaled_np = scaled_t.detach().cpu().permute(1, 2, 0).numpy()
        axes[0, col_idx].imshow(scaled_np)
        axes[0, col_idx].set_title(
            f"Raw Scaled ({cls_name})\nRange: [{scaled_t.min():.2f}, {scaled_t.max():.2f}] | μ={scaled_t.mean():.2f}",
            fontsize=10
        )
        axes[0, col_idx].axis("off")

        # 2. Standardized Z-Score representation (rescaled for display to reveal zero-centered contrast)
        std_norm = (std_t - std_t.min()) / (std_t.max() - std_t.min() + 1e-8)
        std_vis = std_norm.detach().cpu().permute(1, 2, 0).numpy()
        axes[0, col_idx + 1].imshow(std_vis)
        axes[0, col_idx + 1].set_title(
            f"Standardized Z-Score ({cls_name})\nRange: [{std_t.min():.2f}, {std_t.max():.2f}] | μ={std_t.mean():.2f}",
            fontsize=10
        )
        axes[0, col_idx + 1].axis("off")

    # --------------------------------------------------------------------------
    # Row 2: Transformed & Augmented Training Batch Samples from DataLoader
    # --------------------------------------------------------------------------
    train_preview_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transforms)
    preview_loader = DataLoader(train_preview_dataset, batch_size=batch_size, shuffle=True)
    aug_images, aug_labels = next(iter(preview_loader))

    for j in range(min(batch_size, 4)):
        # Denormalize to natural RGB for visual validation of spatial transforms (flipping/resizing)
        img_unnorm = (aug_images[j].cpu() * std_tensor + mean_tensor).clamp(0, 1)
        aug_vis = img_unnorm.permute(1, 2, 0).numpy()
        label_idx = int(aug_labels[j].item())
        label_str = classes[label_idx] if label_idx < len(classes) else str(label_idx)
        z_mean, z_std = aug_images[j].mean().item(), aug_images[j].std().item()

        axes[1, j].imshow(aug_vis)
        axes[1, j].set_title(
            f"Augmented Train Sample [{j+1}]\nClass: {label_str} (224x224)\nZ-Stats: μ={z_mean:.2f}, σ={z_std:.2f}",
            fontsize=10
        )
        axes[1, j].axis("off")

    plt.suptitle("Transformed Images Pipeline: Standardization & Augmentation Inspection", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

    print("✓ Transformed images successfully plotted and verified!")
    #return fig, axes


# ==============================================================================
# 4. Multi-Sample Batch Visual Grid
# ==============================================================================
def plot_random_samples(
    images: torch.Tensor,
    labels: Union[List[Any], torch.Tensor],
    class_names: Optional[Sequence[str]] = ('cat', 'dog'),
    num_samples: int = 10,
    figsize: Tuple[int, int] = (15, 6)
) -> Tuple[plt.Figure, Any]:
    """
    Renders an arbitrary 2x5 grid of batch images from a tensor batch [B, C, H, W].
    """
    total = images.size(0)
    idxs = np.random.choice(total, size=min(num_samples, total), replace=False)
    
    rows = 2
    cols = (len(idxs) + 1) // 2
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten()

    for k, idx in enumerate(idxs):
        img_np = images[idx].detach().cpu().permute(1, 2, 0).numpy()
        if img_np.min() < 0:
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        axs[k].imshow(img_np)
        label_val = labels[idx].item() if isinstance(labels, torch.Tensor) else labels[idx]
        title_str = class_names[label_val] if class_names and isinstance(label_val, int) and label_val < len(class_names) else f"Class {label_val}"
        axs[k].set_title(title_str, fontsize=10)
        axs[k].axis("off")

    for k in range(len(idxs), len(axs)):
        axs[k].axis("off")

    plt.tight_layout()
    plt.show()
    return fig, axs


# ==============================================================================
# 5. Transform Pipeline Architectural Validation Suite
# ==============================================================================
def validate_transforms(
    train_transforms: Any = None,
    eval_transforms: Any = None,
    mean_cpu: Optional[Any] = None,
    std_cpu: Optional[Any] = None,
    *,
    train_transform: Any = None,
    eval_transform: Any = None,
) -> None:
    """
    Automated architectural and operational assertion suite for Session 19 transforms.
    
    Validates:
      1. Instantiation, callability, and container type (v2.Compose).
      2. Strict order of operations:
           - train_transforms: ToImage -> ToDtype(scale=True) -> Resize(224) -> RandomHorizontalFlip -> Normalize(mean, std)
           - eval_transforms:  ToImage -> ToDtype(scale=True) -> Resize(224) -> Normalize(mean, std)
      3. Cardinal Rule of Vision Pipelines:
           - train_transforms contains stochastic RandomHorizontalFlip(p=0.5).
           - eval_transforms strictly OMITS stochastic operations (deterministic audit).
      4. Dynamic range scaling (scale=True) and spatial homogenization (224x224).
      5. Forward propagation sanity on synthetic PIL test images (tensor shape [3, 224, 224], float32).
      6. Terminal normalization invariant (Normalize is the final operator).
    """
    # Accept singular variable aliases if provided
    if train_transforms is None and train_transform is not None:
        train_transforms = train_transform
    if eval_transforms is None and eval_transform is not None:
        eval_transforms = eval_transform

    print("=" * 72)
    print("🧪 RUNNING VISION TRANSFORMATION PIPELINE ASSERTION SUITE")
    print("=" * 72)

    # --------------------------------------------------------------------------
    # [1/5] Existence & Type Validation
    # --------------------------------------------------------------------------
    print("\n[1/5] Verifying Pipeline Instantiation & Container Types...")
    assert train_transforms is not None, (
        "❌ Missing Pipeline: `train_transforms` is None or undefined! "
        "Ensure you have defined `train_transforms = v2.Compose([...])`."
    )
    assert eval_transforms is not None, (
        "❌ Missing Pipeline: `eval_transforms` is None or undefined! "
        "Ensure you have defined `eval_transforms = v2.Compose([...])`."
    )
    assert callable(train_transforms), (
        f"❌ Type Error: `train_transforms` must be callable, got {type(train_transforms)}."
    )
    assert callable(eval_transforms), (
        f"❌ Type Error: `eval_transforms` must be callable, got {type(eval_transforms)}."
    )
    print("      * Both pipelines instantiated and callable ✅")

    # --------------------------------------------------------------------------
    # [2/5] Operator Sequence & Structural Audit
    # --------------------------------------------------------------------------
    print("\n[2/5] Auditing Operator Sequence & Order of Operations...")
    if hasattr(train_transforms, "transforms"):
        t_names = [type(t).__name__ for t in train_transforms.transforms]
        print(f"      * Detected train_transforms: {' -> '.join(t_names)}")
        
        # Check ToImage first
        assert len(t_names) > 0 and t_names[0] == "ToImage", (
            f"❌ Step 1 Error: First transform in `train_transforms` must be `v2.ToImage()`, got {t_names[0] if t_names else 'Empty'}."
        )
        # Check ToDtype second
        assert len(t_names) > 1 and t_names[1] == "ToDtype", (
            f"❌ Step 2 Error: Second transform in `train_transforms` must be `v2.ToDtype(...)`, got {t_names[1] if len(t_names) > 1 else 'Missing'}."
        )
        # Check Resize third
        assert len(t_names) > 2 and t_names[2] == "Resize", (
            f"❌ Step 3 Error: Third transform in `train_transforms` must be `v2.Resize(...)`, got {t_names[2] if len(t_names) > 2 else 'Missing'}."
        )
        # Check RandomHorizontalFlip fourth
        assert "RandomHorizontalFlip" in t_names, (
            "❌ Missing Augmentation: `train_transforms` must include `v2.RandomHorizontalFlip(p=0.5)` for training regularization!"
        )
        assert t_names[3] == "RandomHorizontalFlip", (
            f"❌ Step 4 Error: Fourth transform in `train_transforms` must be `v2.RandomHorizontalFlip(p=0.5)`, got {t_names[3]}."
        )
        # Check Normalize is terminal
        assert t_names[-1] == "Normalize", (
            f"❌ Normalization Order Error: `v2.Normalize(...)` must be the terminal (final) transform! Detected final transform: {t_names[-1]}."
        )
        assert len(t_names) == 5, (
            f"❌ Transform Count: Expected 5 stages in `train_transforms`, found {len(t_names)}: {t_names}."
        )

    if hasattr(eval_transforms, "transforms"):
        e_names = [type(t).__name__ for t in eval_transforms.transforms]
        print(f"      * Detected eval_transforms:  {' -> '.join(e_names)}")

        # Cardinal rule: Evaluation must NOT have RandomHorizontalFlip
        assert "RandomHorizontalFlip" not in e_names and not any("Random" in n for n in e_names), (
            f"❌ Evaluation Leakage: `eval_transforms` must NOT contain stochastic transforms! Found: {[n for n in e_names if 'Random' in n]}. Evaluation must be strictly deterministic."
        )
        # Check sequence: ToImage -> ToDtype -> Resize -> Normalize
        assert len(e_names) == 4, (
            f"❌ Transform Count: Expected 4 stages in `eval_transforms`, found {len(e_names)}: {e_names}."
        )
        assert e_names[0] == "ToImage", f"❌ Step 1 Error: First transform in `eval_transforms` must be `v2.ToImage()`, got {e_names[0]}."
        assert e_names[1] == "ToDtype", f"❌ Step 2 Error: Second transform in `eval_transforms` must be `v2.ToDtype(...)`, got {e_names[1]}."
        assert e_names[2] == "Resize",  f"❌ Step 3 Error: Third transform in `eval_transforms` must be `v2.Resize(...)`, got {e_names[2]}."
        assert e_names[3] == "Normalize", f"❌ Step 4 Error: Terminal transform in `eval_transforms` must be `v2.Normalize(...)`, got {e_names[3]}."

    print("      * Order of operations matches mathematical specification ✅")

    # --------------------------------------------------------------------------
    # [3/5] Dynamic Range & Parameter Hyperparameter Audit
    # --------------------------------------------------------------------------
    print("\n[3/5] Inspecting Dynamic Range Scaling & Spatial Resampling Hyperparameters...")
    if hasattr(train_transforms, "transforms"):
        # Check ToDtype scale=True
        todtype = train_transforms.transforms[1]
        assert getattr(todtype, "scale", False) is True, (
            "❌ ToDtype Error: `v2.ToDtype(torch.float32, scale=True)` must have `scale=True` to scale [0, 255] uint8 to [0.0, 1.0] float32!"
        )
        assert getattr(todtype, "dtype", None) == torch.float32, (
            f"❌ ToDtype Error: Target dtype must be `torch.float32`, got {getattr(todtype, 'dtype', None)}."
        )

        # Check Resize size == (224, 224)
        resize_op = train_transforms.transforms[2]
        r_size = getattr(resize_op, "size", None)
        assert r_size in [(224, 224), [224, 224], 224], (
            f"❌ Resize Error: Expected spatial target size (224, 224) or 224, got {r_size}."
        )

        # Check Normalize mean and std
        norm_op = train_transforms.transforms[-1]
        n_mean = getattr(norm_op, "mean", None)
        n_std = getattr(norm_op, "std", None)
        assert len(n_mean) == 3 and len(n_std) == 3, (
            f"❌ Normalize Error: Expected 3 channel parameters for RGB, got mean={n_mean}, std={n_std}."
        )
        if mean_cpu is not None and std_cpu is not None:
            mean_vals = [float(x) for x in mean_cpu]
            norm_mean = [float(x) for x in n_mean]
            assert np.allclose(norm_mean, mean_vals, atol=1e-3), (
                f"❌ Parameter Mismatch: `Normalize` mean does not match empirical `mean_cpu`!\n"
                f"   Configured: {norm_mean}\n   Expected:   {mean_vals}\n"
                f"Did you pass `mean=mean_cpu`?"
            )
            std_vals = [float(x) for x in std_cpu]
            norm_std = [float(x) for x in n_std]
            assert np.allclose(norm_std, std_vals, atol=1e-3), (
                f"❌ Parameter Mismatch: `Normalize` std does not match empirical `std_cpu`!\n"
                f"   Configured: {norm_std}\n   Expected:   {std_vals}\n"
                f"Did you pass `std=std_cpu`?"
            )
            print("      * Empirical (mean_cpu, std_cpu) parameters match reduction output ✅")

    print("      * Dynamic range scaling (scale=True) and spatial grid (224, 224) verified ✅")

    # --------------------------------------------------------------------------
    # [4/5] Functional Forward Propagation on Synthetic PIL Test Images
    # --------------------------------------------------------------------------
    print("\n[4/5] Testing Forward Propagation on Synthetic PIL Image [300x400]...")
    dummy_img = Image.new("RGB", (400, 300), color=(120, 160, 200))
    
    t_out = train_transforms(dummy_img)
    e_out = eval_transforms(dummy_img)

    assert isinstance(t_out, torch.Tensor), (
        f"❌ Output Type Error: `train_transforms` output must be a torch.Tensor, got {type(t_out)}."
    )
    assert isinstance(e_out, torch.Tensor), (
        f"❌ Output Type Error: `eval_transforms` output must be a torch.Tensor, got {type(e_out)}."
    )

    assert t_out.shape == (3, 224, 224), (
        f"❌ Output Shape Error: `train_transforms` produced shape {t_out.shape}, expected torch.Size([3, 224, 224])."
    )
    assert e_out.shape == (3, 224, 224), (
        f"❌ Output Shape Error: `eval_transforms` produced shape {e_out.shape}, expected torch.Size([3, 224, 224])."
    )

    assert t_out.dtype == torch.float32, (
        f"❌ Output Dtype Error: `train_transforms` produced dtype {t_out.dtype}, expected torch.float32."
    )
    assert e_out.dtype == torch.float32, (
        f"❌ Output Dtype Error: `eval_transforms` produced dtype {e_out.dtype}, expected torch.float32."
    )
    print("      * Synthetic image successfully transformed to planar [3, 224, 224] in float32 ✅")

    # --------------------------------------------------------------------------
    # [5/5] Deterministic Evaluation vs. Stochastic Training Invariance Audit
    # --------------------------------------------------------------------------
    print("\n[5/5] Auditing Deterministic Generalization & Normalization Invariant...")
    # Two successive passes through eval_transforms MUST yield identical bitwise tensors
    e_pass1 = eval_transforms(dummy_img)
    e_pass2 = eval_transforms(dummy_img)
    assert torch.equal(e_pass1, e_pass2), (
        "❌ Non-Deterministic Evaluation Failure! Two successive passes through `eval_transforms` yielded differing tensors. "
        "Evaluation must be strictly deterministic!"
    )
    print("      * Deterministic evaluation invariant confirmed: exact tensor equality over passes ✅")

    # Terminal normalization check: normalized tensor should have negative values for pixels below mean
    assert (e_pass1.min() < 0.0).item(), (
        f"❌ Normalization Failure: Minimum standardized tensor value is {e_pass1.min():.3f} (expected negative z-score). "
        "Did you forget `v2.Normalize(mean=mean_cpu, std=std_cpu)` or scale incorrectly?"
    )
    print("      * Zero-centered Z-score negative dynamic range validated ✅")

    print("\n" + "=" * 72)
    print("🎉 ALL PIPELINE ASSERTIONS PASSED: train_transforms and eval_transforms are production-ready!")
    print("=" * 72)
