"""
Session 15 Lab Utilities: Downsampling Mechanics, Pooling Invariance & Multi-Channel Hierarchies
Course: Deep Learning Foundations — Unit II (Week 5, Friday)
Authors: MSc. Antonio Aguilar (IMC, PUC Chile) & Dr. Luis Aguilar Ibáñez (UNP, Perú)

This module encapsulates:
1. Environment setup and asset management for UNP.jpg.
2. Tensor ingestion, shape formatting, and min-max contrast normalization.
3. Publication-grade visualization suites matching Session 14's light theme:
   - Task 1.2: Empirical Pooling Comparison & Multi-Row Diagnostics on UNP.jpg
   - Task 2.1: Real Stimuli Phase Invariance on UNP Archway Patch (Conv vs. MaxPool vs. GAP)
   - Task 4.1: Multi-Channel Filter Dissection (2D Slices vs. Composite 3D RGB Filter)
   - Task 4.2: ResNet-18 Conv1 Informative Feature Maps Grid
   - Task 4.3: Progressive Downsampling & Receptive Field Cascade
   - Task 4.4: Hierarchical Representation Emergence Across Network Depth
"""

import os
import sys
from typing import Tuple, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# =============================================================================
# 1. Asset & Image Management
# =============================================================================

def download_assets(github_base_url: str = "https://raw.githubusercontent.com/LsAntonio/dl-unp/refs/heads/main/Session-14"):
    """Ensures UNP.jpg exists locally, downloading it from GitHub if running in Colab/remote."""
    if not os.path.exists("UNP.jpg"):
        url = f"{github_base_url}/UNP.jpg"
        try:
            import urllib.request
            print(f"📥 Downloading campus image from {url}...")
            urllib.request.urlretrieve(url, "UNP.jpg")
            print("✅ Downloaded UNP.jpg successfully.")
        except Exception as e:
            print(f"Could not download UNP.jpg: {e}")


def load_unp_image(
    path: str = "UNP.jpg", 
    mode: str = "RGB", 
    resize: Optional[Tuple[int, int]] = None, 
    normalize_imagenet: bool = False
) -> torch.Tensor:
    """
    Loads UNP campus image and returns a 4D PyTorch FloatTensor [1, C, H, W] in [0, 1].
    
    Args:
        path: Path to the image file.
        mode: 'RGB' for 3-channel color or 'L' for 1-channel perceptual luminance.
        resize: Optional tuple (H, W) to resize image (e.g. (224, 224)).
        normalize_imagenet: If True, applies ImageNet mean & std normalization.
    """
    if not os.path.exists(path):
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(pkg_dir, path)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            download_assets()
            if not os.path.exists(path) and os.path.exists(alt_path):
                path = alt_path

    pil_img = Image.open(path).convert(mode)
    
    transform_list = []
    if resize is not None:
        transform_list.append(transforms.Resize(resize))
    transform_list.append(transforms.ToTensor())
    if normalize_imagenet and mode == "RGB":
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                   std=[0.229, 0.224, 0.225]))
        
    pipeline = transforms.Compose(transform_list)
    return pipeline(pil_img).unsqueeze(0)


def to_numpy_image(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Converts a PyTorch tensor ([1, 3, H, W], [3, H, W], [1, 1, H, W], or [H, W])
    into a NumPy array [H, W, C] or [H, W] clamped to [0, 1] for matplotlib display.
    """
    if isinstance(tensor, np.ndarray):
        return np.clip(tensor, 0, 1)
        
    t = tensor.detach().cpu()
    if t.ndim == 4:
        t = t.squeeze(0)  # [C, H, W]
    if t.ndim == 3:
        if t.shape[0] == 3:
            return t.permute(1, 2, 0).clamp(0, 1).numpy()
        elif t.shape[0] == 1:
            return t.squeeze(0).clamp(0, 1).numpy()
    return t.clamp(0, 1).numpy()


def min_max_normalize(data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Normalizes a NumPy array or PyTorch tensor to range [0, 1] for contrast enhancement.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    d_min, d_max = data.min(), data.max()
    return (data - d_min) / (d_max - d_min + 1e-8)


# =============================================================================
# 2. Multi-Row Pooling Comparison Suite (Task 1.2)
# =============================================================================

def plot_pooling_comparison_diagnostics(
    x_unp: torch.Tensor,
    x_max: torch.Tensor,
    x_avg: torch.Tensor,
    crop_orig: torch.Tensor,
    crop_max: torch.Tensor,
    crop_avg: torch.Tensor,
    div_map: np.ndarray,
    err_avg: np.ndarray,
    line_orig: np.ndarray,
    line_max: np.ndarray,
    line_avg: np.ndarray,
    slice_row: int = 18
):
    """
    Renders publication-grade 3x3 diagnostic visualization comparing Max Pooling vs Average Pooling.
    Row 1: Full-Canvas Perspective (Original, MaxPool, AvgPool)
    Row 2: Zoomed Sub-Pixel Analysis (Archway Lettering)
    Row 3: Diagnostic Analytics (Operator Divergence, Blur Residual, 1D Scanline Profiles)
    """
    fig = plt.figure(figsize=(15.5, 11), facecolor="#ffffff")
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.28, wspace=0.28)

    # --- Row 1: Full-Canvas Perspective ---
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])

    ax00.imshow(to_numpy_image(x_unp))
    ax00.set_title(f"Original UNP Image\n[{x_unp.shape[2]} × {x_unp.shape[3]} px | Full Resolution]", 
                   fontsize=10.5, fontweight="bold", pad=5)
    ax00.axis("off")

    ax01.imshow(to_numpy_image(x_max))
    ax01.set_title(f"Max Pooling (K=2, S=2)\n[{x_max.shape[2]} × {x_max.shape[3]} px | 4× Memory Reduction]", 
                   fontsize=10.5, fontweight="bold", pad=5)
    ax01.axis("off")

    ax02.imshow(to_numpy_image(x_avg))
    ax02.set_title(f"Average Pooling (K=2, S=2)\n[{x_avg.shape[2]} × {x_avg.shape[3]} px | 4× Memory Reduction]", 
                   fontsize=10.5, fontweight="bold", pad=5)
    ax02.axis("off")

    # --- Row 2: Zoomed Sub-Pixel Analysis (Archway Lettering) ---
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])

    ax10.imshow(to_numpy_image(crop_orig))
    ax10.set_title("Zoomed Original [90 × 140 px]\n[Sharp Lettering & Pickets]", 
                   fontsize=10, fontweight="bold", pad=5)
    ax10.axis("off")

    ax11.imshow(to_numpy_image(crop_max), interpolation="nearest")
    ax11.set_title("Zoomed MaxPool [45 × 70 px]\n[Supremum: Lettering & Edges Preserved]", 
                   fontsize=10, fontweight="bold", pad=5)
    ax11.axis("off")

    ax12.imshow(to_numpy_image(crop_avg), interpolation="nearest")
    ax12.set_title("Zoomed AvgPool [45 × 70 px]\n[Low-Pass Mean: Lettering Attenuated]", 
                   fontsize=10, fontweight="bold", pad=5)
    ax12.axis("off")

    # --- Row 3: Diagnostic Analytics (Where & What Changed) ---
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])
    ax22 = fig.add_subplot(gs[2, 2])

    im_div = ax20.imshow(div_map, cmap="inferno")
    ax20.set_title(r"Operator Divergence $|Y_{max} - Y_{avg}|$" + "\n[Bright = High Disagreement at Edges]", 
                   fontsize=9.5, fontweight="bold", pad=5)
    ax20.axis("off")
    cb1 = fig.colorbar(im_div, ax=ax20, location="left", fraction=0.046, pad=0.06)
    cb1.ax.tick_params(labelsize=8)

    im_res = ax21.imshow(err_avg, cmap="magma")
    ax21.set_title(r"AvgPool Blur Error $|X_{orig} - Y_{avg}^{\uparrow}|$" + "\n[Energy Loss Along Text & Boundaries]", 
                   fontsize=9.5, fontweight="bold", pad=5)
    ax21.axis("off")
    cb2 = fig.colorbar(im_res, ax=ax21, location="left", fraction=0.046, pad=0.06)
    cb2.ax.tick_params(labelsize=8)

    # 1D Cross-Section Profiles
    cols_orig = np.arange(len(line_orig))
    cols_pool = np.arange(len(line_max)) * 2

    ax22.plot(cols_orig, line_orig, color="#0284c7", linewidth=1.8, label="Original Signal (Fine Edges)")
    ax22.plot(cols_pool, line_max, color="#e11d48", linewidth=2.0, linestyle="--", marker="o", markersize=3, label="MaxPool (Preserves Peaks)")
    ax22.plot(cols_pool, line_avg, color="#059669", linewidth=2.0, linestyle=":", marker="s", markersize=3, label="AvgPool (Flattens Peaks)")
    ax22.set_title(f"1D Luminance Scanline Profile (Row {slice_row})\n[Peak vs. Mean Attenuation Across Letters]", 
                   fontsize=9.5, fontweight="bold", pad=5)
    ax22.set_xlabel("Horizontal Pixel Coordinate (Zoom Crop)", fontsize=8.5, fontweight="bold")
    ax22.set_ylabel("Normalized Luminance", fontsize=8.5, fontweight="bold")
    ax22.set_ylim(-0.05, 1.05)
    ax22.grid(True, linestyle="--", alpha=0.5)
    ax22.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax22.tick_params(labelsize=8)

    plt.show()


# =============================================================================
# 3. Phase Invariance Visualizations (Tasks 2.1 & 2.2)
# =============================================================================

def plot_synthetic_phase_invariance(
    edge_orig: torch.Tensor,
    edge_shifted: torch.Tensor,
    conv_orig: torch.Tensor,
    conv_shifted: torch.Tensor,
    pool_orig: torch.Tensor,
    pool_shifted: torch.Tensor,
    gap_orig: float,
    gap_shifted: float
):
    """
    Renders 2x4 visual proof of symmetry transformations:
    - Col 1: Input canvas [8x8] with explicit bounding border
    - Col 2: Convolution [8x8] showing point-to-point translation equivariance
    - Col 3: MaxPool [4x4] showing local phase invariance under intra-window shift
    - Col 4: Global Average Pooling showing exact global translation invariance
    """
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15.5, 7.5), gridspec_kw={"hspace": 0.35, "wspace": 0.25})

    streams = [
        (edge_orig, conv_orig, pool_orig, gap_orig, "Original Input X", "Centered Step Edge", 
         "Equivariant Edge Response", 0),
        (edge_shifted, conv_shifted, pool_shifted, gap_shifted, "Shifted Input T_g(X)", "Translated by +1 Pixel", 
         "Response Shifts Concurrently", 1)
    ]

    for row, (e, c, p, g, name, sub, conv_status, r_idx) in enumerate(streams):
        # Col 0: Input with clean blue border
        ax_in = axes[row, 0]
        ax_in.imshow(e.squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
        ax_in.set_title(f"{name}\n[{sub}]", fontsize=10.5, fontweight="bold", pad=6)
        for spine in ax_in.spines.values():
            spine.set_edgecolor("#0284c7")
            spine.set_linewidth(2.0)
        ax_in.set_xticks(range(8))
        ax_in.set_yticks(range(8))
        ax_in.tick_params(labelsize=7)

        # Col 1: Conv Equivariance
        ax_c = axes[row, 1]
        ax_c.imshow(c.squeeze().cpu(), cmap="magma")
        ax_c.set_title(f"Conv({name.split()[0]})\n[{conv_status}]", fontsize=10.5, fontweight="bold", pad=6)
        ax_c.set_xticks(range(8))
        ax_c.set_yticks(range(8))
        ax_c.tick_params(labelsize=7)

        # Col 2: MaxPool Local Invariance
        ax_p = axes[row, 2]
        ax_p.imshow(p.squeeze().cpu(), cmap="viridis")
        ax_p.set_title(f"MaxPool(Conv({name.split()[0]})) [4x4]\n[Local Phase Invariance]", fontsize=10.5, fontweight="bold", pad=6)
        ax_p.set_xticks(range(4))
        ax_p.set_yticks(range(4))
        ax_p.tick_params(labelsize=8)

        # Col 3: GAP Strict Global Invariance
        ax_g = axes[row, 3]
        ax_g.set_facecolor("#f8fafc")
        for spine in ax_g.spines.values():
            spine.set_edgecolor("#059669")
            spine.set_linewidth(2.0)
        ax_g.set_xticks([])
        ax_g.set_yticks([])
        ax_g.text(0.5, 0.65, f"GAP Scalar:\n{g:.3f}", fontsize=13, fontweight="bold",
                  color="#065f46", ha="center", va="center")
        ax_g.text(0.5, 0.25, "Global Invariance\n|GAP(X) - GAP(T_gX)| = 0", fontsize=8.5,
                  color="#047857", ha="center", va="center")
        ax_g.set_title("Global Average Pooling\n[Strict Invariance]", fontsize=10.5, fontweight="bold", pad=6)

    plt.suptitle("Synthetic Micro-Grid Experiment: Translation Equivariance vs. Local & Global Invariance",
                 fontsize=12.5, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_real_stimuli_invariance(
    patch_orig: torch.Tensor,
    patch_shifted: torch.Tensor,
    c_orig: torch.Tensor,
    c_shifted: torch.Tensor,
    p_orig: torch.Tensor,
    p_shifted: torch.Tensor,
    shift_px: int = 6
):
    """Renders 2x3 visualization of real stimuli equivariance vs. invariance on UNP.jpg."""
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8.5), gridspec_kw={"hspace": 0.35, "wspace": 0.18})

    # Row 1: Unshifted Reference Stream
    axes[0, 0].imshow(to_numpy_image(patch_orig), cmap="gray")
    axes[0, 0].set_title("Original UNP Archway Patch X\n[University Logo Shield & Lettering]", 
                         fontsize=10.5, fontweight="bold", pad=8)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(to_numpy_image(c_orig), cmap="magma")
    axes[0, 1].set_title("Conv(X) [Sobel Vertical Filter]\nEquivariant Edge Response", 
                         fontsize=10.5, fontweight="bold", pad=8)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(to_numpy_image(p_orig), cmap="viridis")
    axes[0, 2].set_title("MaxPool(Conv(X)) [K=2, S=2]\nLocal Supremum Reduction", 
                         fontsize=10.5, fontweight="bold", pad=8)
    axes[0, 2].axis("off")

    # Row 2: Shifted Stream
    axes[1, 0].imshow(to_numpy_image(patch_shifted), cmap="gray")
    axes[1, 0].set_title(f"Shifted UNP Patch T_g(X)\n[Horizontal Translation Δx = {shift_px} px]", 
                         fontsize=10.5, fontweight="bold", pad=8)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(to_numpy_image(c_shifted), cmap="magma")
    axes[1, 1].set_title(f"Conv(T_g(X)) [Shifted by {shift_px} px]\nExact Translation Equivariance", 
                         fontsize=10.5, fontweight="bold", pad=8)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(to_numpy_image(p_shifted), cmap="viridis")
    axes[1, 2].set_title("MaxPool(Conv(T_g(X)))\nDampens Spatial Jitter (Local Invariance)", 
                         fontsize=10.5, fontweight="bold", pad=8)
    axes[1, 2].axis("off")

    plt.show()


# =============================================================================
# 4. Multi-Channel Filter Dissection & Feature Emergence Suites (Task 4)
# =============================================================================

def plot_filter_dissection(
    conv1_weights: torch.Tensor, 
    selected_filters: List[int]
):
    """
    Renders 8x4 grid dissecting 3D convolutional filters into Red, Green, Blue slices
    alongside composite 3D RGB filters to visualize Gabor and color-opponent kernels.
    """
    fig, axes = plt.subplots(nrows=len(selected_filters), ncols=4, figsize=(10, 2.3 * len(selected_filters)))

    col_headers = [
        r"Red Slice $W[k, 0, :, :]$", 
        r"Green Slice $W[k, 1, :, :]$", 
        r"Blue Slice $W[k, 2, :, :]$", 
        r"Composite 3D Filter (RGB)"
    ]
    for col, header in enumerate(col_headers):
        axes[0, col].set_title(header, fontsize=11, fontweight='bold', pad=10)

    for row, f_idx in enumerate(selected_filters):
        w_filter = conv1_weights[f_idx]  # Shape: [3, 7, 7]
        vmax = max(abs(w_filter.min().item()), abs(w_filter.max().item()))
        
        # Plot individual 2D slices (Red, Green, Blue)
        for c in range(3):
            ax = axes[row, c]
            ax.imshow(w_filter[c], cmap='coolwarm', vmin=-vmax, vmax=vmax, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"Filter #{f_idx}", fontsize=10, fontweight='bold')
                
        # Plot composite 3D RGB filter (min-max normalized into [0, 1])
        ax_rgb = axes[row, 3]
        w_rgb_norm = min_max_normalize(w_filter.permute(1, 2, 0))
        ax_rgb.imshow(w_rgb_norm, interpolation='nearest')
        ax_rgb.set_xticks([]); ax_rgb.set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_informative_feature_maps(
    f_maps: np.ndarray, 
    selected_channels: List[int], 
    channel_activity: np.ndarray
):
    """Renders 4x4 grid displaying top 16 contrast-ranked feature maps from ResNet-18 conv1."""
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    fig.suptitle("ResNet-18 Conv1 Top-16 Informative Feature Maps on UNP.jpg (112 x 112)", 
                 fontsize=13, fontweight='bold', y=0.99)

    for rank, (ch_idx, ax) in enumerate(zip(selected_channels, axes.flat)):
        ch_norm = min_max_normalize(f_maps[ch_idx])
        ax.imshow(ch_norm, cmap='magma')
        ax.set_title(f"Channel {ch_idx} (σ={channel_activity[ch_idx]:.2f})", fontsize=10, fontweight='bold', pad=3)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def extract_downsampling_stages(
    model: nn.Module, 
    img_tensor: torch.Tensor, 
    raw_display: Optional[np.ndarray] = None
) -> List[Tuple]:
    """
    Progressively extracts internal feature representations across ResNet-18 stages:
    1. Input Canvas (224x224, J=1, RF=1px)
    2. Conv1 Stem (112x112, J=2, RF=7px)
    3. MaxPool (56x56, J=4, RF=11px)
    4. Layer 2 (28x28, J=8, RF=75px)
    5. Layer 3 (14x14, J=16, RF=155px)
    6. Layer 4 (7x7, J=32, RF=315px)
    
    Aggregates activation energy by taking the channel-wise mean of absolute values.
    """
    if raw_display is None:
        raw_display = to_numpy_image(load_unp_image(mode="RGB", resize=(224, 224)))

    with torch.no_grad():
        s1_conv = model.conv1(img_tensor)
        s1_act = model.relu(model.bn1(s1_conv))
        s2_pool = model.maxpool(s1_act)
        s3_layer1 = model.layer1(s2_pool)
        s4_layer2 = model.layer2(s3_layer1)
        s5_layer3 = model.layer3(s4_layer2)
        s6_layer4 = model.layer4(s5_layer3)

    stages = [
        ("Input Image", raw_display, "224 x 224", "J = 1", "RF = 1 px"),
        ("Conv1 Stem", s1_act.squeeze(0).abs().mean(dim=0).cpu().numpy(), "112 x 112", "J = 2", "RF = 7 px"),
        ("MaxPool", s2_pool.squeeze(0).abs().mean(dim=0).cpu().numpy(), "56 x 56", "J = 4", "RF = 11 px"),
        ("Layer 2", s4_layer2.squeeze(0).abs().mean(dim=0).cpu().numpy(), "28 x 28", "J = 8", "RF = 75 px"),
        ("Layer 3", s5_layer3.squeeze(0).abs().mean(dim=0).cpu().numpy(), "14 x 14", "J = 16", "RF = 155 px"),
        ("Layer 4", s6_layer4.squeeze(0).abs().mean(dim=0).cpu().numpy(), "7 x 7", "J = 32", "RF = 315 px"),
    ]
    return stages


def print_downsampling_table():
    """Prints the architectural downsampling & receptive field progression table."""
    print("=" * 82)
    print(f"{'Stage / Layer':<18} | {'Spatial Resolution':<18} | {'Cumulative Stride':<18} | {'Receptive Field':<16}")
    print("=" * 82)
    print(f"{'Input Canvas':<18} | {'224 x 224':<18} | {'J = 1':<18} | {'RF = 1 px':<16}")
    print(f"{'Conv1 (Stem)':<18} | {'112 x 112':<18} | {'J = 2':<18} | {'RF = 7 px':<16}")
    print(f"{'MaxPool':<18} | {'56 x 56':<18} | {'J = 4':<18} | {'RF = 11 px':<16}")
    print(f"{'Layer 1':<18} | {'56 x 56':<18} | {'J = 4':<18} | {'RF = 35 px':<16}")
    print(f"{'Layer 2':<18} | {'28 x 28':<18} | {'J = 8':<18} | {'RF = 75 px':<16}")
    print(f"{'Layer 3':<18} | {'14 x 14':<18} | {'J = 16':<18} | {'RF = 155 px':<16}")
    print(f"{'Layer 4':<18} | {'7 x 7':<18} | {'J = 32':<18} | {'RF = 315 px':<16}")
    print(f"{'AvgPool (GAP)':<18} | {'1 x 1':<18} | {'J = 32':<18} | {'RF = Global (224+)':<16}")
    print("=" * 82)
    print("💡 Key Takeaway: As spatial resolution downsamples by 32x (224 -> 7), Receptive Field expands")
    print("   by >300x. High-frequency pixel variations are collapsed into semantically invariant")
    print("   object representations, confirming the mathematical thesis of Session 15!")


def plot_downsampling_cascade(stages_or_model: Union[List[Tuple], nn.Module], img_tensor: Optional[torch.Tensor] = None):
    """
    Renders 1x6 cascade showing spatial resolution vs. receptive field expansion.
    Can be called either with precomputed `stages` list or directly with `(model, img_tensor)`.
    """
    if isinstance(stages_or_model, list):
        stages = stages_or_model
    else:
        if img_tensor is None:
            raise ValueError("img_tensor must be provided when passing a model to plot_downsampling_cascade")
        stages = extract_downsampling_stages(stages_or_model, img_tensor)

    fig, axes = plt.subplots(nrows=1, ncols=len(stages), figsize=(18, 3.8))

    for idx, (title, data, res, jump, rf) in enumerate(stages):
        ax = axes[idx]
        if idx == 0:
            ax.imshow(data)
        else:
            norm_data = min_max_normalize(data)
            ax.imshow(norm_data, cmap='inferno')
            
        ax.set_title(f"{title}\n[{res}]\n{jump} | {rf}", fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')

    plt.suptitle("The Downsampling & Receptive Field Hierarchy on UNP.jpg (ResNet-18)", 
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


def plot_representation_hierarchy(stages_hierarchy: List[Tuple]):
    """
    Renders 4x5 deep feature emergence grid:
    Column 1: Stage description badge with blue card spines
    Columns 2-5: 4 representative semantic channels with labels
    """
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle("The Visual Representation Hierarchy in Deep CNNs: From Edges to Semantics (ResNet-18 on UNP.jpg)", 
                 fontsize=13.5, fontweight='bold', y=0.985)

    gs = GridSpec(nrows=4, ncols=5, width_ratios=[1.35, 1, 1, 1, 1], wspace=0.18, hspace=0.28,
                  left=0.03, right=0.98, top=0.93, bottom=0.03)

    for row, (desc, f_map, ch_info) in enumerate(stages_hierarchy):
        # Stage description badge
        ax_desc = fig.add_subplot(gs[row, 0])
        ax_desc.set_facecolor('#f8fafc')
        ax_desc.set_xticks([]); ax_desc.set_yticks([])
        for spine in ax_desc.spines.values():
            spine.set_color('#0284c7')
            spine.set_linewidth(1.5)
        ax_desc.text(0.08, 0.5, desc, color='#0f172a', fontsize=9.5, fontweight='bold', 
                     va='center', ha='left', transform=ax_desc.transAxes, linespacing=1.35)
        
        # 4 Representative Feature Channels
        for col, (ch_idx, label) in enumerate(ch_info):
            ax = fig.add_subplot(gs[row, col + 1])
            ch_norm = min_max_normalize(f_map[ch_idx])
            ax.imshow(ch_norm, cmap='inferno')
            ax.set_title(f"Ch {ch_idx}: {label}", fontsize=9, fontweight='bold', pad=4)
            ax.axis('off')

    plt.show()
