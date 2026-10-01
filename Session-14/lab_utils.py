"""
Session 14 Lab Utilities: 2D Convolutions, Classical & Cortical Filter Banks, and Spatial Feature Auditing
Course: Deep Learning: Foundations, Systems & Scientific AI Auditing
Authors: MSc. Antonio Aguilar (IMC, PUC Chile) & Dr. Luis Aguilar Ibáñez (UNP, Perú)

This module encapsulates:
1. Environment setup and helper resource management (UNP image & assets).
2. Image ingestion and preprocessing (RGB color tensor and luminance grayscale).
3. Handcrafted filter bank generators (Sobel, Laplacian, Gaussian blur, and cortical Gabor wavelets).
4. Functional convolution helper (apply_filter with automatic 4D reshaping).
5. Theoretical spatial dimension calculators (master formula and assertion suites).
6. Comprehensive publication-grade visualization suites for zero-padding, filter responses,
   cortical firing rates, depthwise RGB convolutions, non-linear ReLU rectification,
   and student coding challenges.
"""

import os
import sys
from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# ============================================================================
# 1. Environment & Asset Management
# ============================================================================

def download_assets(github_base_url: str = "https://raw.githubusercontent.com/LsAntonio/dl-unp/refs/heads/main/Session-14"):
    """Ensures UNP.jpg exists locally, downloading it from GitHub if running in Colab/remote."""
    if not os.path.exists("UNP.jpg"):
        url = f"{github_base_url}/UNP.jpg"
        try:
            import urllib.request
            urllib.request.urlretrieve(url, "UNP.jpg")
            print("Downloaded UNP.jpg successfully.")
        except Exception as e:
            print(f"Could not download UNP.jpg: {e}")


def load_unp_image(path: str = "UNP.jpg") -> Tuple[torch.Tensor, torch.Tensor, Image.Image]:
    """
    Loads UNP campus image and returns:
    - color_tensor: 3D FloatTensor of shape [3, H, W] with values in [0, 1]
    - gray_tensor: 4D FloatTensor of shape [1, 1, H, W] computed via standard luminance weights
    - pil_image: Original PIL RGB Image
    """
    if not os.path.exists(path):
        # Also check in same directory as lab_utils.py
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(pkg_dir, path)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            download_assets()
            if not os.path.exists(path) and os.path.exists(alt_path):
                path = alt_path
        
    pil_image = Image.open(path).convert("RGB")
    color_tensor = transforms.ToTensor()(pil_image)  # [3, H, W]
    
    # Standard ITU-R BT.601 perceptual luminance weights: Y = 0.2989*R + 0.5870*G + 0.1140*B
    gray_2d = 0.2989 * color_tensor[0] + 0.5870 * color_tensor[1] + 0.1140 * color_tensor[2]
    gray_tensor = gray_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    return color_tensor, gray_tensor, pil_image


# ============================================================================
# 2. Filter Bank Synthesis & Convolution Operations
# ============================================================================

def make_gaussian_blur(ksize: int = 3, sigma: float = 1.0) -> torch.Tensor:
    """Generates a normalized 2D Gaussian smoothing kernel."""
    ax = torch.arange(-(ksize // 2), ksize // 2 + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def create_gabor_filter(
    ksize: int = 7, 
    theta: float = 0.0, 
    lambda_val: float = 3.5, 
    psi: float = 0.0, 
    sigma: float = 2.0, 
    gamma: float = 0.5
) -> torch.Tensor:
    """
    Generates a 2D Gabor wavelet matching mammalian V1 primary visual cortex simple cells:
    G(x, y) = exp(-(x'^2 + gamma^2 * y'^2) / (2 * sigma^2)) * cos(2 * pi * x' / lambda + psi)
    Includes zero-mean DC suppression and unit-energy normalization.
    """
    radius = ksize // 2
    y, x = torch.meshgrid(
        torch.linspace(-radius, radius, ksize),
        torch.linspace(-radius, radius, ksize),
        indexing="ij"
    )
    
    # Coordinate rotation by angle theta
    x_prime = x * np.cos(theta) + y * np.sin(theta)
    y_prime = -x * np.sin(theta) + y * np.cos(theta)
    
    gaussian = torch.exp(-(x_prime**2 + (gamma**2) * (y_prime**2)) / (2 * (sigma**2)))
    sinusoid = torch.cos(2 * np.pi * x_prime / lambda_val + psi)
    
    kernel = gaussian * sinusoid
    kernel = kernel - kernel.mean()            # Zero-mean DC suppression
    kernel = kernel / (kernel.norm() + 1e-8)   # Unit-energy normalization
    return kernel


def get_classical_filter_bank() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns canonical (sobel_x, sobel_y, laplacian, gaussian_blur) filter tensors."""
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
    laplacian = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    gaussian_blur = (1.0 / 16.0) * torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
    return sobel_x, sobel_y, laplacian, gaussian_blur


def get_v1_gabor_bank() -> Tuple[List[torch.Tensor], List[str]]:
    """Returns standard 4-orientation V1 cortex Gabor filters and their label strings."""
    thetas = ["0° (Horizontal)", "45° (Diagonal)", "90° (Vertical)", "135° (Anti-Diagonal)"]
    angles = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    kernels = [create_gabor_filter(ksize=7, theta=a) for a in angles]
    return kernels, thetas


def apply_filter(image_4d: torch.Tensor, kernel_2d: torch.Tensor, padding: int = 1) -> torch.Tensor:
    """
    Applies a 2D filter kernel to a 4D single-channel image tensor [1, 1, H, W] via F.conv2d.
    Returns squeezed 2D feature map [H_out, W_out].
    """
    kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0)
    return F.conv2d(image_4d, kernel_4d, padding=padding).squeeze().detach()


# ============================================================================
# 3. Spatial Dimension Formula Calculations
# ============================================================================

def compute_output_dim(dim_in: int, k: int, p: int, s: int, d: int = 1) -> int:
    """Calculates exact spatial output dimension for a 2D convolution along one axis."""
    k_eff = d * (k - 1) + 1
    return ((dim_in + 2 * p - k_eff) // s) + 1


def compute_conv2d_output_shape(img_shape: tuple, kernel_shape: tuple, p: int = 0, s: int = 1, d: int = 1) -> tuple:
    """Calculates (H_out, W_out) for arbitrary 2D rectangular images and kernels."""
    h_in, w_in = img_shape
    kh, kw = kernel_shape
    h_out = compute_output_dim(h_in, kh, p, s, d)
    w_out = compute_output_dim(w_in, kw, p, s, d)
    return (h_out, w_out)


# ============================================================================
# 4. Publication-Grade Visualization Suites
# ============================================================================

def plot_padding_audit(img_unp: torch.Tensor, pad_1: torch.Tensor, pad_2: torch.Tensor):
    """Visualizes full images and magnified top-left corner grids with strict pixel boundary alignment."""
    corner_orig = img_unp[:, :12, :12]
    corner_pad1 = pad_1[:, :12, :12]
    corner_pad2 = pad_2[:, :12, :12]
    
    fig = plt.figure(figsize=(18, 9))
    
    # Row 1: Full Images
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(img_unp.permute(1, 2, 0).cpu().numpy())
    ax1.set_title(f"Original UNP Image (P=0)\nShape: {list(img_unp.shape[1:])}", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(pad_1.permute(1, 2, 0).cpu().numpy())
    ax2.set_title(f"PyTorch F.pad (P=1)\nShape: {list(pad_1.shape[1:])} (+2 on H & W)", fontsize=11, fontweight="bold")
    ax2.axis("off")

    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(pad_2.permute(1, 2, 0).cpu().numpy())
    ax3.set_title(f"PyTorch F.pad (P=2)\nShape: {list(pad_2.shape[1:])} (+4 on H & W)", fontsize=11, fontweight="bold")
    ax3.axis("off")

    # Helper for pixel boundary grid alignment
    def setup_corner_ax(ax, img_data, p_val, title):
        ax.imshow(img_data.permute(1, 2, 0).cpu().numpy(), extent=(-0.5, 11.5, 11.5, -0.5))
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_xticks(range(12))
        ax.set_yticks(range(12))
        ax.set_xticklabels(range(12), fontsize=9)
        ax.set_yticklabels(range(12), fontsize=9)
        
        # Grid lines strictly at pixel boundaries
        for boundary in np.arange(-0.5, 12.5, 1.0):
            ax.axvline(boundary, color="#64748b", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.axhline(boundary, color="#64748b", linestyle=":", linewidth=0.8, alpha=0.6)
            
        if p_val > 0:
            rect = patches.Rectangle((-0.5 + p_val, -0.5 + p_val), 12 - p_val, 12 - p_val,
                                     linewidth=2.2, edgecolor="#f43f5e", facecolor="none", linestyle="--")
            ax.add_patch(rect)
            ax.text(p_val + 0.2, p_val + 0.6, f"Original Image\nstarts at ({p_val}, {p_val})",
                    color="#f43f5e", fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#f43f5e", alpha=0.9))
        else:
            ax.text(0.2, 0.6, "Original Image\nstarts at (0, 0)",
                    color="#0284c7", fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#0284c7", alpha=0.9))
        ax.set_xlim(-0.5, 11.5)
        ax.set_ylim(11.5, -0.5)

    ax4 = plt.subplot(2, 3, 4)
    setup_corner_ax(ax4, corner_orig, 0, "Top-Left Corner (P=0)\n[No zero boundary — image pixels start at (0,0)]")

    ax5 = plt.subplot(2, 3, 5)
    setup_corner_ax(ax5, corner_pad1, 1, "Top-Left Corner (P=1)\n[Exactly 1-pixel zero-border: row 0 & col 0]")

    ax6 = plt.subplot(2, 3, 6)
    setup_corner_ax(ax6, corner_pad2, 2, "Top-Left Corner (P=2)\n[Exactly 2-pixel zero-border: rows 0-1 & cols 0-1]")

    plt.tight_layout()
    plt.show()


def plot_classical_responses(
    img_tensor: torch.Tensor, 
    resp_sobel_x: torch.Tensor, 
    resp_sobel_y: torch.Tensor, 
    resp_sobel_mag: torch.Tensor, 
    resp_gaussian: torch.Tensor, 
    class_name: str = "UNP Campus Gate"
):
    """Visualizes the 5 classical filter response maps."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

    axes[0].imshow(img_tensor.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title(f"Original: {class_name}", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(resp_sobel_x.cpu().numpy(), cmap="gray")
    axes[1].set_title("Sobel X (Vertical Edges)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(resp_sobel_y.cpu().numpy(), cmap="gray")
    axes[2].set_title("Sobel Y (Horizontal Edges)", fontsize=12)
    axes[2].axis("off")

    axes[3].imshow(resp_sobel_mag.cpu().numpy(), cmap="hot")
    axes[3].set_title("Combined Edge Gradient", fontsize=12, fontweight="bold")
    axes[3].axis("off")

    axes[4].imshow(resp_gaussian.cpu().numpy(), cmap="gray")
    axes[4].set_title("Gaussian Low-Pass Blur", fontsize=12)
    axes[4].axis("off")

    plt.tight_layout()
    plt.show()


def plot_gabor_responses(kernels: list, resps: list, thetas: list):
    """
    Visualizes the 3x4 V1 Cortex Gabor Filter Bank:
    - Row 0: Handcrafted Receptive Fields (cmap='bwr')
    - Row 1: Linear Signed Convolution (cmap='gray', synchronized with markdown)
    - Row 2: Rectified Cortical Activation (ReLU + cmap='viridis')
    """
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))

    for col in range(4):
        # Row 0: Handcrafted V1 Receptive Fields
        axes[0, col].imshow(kernels[col].detach().cpu().numpy(), cmap="bwr")
        axes[0, col].set_title(f"V1 Kernel (θ = {thetas[col]})", fontsize=11, fontweight="bold")
        axes[0, col].axis("off")
        
        # Row 1: Linear Signed Convolution (Gray, neutral 0 = mid-gray)
        axes[1, col].imshow(resps[col].detach().cpu().numpy(), cmap="gray")
        axes[1, col].set_title(f"Linear Response (Gray) θ = {thetas[col]}", fontsize=10)
        axes[1, col].axis("off")
        
        # Row 2: Rectified Cortical Activation (ReLU + Viridis)
        relu_act = F.relu(resps[col])
        axes[2, col].imshow(relu_act.detach().cpu().numpy(), cmap="viridis")
        axes[2, col].set_title(f"Cortical Firing (ReLU + Viridis) θ = {thetas[col]}", fontsize=10, fontweight="bold")
        axes[2, col].axis("off")

    plt.tight_layout()
    plt.show()


def plot_multichannel_gradients(img_tensor: torch.Tensor, mag_rgb: torch.Tensor, combined_color_mag: torch.Tensor):
    """Visualizes Depthwise RGB edge maps and combined color magnitude."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    axes[0].imshow(img_tensor.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original RGB (UNP Gate)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(mag_rgb[0].cpu().numpy(), cmap="Reds")
    axes[1].set_title("Red Channel Edges (Cones)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(mag_rgb[1].cpu().numpy(), cmap="Greens")
    axes[2].set_title("Green Channel Edges (Trees)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(mag_rgb[2].cpu().numpy(), cmap="Blues")
    axes[3].set_title("Blue Channel Edges (Sign/Sky)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    axes[4].imshow(combined_color_mag.cpu().numpy(), cmap="magma")
    axes[4].set_title("Combined Multi-Channel Magnitude", fontsize=11, fontweight="bold")
    axes[4].axis("off")

    plt.tight_layout()
    plt.show()


def plot_linear_vs_relu_comparison(resps_linear: list, resps_relu: list):
    """Visualizes strict 1-to-1 Column Comparison between Linear and Rectified (ReLU) Filters."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    titles_row1 = [
        "Linear Sobel X (Signed, Vertical Edges)",
        "Linear Sobel Y (Signed, Horizontal Edges)",
        "Linear Laplacian (Signed, Zero-Crossings)",
        "Linear Gaussian Blur (Non-negative)"
    ]
    titles_row2 = [
        "ReLU(Sobel X): Vertical ON-Edges",
        "ReLU(Sobel Y): Horizontal ON-Edges",
        "ReLU(Laplacian): Positive Ridges",
        "ReLU(Gaussian): Exact Identity Mapping"
    ]

    for j in range(4):
        # Row 1: Linear Responses (mid-gray = 0, dark = negative, bright = positive)
        axes[0, j].imshow(resps_linear[j].cpu().numpy(), cmap="gray")
        axes[0, j].set_title(titles_row1[j], fontsize=11, fontweight="bold")
        axes[0, j].axis("off")
        
        # Row 2: Rectified Activations (black = 0 / silent, white = active excitation)
        axes[1, j].imshow(resps_relu[j].cpu().numpy(), cmap="gray")
        axes[1, j].set_title(titles_row2[j], fontsize=11, fontweight="bold")
        axes[1, j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_polarity_recombination(
    gray_img: torch.Tensor, 
    resp_sobel_x: torch.Tensor, 
    relu_sobel_x_pos: torch.Tensor, 
    relu_sobel_x_neg: torch.Tensor, 
    recombined_edges: torch.Tensor
):
    """Visualizes the polarity selectivity story (+Sobel X vs -Sobel X) and dual-channel recombination."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    axes[0].imshow(gray_img.squeeze().cpu().numpy(), cmap="gray")
    axes[0].set_title("Input (Grayscale UNP)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(resp_sobel_x.cpu().numpy(), cmap="gray")
    axes[1].set_title("Linear Sobel X\n(Both Polarities Signed)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(relu_sobel_x_pos.cpu().numpy(), cmap="gray")
    axes[2].set_title("ReLU(+Sobel X): ON-Edges\n(Dark → Light Transitions)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(relu_sobel_x_neg.cpu().numpy(), cmap="gray")
    axes[3].set_title("ReLU(-Sobel X): OFF-Edges\n(Light → Dark Transitions)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    axes[4].imshow(recombined_edges.cpu().numpy(), cmap="gray")
    axes[4].set_title("Recombination: ReLU(+Gx) + ReLU(-Gx)\n(Full Bidirectional Boundary Recovered)", fontsize=11, fontweight="bold")
    axes[4].axis("off")

    plt.tight_layout()
    plt.show()


def plot_filter_challenge(
    gray_img: torch.Tensor, 
    out_filtered: torch.Tensor, 
    title: str, 
    ref_filtered: Optional[torch.Tensor] = None, 
    ref_title: str = "Reference Comparison"
):
    """Harness to plot student challenge results against original grayscale and optional reference."""
    if ref_filtered is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(gray_img.squeeze().cpu().numpy(), cmap="gray")
        axes[0].set_title("Original Grayscale (UNP)", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(out_filtered.cpu().numpy(), cmap="gray")
        axes[1].set_title(title, fontsize=11, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(ref_filtered.cpu().numpy(), cmap="gray")
        axes[2].set_title(ref_title, fontsize=11, fontweight="bold")
        axes[2].axis("off")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(gray_img.squeeze().cpu().numpy(), cmap="gray")
        axes[0].set_title("Original Grayscale (UNP)", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(out_filtered.cpu().numpy(), cmap="gray")
        axes[1].set_title(title, fontsize=12, fontweight="bold")
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()
