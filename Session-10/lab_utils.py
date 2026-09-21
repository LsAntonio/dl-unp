"""
Session 10 Lab Utilities: Stochastic Noise Corruption & Robust Representations Lab
Course: Deep Learning: Foundations, Systems & Scientific AI Auditing
Authors: MSc. Antonio Aguilar & Dr. Luis Aguilar Ibáñez
National University of Piura (UNP) & Pontifical Catholic University of Chile (UC)

This module encapsulates boilerplate visualization suites, pixel distribution auditing,
throughput benchmarking, modular BaselineDAE architecture components (Encoder, Decoder, BaselineDAE),
in-VRAM training routines, and the interactive Model m breakdown auditing harness.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# 1. Environment & Reproducibility Setup
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Configures deterministic pseudo-random number generator (PRNG) states
    across CPU, CUDA backends, and NumPy runtime for exact scientific reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Selects the optimal active hardware accelerator (CUDA GPU or CPU fallback)
    and prints hardware execution telemetry.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ PyTorch Version:       {torch.__version__}")
    print(f"💻 Active Compute Device: {device}")
    if torch.cuda.is_available():
        print(f"🚀 GPU Device Name:       {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"📊 Available GPU VRAM:    {vram_gb:.2f} GB")
    return device


# ============================================================================
# 2. Standalone Noise Implementations (Functional Fallbacks)
# ============================================================================

def add_gaussian_noise(x: torch.Tensor, sigma: float = 0.3) -> torch.Tensor:
    """Zero-mean Additive White Gaussian Noise (AWGN) with boundary clamping."""
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, 0.0, 1.0)


def add_salt_and_pepper_noise(x: torch.Tensor, prob: float = 0.1) -> torch.Tensor:
    """Discrete Salt-and-Pepper impulse noise using uniform random partitioning."""
    prob = max(0.0, min(1.0, float(prob)))
    noisy = x.clone()
    rand_tensor = torch.rand_like(x)
    pepper_mask = rand_tensor < (prob / 2.0)
    salt_mask = (rand_tensor >= (prob / 2.0)) & (rand_tensor < prob)
    noisy[pepper_mask] = 0.0
    noisy[salt_mask] = 1.0
    return noisy


def add_masking_noise(x: torch.Tensor, drop_prob: float = 0.25, **kwargs) -> torch.Tensor:
    """Vectorized masking corruption using Bernoulli dropout (accepts drop_prob or prob)."""
    if 'prob' in kwargs:
        drop_prob = kwargs['prob']
    drop_prob = max(0.0, min(1.0, float(drop_prob)))
    keep_prob = 1.0 - drop_prob
    mask = torch.bernoulli(torch.full_like(x, keep_prob))
    return x * mask


class NoiseInjector:
    """
    Functional noise injection transform suite for PyTorch tensors.
    All methods operate out-of-place and preserve device, dtype, and gradient graphs.
    Includes defensive probability bounding to prevent CUDA device-side assertions.
    """
    
    @staticmethod
    def gaussian(x: torch.Tensor, sigma: float = 0.3) -> torch.Tensor:
        """Injects Additive White Gaussian Noise with [0, 1] boundary clamping."""
        safe_sigma = max(0.0, float(sigma))
        noise = torch.randn_like(x) * safe_sigma
        return torch.clamp(x + noise, min=0.0, max=1.0)
    
    @staticmethod
    def salt_and_pepper(x: torch.Tensor, prob: float = 0.1) -> torch.Tensor:
        """Injects discrete Bernoulli impulse (salt and pepper) noise."""
        safe_prob = max(0.0, min(1.0, float(prob)))
        x_noisy = x.clone()
        rand = torch.rand_like(x)
        x_noisy[rand < (safe_prob / 2.0)] = 0.0
        x_noisy[(rand >= (safe_prob / 2.0)) & (rand < safe_prob)] = 1.0
        return x_noisy

    @staticmethod
    def masking(x: torch.Tensor, drop_prob: float = 0.25, **kwargs) -> torch.Tensor:
        """Zeros out pixels randomly with Bernoulli sampling, safely bounded to [0.0, 1.0]."""
        if 'prob' in kwargs:
            drop_prob = kwargs['prob']
        safe_drop = max(0.0, min(1.0, float(drop_prob)))
        keep_prob = 1.0 - safe_drop
        mask = torch.bernoulli(torch.full_like(x, keep_prob))
        return x * mask


# ============================================================================
# 3. Visualization Suites
# ============================================================================

def plot_clean_samples(images: torch.Tensor, labels: torch.Tensor, n: int = 8) -> None:
    """Plots pristine reference digits with their ground truth class labels."""
    fig, axes = plt.subplots(1, n, figsize=(n * 1.8, 2.2))
    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f"Label: {labels[i].item()}", fontsize=11, fontweight='bold', color='#38bdf8')
        axes[i].axis('off')
    plt.suptitle("Pristine Ground-Truth MNIST Reference Batch", fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()


def plot_gaussian_spectrum(
    digit: torch.Tensor,
    sigma_levels: Optional[Any] = None,
    noise_fn: Optional[Callable] = None
) -> None:
    """
    Visualizes a digit progressively corrupted across any arbitrary sequence of Gaussian noise levels (sigma).
    Dynamically adapts figure layout to 1, 3, 5, 8, or any arbitrary number of levels.
    """
    if sigma_levels is None:
        sigma_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
    elif isinstance(sigma_levels, (int, float)):
        sigma_levels = [float(sigma_levels)]
    else:
        sigma_levels = list(sigma_levels)

    if not sigma_levels:
        print("⚠️ Warning: sigma_levels is empty. Please provide at least one noise level.")
        return

    n_levels = len(sigma_levels)
    if noise_fn is None:
        noise_fn = add_gaussian_noise

    single_digit = digit[0:1] if digit.dim() == 4 else digit.unsqueeze(0)
    fig, axes = plt.subplots(1, n_levels, figsize=(max(3.0, n_levels * 2.0), 2.6), squeeze=False)
    
    for idx, sigma in enumerate(sigma_levels):
        corrupted = noise_fn(single_digit, sigma=sigma)
        axes[0, idx].imshow(corrupted.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0, idx].set_title(f"σ = {sigma:.2f}", fontsize=11, fontweight='bold')
        axes[0, idx].axis('off')
        
    plt.suptitle("Gaussian Noise Spectrum: From Imperceptible Static to Feature Annihilation", 
                 fontsize=13, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.show()


def plot_gaussian_diagnostic(
    clean_imgs: torch.Tensor,
    sigma: float = 0.35,
    n_samples: int = 6,
    noise_fn: Optional[Callable] = None
) -> None:
    """Visualizes Clean, Noisy, and Isolated Residual difference maps side-by-side using bwr colormap."""
    if noise_fn is None:
        noise_fn = add_gaussian_noise

    imgs = clean_imgs[:n_samples]
    noisy_imgs = noise_fn(imgs, sigma=sigma)
    residuals = noisy_imgs - imgs
    
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 2.0, 6.2))
    
    for i in range(n_samples):
        # Row 1: Pristine Clean Target
        axes[0, i].imshow(imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"Digit {i+1}", fontsize=11, fontweight='bold')
        
        # Row 2: Corrupted Input
        axes[1, i].imshow(noisy_imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        
        # Row 3: Isolated Residual Difference (bwr centered at 0.0)
        im = axes[2, i].imshow(residuals[i].squeeze(), cmap='bwr', vmin=-1.0, vmax=1.0)
        
        for r in range(3):
            axes[r, i].axis('off')
            
    axes[0, 0].text(-0.35, 0.5, "1. Clean Target (x)", transform=axes[0, 0].transAxes,
                    rotation=90, va='center', ha='center', fontsize=11, fontweight='bold', color='#10b981')
    axes[1, 0].text(-0.35, 0.5, f"2. Corrupted (x̃)\n[σ={sigma}]", transform=axes[1, 0].transAxes,
                    rotation=90, va='center', ha='center', fontsize=11, fontweight='bold', color='#f59e0b')
    axes[2, 0].text(-0.35, 0.5, "3. Residual (x̃ - x)", transform=axes[2, 0].transAxes,
                    rotation=90, va='center', ha='center', fontsize=11, fontweight='bold', color='#38bdf8')
    
    plt.suptitle("The 3-Way Inspection Protocol: Isolating the Gaussian Perturbation Signature", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def plot_pixel_distribution_shift(
    clean_batch: torch.Tensor,
    sigma: float = 0.3,
    noise_fn: Optional[Callable] = None
) -> None:
    """Plots probability density histograms comparing clean vs. corrupted pixel intensities."""
    if noise_fn is None:
        noise_fn = add_gaussian_noise

    clean_flat = clean_batch.flatten().cpu().numpy()
    noisy_flat = noise_fn(clean_batch, sigma=sigma).flatten().cpu().numpy()

    plt.figure(figsize=(9, 3.8))
    plt.hist(clean_flat, bins=50, density=True, alpha=0.6, color='#10b981', label='Clean Pristine Tensors')
    plt.hist(noisy_flat, bins=50, density=True, alpha=0.5, color='#f43f5e', label=f'Gaussian Corrupted (σ={sigma})')
    plt.title("Pixel Intensity Probability Density: Clean vs. Gaussian Corrupted", fontsize=12, fontweight='bold')
    plt.xlabel("Pixel Intensity Value [0.0, 1.0]", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_sp_spectrum(
    digit: torch.Tensor,
    p_levels: Optional[Any] = None,
    noise_fn: Optional[Callable] = None
) -> None:
    """
    Visualizes a digit progressively corrupted across any arbitrary sequence of Salt & Pepper probabilities.
    Dynamically adapts figure layout to 1, 3, 5, 7, or any arbitrary number of levels.
    """
    if p_levels is None:
        p_levels = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70]
    elif isinstance(p_levels, (int, float)):
        p_levels = [float(p_levels)]
    else:
        p_levels = list(p_levels)

    if not p_levels:
        print("⚠️ Warning: p_levels is empty. Please provide at least one noise level.")
        return

    n_levels = len(p_levels)
    if noise_fn is None:
        noise_fn = add_salt_and_pepper_noise

    single_digit = digit[0:1] if digit.dim() == 4 else digit.unsqueeze(0)
    fig, axes = plt.subplots(1, n_levels, figsize=(max(3.0, n_levels * 2.0), 2.6), squeeze=False)
    
    for idx, p in enumerate(p_levels):
        corrupted = noise_fn(single_digit, prob=p)
        axes[0, idx].imshow(corrupted.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0, idx].set_title(f"p = {p:.2f}", fontsize=11, fontweight='bold')
        axes[0, idx].axis('off')
        
    plt.suptitle("Salt & Pepper Impulse Spectrum: Increasing Density of Bipolar Sensor Bitflips", 
                 fontsize=13, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.show()


def plot_sp_diagnostic(
    clean_imgs: torch.Tensor,
    prob: float = 0.15,
    n_samples: int = 6,
    noise_fn: Optional[Callable] = None
) -> None:
    """Visualizes Clean, Salt-and-Pepper corrupted, and Residual maps."""
    if noise_fn is None:
        noise_fn = add_salt_and_pepper_noise

    imgs = clean_imgs[:n_samples]
    noisy_imgs = noise_fn(imgs, prob=prob)
    residuals = noisy_imgs - imgs
    
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 2.0, 6.2))
    
    for i in range(n_samples):
        axes[0, i].imshow(imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"Sample {i+1}", fontsize=11, fontweight='bold')
        axes[1, i].imshow(noisy_imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[2, i].imshow(residuals[i].squeeze(), cmap='bwr', vmin=-1.0, vmax=1.0)
        
        for r in range(3):
            axes[r, i].axis('off')
            
    axes[0, 0].text(-0.35, 0.5, "1. Clean Target (x)", transform=axes[0, 0].transAxes,
                    rotation=90, va='center', ha='center', fontsize=11, fontweight='bold', color='#10b981')
    axes[1, 0].text(-0.35, 0.5, f"2. Salt & Pepper\n[p={prob}]", transform=axes[1, 0].transAxes,
                    rotation=90, va='center', ha='center', fontsize=11, fontweight='bold', color='#f59e0b')
    axes[2, 0].text(-0.35, 0.5, "3. Residual (x̃ - x)", transform=axes[2, 0].transAxes,
                    rotation=90, va='center', ha='center', fontsize=11, fontweight='bold', color='#38bdf8')
    
    plt.suptitle("Salt & Pepper 3-Way Inspection: Identifying Extreme Bipolar Perturbation Spikes", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def plot_masking_spectrum(
    digit: torch.Tensor,
    drop_levels: Optional[Any] = None,
    noise_fn: Optional[Callable] = None
) -> None:
    """
    Visualizes a digit progressively corrupted across any arbitrary sequence of Bernoulli masking drop probabilities.
    Dynamically adapts figure layout to 1, 3, 5, 6, or any arbitrary number of levels.
    """
    if drop_levels is None:
        drop_levels = [0.10, 0.25, 0.40, 0.60, 0.80, 0.95]
    elif isinstance(drop_levels, (int, float)):
        drop_levels = [float(drop_levels)]
    else:
        drop_levels = list(drop_levels)

    if not drop_levels:
        print("⚠️ Warning: drop_levels is empty. Please provide at least one noise level.")
        return

    n_levels = len(drop_levels)
    if noise_fn is None:
        noise_fn = add_masking_noise

    single_digit = digit[0:1] if digit.dim() == 4 else digit.unsqueeze(0)
    fig, axes = plt.subplots(1, n_levels, figsize=(max(3.0, n_levels * 2.0), 2.6), squeeze=False)
    
    for idx, p in enumerate(drop_levels):
        corrupted = noise_fn(single_digit, drop_prob=p)
        axes[0, idx].imshow(corrupted.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0, idx].set_title(f"Drop p = {p:.2f}", fontsize=11, fontweight='bold')
        axes[0, idx].axis('off')
        
    plt.suptitle("Masking Noise Spectrum: Random Pixel Dropout & Inpainting Challenges", 
                 fontsize=13, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.show()


def plot_masking_diagnostic(
    clean_imgs: torch.Tensor,
    drop_prob: float = 0.35,
    n_samples: int = 6,
    noise_fn: Optional[Callable] = None,
    **kwargs
) -> None:
    """
    Visualizes Clean, Masked corrupted, and Isolated Residual difference maps side-by-side using bwr colormap.
    For Bernoulli masking (multiplicative dropout), the residual (x̃ - x) is strictly non-positive (<= 0):
    0 where untouched/background (white in 'bwr'), and -x where erased (blue in 'bwr').
    There are strictly zero positive/red pixels because masking never injects additive energy.
    """
    if 'prob' in kwargs:
        drop_prob = kwargs['prob']
    if noise_fn is None:
        noise_fn = add_masking_noise

    imgs = clean_imgs[:n_samples]
    try:
        noisy_imgs = noise_fn(imgs, drop_prob=drop_prob)
    except TypeError:
        noisy_imgs = noise_fn(imgs, prob=drop_prob)
    residuals = noisy_imgs - imgs
    
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 2.0, 6.2))
    
    for i in range(n_samples):
        # Row 1: Pristine Clean Target
        axes[0, i].imshow(imgs[i].squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"Sample {i+1}", fontsize=11, fontweight='bold')
        
        # Row 2: Corrupted Input
        axes[1, i].imshow(noisy_imgs[i].squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        
        # Row 3: Isolated Residual Difference (bwr centered at 0.0)
        axes[2, i].imshow(residuals[i].squeeze().cpu(), cmap='bwr', vmin=-1.0, vmax=1.0)
        
        for r in range(3):
            axes[r, i].axis('off')
            
    axes[0, 0].text(-0.35, 0.5, "1. Clean Target (x)", transform=axes[0, 0].transAxes,
                    rotation=90, va='center', ha='center', fontsize=11, fontweight='bold', color='#10b981')
    axes[1, 0].text(-0.35, 0.5, f"2. Masked (x̃)\n[p={drop_prob:.2f}]", transform=axes[1, 0].transAxes,
                    rotation=90, va='center', ha='center', fontsize=11, fontweight='bold', color='#f59e0b')
    axes[2, 0].text(-0.35, 0.5, "3. Residual (x̃ - x)", transform=axes[2, 0].transAxes,
                    rotation=90, va='center', ha='center', fontsize=11, fontweight='bold', color='#38bdf8')
    
    plt.suptitle("Masking Noise 3-Way Inspection: Isolating Multiplicative Zero-Out Deletions", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def plot_multi_noise_matrix(
    clean_imgs: torch.Tensor,
    n_samples: int = 6,
    gaussian_fn: Optional[Callable] = None,
    sp_fn: Optional[Callable] = None,
    mask_fn: Optional[Callable] = None
) -> None:
    """Generates a comprehensive comparative grid across all 3 noise families."""
    if gaussian_fn is None:
        gaussian_fn = add_gaussian_noise
    if sp_fn is None:
        sp_fn = add_salt_and_pepper_noise
    if mask_fn is None:
        mask_fn = add_masking_noise

    imgs = clean_imgs[:n_samples]
    gaussian_imgs = gaussian_fn(imgs, sigma=0.35)
    sp_imgs = sp_fn(imgs, prob=0.15)
    masked_imgs = mask_fn(imgs, drop_prob=0.35)
    
    fig, axes = plt.subplots(4, n_samples, figsize=(n_samples * 2.0, 8.5))
    
    for i in range(n_samples):
        axes[0, i].imshow(imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"Sample {i+1}", fontsize=11, fontweight='bold')
        axes[1, i].imshow(gaussian_imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[2, i].imshow(sp_imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[3, i].imshow(masked_imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        
        for r in range(4):
            axes[r, i].axis('off')
            
    row_labels = ["1. Clean Target (x)", "2. Gaussian (σ=0.35)", "3. Salt & Pepper (p=0.15)", "4. Masking (p=0.35)"]
    colors = ['#10b981', '#f59e0b', '#ec4899', '#3b82f6']
    
    for r in range(4):
        axes[r, 0].text(-0.35, 0.5, row_labels[r], transform=axes[r, 0].transAxes,
                        rotation=90, va='center', ha='center', fontsize=11, fontweight='bold', color=colors[r])
        
    plt.suptitle("Comparative Noise Matrix: Distinct Mathematical Degradation Profiles", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def plot_recognition_threshold(
    image: torch.Tensor,
    sigma: float = 0.45,
    sp_prob: float = 0.25,
    mask_prob: Optional[float] = None
) -> None:
    """
    Interactive visualization allowing students to test specific Gaussian (sigma),
    Salt & Pepper (prob), and optional Masking corruption intensities against the clean reference.
    Evaluates the human perceptual recognition threshold.
    """
    single_digit = image[0:1] if image.dim() == 4 else image.unsqueeze(0)
    
    g_test = NoiseInjector.gaussian(single_digit, sigma=sigma)
    sp_test = NoiseInjector.salt_and_pepper(single_digit, prob=sp_prob)
    
    num_plots = 4 if mask_prob is not None else 3
    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 3.0, 3.2))
    
    # 1. Clean Reference
    axes[0].imshow(single_digit.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Clean Reference", fontweight='bold', color='#10b981', fontsize=11)
    axes[0].axis('off')
    
    # 2. Gaussian
    axes[1].imshow(g_test.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Gaussian (σ = {sigma:.2f})", fontweight='bold', color='#38bdf8', fontsize=11)
    axes[1].axis('off')
    
    # 3. Salt & Pepper
    axes[2].imshow(sp_test.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"S&P (p = {sp_prob:.2f})", fontweight='bold', color='#f59e0b', fontsize=11)
    axes[2].axis('off')
    
    # 4. Optional Masking
    if mask_prob is not None:
        m_test = NoiseInjector.masking(single_digit, drop_prob=mask_prob)
        axes[3].imshow(m_test.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[3].set_title(f"Masking (p = {mask_prob:.2f})", fontweight='bold', color='#ec4899', fontsize=11)
        axes[3].axis('off')
        
    plt.suptitle("Interactive Human Recognition Threshold Test", fontsize=12, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.show()


# Convenient alias
plot_human_recognition_threshold = plot_recognition_threshold


def benchmark_corruption_throughput(
    train_loader: DataLoader,
    noise_fn: Optional[Callable] = None,
    num_batches: int = 50,
    sigma: float = 0.3
) -> None:
    """Benchmarks on-the-fly corruption streaming performance across batches."""
    if noise_fn is None:
        noise_fn = add_gaussian_noise

    start_time = time.time()
    total_samples = 0

    for idx, (clean_images, _) in enumerate(train_loader):
        if idx >= num_batches:
            break
        corrupted_images = noise_fn(clean_images, sigma=sigma)
        total_samples += clean_images.size(0)
        assert clean_images.shape == corrupted_images.shape

    elapsed = time.time() - start_time
    throughput = total_samples / elapsed
    print(f"⚡ Corruption Benchmark Completed:")
    print(f"   Processed {total_samples:,} images across {num_batches} batches in {elapsed:.3f}s")
    print(f"   Throughput: {throughput:,.1f} images/second ({num_batches/elapsed:.1f} batches/second)")
    print(f"   Average Latency per Image: {(elapsed / total_samples) * 1000:.3f} ms")


# ============================================================================
# 4. Model m Architecture & Fast Training Suite (Section 9)
# ============================================================================

class Encoder(nn.Module):
    """Compresses image tensors into a low-dimensional bottleneck code."""
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        z = self.relu2(self.fc2(x))
        return z


class Decoder(nn.Module):
    """Reconstructs full-dimensional image vectors from latent codes."""
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 784)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.relu1(self.fc1(z))
        out = self.sigmoid(self.fc2(z))
        return out


class BaselineDAE(nn.Module):
    """Unified Denoising Autoencoder assembling modular Encoder and Decoder classes."""
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_recon_flat = self.decoder(z)
        if x.dim() == 4:
            return x_recon_flat.view(x.size(0), 1, 28, 28)
        return x_recon_flat


def train_baseline_model_m(
    train_dataset: Dataset,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    latent_dim: int = 32,
    hidden_dim: int = 128
) -> BaselineDAE:
    """
    High-throughput training of Baseline DAE Model m directly in VRAM/CPU memory.
    Trained under mild Gaussian corruption (sigma=0.30) to establish a baseline model for student auditing.
    """
    print(f"🚀 Training Baseline DAE Model m (Encoder + Decoder, d={latent_dim}) on {device}...")
    model_m = BaselineDAE(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model_m.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Extract training data as high-throughput contiguous tensor: (60000, 1, 28, 28)
    if hasattr(train_dataset, 'data'):
        X_train_tensor = train_dataset.data.float().div(255.0).unsqueeze(1).to(device)
    else:
        # Fallback for DataLoader or Subset
        loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        X_train_tensor = next(iter(loader))[0].to(device)

    N = X_train_tensor.size(0)
    model_m.train()
    t0 = time.time()
    
    for ep in range(1, epochs + 1):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            clean = X_train_tensor[idx]
            noisy = add_gaussian_noise(clean, sigma=0.3)
            
            optimizer.zero_grad(set_to_none=True)
            recon = model_m(noisy)
            loss = criterion(recon, clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        
        train_mse = total_loss / N
        if ep % 2 == 0 or ep == epochs:
            print(f"   \rEpoch [{ep:02d}/{epochs:02d}] | Train MSE: {train_mse:.5f}", end = "")
            
    print(f"✅ Baseline Model m successfully trained in {time.time()-t0:.2f}s! Ready for student auditing.\n")
    
    total_params = sum(p.numel() for p in model_m.parameters())
    enc_params = sum(p.numel() for p in model_m.encoder.parameters())
    dec_params = sum(p.numel() for p in model_m.decoder.parameters())
    print(model_m)
    print(f"\n🔢 Encoder Parameters: {enc_params:,}")
    print(f"🔢 Decoder Parameters: {dec_params:,}")
    print(f"🔢 Total Parameters:   {total_params:,}")
    return model_m


# ============================================================================
# 5. Interactive Auditing & PSNR Analysis Suite
# ============================================================================

def compute_batch_psnr(
    clean: torch.Tensor,
    recon: torch.Tensor,
    max_val: float = 1.0,
    eps: float = 1e-10
) -> float:
    """Computes average Peak Signal-to-Noise Ratio (PSNR) in dB over a batch."""
    mse = torch.mean((clean - recon) ** 2).item()
    if mse < eps:
        return 100.0
    return 10.0 * np.log10((max_val ** 2) / mse)


def run_model_m_noise_audit(
    model: nn.Module,
    test_images: torch.Tensor,
    noise_type: str,
    noise_levels: List[float],
    device: torch.device,
    sample_idx: int = 0,
    clean_labels: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Audits Model m across a parameter sweep for Gaussian, Salt & Pepper, or Masking noise.
    Plots side-by-side reconstruction grids for any selected digit sample and the PSNR degradation curve.

    Args:
        sample_idx: The target digit class (0 to 9) to audit. Inputs outside [0, 9] are clamped
                    to the closest valid boundary (<0 -> 0, >9 -> 9) with an assertion guarantee.
        clean_labels: Optional labels tensor used to locate the requested digit in test_images.
    """
    if not noise_levels:
        print("⚠️ Warning: noise_levels list is empty! Please provide at least 2 parameter levels to audit.")
        return {'noise_type': noise_type, 'levels': [], 'psnr': []}

    model.eval()
    test_images = test_images.to(device)
    num_levels = len(noise_levels)
    
    # 1. Digit Selection & Defensive Clamping: Map sample_idx to digit interval [0, 9]
    requested_digit = int(sample_idx)
    target_digit = max(0, min(requested_digit, 9))
    
    if requested_digit < 0:
        print(f"⚠️ Notice: Requested digit {requested_digit} < 0 is outside [0, 9]. Clamping to closest valid digit: 0")
    elif requested_digit > 9:
        print(f"⚠️ Notice: Requested digit {requested_digit} > 9 is outside [0, 9]. Clamping to closest valid digit: 9")
        
    assert 0 <= target_digit <= 9, f"Target digit must be in interval [0, 9], got {target_digit}"

    # 2. Locate the first sample corresponding to the target digit class
    if clean_labels is not None:
        matching_indices = (clean_labels == target_digit).nonzero(as_tuple=True)[0]
        if len(matching_indices) > 0:
            actual_sample_idx = matching_indices[0].item()
        else:
            print(f"⚠️ Warning: Digit '{target_digit}' not found in test batch labels. Falling back to index 0.")
            actual_sample_idx = 0
    else:
        # Fallback if labels are not provided: clamp to available batch index
        actual_sample_idx = max(0, min(target_digit, test_images.size(0) - 1))
    
    psnr_scores = []
    corrupted_samples = []
    reconstructed_samples = []
    
    with torch.no_grad():
        for val in noise_levels:
            if noise_type == 'gaussian':
                noisy = add_gaussian_noise(test_images, sigma=val)
            elif noise_type in ('salt_and_pepper', 'salt_pepper'):
                noisy = add_salt_and_pepper_noise(test_images, prob=val)
            elif noise_type == 'masking':
                noisy = add_masking_noise(test_images, drop_prob=val)
            else:
                raise ValueError(f"Unknown noise type: {noise_type}. Choose from 'gaussian', 'salt_and_pepper', 'masking'.")
            
            recon = model(noisy)
            psnr = compute_batch_psnr(test_images, recon)
            psnr_scores.append(psnr)
            corrupted_samples.append(noisy[actual_sample_idx, 0].cpu().numpy())
            reconstructed_samples.append(recon[actual_sample_idx, 0].cpu().numpy())

    # 1. Multi-Row Visual Grid
    clean_sample = test_images[actual_sample_idx, 0].cpu().numpy()
    fig, axes = plt.subplots(3, num_levels, figsize=(3 * num_levels, 8))
    row_labels = ["Pristine Clean (x)", f"Noisy ({noise_type})", "Model m Output (x̂)"]
    
    for j, val in enumerate(noise_levels):
        axes[0, j].imshow(clean_sample, cmap='gray', vmin=0, vmax=1)
        axes[0, j].set_title(f"Level: {val}\nPSNR: {psnr_scores[j]:.1f} dB", fontsize=11, fontweight='bold')
        axes[1, j].imshow(corrupted_samples[j], cmap='gray', vmin=0, vmax=1)
        axes[2, j].imshow(reconstructed_samples[j], cmap='gray', vmin=0, vmax=1)
        
        for i in range(3):
            if j == 0:
                axes[i, 0].set_ylabel(row_labels[i], fontsize=12, fontweight='bold', labelpad=6)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    actual_digit = clean_labels[actual_sample_idx].item() if clean_labels is not None else target_digit
    plt.suptitle(f"Baseline DAE Model m Auditing — Sensitivity to {noise_type.upper()} Noise (Digit '{actual_digit}' / Batch Index {actual_sample_idx})", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 2. PSNR Degradation Curve
    plt.figure(figsize=(8, 4))
    plt.plot(noise_levels, psnr_scores, 'o-', color='#0284c7', linewidth=2.5, markersize=8, label='Reconstruction PSNR (dB)')
    for x, y in zip(noise_levels, psnr_scores):
        plt.annotate(f"{y:.1f} dB", (x, y), textcoords="offset points", xytext=(0, 10), 
                     ha='center', fontsize=10, fontweight='bold')
    
    plt.title(f"Model m Recovery Fidelity vs. {noise_type.capitalize()} Severity", fontsize=13, fontweight='bold', pad=18)
    plt.xlabel(f"{noise_type.capitalize()} Parameter Value", fontsize=11)
    plt.ylabel("Peak Signal-to-Noise Ratio (PSNR dB)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, fontsize=11)
    plt.margins(y=0.18)
    plt.tight_layout()
    plt.show()
    
    return {'noise_type': noise_type, 'levels': noise_levels, 'psnr': psnr_scores}
