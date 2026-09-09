"""
lab_utils.py - Helper Utilities for Session 5 The Canonical PyTorch Training Loop Lab
======================================================================================
"""

import os
from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================================
# 1. Environment & Reproducibility Helpers
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """Sets random seeds for exact reproducibility across PyTorch and NumPy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Detects and returns active compute device (CUDA GPU or CPU) with banner printout."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ PyTorch Version:     {torch.__version__}")
    print(f"💻 Active Device:       {device}")
    print(f"🎯 Reproducibility Seed: 42")
    return device


# ============================================================================
# 2. MNIST Data Ingestion Pipeline Helper
# ============================================================================

def get_mnist_dataloaders(
    batch_size: int = 64,
    root: str = "./data",
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepares standard normalized MNIST training and test DataLoaders.

    Transformations:
      1. ToTensor(): Scales raw uint8 [0, 255] pixels to float32 [0.0, 1.0].
      2. Normalize(mean=(0.1307,), std=(0.3081,)): Standardizes pixel distribution.

    Returns:
      train_loader (shuffled), test_loader (unshuffled)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


# ============================================================================
# 3. Visualization: Training Loss Dynamics (Batch vs. Epoch)
# ============================================================================

def plot_training_loss(
    batch_loss_history: List[float],
    epoch_loss_history: List[float],
    num_epochs: int,
    window: int = 50,
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Plots side-by-side training convergence diagnostics:
      Left:  Noisy raw batch losses with an overlaid rolling moving average.
      Right: Monotonically decreasing smooth epoch loss averages.
    """
    plt.figure(figsize=figsize)

    # Subplot 1: Raw Batch Loss (Noisy dynamics)
    plt.subplot(1, 2, 1)
    plt.plot(batch_loss_history, color="#0284c7", alpha=0.4, label="Batch Loss")
    
    if len(batch_loss_history) >= window:
        moving_avg = np.convolve(batch_loss_history, np.ones(window) / window, mode="valid")
        plt.plot(
            range(window - 1, len(batch_loss_history)),
            moving_avg,
            color="#e11d48",
            linewidth=2,
            label=f"{window}-Batch Moving Avg"
        )
    
    plt.title("Batch Loss History (938 Batches/Epoch)", fontsize=12, fontweight="bold")
    plt.xlabel("Iteration Step", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Smooth Epoch Loss
    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, num_epochs + 1),
        epoch_loss_history,
        "o-",
        color="#10b981",
        linewidth=2.5,
        markersize=8
    )
    plt.title("Epoch Average Loss Convergence", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Average Loss", fontsize=11)
    plt.xticks(range(1, num_epochs + 1))
    plt.grid(True, alpha=0.3)

    plt.suptitle("MNIST 3-Layer MLP Training Loss Dynamics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================================
# 4. Visualization: Impact of Forgetting optimizer.zero_grad()
# ============================================================================

def plot_zero_grad_impact(
    buggy_loss_history: List[float],
    normal_loss_history: List[float],
    max_batches: int = 150,
    figsize: Tuple[int, int] = (8, 4)
) -> None:
    """
    Compares the diverging loss trajectory of training without zero_grad()
    against normal healthy training with zero_grad().
    """
    plt.figure(figsize=figsize)
    plt.plot(
        buggy_loss_history[:max_batches],
        color="#f43f5e",
        linewidth=2,
        label="No zero_grad() (Oscillating/Diverging)"
    )
    plt.plot(
        normal_loss_history[:max_batches],
        color="#10b981",
        linewidth=2,
        label="Normal zero_grad() (Smooth Descent)"
    )
    plt.title("Impact of Forgetting optimizer.zero_grad()", fontsize=13, fontweight="bold")
    plt.xlabel("Batch Step", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
