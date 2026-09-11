"""
lab_utils.py - Helper Utilities for Session 6: Evaluation, Metrics & Overfitting Diagnostics Lab
================================================================================================
Course: Deep Learning: Foundations, Systems & Scientific AI Auditing
Instructors: MSc. Antonio Aguilar & Dr. Luis Aguilar Ibáñez
"""

import os
from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset


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
# 2. Tensor Un-Normalization Utility
# ============================================================================

def unnormalize_img(tensor: torch.Tensor, mean: float = 0.1307, std: float = 0.3082) -> torch.Tensor:
    """Inverts z-score standardization back to raw display range [0.0, 1.0]."""
    return (tensor * std + mean).clamp(0.0, 1.0)


# ============================================================================
# 3. Section 1 Visualization: Batch Digits Grid & Class Distribution
# ============================================================================

def plot_batch_and_distribution(
    images_batch: torch.Tensor,
    labels_batch: torch.Tensor,
    mean: float = 0.1307,
    std: float = 0.3082,
    num_samples: int = 16
) -> None:
    """
    Renders two complementary diagnostic panels for incoming streaming data:
      Panel 1: A 2x8 grid of un-normalized sample digits with clear vertical padding.
      Panel 2: An empirical bar chart of class frequencies within this mini-batch
               compared against theoretical expectation E[k_c] = B * (1/10) under Multinomial(B, 0.1).
    """
    # 1. Plot 16 digits in a 2x8 grid
    fig, axes = plt.subplots(2, 8, figsize=(14, 4.8))
    fig.suptitle(f"Inspection of Streamed Mini-Batch: First {num_samples} Standardized Digits",
                 fontsize=13, fontweight='bold')
    plt.subplots_adjust(top=0.88, bottom=0.05, hspace=0.48, wspace=0.25)

    for i in range(num_samples):
        row, col = i // 8, i % 8
        ax = axes[row, col]
        img_unnorm = unnormalize_img(images_batch[i], mean=mean, std=std).squeeze().cpu().numpy()
        ax.imshow(img_unnorm, cmap='gray')
        ax.set_title(f"Label: {labels_batch[i].item()}", fontsize=10, fontweight='bold', pad=6)
        ax.axis('off')

    plt.show()

    # 2. Plot empirical class distribution
    plt.figure(figsize=(10, 3.8))
    batch_size = len(labels_batch)
    counts = torch.bincount(labels_batch.cpu(), minlength=10).numpy()
    expected_val = batch_size / 10.0

    plt.bar(range(10), counts, color='#0284c7', alpha=0.8, edgecolor='black', label=f"Observed Counts ($B={batch_size}$)")
    plt.axhline(expected_val, color='#ef4444', linestyle='--', linewidth=2,
                label=f"Expected Mean $\\mathbb{{E}}[k_c]={expected_val:.1f}$ (Uniform Prior)")

    plt.title(f"Empirical Class Distribution in Single Mini-Batch ($B={batch_size}$)", fontsize=12, fontweight='bold')
    plt.xlabel("Digit Class Index ($c \\in \\{0, \\dots, 9\\}$)", fontsize=10)
    plt.ylabel("Sample Frequency ($k_c$)", fontsize=10)
    plt.xticks(range(10))
    plt.grid(axis='y', alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 4. Section 4 Visualization: Dual Learning Curves
# ============================================================================

def plot_dual_learning_curves(
    history: Dict[str, List[float]],
    num_epochs: int,
    figsize: Tuple[int, int] = (13, 5)
) -> None:
    """
    Plots side-by-side training vs. validation learning curves:
      Panel 1: Cross-Entropy Loss convergence (detects overfitting divergence).
      Panel 2: Classification Accuracy progression (in %).
    """
    plt.figure(figsize=figsize)
    epochs_range = range(1, num_epochs + 1)

    # Panel 1: Loss Curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], 'o-', color='#0284c7', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, history['val_loss'], 's--', color='#e11d48', label='Validation Loss', linewidth=2)
    min_val_loss_epoch = int(np.argmin(history['val_loss'])) + 1
    min_val_loss = min(history['val_loss'])
    plt.axvline(min_val_loss_epoch, color='gray', linestyle=':', alpha=0.6,
                label=f'Min Val Loss (Ep {min_val_loss_epoch}: {min_val_loss:.4f})')
    plt.title("Cross-Entropy Loss Trajectory", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.xticks(epochs_range)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Panel 2: Accuracy Curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], 'o-', color='#0284c7', label='Training Acc', linewidth=2)
    plt.plot(epochs_range, history['val_acc'], 's--', color='#10b981', label='Validation Acc', linewidth=2)
    max_val_acc_epoch = int(np.argmax(history['val_acc'])) + 1
    max_val_acc = max(history['val_acc'])
    plt.axvline(max_val_acc_epoch, color='gray', linestyle=':', alpha=0.6,
                label=f'Max Val Acc (Ep {max_val_acc_epoch}: {max_val_acc:.2f}%)')
    plt.title("Classification Accuracy Trajectory (%)", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.xticks(epochs_range)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.suptitle("MNIST 3-Layer MLP Training & Validation Trajectory", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
# 5. Section 6 Visualization: Confusion Matrix Heatmap
# ============================================================================

def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Renders an annotated 10x10 heatmap for multi-class error cluster auditing.
    """
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=0.5, linecolor='gray')

    plt.title(r"MNIST Test Set Confusion Matrix ($10 \times 10$ Heatmap)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight='bold')
    plt.ylabel("True Ground-Truth Class", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
# 6. Section 7 Visualization: Misclassified Images Inspection
# ============================================================================

def plot_misclassified_audit(
    misclassified_indices: np.ndarray,
    test_dataset: Dataset,
    all_preds: np.ndarray,
    all_probs: np.ndarray,
    num_to_plot: int = 12,
    mean: float = 0.1307,
    std: float = 0.3082,
    figsize: Tuple[int, int] = (12, 9)
) -> None:
    """
    Renders a grid of misclassified test digits along with true label,
    predicted label, and prediction confidence percentage.
    """
    num_to_plot = min(num_to_plot, len(misclassified_indices))
    rows = (num_to_plot + 3) // 4
    plt.figure(figsize=figsize)

    for i in range(num_to_plot):
        idx = misclassified_indices[i]
        img, true_lbl = test_dataset[idx]
        pred_lbl = all_preds[idx]
        confidence = all_probs[idx][pred_lbl] * 100.0

        img_display = unnormalize_img(img, mean=mean, std=std).squeeze().cpu().numpy()

        plt.subplot(rows, 4, i + 1)
        plt.imshow(img_display, cmap='gray')
        plt.title(f"True: {true_lbl} | Pred: {pred_lbl}\nConf: {confidence:.1f}%",
                  fontsize=10, color='red' if true_lbl != pred_lbl else 'black', pad=6)
        plt.axis('off')

    plt.suptitle("Audit of Misclassified Test Digits (Ambiguous Visual Geometry)", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=1.6)
    plt.show()
