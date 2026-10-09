"""
lab_utils.py
Session 18: Optimization Dynamics, Normalization & Learning Rate Schedules Lab
Institutional Standard: MIT EECS / ETH Zürich D-INFK

This module encapsulates all infrastructure and visualization boilerplate:
  1. Deterministic Seeding & Accelerator Profiling
  2. High-Speed CIFAR-10 Ingestion, Standardization & In-VRAM Pre-Allocation
  3. Matplotlib Visual Diagnostics & Figures
  4. Ultra-Fast Zero-Overhead In-VRAM Training & Evaluation Engines
  5. Multi-Model Batch-Synchronous Lock-Step Tournament Runners
  6. Curvature Auditing (1D Filter-Normalized Slices & SAM Adversarial Sharpness)
"""

import os
import sys
import time
import math
import copy
import random
import tarfile
import pickle
import urllib.request
from typing import Dict, Tuple, List, Optional, Any, Callable

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR
)


# ==============================================================================
# 1. Environment Setup & Hardware Profiling
# ==============================================================================
def setup_environment(seed: int = 42) -> torch.device:
    """
    Enforces deterministic seeding and profiles hardware accelerators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("⚡ COMPUTE ENVIRONMENT & HARDWARE ACCELERATOR")
    print("=" * 70)
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Active GPU Device : {gpu_name}")
        print(f"Compute Capability: {capability[0]}.{capability[1]}")
        print(f"Total VRAM        : {vram_gb:.2f} GB")
    else:
        print("Running on CPU.")
    print("=" * 70)
    return device


def create_grad_scaler(device_type: str = "cuda") -> Any:
    """
    Instantiates PyTorch's modern device-agnostic GradScaler (PyTorch 2.3+ standard)
    to eliminate FutureWarning notices, while maintaining backward compatibility.
    """
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type)
    return torch.cuda.amp.GradScaler()


# ==============================================================================
# 2. CIFAR-10 Ingestion, Standardization & VRAM Resident Pipeline
# ==============================================================================
def load_cifar10_vram(
    data_dir: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Downloads/loads raw CIFAR-10 python tarball, standardizes images into channel-first
    tensors, deterministically partitions (45k train, 5k val, 10k test), and pre-allocates
    the entire dataset directly resident inside GPU VRAM.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidate_dirs = [
        data_dir,
        "/home/antonio/Documents/UNP-Course/materials/Unit-II/Week-2/2_Wednesday/NotebookLab/data",
        "./data",
        "../data"
    ]
    target_dir = "./data"
    archive_path = None

    for cdir in candidate_dirs:
        if cdir and os.path.exists(cdir):
            candidate_archive = os.path.join(cdir, "cifar-10-python.tar.gz")
            if os.path.exists(candidate_archive):
                target_dir = cdir
                archive_path = candidate_archive
                break

    os.makedirs(target_dir, exist_ok=True)
    if archive_path is None:
        archive_path = os.path.join(target_dir, "cifar-10-python.tar.gz")

    cifar_extracted_dir = os.path.join(target_dir, "cifar-10-batches-py")
    needs_download = not (os.path.exists(archive_path) and os.path.getsize(archive_path) > 100_000_000)
    needs_extract = not (os.path.exists(cifar_extracted_dir) and os.path.exists(os.path.join(cifar_extracted_dir, "data_batch_1")))

    mirror_url = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"

    if needs_download:
        print(f"📥 Downloading CIFAR-10 from BrainChip mirror ({mirror_url})...")
        t0 = time.time()
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(mirror_url, archive_path)
        print(f"✓ Download complete in {time.time() - t0:.2f}s ({os.path.getsize(archive_path) / (1024**2):.1f} MB).")

    if needs_extract:
        print(f"📦 Extracting raw archives to: '{target_dir}'...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=target_dir)

    # Load 5 training batches
    train_images, train_labels = [], []
    for batch_id in range(1, 6):
        batch_file = os.path.join(cifar_extracted_dir, f"data_batch_{batch_id}")
        with open(batch_file, "rb") as f:
            entry = pickle.load(f, encoding="bytes")
            train_images.append(entry[b"data"])
            train_labels.extend(entry[b"labels"])

    train_images = np.vstack(train_images).reshape(-1, 3, 32, 32)
    train_labels = np.array(train_labels, dtype=np.int64)

    # Load test batch
    test_file = os.path.join(cifar_extracted_dir, "test_batch")
    with open(test_file, "rb") as f:
        test_entry = pickle.load(f, encoding="bytes")
        test_images = test_entry[b"data"].reshape(-1, 3, 32, 32)
        test_labels = np.array(test_entry[b"labels"], dtype=np.int64)

    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print("\n🚀 Standardizing and transferring all 60,000 images directly into GPU VRAM...")
    t_vram_start = time.time()

    x_train_raw = torch.from_numpy(train_images).float() / 255.0
    y_train_raw = torch.from_numpy(train_labels).long()
    x_test_raw  = torch.from_numpy(test_images).float() / 255.0
    y_test_raw  = torch.from_numpy(test_labels).long()

    # Channel-wise Standardization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

    x_train_norm = (x_train_raw - mean) / std
    x_test_norm  = (x_test_raw - mean) / std

    # Deterministic Train/Validation Partition (45k train, 5k val)
    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(50000, generator=gen)
    train_idx, val_idx = perm[:45000], perm[45000:]

    x_train_gpu = x_train_norm[train_idx].contiguous().to(device)
    y_train_gpu = y_train_raw[train_idx].contiguous().to(device)
    x_val_gpu   = x_train_norm[val_idx].contiguous().to(device)
    y_val_gpu   = y_train_raw[val_idx].contiguous().to(device)
    x_test_gpu  = x_test_norm.contiguous().to(device)
    y_test_gpu  = y_test_raw.contiguous().to(device)

    total_mb = (x_train_gpu.nbytes + y_train_gpu.nbytes +
                x_val_gpu.nbytes + y_val_gpu.nbytes +
                x_test_gpu.nbytes + y_test_gpu.nbytes) / (1024 ** 2)

    print(f"✓ VRAM Transfer Complete in {time.time() - t_vram_start:.2f}s!")
    print(f"  - x_train_gpu : {tuple(x_train_gpu.shape)} ({x_train_gpu.nbytes / (1024**2):.1f} MB)")
    print(f"  - x_val_gpu   : {tuple(x_val_gpu.shape)}   ({x_val_gpu.nbytes / (1024**2):.1f} MB)")
    print(f"  - x_test_gpu  : {tuple(x_test_gpu.shape)}  ({x_test_gpu.nbytes / (1024**2):.1f} MB)")
    print(f"  - Total In-VRAM Footprint: {total_mb:.1f} MB (Zero PCIe bus latency during training)")

    return x_train_gpu, y_train_gpu, x_val_gpu, y_val_gpu, x_test_gpu, y_test_gpu, cifar_classes


# ==============================================================================
# 3. Visualization Helpers
# ==============================================================================
def plot_cifar10_exploration(
    x_val_gpu: torch.Tensor,
    y_val_gpu: torch.Tensor,
    cifar_classes: List[str]
) -> None:
    """
    Renders 10 class samples alongside RGB raw vs standardized histograms.
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(x_val_gpu.device)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(x_val_gpu.device)

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 2.8])

    # Sample Grid
    for cls_idx in range(10):
        r, c = cls_idx // 5, cls_idx % 5
        sample_loc = (y_val_gpu == cls_idx).nonzero(as_tuple=True)[0][0]
        img_norm = x_val_gpu[sample_loc]
        img_denorm = (img_norm * std[0] + mean[0]).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

        ax = fig.add_subplot(gs[r, c])
        ax.imshow(img_denorm)
        ax.set_title(f"{cifar_classes[cls_idx]}", fontsize=10, fontweight="bold")
        ax.axis("off")

    # Histograms
    ax_hist_raw = fig.add_subplot(gs[0, 5])
    ax_hist_std = fig.add_subplot(gs[1, 5])

    sample_pool = x_val_gpu[:1000]
    sample_pool_raw = (sample_pool * std + mean).clamp(0, 1).cpu().numpy()
    sample_pool_std = sample_pool.cpu().numpy()

    colors = ['#e74c3c', '#2ecc71', '#3498db']
    c_names = ['R Channel', 'G Channel', 'B Channel']

    for c_i in range(3):
        ax_hist_raw.hist(sample_pool_raw[:, c_i].flatten(), bins=50, density=True,
                         alpha=0.45, color=colors[c_i], label=c_names[c_i])
        ax_hist_std.hist(sample_pool_std[:, c_i].flatten(), bins=50, density=True,
                         alpha=0.45, color=colors[c_i], label=c_names[c_i])

    ax_hist_raw.set_title("Raw Pixel Intensities $[0, 1]$", fontsize=10, fontweight="bold")
    ax_hist_raw.legend(loc="upper right", fontsize=8)
    ax_hist_std.set_title(r"Standardized Distribution ($\mu \approx 0, \sigma \approx 1$)", fontsize=10, fontweight="bold")
    ax_hist_std.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 4. In-VRAM Core Execution Engines
# ==============================================================================
def train_epoch_vram(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    batch_size: int = 256,
    scheduler: Optional[Any] = None,
    step_per_batch: bool = False
) -> Tuple[float, float, List[float]]:
    """Executes 1 full training epoch directly inside GPU VRAM."""
    model.train()
    n_samples = x_data.shape[0]
    indices = torch.randperm(n_samples, device=x_data.device)
    criterion = nn.CrossEntropyLoss()
    total_loss, correct = 0.0, 0
    step_lrs = []

    for start_idx in range(0, n_samples, batch_size):
        b_idx = indices[start_idx : start_idx + batch_size]
        bx, by = x_data[b_idx], y_data[b_idx]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(bx)
            loss = criterion(logits, by)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step_lrs.append(optimizer.param_groups[0]["lr"])
        if scheduler is not None and step_per_batch:
            scheduler.step()

        total_loss += loss.item() * bx.size(0)
        correct += (logits.argmax(dim=1) == by).sum().item()

    if scheduler is not None and not step_per_batch:
        scheduler.step()

    return total_loss / n_samples, correct / n_samples, step_lrs


@torch.no_grad()
def eval_epoch_vram(
    model: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    batch_size: int = 512
) -> Tuple[float, float]:
    """Fast batch-wise validation directly inside GPU VRAM."""
    model.eval()
    n_samples = x_data.shape[0]
    criterion = nn.CrossEntropyLoss()
    total_loss, correct = 0.0, 0

    for start_idx in range(0, n_samples, batch_size):
        bx = x_data[start_idx : start_idx + batch_size]
        by = y_data[start_idx : start_idx + batch_size]

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(bx)
            loss = criterion(logits, by)

        total_loss += loss.item() * bx.size(0)
        correct += (logits.argmax(dim=1) == by).sum().item()

    return total_loss / n_samples, correct / n_samples


# ==============================================================================
# 5. Lock-Step Normalization Tournament Runner & Plotter
# ==============================================================================
def run_lockstep_normalization_experiment(
    model_bn: nn.Module,
    model_ln: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 25,
    batch_size: int = 256
) -> Dict[str, Any]:
    """
    Executes a 2-way batch-synchronous lockstep tournament between AlexNet-BN and AlexNet-LN.
    """
    print(f"\n▶ Executing Batch-Synchronous Lock-Step Tournament for {epochs} Epochs...")
    print("  Protocol: Bitwise identical mini-batch sequence (bx, by) streamed to both architectures.\n")

    opt_bn = AdamW(model_bn.parameters(), lr=1e-3, weight_decay=1e-4)
    opt_ln = AdamW(model_ln.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler_bn = create_grad_scaler("cuda")
    scaler_ln = create_grad_scaler("cuda")
    criterion = nn.CrossEntropyLoss()

    results = {
        "AlexNet-BN": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "epoch_times": []},
        "AlexNet-LN": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "epoch_times": []}
    }

    n_samples = x_train.shape[0]

    for ep in range(1, epochs + 1):
        model_bn.train()
        model_ln.train()

        indices = torch.randperm(n_samples, device=x_train.device)
        loss_bn_acc, corr_bn_acc, time_bn_acc = 0.0, 0, 0.0
        loss_ln_acc, corr_ln_acc, time_ln_acc = 0.0, 0, 0.0

        for start_idx in range(0, n_samples, batch_size):
            b_idx = indices[start_idx : start_idx + batch_size]
            bx, by = x_train[b_idx], y_train[b_idx]
            bs = bx.size(0)

            # Model BN
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt_bn.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits_bn = model_bn(bx)
                loss_bn = criterion(logits_bn, by)
            scaler_bn.scale(loss_bn).backward()
            scaler_bn.step(opt_bn)
            scaler_bn.update()
            torch.cuda.synchronize()
            time_bn_acc += (time.perf_counter() - t0)
            loss_bn_acc += loss_bn.item() * bs
            corr_bn_acc += (logits_bn.argmax(dim=1) == by).sum().item()

            # Model LN (Identical bx, by warm in L2 cache!)
            t0 = time.perf_counter()
            opt_ln.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits_ln = model_ln(bx)
                loss_ln = criterion(logits_ln, by)
            scaler_ln.scale(loss_ln).backward()
            scaler_ln.step(opt_ln)
            scaler_ln.update()
            torch.cuda.synchronize()
            time_ln_acc += (time.perf_counter() - t0)
            loss_ln_acc += loss_ln.item() * bs
            corr_ln_acc += (logits_ln.argmax(dim=1) == by).sum().item()

        val_loss_bn, val_acc_bn = eval_epoch_vram(model_bn, x_val, y_val, batch_size=512)
        val_loss_ln, val_acc_ln = eval_epoch_vram(model_ln, x_val, y_val, batch_size=512)

        results["AlexNet-BN"]["train_loss"].append(loss_bn_acc / n_samples)
        results["AlexNet-BN"]["val_loss"].append(val_loss_bn)
        results["AlexNet-BN"]["train_acc"].append((corr_bn_acc / n_samples) * 100)
        results["AlexNet-BN"]["val_acc"].append(val_acc_bn * 100)
        results["AlexNet-BN"]["epoch_times"].append(time_bn_acc)

        results["AlexNet-LN"]["train_loss"].append(loss_ln_acc / n_samples)
        results["AlexNet-LN"]["val_loss"].append(val_loss_ln)
        results["AlexNet-LN"]["train_acc"].append((corr_ln_acc / n_samples) * 100)
        results["AlexNet-LN"]["val_acc"].append(val_acc_ln * 100)
        results["AlexNet-LN"]["epoch_times"].append(time_ln_acc)

        if ep % 5 == 0 or ep == epochs:
            print(f"  [Epoch {ep:02d}/{epochs}] "
                  f"BN: Loss {loss_bn_acc/n_samples:.4f} | Acc {(corr_bn_acc/n_samples)*100:5.2f}% | Val {val_acc_bn*100:5.2f}% ({time_bn_acc:.2f}s)  <--->  "
                  f"LN: Loss {loss_ln_acc/n_samples:.4f} | Acc {(corr_ln_acc/n_samples)*100:5.2f}% | Val {val_acc_ln*100:5.2f}% ({time_ln_acc:.2f}s)")

    results["AlexNet-BN"]["total_time"] = sum(results["AlexNet-BN"]["epoch_times"])
    results["AlexNet-BN"]["model"] = model_bn
    results["AlexNet-LN"]["total_time"] = sum(results["AlexNet-LN"]["epoch_times"])
    results["AlexNet-LN"]["model"] = model_ln

    print(f"\n✓ AlexNet-BN Finished in {results['AlexNet-BN']['total_time']:.1f}s (Avg: {np.mean(results['AlexNet-BN']['epoch_times']):.2f}s/epoch) | Best Val Acc: {max(results['AlexNet-BN']['val_acc']):.2f}%")
    print(f"✓ AlexNet-LN Finished in {results['AlexNet-LN']['total_time']:.1f}s (Avg: {np.mean(results['AlexNet-LN']['epoch_times']):.2f}s/epoch) | Best Val Acc: {max(results['AlexNet-LN']['val_acc']):.2f}%")
    return results


def plot_normalization_comparison(results: Dict[str, Any], epochs: int = 25) -> None:
    """Renders 3-panel comparison figure between AlexNet-BN and AlexNet-LN."""
    epochs_range = range(1, epochs + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))
    colors_norm = {"AlexNet-BN": "#2980b9", "AlexNet-LN": "#27ae60"}

    # Panel 1: Loss
    for name, hist in results.items():
        c = colors_norm[name]
        ax1.plot(epochs_range, hist["train_loss"], label=f"{name} (Train)", color=c, linestyle="--", alpha=0.8)
        ax1.plot(epochs_range, hist["val_loss"], label=f"{name} (Val)", color=c, linewidth=2.0)
    ax1.set_title("Cross-Entropy Loss Dynamics", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Panel 2: Accuracy
    for name, hist in results.items():
        c = colors_norm[name]
        ax2.plot(epochs_range, hist["train_acc"], label=f"{name} (Train)", color=c, linestyle="--", alpha=0.8)
        ax2.plot(epochs_range, hist["val_acc"], label=f"{name} (Val)", color=c, linewidth=2.0)
    ax2.set_title("Top-1 Accuracy Progression (%)", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    # Panel 3: Latency
    model_names = list(results.keys())
    avg_times = [np.mean(results[m]["epoch_times"]) for m in model_names]
    best_accs = [max(results[m]["val_acc"]) for m in model_names]

    bar_x = np.arange(len(model_names))
    bars = ax3.bar(bar_x, avg_times, color=[colors_norm[m] for m in model_names], width=0.5)
    ax3.set_xticks(bar_x)
    ax3.set_xticklabels(model_names, fontweight="semibold")
    ax3.set_ylabel("Seconds per Epoch (In-VRAM)")
    ax3.set_title("Wall-Clock Epoch Latency", fontweight="bold")

    for bar, acc in zip(bars, best_accs):
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}s/ep\n({acc:.1f}% Acc)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 6. Schedulers Simulation & 4-Way Lock-Step Grand Prix
# ==============================================================================
def simulate_and_plot_schedules(epochs: int = 25, steps_per_epoch: int = 176) -> None:
    """Generates the theoretical schedule trajectory curves and renders comparison figure."""
    total_steps = epochs * steps_per_epoch
    dummy_model = nn.Linear(10, 2)
    schedules = {}

    # StepLR
    opt = SGD(dummy_model.parameters(), lr=1e-3)
    opt.step()
    s = StepLR(opt, step_size=8, gamma=0.5)
    schedules["StepLR"] = [opt.param_groups[0]["lr"] for _ in range(epochs) for _ in [None] if [opt.step(), s.step()]]

    # CosineAnnealingLR
    opt = SGD(dummy_model.parameters(), lr=1e-3)
    opt.step()
    s = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    schedules["CosineAnnealingLR"] = [opt.param_groups[0]["lr"] for _ in range(epochs) for _ in [None] if [opt.step(), s.step()]]

    # CosineWarmRestarts
    opt = SGD(dummy_model.parameters(), lr=1e-3)
    opt.step()
    s = CosineAnnealingWarmRestarts(opt, T_0=7, T_mult=2, eta_min=1e-5)
    schedules["CosineWarmRestarts"] = [opt.param_groups[0]["lr"] for _ in range(epochs) for _ in [None] if [opt.step(), s.step()]]

    # OneCycleLR
    opt = SGD(dummy_model.parameters(), lr=1e-3)
    s = OneCycleLR(opt, max_lr=2e-3, total_steps=total_steps, pct_start=0.3)
    schedules["OneCycleLR"] = [opt.param_groups[0]["lr"] for _ in range(total_steps) for _ in [None] if [opt.step(), s.step()]]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    colors = ["#e74c3c", "#2980b9", "#f39c12", "#27ae60"]

    for i, (name, traj) in enumerate(schedules.items()):
        if name == "OneCycleLR":
            x_axis = np.linspace(1, epochs, len(traj))
            axes[i].plot(x_axis, traj, color=colors[i], linewidth=2.2)
            axes[i].set_xlabel("Epoch (Sub-stepped per batch)")
        else:
            axes[i].step(range(1, epochs + 1), traj, where="post", color=colors[i], linewidth=2.2)
            axes[i].set_xlabel("Epoch (Stepped per epoch)")

        axes[i].set_title(name, fontweight="bold")
        axes[i].set_ylabel("Learning Rate")
        axes[i].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Theoretical Learning Rate Schedules Dynamics (Simulation)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()


def run_lockstep_tournament(
    model_cls: Callable[..., nn.Module],
    configs: List[Dict[str, Any]],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 25,
    batch_size: int = 256,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Executes a 4-way batch-synchronous lock-step tournament across learning rate schedules.
    """
    if device is None:
        device = x_train.device

    print(f"\n▶ Executing 4-Way Batch-Synchronous Tournament for {epochs} Epochs...")
    print("  Protocol: Bitwise identical mini-batch sequence (bx, by) streamed to all 4 models.\n")

    contestants = {}
    for cfg in configs:
        name = cfg["name"]
        torch.manual_seed(42)
        model = model_cls(norm_type="ln").to(device)
        optimizer = AdamW(model.parameters(), lr=cfg["init_lr"], weight_decay=1e-4)
        scheduler = cfg["scheduler_fn"](optimizer)
        scaler = create_grad_scaler("cuda")
        contestants[name] = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "step_per_batch": cfg["step_per_batch"],
            "history": {
                "train_loss": [], "val_loss": [],
                "train_acc": [], "val_acc": [],
                "all_step_lrs": [], "epoch_times": []
            }
        }

    criterion = nn.CrossEntropyLoss()
    n_samples = x_train.shape[0]

    for ep in range(1, epochs + 1):
        for c in contestants.values():
            c["model"].train()

        indices = torch.randperm(n_samples, device=device)
        batch_losses = {name: 0.0 for name in contestants}
        batch_corrects = {name: 0 for name in contestants}
        batch_times = {name: 0.0 for name in contestants}

        for start_idx in range(0, n_samples, batch_size):
            b_idx = indices[start_idx : start_idx + batch_size]
            bx, by = x_train[b_idx], y_train[b_idx]
            bs = bx.size(0)

            for name, c in contestants.items():
                torch.cuda.synchronize()
                t0 = time.perf_counter()

                c["optimizer"].zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = c["model"](bx)
                    loss = criterion(logits, by)
                c["scaler"].scale(loss).backward()
                c["scaler"].step(c["optimizer"])
                c["scaler"].update()

                c["history"]["all_step_lrs"].append(c["optimizer"].param_groups[0]["lr"])
                if c["step_per_batch"]:
                    c["scheduler"].step()

                torch.cuda.synchronize()
                batch_times[name] += (time.perf_counter() - t0)
                batch_losses[name] += loss.item() * bs
                batch_corrects[name] += (logits.argmax(dim=1) == by).sum().item()

        for name, c in contestants.items():
            if not c["step_per_batch"]:
                c["scheduler"].step()

        val_accs_summary = {}
        for name, c in contestants.items():
            train_loss = batch_losses[name] / n_samples
            train_acc  = batch_corrects[name] / n_samples
            val_loss, val_acc = eval_epoch_vram(c["model"], x_val, y_val, batch_size=512)

            c["history"]["train_loss"].append(train_loss)
            c["history"]["val_loss"].append(val_loss)
            c["history"]["train_acc"].append(train_acc * 100)
            c["history"]["val_acc"].append(val_acc * 100)
            c["history"]["epoch_times"].append(batch_times[name])
            val_accs_summary[name] = val_acc * 100

        if ep % 5 == 0 or ep == epochs:
            lead_name = max(val_accs_summary, key=val_accs_summary.get)
            print(f"  [Ep {ep:02d}/{epochs}] "
                  f"Step: {val_accs_summary.get('StepLR', 0):4.1f}% | "
                  f"Cosine: {val_accs_summary.get('CosineAnnealingLR', 0):4.1f}% | "
                  f"Restarts: {val_accs_summary.get('CosineWarmRestarts', 0):4.1f}% | "
                  f"OneCycle: {val_accs_summary.get('OneCycleLR', 0):4.1f}% "
                  f"★ Leader: {lead_name}")

    results = {}
    for name, c in contestants.items():
        hist = c["history"]
        hist["total_time"] = sum(hist["epoch_times"])
        hist["model"] = c["model"]
        hist["best_val_acc"] = max(hist["val_acc"])
        hist["final_val_acc"] = hist["val_acc"][-1]
        results[name] = hist
        print(f"  ✓ {name:20s} Finished in {hist['total_time']:.1f}s | Best Val Acc: {hist['best_val_acc']:.2f}% | Final Val Acc: {hist['final_val_acc']:.2f}%")

    return results


def plot_schedulers_tournament(
    results: Dict[str, Any],
    epochs: int = 25,
    configs: Optional[List[Dict[str, Any]]] = None
) -> None:
    """Renders 4-panel tournament benchmark suite."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Session 18: Learning Rate Schedulers Benchmark on AlexNet-LN", fontsize=16, fontweight="bold")
    epochs_axis = range(1, epochs + 1)

    default_colors = {"StepLR": "#e74c3c", "CosineAnnealingLR": "#2980b9", "CosineWarmRestarts": "#f39c12", "OneCycleLR": "#27ae60"}
    color_map = {cfg["name"]: cfg["color"] for cfg in configs} if configs else default_colors

    # Panel 1: LR
    for name, hist in results.items():
        lrs = hist["all_step_lrs"]
        step_x = np.linspace(1, epochs, len(lrs))
        axes[0, 0].plot(step_x, lrs, label=name, color=color_map.get(name, "#333"), linewidth=2.0)
    axes[0, 0].set_title(r"Learning Rate ($\eta$) Dynamics per Step", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Learning Rate")
    axes[0, 0].legend()

    # Panel 2: Val Loss
    for name, hist in results.items():
        axes[0, 1].plot(epochs_axis, hist["val_loss"], label=name, color=color_map.get(name, "#333"), linewidth=2.0)
    axes[0, 1].set_title("Validation Loss Progression", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Validation Cross-Entropy Loss")
    axes[0, 1].legend()

    # Panel 3: Val Acc
    for name, hist in results.items():
        axes[1, 0].plot(epochs_axis, hist["val_acc"], label=name, color=color_map.get(name, "#333"), linewidth=2.2)
    axes[1, 0].set_title("Validation Accuracy Progression (%)", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].legend()

    # Panel 4: Bar Chart Leaderboard
    sched_names = list(results.keys())
    best_accs = [results[s]["best_val_acc"] for s in sched_names]
    final_accs = [results[s]["final_val_acc"] for s in sched_names]

    x = np.arange(len(sched_names))
    w = 0.35
    axes[1, 1].bar(x - w/2, best_accs, width=w, label="Peak Val Acc (%)", color="#3498db")
    axes[1, 1].bar(x + w/2, final_accs, width=w, label="Final Val Acc (%)", color="#2ecc71")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(sched_names, rotation=15, fontweight="semibold")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_title("Tournament Leaderboard Summary", fontweight="bold")
    axes[1, 1].set_ylim(min(final_accs) - 5, max(best_accs) + 3)
    axes[1, 1].legend()

    for i in range(len(sched_names)):
        axes[1, 1].text(x[i] - w/2, best_accs[i] + 0.3, f"{best_accs[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
        axes[1, 1].text(x[i] + w/2, final_accs[i] + 0.3, f"{final_accs[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 7. Curvature & Sharpness Audits
# ==============================================================================
@torch.no_grad()
def compute_filter_normalized_slice(
    model: nn.Module,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    alphas: np.ndarray,
    batch_size: int = 1024
) -> Tuple[np.ndarray, List[float]]:
    """Evaluates loss along a filter-normalized 1D direction ray (Li et al., NeurIPS 2018)."""
    model.eval()
    direction = {}
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            v = torch.randn_like(param)
            shape_view = [param.size(0)] + [1] * (param.dim() - 1)
            v_norm = v.view(param.size(0), -1).norm(dim=1).view(*shape_view)
            p_norm = param.view(param.size(0), -1).norm(dim=1).view(*shape_view)
            direction[name] = v / (v_norm + 1e-7) * p_norm
        else:
            direction[name] = torch.zeros_like(param)

    orig_params = {n: p.clone() for n, p in model.named_parameters()}
    loss_curve = []

    for a in alphas:
        for name, param in model.named_parameters():
            param.copy_(orig_params[name] + a * direction[name])
        eval_loss, _ = eval_epoch_vram(model, x_eval, y_eval, batch_size=batch_size)
        loss_curve.append(eval_loss)

    for name, param in model.named_parameters():
        param.copy_(orig_params[name])

    return alphas, loss_curve


def compute_adversarial_sharpness(
    model: nn.Module,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    rhos: List[float],
    batch_size: int = 256,
    eval_batch_size: int = 1024
) -> Dict[str, List[float]]:
    """Computes SAM-style adversarial epsilon-sharpness by taking a projected gradient ascent step."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    sub_x = x_eval[:batch_size]
    sub_y = y_eval[:batch_size]

    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        base_logits = model(sub_x)
        base_loss = criterion(base_logits, sub_y)
    base_loss.backward()

    base_val_loss, base_val_acc = eval_epoch_vram(model, x_eval, y_eval, batch_size=eval_batch_size)
    grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)).item()
    orig_params = {n: p.clone() for n, p in model.named_parameters()}
    results = {"rhos": rhos, "delta_loss": [], "acc_drop": [], "perturbed_acc": []}

    for r in rhos:
        for n, p in model.named_parameters():
            if p.grad is not None:
                p.data.add_(r * p.grad / (grad_norm + 1e-7))

        adv_loss, adv_acc = eval_epoch_vram(model, x_eval, y_eval, batch_size=eval_batch_size)
        acc_drop = (base_val_acc - adv_acc) * 100
        results["delta_loss"].append(adv_loss - base_val_loss)
        results["acc_drop"].append(acc_drop)
        results["perturbed_acc"].append(adv_acc * 100)

        for n, p in model.named_parameters():
            p.data.copy_(orig_params[n])

    model.zero_grad(set_to_none=True)
    return results


def run_curvature_audits(
    models_dict: Dict[str, nn.Module],
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    alphas: Optional[np.ndarray] = None,
    rhos: Optional[List[float]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Runs filter-normalized slices and adversarial sharpness across models."""
    if alphas is None:
        alphas = np.linspace(-0.6, 0.6, 25)
    if rhos is None:
        rhos = [0.01, 0.03, 0.05, 0.08, 0.10]

    print("🔬 Running Loss Landscape Curvature & Adversarial Accuracy Degradation Audits on GPU...")
    audit_data = {}

    for name, model in models_dict.items():
        _, slice_losses = compute_filter_normalized_slice(model, x_eval, y_eval, alphas=alphas, batch_size=1024)
        sharpness_data = compute_adversarial_sharpness(model, x_eval, y_eval, rhos=rhos, batch_size=256, eval_batch_size=1024)
        audit_data[name] = {
            "slice": slice_losses,
            "sharpness": sharpness_data
        }
        print(f"  ✓ Completed curvature & accuracy degradation profiling for: {name}")

    print("✓ All models profiled successfully across filter-normalized slices and adversarial balls.")
    return alphas, audit_data


def plot_loss_landscape_geometry(
    audit_data: Dict[str, Any],
    alphas: np.ndarray,
    color_map: Optional[Dict[str, str]] = None
) -> None:
    """Renders 2-panel Loss Landscape Geometry: 1D Slices & Accuracy Drop Curve."""
    if color_map is None:
        color_map = {"StepLR": "#e74c3c", "CosineAnnealingLR": "#2980b9", "CosineWarmRestarts": "#f39c12", "OneCycleLR": "#27ae60"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    fig.suptitle("Loss Landscape Geometry Benchmark: Flat vs. Sharp Minima Across Schedulers", fontsize=15, fontweight="bold")

    # Panel 1: Slices
    for name, data in audit_data.items():
        c = color_map.get(name, "#333")
        ax1.plot(alphas, data["slice"], label=name, color=c, linewidth=2.4)

    ax1.set_title(r"Experiment A: 1D Filter-Normalized Loss Slices ($\mathcal{L}(\theta^* + \alpha d)$)", fontweight="bold")
    ax1.set_xlabel(r"Normalized Perturbation Distance $\alpha$")
    ax1.set_ylabel("Validation Loss")
    ax1.set_ylim(bottom=0.5, top=max(max(d["slice"]) for d in audit_data.values()) * 1.05)
    ax1.legend(loc="upper center")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Accuracy Drop
    for name, data in audit_data.items():
        c = color_map.get(name, "#333")
        rhos = data["sharpness"]["rhos"]
        acc_drop = data["sharpness"]["acc_drop"]
        ax2.plot(rhos, acc_drop, marker="o", label=name, color=c, linewidth=2.2, markersize=6)

    ax2.set_title(r"Experiment B: Adversarial Robustness — Accuracy Drop ($\Delta \mathrm{Acc}$ under Radius $\rho$)", fontweight="bold")
    ax2.set_xlabel(r"Perturbation Radius $\rho$ (Gradient Ascent Step)")
    ax2.set_ylabel(r"Validation Accuracy Drop ($\%$)")
    max_drop = max(max(d["sharpness"]["acc_drop"]) for d in audit_data.values())
    ax2.set_ylim(bottom=0, top=max(max_drop * 1.15, 1.0))
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("🎯 LOSS LANDSCAPE GEOMETRY ANALYSIS: ACCURACY ROBUSTNESS")
    print("=" * 70)
    for name, data in audit_data.items():
        sharp_at_max = data["sharpness"]["delta_loss"][-1]
        acc_drop_max = data["sharpness"]["acc_drop"][-1]
        pert_acc_max = data["sharpness"]["perturbed_acc"][-1]
        print(f"  • {name:<22}: Acc Drop @ ρ=0.10: -{acc_drop_max:5.1f}% | Perturbed Acc: {pert_acc_max:5.1f}% | ΔLoss: {sharp_at_max:+.3f}")
    print("=" * 70)


# ==============================================================================
# 8. Fine-Tuning Sweep Engine & Plotter
# ==============================================================================
def run_lockstep_finetuning(
    model_cls: Callable[..., nn.Module],
    candidates: List[Dict[str, Any]],
    steps_per_epoch: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 25,
    batch_size: int = 256,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Executes a 3-way lockstep fine-tuning sweep across OneCycleLR peak learning rates."""
    if device is None:
        device = x_train.device

    print(f"\n▶ Executing 3-Way Batch-Synchronous Sweep for {epochs} Epochs...")
    print("  Protocol: Bitwise identical mini-batch sequence (bx, by) streamed to all candidate models.\n")

    sweep_models = {}
    for cand in candidates:
        label = cand["label"]
        torch.manual_seed(42)
        model = model_cls(norm_type="ln").to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = OneCycleLR(optimizer, max_lr=cand["max_lr"], steps_per_epoch=steps_per_epoch, epochs=epochs, pct_start=0.3)
        scaler = create_grad_scaler("cuda")
        sweep_models[label] = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "history": {
                "train_loss": [], "val_loss": [],
                "train_acc": [], "val_acc": [],
                "step_lrs": [], "epoch_times": []
            }
        }

    criterion = nn.CrossEntropyLoss()
    n_samples = x_train.shape[0]

    for ep in range(1, epochs + 1):
        for m in sweep_models.values():
            m["model"].train()

        indices = torch.randperm(n_samples, device=device)
        batch_losses = {label: 0.0 for label in sweep_models}
        batch_corrects = {label: 0 for label in sweep_models}
        batch_times = {label: 0.0 for label in sweep_models}

        for start_idx in range(0, n_samples, batch_size):
            b_idx = indices[start_idx : start_idx + batch_size]
            bx, by = x_train[b_idx], y_train[b_idx]
            bs = bx.size(0)

            for label, m in sweep_models.items():
                torch.cuda.synchronize()
                t0 = time.perf_counter()

                m["optimizer"].zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = m["model"](bx)
                    loss = criterion(logits, by)
                m["scaler"].scale(loss).backward()
                m["scaler"].step(m["optimizer"])
                m["scaler"].update()

                m["scheduler"].step()
                m["history"]["step_lrs"].append(m["optimizer"].param_groups[0]["lr"])

                torch.cuda.synchronize()
                batch_times[label] += (time.perf_counter() - t0)
                batch_losses[label] += loss.item() * bs
                batch_corrects[label] += (logits.argmax(dim=1) == by).sum().item()

        val_accs_summary = {}
        for label, m in sweep_models.items():
            train_loss = batch_losses[label] / n_samples
            train_acc  = batch_corrects[label] / n_samples
            val_loss, val_acc = eval_epoch_vram(m["model"], x_val, y_val, batch_size=1024)

            m["history"]["train_loss"].append(train_loss)
            m["history"]["val_loss"].append(val_loss)
            m["history"]["train_acc"].append(train_acc * 100)
            m["history"]["val_acc"].append(val_acc * 100)
            m["history"]["epoch_times"].append(batch_times[label])
            val_accs_summary[label] = val_acc * 100

        if ep % 5 == 0 or ep == epochs:
            lead_cand = max(val_accs_summary, key=val_accs_summary.get)
            print(f"  [Ep {ep:02d}/{epochs}] "
                  f"1e-3: {val_accs_summary.get('Conservative (max_lr = 1e-3)', 0):4.1f}% | "
                  f"2e-3: {val_accs_summary.get('Standard     (max_lr = 2e-3)', 0):4.1f}% | "
                  f"4e-3: {val_accs_summary.get('Aggressive   (max_lr = 4e-3)', 0):4.1f}% "
                  f"★ Leader: {lead_cand[:12]}")

    results = {}
    for label, m in sweep_models.items():
        hist = m["history"]
        hist["total_time"] = sum(hist["epoch_times"])
        hist["model"] = m["model"]
        hist["best_val_acc"] = max(hist["val_acc"])
        results[label] = hist
        print(f"  ✓ {label} | Best Val Acc: {hist['best_val_acc']:.2f}% in {hist['total_time']:.1f}s")

    return results


def plot_finetune_sweep(
    results: Dict[str, Any],
    epochs: int = 25,
    candidates: Optional[List[Dict[str, Any]]] = None
) -> Tuple[str, nn.Module]:
    """Renders 2-panel fine-tuning sweep comparison figure and returns champion configuration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fine-Tuning OneCycleLR Peak Learning Rate on AlexNet-LN", fontsize=15, fontweight="bold")
    epochs_axis = range(1, epochs + 1)

    default_colors = {"Conservative (max_lr = 1e-3)": "#95a5a6", "Standard     (max_lr = 2e-3)": "#2980b9", "Aggressive   (max_lr = 4e-3)": "#e67e22"}
    colors_ft = {cand["label"]: cand["color"] for cand in candidates} if candidates else default_colors

    for label, hist in results.items():
        c = colors_ft.get(label, "#333")
        ax1.plot(epochs_axis, hist["val_loss"], label=f"{label} (Best: {min(hist['val_loss']):.3f})", color=c, linewidth=2.2)
        ax2.plot(epochs_axis, hist["val_acc"], label=f"{label} (Peak: {hist['best_val_acc']:.2f}%)", color=c, linewidth=2.2)

    ax1.set_title(r"Validation Loss Trajectories Across $\eta_{\max}$", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Loss")
    ax1.legend()

    ax2.set_title(r"Validation Accuracy Across $\eta_{\max}$", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    champion_label = max(results, key=lambda k: results[k]["best_val_acc"])
    champion_model = results[champion_label]["model"]
    print(f"\n🏆 CHAMPION SCHEDULE CONFIGURATION: {champion_label}")
    print(f"   Peak Validation Accuracy: {results[champion_label]['best_val_acc']:.2f}%")
    return champion_label, champion_model
