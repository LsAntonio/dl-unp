"""
Session 7 Lab Utilities: Optimization & Deep Regularization
Course: Deep Learning: Foundations, Systems & Scientific AI Auditing
Authors: MSc. Antonio Aguilar & Dr. Luis Aguilar Ibáñez
National University of Piura (UNP) & Pontifical Catholic University of Chile (UC)

This module encapsulates boilerplate data caching, high-throughput GPU training loops,
multi-seed statistical execution harnesses, and diagnostic visualization suites
for studying Weight Decay (L2) and Inverted Dropout on high-dimensional vision benchmarks.
"""

from typing import Dict, List, Tuple, Optional, Any, Type
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# ============================================================================
# 1. Environment & Reproducibility Setup
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Configures deterministic pseudo-random number generator (PRNG) states
    across CPU, CUDA backends, and Python runtime for exact scientific reproducibility.
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
# 2. High-Throughput In-VRAM GPU Cache & Canonical Training Heartbeat
# ============================================================================

_GPU_DATA_CACHE: Dict[Any, Tuple[torch.Tensor, torch.Tensor]] = {}

def _get_gpu_tensor_data(loader: DataLoader, dev: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Caches full DataLoader partitions directly into GPU VRAM for instant tensor slicing,
    eliminating 469 CPU-GPU PCIe bus transfers per epoch on small-scale datasets (e.g. MNIST).
    """
    if dev.type != 'cuda':
        return None, None
    if loader not in _GPU_DATA_CACHE:
        all_x, all_y = [], []
        for x, y in loader:
            all_x.append(x)
            all_y.append(y)
        _GPU_DATA_CACHE[loader] = (
            torch.cat(all_x, dim=0).to(dev),
            torch.cat(all_y, dim=0).to(dev)
        )
    return _GPU_DATA_CACHE[loader]


def train_and_evaluate(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    epochs: int = 6,
    name: str = "Model",
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Executes the canonical 5-step optimization heartbeat with non-blocking GPU tensor
    accumulation (loss.detach()), fast gradient clearing (set_to_none=True), and
    instant 1-shot matrix multiplication validation directly in GPU VRAM.

    Args:
        model: PyTorch nn.Module neural architecture.
        optimizer: Configured gradient optimizer (SGD, Adam, AdamW).
        train_loader: Streaming DataLoader for empirical training samples.
        val_loader: Streaming DataLoader for out-of-sample validation samples.
        criterion: Cross-entropy loss function.
        epochs: Number of complete dataset optimization passes.
        name: Telemetry display identifier.
        batch_size: Mini-batch gradient slice size (default: 256).
        device: Active compute device (defaults to auto-detected device).
        verbose: If True, streams per-epoch progress logs.

    Returns:
        history: Dictionary containing 'train_loss', 'val_loss', and 'val_acc'.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    if verbose:
        print(f"\n🚀 Training [{name}] for {epochs} Epochs on {device}...")
    start_time = time.time()
    
    # Check if tensors can be cached and sliced directly in GPU VRAM
    X_train_gpu, y_train_gpu = _get_gpu_tensor_data(train_loader, device)
    X_val_gpu, y_val_gpu     = _get_gpu_tensor_data(val_loader, device)
    use_vram = (X_train_gpu is not None and X_val_gpu is not None)
    
    if batch_size is None:
        batch_size = train_loader.batch_size if train_loader.batch_size else 256
    N_train = len(train_loader.dataset)
    N_val   = len(val_loader.dataset)
    
    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        
        if use_vram:
            perm = torch.randperm(N_train, device=device)
            running_loss_tensor = torch.zeros(1, device=device)
            for i in range(0, N_train, batch_size):
                idx = perm[i : i + batch_size]
                images, labels = X_train_gpu[idx], y_train_gpu[idx]
                
                # Canonical 5-Step Training Loop (Non-blocking)
                optimizer.zero_grad(set_to_none=True)     # 1. Fast zeroing (bypasses memory overhead)
                outputs = model(images)                   # 2. Forward pass
                loss = criterion(outputs, labels)         # 3. Loss error
                loss.backward()                           # 4. Autograd backward
                optimizer.step()                          # 5. Parameter update
                
                running_loss_tensor += loss.detach() * len(idx)
            
            # Single GPU -> CPU scalar synchronization per epoch
            epoch_train_loss = (running_loss_tensor / N_train).item()
        else:
            running_train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * images.size(0)
            epoch_train_loss = running_train_loss / N_train
        
        # 2. Validation Phase (Evaluation Mode)
        model.eval()
        with torch.no_grad():
            if use_vram:
                # Instant 1-shot full validation matrix multiplication directly in VRAM (~31 MB)
                val_outputs = model(X_val_gpu)
                epoch_val_loss = criterion(val_outputs, y_val_gpu).item()
                epoch_val_acc = (val_outputs.argmax(dim=1) == y_val_gpu).float().mean().item() * 100.0
            else:
                running_val_loss = 0.0
                correct = 0
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                epoch_val_loss = running_val_loss / N_val
                epoch_val_acc = (correct / N_val) * 100.0
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        if verbose:
            print(f"  \rEpoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%", end="")
            
    elapsed = time.time() - start_time
    final_acc = history['val_acc'][-1]
    if verbose:
        print(f"\n⏱️ Training completed in {elapsed:.1f}s | Final Val Acc: {final_acc:.2f}%")
    return history


# ============================================================================
# 3. Section 3: Weight Decay Exploration & Visualization Utilities
# ============================================================================

def plot_weight_decay_exploration(
    weight_decay_values: List[float],
    adamw_experiments: Dict[float, Dict[str, List[float]]],
    epochs_to_train: int,
    figsize: Tuple[int, int] = (16, 5.5)
) -> None:
    """
    Renders dual diagnostic panels comparing validation loss descent (Log Scale)
    and empirical generalization gaps (Val Loss - Train Loss) alongside validation accuracies.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=120)
    colors = {0.0: '#ef4444', 1e-4: '#f59e0b', 1e-2: '#10b981', 1e-1: '#6366f1'}
    epochs_range = range(1, epochs_to_train + 1)

    # Left Panel: Validation Loss Trajectories on Log Scale
    for wd in weight_decay_values:
        label_str = f"λ = {wd} (Pure Adam)" if wd == 0.0 else f"λ = {wd}"
        ax1.plot(epochs_range, adamw_experiments[wd]['val_loss'], 
                 label=label_str, color=colors.get(wd, '#64748b'), linewidth=2.2, marker='o', markersize=3.5)

    ax1.set_yscale('log')
    ax1.set_title('Validation Loss Trajectory vs. Weight Decay λ (Log Scale)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Epochs', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Validation Loss (Log Scale)', fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5, which='both')
    ax1.legend(frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=10)

    # Right Panel: Final Generalization Gap & Validation Accuracy Bar Chart
    wd_labels = [f"λ = {wd}" for wd in weight_decay_values]
    final_train = [adamw_experiments[wd]['train_loss'][-1] for wd in weight_decay_values]
    final_val = [adamw_experiments[wd]['val_loss'][-1] for wd in weight_decay_values]
    final_acc = [adamw_experiments[wd]['val_acc'][-1] for wd in weight_decay_values]

    x_pos = np.arange(len(weight_decay_values))
    width = 0.35

    rects1 = ax2.bar(x_pos - width/2, final_train, width, label='Final Train Loss', color='#ef4444', alpha=0.85, edgecolor='#b91c1c')
    rects2 = ax2.bar(x_pos + width/2, final_val, width, label='Final Val Loss', color='#0284c7', alpha=0.85, edgecolor='#0369a1')

    max_h = max(max(final_train), max(final_val))
    ax2.set_ylim(0, max_h * 1.35)

    for i in range(len(weight_decay_values)):
        gap = abs(final_val[i] - final_train[i])
        higher = max(final_val[i], final_train[i])
        ax2.text(i, higher + 0.008, f"Gap: {gap:.3f}\nAcc: {final_acc[i]:.2f}%", 
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1e293b',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.9))

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(wd_labels, fontsize=10.5, fontweight='bold')
    ax2.set_title('Final Generalization Gap & Accuracy Across Weight Decay λ', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('Weight Decay Parameter (λ)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cross-Entropy Loss', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.show()


def run_multi_seed_wd(
    model_cls: Type[nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    weight_decay_values: List[float],
    seeds: List[int],
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: Optional[torch.device] = None
) -> Dict[float, Dict[str, List[Any]]]:
    """
    Executes a Full Factorial Randomized Block Experiment across K random seeds
    for all specified weight decay parameters (λ).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    multi_seed_wd_results: Dict[float, Dict[str, List[Any]]] = {
        wd: {
            'train_loss': [], 'val_loss': [], 'val_acc': [],
            'final_train_loss': [], 'final_val_loss': [], 'final_val_acc': [], 'final_gap': []
        }
        for wd in weight_decay_values
    }

    print(f"🚀 Launching Multi-Seed Experiment across λ in {weight_decay_values}...")
    print(f"• Seeds ({len(seeds)} total): {seeds}")
    print(f"• Total runs: {len(seeds) * len(weight_decay_values)} runs ({len(seeds) * len(weight_decay_values) * epochs} epochs total)\n")

    t0 = time.time()
    for s_idx, seed in enumerate(seeds, 1):
        print(f"========== [Seed Block {s_idx}/{len(seeds)}: {seed}] ==========")
        for wd in weight_decay_values:
            set_seed(seed)
            model = model_cls().to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            lbl = f"λ={wd}" if wd > 0 else "λ=0.0 (Pure Adam)"
            
            hist = train_and_evaluate(
                model, optimizer, train_loader, val_loader, criterion,
                epochs=epochs, name=f"Seed {seed} | {lbl}", batch_size=batch_size,
                device=device, verbose=False
            )
            
            multi_seed_wd_results[wd]['train_loss'].append(hist['train_loss'])
            multi_seed_wd_results[wd]['val_loss'].append(hist['val_loss'])
            multi_seed_wd_results[wd]['val_acc'].append(hist['val_acc'])
            
            final_tr  = hist['train_loss'][-1]
            final_vl  = hist['val_loss'][-1]
            final_acc = hist['val_acc'][-1]
            final_gap = final_vl - final_tr
            
            multi_seed_wd_results[wd]['final_train_loss'].append(final_tr)
            multi_seed_wd_results[wd]['final_val_loss'].append(final_vl)
            multi_seed_wd_results[wd]['final_val_acc'].append(final_acc)
            multi_seed_wd_results[wd]['final_gap'].append(final_gap)
            
            print(f"  [λ={wd:<5}] Final Val Loss: {final_vl:.4f} | Val Acc: {final_acc:.2f}% | Gap: {final_gap:.4f}")

    total_time = time.time() - t0
    print(f"\n✅ All {len(seeds) * len(weight_decay_values)} runs finished in {total_time:.1f}s ({total_time/60:.1f} min)!")
    return multi_seed_wd_results


def plot_multi_seed_wd(
    multi_seed_wd_results: Dict[float, Dict[str, List[Any]]],
    weight_decay_values: List[float],
    seeds: List[int],
    epochs: int = 50,
    figsize: Tuple[int, int] = (16, 5.5)
) -> None:
    """
    Renders empirical mean trajectories with ±1σ confidence ribbons and matched-seed bar charts.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=120)
    palette = {0.0: '#ef4444', 1e-4: '#f59e0b', 1e-2: '#10b981', 1e-1: '#6366f1'}
    epochs_arr = np.arange(1, epochs + 1)

    # Panel 1: Trajectory with mean ± 1 std confidence bands (Log Scale)
    for wd in weight_decay_values:
        val_loss_matrix = np.array(multi_seed_wd_results[wd]['val_loss'])  # Shape: (K, epochs)
        mean_traj = np.mean(val_loss_matrix, axis=0)
        std_traj  = np.std(val_loss_matrix, axis=0)
        
        lbl = f"λ = {wd}" if wd > 0 else "λ = 0.0 (Pure Adam)"
        c = palette.get(wd, '#64748b')
        ax1.plot(epochs_arr, mean_traj, label=lbl, color=c, lw=2.2)
        ax1.fill_between(epochs_arr, mean_traj - std_traj, mean_traj + std_traj, color=c, alpha=0.15)

    ax1.set_title(f"Validation Loss Trajectory Across {len(seeds)} Seeds (Mean ± 1σ, Log Scale)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epochs", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Validation Loss (Log Scale)", fontsize=11, fontweight='bold')
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.5)

    # Panel 2: Grouped Metrics (Mean ± Std) + Matched Seed Lines
    wd_labels = [f"λ = {wd}" for wd in weight_decay_values]
    x_coords = np.arange(len(weight_decay_values))

    means_val_loss = [np.mean(multi_seed_wd_results[wd]['final_val_loss']) for wd in weight_decay_values]
    stds_val_loss  = [np.std(multi_seed_wd_results[wd]['final_val_loss'], ddof=1) for wd in weight_decay_values]
    means_val_acc  = [np.mean(multi_seed_wd_results[wd]['final_val_acc']) for wd in weight_decay_values]
    stds_val_acc   = [np.std(multi_seed_wd_results[wd]['final_val_acc'], ddof=1) for wd in weight_decay_values]

    bars = ax2.bar(x_coords, means_val_loss, yerr=stds_val_loss, capsize=6,
                   color=[palette.get(wd, '#64748b') for wd in weight_decay_values], alpha=0.75, edgecolor='#334155', lw=1.5)

    # Overlay individual seed points jittered and connect paired seeds
    for s_idx in range(len(seeds)):
        seed_losses = [multi_seed_wd_results[wd]['final_val_loss'][s_idx] for wd in weight_decay_values]
        ax2.plot(x_coords, seed_losses, color='#64748b', alpha=0.4, linestyle='--', marker='o', markersize=5)

    # Annotate bars with Mean Loss & Acc
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + stds_val_loss[i] + 0.005,
                 f"Loss: {means_val_loss[i]:.4f}\nAcc: {means_val_acc[i]:.2f}±{stds_val_acc[i]:.2f}%",
                 ha='center', va='bottom', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.9))

    ax2.set_xticks(x_coords)
    ax2.set_xticklabels(wd_labels, fontsize=10.5, fontweight='bold')
    ax2.set_title(f"Final Val Loss & Acc Across {len(seeds)} Matched Seeds", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Weight Decay Parameter (λ)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Validation Cross-Entropy Loss", fontsize=11, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.set_ylim(0, max(means_val_loss) * 1.45)

    plt.tight_layout()
    plt.show()


# ============================================================================
# 4. Section 3.3 & 3.4: Multi-Seed Ablation Study (Baseline vs. Optimal Regularizer)
# ============================================================================

def run_multi_seed_ablation(
    model_cls: Type[nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    seeds: List[int],
    epochs: int = 50,
    selected_l2_wd: float = 1e-2,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: Optional[torch.device] = None
) -> Tuple[List[Dict[str, List[float]]], List[Dict[str, List[float]]]]:
    """
    Conducts head-to-head training across identical random seeds for
    Unregularized Baseline (λ=0.0) vs. Selected AdamW (λ=selected_l2_wd).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed_results_baseline: List[Dict[str, List[float]]] = []
    seed_results_adamw: List[Dict[str, List[float]]] = []

    print("🚀 Launching Multi-Seed Ablation Study...")
    print(f"• Seeds ({len(seeds)} total): {seeds}")
    print(f"• Epochs per run: {epochs}")
    print(f"• Total epochs: {len(seeds) * 2 * epochs} across {len(seeds) * 2} training runs directly in GPU VRAM\n")

    t0 = time.time()
    for i, seed in enumerate(seeds, 1):
        print(f"--- [Seed {i:02d}/{len(seeds)}: {seed}] ---")
        
        # 1. Train Unregularized Baseline (Pure Adam, λ=0.0)
        set_seed(seed)
        model_base = model_cls().to(device)
        opt_base = optim.Adam(model_base.parameters(), lr=lr)
        hist_base = train_and_evaluate(
            model_base, opt_base, train_loader, val_loader, criterion,
            epochs=epochs, name=f"Baseline (Seed {seed})", batch_size=batch_size,
            device=device, verbose=False
        )
        seed_results_baseline.append(hist_base)
        
        # 2. Train Regularized Model (AdamW, λ=selected_l2_wd)
        set_seed(seed)
        model_adamw = model_cls().to(device)
        opt_adamw = optim.AdamW(model_adamw.parameters(), lr=lr, weight_decay=selected_l2_wd)
        hist_adamw = train_and_evaluate(
            model_adamw, opt_adamw, train_loader, val_loader, criterion,
            epochs=epochs, name=f"AdamW (Seed {seed})", batch_size=batch_size,
            device=device, verbose=False
        )
        seed_results_adamw.append(hist_adamw)
        
        print(f"  Baseline Val Loss: {hist_base['val_loss'][-1]:.4f} | Acc: {hist_base['val_acc'][-1]:.2f}%")
        print(f"  AdamW    Val Loss: {hist_adamw['val_loss'][-1]:.4f} | Acc: {hist_adamw['val_acc'][-1]:.2f}%")

    total_time = time.time() - t0
    print(f"\n✅ Finished {len(seeds) * 2} runs in {total_time:.1f}s ({total_time/60:.1f} min)!")
    return seed_results_baseline, seed_results_adamw


def plot_multi_seed_ablation(
    seed_results_baseline: List[Dict[str, List[float]]],
    seed_results_adamw: List[Dict[str, List[float]]],
    seeds: List[int],
    epochs: int = 50,
    selected_l2_wd: float = 1e-2,
    figsize: Tuple[int, int] = (16, 5.5)
) -> None:
    """
    Renders comparative trajectory confidence ribbons and generalization gap evolutions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=120)
    epochs_range = np.arange(1, epochs + 1)

    # Stack trajectories into matrices of shape (num_seeds, epochs)
    base_val_loss = np.array([run["val_loss"] for run in seed_results_baseline])
    adamw_val_loss = np.array([run["val_loss"] for run in seed_results_adamw])

    base_train_loss = np.array([run["train_loss"] for run in seed_results_baseline])
    adamw_train_loss = np.array([run["train_loss"] for run in seed_results_adamw])

    base_gap = base_val_loss - base_train_loss
    adamw_gap = adamw_val_loss - adamw_train_loss

    # Compute empirical Mean and Standard Deviation across seeds
    base_val_mean, base_val_std = np.mean(base_val_loss, axis=0), np.std(base_val_loss, axis=0)
    adamw_val_mean, adamw_val_std = np.mean(adamw_val_loss, axis=0), np.std(adamw_val_loss, axis=0)

    base_gap_mean, base_gap_std = np.mean(base_gap, axis=0), np.std(base_gap, axis=0)
    adamw_gap_mean, adamw_gap_std = np.mean(adamw_gap, axis=0), np.std(adamw_gap, axis=0)

    # Panel 1: Validation Loss Trajectories on Log Scale with ±1 Std Error Bands
    for k in range(len(seeds)):
        ax1.plot(epochs_range, base_val_loss[k], color="#ef4444", alpha=0.15, linewidth=0.9)
        ax1.plot(epochs_range, adamw_val_loss[k], color="#10b981", alpha=0.15, linewidth=0.9)

    ax1.plot(epochs_range, base_val_mean, color="#ef4444", linewidth=2.6, 
             label=f"Baseline (Pure Adam, λ=0.0) Mean (N={len(seeds)})")
    ax1.fill_between(epochs_range, base_val_mean - base_val_std, base_val_mean + base_val_std, 
                     color="#ef4444", alpha=0.18, label="Baseline ±1σ Band")

    ax1.plot(epochs_range, adamw_val_mean, color="#10b981", linewidth=2.6, 
             label=f"Regularized AdamW (λ={selected_l2_wd}) Mean (N={len(seeds)})")
    ax1.fill_between(epochs_range, adamw_val_mean - adamw_val_std, adamw_val_mean + adamw_val_std, 
                     color="#10b981", alpha=0.18, label="AdamW ±1σ Band")

    ax1.set_yscale("log")
    ax1.set_title(f"Validation Loss Trajectory Across {len(seeds)} Seeds (Log Scale)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epochs", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Validation Loss (Log Scale)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5, which="both")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.5, loc="upper center")

    # Panel 2: Mean Generalization Gap (Val Loss - Train Loss)
    for k in range(len(seeds)):
        ax2.plot(epochs_range, base_gap[k], color="#ef4444", alpha=0.15, linewidth=0.9)
        ax2.plot(epochs_range, adamw_gap[k], color="#10b981", alpha=0.15, linewidth=0.9)

    ax2.plot(epochs_range, base_gap_mean, color="#ef4444", linewidth=2.6, label=f"Baseline Mean Gap (N={len(seeds)})")
    ax2.fill_between(epochs_range, base_gap_mean - base_gap_std, base_gap_mean + base_gap_std, 
                     color="#ef4444", alpha=0.18)

    ax2.plot(epochs_range, adamw_gap_mean, color="#10b981", linewidth=2.6, label=f"AdamW (λ={selected_l2_wd}) Mean Gap (N={len(seeds)})")
    ax2.fill_between(epochs_range, adamw_gap_mean - adamw_gap_std, adamw_gap_mean + adamw_gap_std, 
                     color="#10b981", alpha=0.18)

    ax2.axhline(0, color="#64748b", linestyle=":", linewidth=1.2)
    ax2.set_title(f"Generalization Gap (Val Loss - Train Loss) Across {len(seeds)} Seeds", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Epochs", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Generalization Gap (Cross-Entropy)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.5, loc="upper left")

    plt.tight_layout()
    plt.show()

    # Print Statistical Summary Table
    base_final_accs = [run["val_acc"][-1] for run in seed_results_baseline]
    adamw_final_accs = [run["val_acc"][-1] for run in seed_results_adamw]

    print("\n" + "="*78)
    print(f"📊 MULTI-SEED STATISTICAL SUMMARY AT EPOCH {epochs} (N = {len(seeds)} Seeds)")
    print("="*78)
    print(f"{'Metric':<30} | {'Baseline (λ=0.0)':<20} | {f'AdamW (λ={selected_l2_wd})':<20}")
    print("-"*78)
    print(f"{'Validation Loss (μ ± σ)':<30} | {base_val_mean[-1]:.4f} ± {base_val_std[-1]:.4f}     | {adamw_val_mean[-1]:.4f} ± {adamw_val_std[-1]:.4f}")
    print(f"{'Validation Accuracy (μ ± σ)':<30} | {np.mean(base_final_accs):.2f}% ± {np.std(base_final_accs):.2f}%     | {np.mean(adamw_final_accs):.2f}% ± {np.std(adamw_final_accs):.2f}%")
    print(f"{'Generalization Gap (μ ± σ)':<30} | {base_gap_mean[-1]:.4f} ± {base_gap_std[-1]:.4f}     | {adamw_gap_mean[-1]:.4f} ± {adamw_gap_std[-1]:.4f}")
    print("="*78)


# ============================================================================
# 5. Section 5 & 6: Inverted Dropout Exploration & Multi-Seed Utilities
# ============================================================================

def plot_dropout_tournament(
    tournament_results: Dict[str, Dict[str, List[float]]],
    epochs_to_train: int,
    figsize: Tuple[int, int] = (16, 5.5)
) -> None:
    """
    Renders multi-optimizer trajectory comparisons for the Optimizer Tournament.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=120)

    colors = {
        'Vanilla SGD': '#ef4444',
        'SGD + Momentum': '#f59e0b',
        'Standard Adam': '#0284c7',
        'Modern AdamW': '#10b981'
    }
    styles = {
        'Vanilla SGD': '--',
        'SGD + Momentum': '-.',
        'Standard Adam': ':',
        'Modern AdamW': '-'
    }
    markers = {
        'Vanilla SGD': 'o',
        'SGD + Momentum': 's',
        'Standard Adam': '^',
        'Modern AdamW': 'D'
    }

    epochs_range = range(1, epochs_to_train + 1)

    # Panel 1: Training Loss
    for name, hist in tournament_results.items():
        ax1.plot(epochs_range, hist['train_loss'], label=name, 
                 color=colors.get(name, '#64748b'), linestyle=styles.get(name, '-'),
                 marker=markers.get(name, 'o'), markersize=5, linewidth=2.2)
    ax1.set_title('Training Loss Descent by Optimizer', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=10, loc='upper right')
    ax1.set_yscale('log')

    # Panel 2: Validation Accuracy
    for name, hist in tournament_results.items():
        ax2.plot(epochs_range, hist['val_acc'], label=f"{name} ({hist['val_acc'][-1]:.2f}%)", 
                 color=colors.get(name, '#64748b'), linestyle=styles.get(name, '-'),
                 marker=markers.get(name, 'o'), markersize=5, linewidth=2.2)
    ax2.set_title('Validation Accuracy Progression by Optimizer', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=10, loc='lower right')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.show()


def plot_dropout_gap_comparison(
    dropout_rates: List[float],
    dropout_experiments: Dict[float, Dict[str, List[float]]],
    figsize: Tuple[int, int] = (12, 5.5)
) -> None:
    """
    Renders grouped bar charts displaying the Generalization Gap (Val Loss vs. Train Loss)
    and validation accuracy across distinct dropout rates p.
    """
    plt.figure(figsize=figsize, dpi=120)

    p_vals = list(dropout_experiments.keys())
    final_train_losses = [dropout_experiments[p]['train_loss'][-1] for p in p_vals]
    final_val_losses = [dropout_experiments[p]['val_loss'][-1] for p in p_vals]
    final_val_accs = [dropout_experiments[p]['val_acc'][-1] for p in p_vals]

    x_pos = np.arange(len(p_vals))
    width = 0.35

    rects1 = plt.bar(x_pos - width/2, final_train_losses, width, label='Final Train Loss', color='#ef4444', alpha=0.85, edgecolor='#b91c1c')
    rects2 = plt.bar(x_pos + width/2, final_val_losses, width, label='Final Val Loss', color='#0284c7', alpha=0.85, edgecolor='#0369a1')

    # Add headroom above highest bar
    max_val = max(max(final_train_losses), max(final_val_losses))
    plt.ylim(0, max_val * 1.35)

    for i in range(len(p_vals)):
        gap = abs(final_val_losses[i] - final_train_losses[i])
        higher_bar = max(final_val_losses[i], final_train_losses[i])
        plt.text(i, higher_bar + 0.015, f"Gap: {gap:.3f}\nAcc: {final_val_accs[i]:.1f}%", 
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1e293b',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.9))

    plt.xticks(x_pos, [f"p = {p}" for p in p_vals], fontsize=11, fontweight='bold')
    plt.title('Generalization Gap (Validation vs. Training Loss) Across Dropout Rates', fontsize=13, fontweight='bold', pad=14)
    plt.xlabel('Dropout Probability (p)', fontsize=11, fontweight='bold')
    plt.ylabel('Cross-Entropy Loss', fontsize=11, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.show()


def run_multi_seed_dropout(
    model_cls: Type[nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    dropout_rates: List[float],
    seeds: List[int],
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: Optional[torch.device] = None
) -> Dict[float, Dict[str, List[Any]]]:
    """
    Executes a Full Factorial Randomized Block Experiment across K seeds
    for all specified dropout probabilities (p) with isolated weight decay (λ=0.0).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    multi_seed_do_results: Dict[float, Dict[str, List[Any]]] = {
        p: {
            'train_loss': [], 'val_loss': [], 'val_acc': [],
            'final_train_loss': [], 'final_val_loss': [], 'final_val_acc': [], 'final_gap': []
        }
        for p in dropout_rates
    }

    print(f"🚀 Launching Multi-Seed Experiment across Dropout p in {dropout_rates} (λ=0.0)...")
    print(f"• Seeds ({len(seeds)} total): {seeds}")
    print(f"• Total runs: {len(seeds) * len(dropout_rates)} runs ({len(seeds) * len(dropout_rates) * epochs} epochs total)\n")

    t0 = time.time()
    for s_idx, seed in enumerate(seeds, 1):
        print(f"========== [Seed Block {s_idx}/{len(seeds)}: {seed}] ==========")
        for p in dropout_rates:
            set_seed(seed)
            # Instantiate model supporting dropout_p, dropout_rate, or positional parameter
            try:
                model = model_cls(dropout_p=p).to(device)
            except TypeError:
                try:
                    model = model_cls(dropout_rate=p).to(device)
                except TypeError:
                    try:
                        model = model_cls(p=p).to(device)
                    except TypeError:
                        model = model_cls(p).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
            
            hist = train_and_evaluate(
                model, optimizer, train_loader, val_loader, criterion,
                epochs=epochs, name=f"Seed {seed} | Dropout p={p}", batch_size=batch_size,
                device=device, verbose=False
            )
            
            multi_seed_do_results[p]['train_loss'].append(hist['train_loss'])
            multi_seed_do_results[p]['val_loss'].append(hist['val_loss'])
            multi_seed_do_results[p]['val_acc'].append(hist['val_acc'])
            
            final_tr  = hist['train_loss'][-1]
            final_vl  = hist['val_loss'][-1]
            final_acc = hist['val_acc'][-1]
            final_gap = final_vl - final_tr
            
            multi_seed_do_results[p]['final_train_loss'].append(final_tr)
            multi_seed_do_results[p]['final_val_loss'].append(final_vl)
            multi_seed_do_results[p]['final_val_acc'].append(final_acc)
            multi_seed_do_results[p]['final_gap'].append(final_gap)
            
            print(f"  [Dropout p={p:<3}] Final Val Loss: {final_vl:.4f} | Val Acc: {final_acc:.2f}% | Gap: {final_gap:.4f}")

    total_time = time.time() - t0
    print(f"\n✅ All {len(seeds) * len(dropout_rates)} runs finished in {total_time:.1f}s ({total_time/60:.1f} min)!")
    return multi_seed_do_results


def plot_multi_seed_dropout(
    multi_seed_do_results: Dict[float, Dict[str, List[Any]]],
    dropout_rates: List[float],
    seeds: List[int],
    epochs: int = 50,
    figsize: Tuple[int, int] = (16, 5.5)
) -> None:
    """
    Renders empirical mean validation trajectories with ±1σ confidence ribbons and matched-seed bar charts for Dropout.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=120)
    do_palette = {0.1: '#06b6d4', 0.3: '#10b981', 0.5: '#f59e0b', 0.7: '#ef4444'}
    epochs_arr = np.arange(1, epochs + 1)

    # Panel 1: Trajectory with mean ± 1 std confidence bands (Log Scale)
    for p in dropout_rates:
        val_loss_matrix = np.array(multi_seed_do_results[p]['val_loss'])  # Shape: (K, epochs)
        mean_traj = np.mean(val_loss_matrix, axis=0)
        std_traj  = np.std(val_loss_matrix, axis=0)
        
        c = do_palette.get(p, '#64748b')
        ax1.plot(epochs_arr, mean_traj, label=f"Dropout p = {p}", color=c, lw=2.2)
        ax1.fill_between(epochs_arr, mean_traj - std_traj, mean_traj + std_traj, color=c, alpha=0.15)

    ax1.set_title(f"Validation Loss Trajectory Across {len(seeds)} Seeds (Mean ± 1σ, Log Scale)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epochs", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Validation Loss (Log Scale)", fontsize=11, fontweight='bold')
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.5)

    # Panel 2: Grouped Metrics (Mean ± Std) + Matched Seed Lines
    p_labels = [f"p = {p}" for p in dropout_rates]
    x_coords = np.arange(len(dropout_rates))

    means_val_loss = [np.mean(multi_seed_do_results[p]['final_val_loss']) for p in dropout_rates]
    stds_val_loss  = [np.std(multi_seed_do_results[p]['final_val_loss'], ddof=1) for p in dropout_rates]
    means_val_acc  = [np.mean(multi_seed_do_results[p]['final_val_acc']) for p in dropout_rates]
    stds_val_acc   = [np.std(multi_seed_do_results[p]['final_val_acc'], ddof=1) for p in dropout_rates]

    bars = ax2.bar(x_coords, means_val_loss, yerr=stds_val_loss, capsize=6,
                   color=[do_palette.get(p, '#64748b') for p in dropout_rates], alpha=0.75, edgecolor='#334155', lw=1.5)

    # Overlay individual seed points jittered and connect paired seeds
    for s_idx in range(len(seeds)):
        seed_losses = [multi_seed_do_results[p]['final_val_loss'][s_idx] for p in dropout_rates]
        ax2.plot(x_coords, seed_losses, color='#64748b', alpha=0.35, linestyle='--', marker='o', markersize=4.5)

    # Annotate bars with Mean Loss & Acc
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + stds_val_loss[i] + 0.005,
                 f"Loss: {means_val_loss[i]:.4f}\nAcc: {means_val_acc[i]:.2f}±{stds_val_acc[i]:.2f}%",
                 ha='center', va='bottom', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.9))

    ax2.set_xticks(x_coords)
    ax2.set_xticklabels(p_labels, fontsize=10.5, fontweight='bold')
    ax2.set_title(f"Final Val Loss & Acc Across {len(seeds)} Matched Seeds", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Dropout Probability (p)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Validation Cross-Entropy Loss", fontsize=11, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.set_ylim(0, max(means_val_loss) * 1.45)

    plt.tight_layout()
    plt.show()


# ============================================================================
# 6. Section 7: Diagnostic & Confusion Matrix Heatmap
# ============================================================================

def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    accuracy: Optional[float] = None,
    class_names: Optional[List[Any]] = None,
    figsize: Tuple[int, int] = (9, 7.5)
) -> None:
    """
    Renders an annotated 10x10 confusion matrix heatmap on the test set.
    """
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=figsize, dpi=120)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                annot_kws={"size": 10.5, "weight": "bold"},
                linewidths=0.5, linecolor='#e2e8f0',
                xticklabels=class_names, yticklabels=class_names)
    
    title = "Test Set Confusion Matrix"
    if accuracy is not None:
        title += f" (Accuracy: {accuracy:.2f}%)"
    plt.title(title, fontsize=13, fontweight='bold', pad=14)
    plt.xlabel('Predicted Digit', fontsize=11, fontweight='bold')
    plt.ylabel('True Digit', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()
