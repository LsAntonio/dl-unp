"""
Session 8 Lab Utilities: Autoencoder Architecture & Latent Spaces Lab
Course: Deep Learning: Foundations, Systems & Scientific AI Auditing
Authors: MSc. Antonio Aguilar & Dr. Luis Aguilar Ibáñez
National University of Piura (UNP) & Pontifical Catholic University of Chile (UC)

This module encapsulates boilerplate data caching, high-throughput GPU training loops,
bottleneck capacity benchmarking, latent feature visualization, and unsupervised
anomaly detection suites for Autoencoder architectures on vision benchmarks.
"""

from typing import Dict, List, Tuple, Optional, Any, Type
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


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
# 2. High-Throughput In-VRAM GPU Cache & Autoencoder Training Heartbeat
# ============================================================================

_GPU_DATA_CACHE: Dict[Any, Tuple[torch.Tensor, torch.Tensor]] = {}

def get_gpu_tensor_data(loader: DataLoader, dev: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Caches full DataLoader partitions directly into GPU VRAM for instant tensor slicing,
    eliminating 469 CPU-GPU PCIe bus transfers per epoch on small-scale vision datasets.
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


def train_and_evaluate_ae(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    epochs: int = 5,
    name: str = "Autoencoder",
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Executes the self-supervised continuous pixel regression training heartbeat
    with non-blocking GPU tensor accumulation (loss.detach()), fast gradient clearing
    (set_to_none=True), and 1-shot matrix validation directly in GPU VRAM.

    Args:
        model: PyTorch Autoencoder architecture.
        optimizer: Configured gradient optimizer (AdamW, Adam, etc.).
        train_loader: Streaming DataLoader for training samples.
        val_loader: Streaming DataLoader for validation samples.
        criterion: Reconstruction loss function (typically nn.MSELoss).
        epochs: Number of complete dataset training passes.
        name: Telemetry display identifier.
        batch_size: Mini-batch gradient slice size (default: 256).
        device: Active compute device.
        verbose: If True, streams per-epoch progress logs.

    Returns:
        history: Dictionary containing 'train_loss' and 'val_loss'.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': []}

    if verbose:
        print(f"\n🚀 Training [{name}] for {epochs} Epochs on {device}...")
    start_time = time.time()

    X_train_gpu, _ = get_gpu_tensor_data(train_loader, device)
    X_val_gpu, _   = get_gpu_tensor_data(val_loader, device)
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
                images = X_train_gpu[idx]

                optimizer.zero_grad(set_to_none=True)
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                loss.backward()
                optimizer.step()

                running_loss_tensor += loss.detach() * len(idx)

            epoch_train_loss = (running_loss_tensor / N_train).item()
        else:
            running_train_loss = 0.0
            for images, _ in train_loader:
                images = images.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * images.size(0)
            epoch_train_loss = running_train_loss / N_train
        history['train_loss'].append(epoch_train_loss)

        # 2. Validation Phase
        model.eval()
        with torch.no_grad():
            if use_vram:
                if N_val <= 10000:
                    val_recon = model(X_val_gpu)
                    epoch_val_loss = criterion(val_recon, X_val_gpu).item()
                else:
                    running_val_tensor = torch.zeros(1, device=device)
                    for i in range(0, N_val, batch_size):
                        images = X_val_gpu[i : i + batch_size]
                        loss = criterion(model(images), images)
                        running_val_tensor += loss.detach() * len(images)
                    epoch_val_loss = (running_val_tensor / N_val).item()
            else:
                running_val_loss = 0.0
                for images, _ in val_loader:
                    images = images.to(device)
                    reconstructed = model(images)
                    loss = criterion(reconstructed, images)
                    running_val_loss += loss.item() * images.size(0)
                epoch_val_loss = running_val_loss / N_val
        history['val_loss'].append(epoch_val_loss)

        if verbose:
            print(f"\rEpoch [{epoch:02d}/{epochs:02d}] | Train MSE: {epoch_train_loss:.5f} | Val MSE: {epoch_val_loss:.5f}", end = "")

    total_time = time.time() - start_time
    if verbose:
        print(f"\n✅ Training completed in {total_time:.2f}s | Final Val MSE: {history['val_loss'][-1]:.5f}")

    return history


# ============================================================================
# 3. Diagnostic & Visualization Suites
# ============================================================================

def plot_sample_canvas(sample_images: torch.Tensor, sample_labels: torch.Tensor) -> None:
    """Renders a 2x5 grid of sample input images for reconstruction auditing."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Sample Input MNIST Images (Targets for Self-Supervised Reconstruction)', fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        img = sample_images[i].squeeze().cpu().numpy()
        im = ax.imshow(img, cmap='magma', vmin=0.0, vmax=1.0)
        ax.set_title(f"Digit: {sample_labels[i].item()}", fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"🔍 Tensor Shape per Batch: {sample_images.shape} -> (Batch_Size, Channels, Height, Width)")
    print(f"🔍 Min Pixel Intensity:   {sample_images.min().item():.3f}")
    print(f"🔍 Max Pixel Intensity:   {sample_images.max().item():.3f}")


def plot_loss_curves(train_history: List[float], val_history: List[float]) -> None:
    """Plots Training and Validation Reconstruction Loss (MSE) trajectories."""
    epochs_range = range(1, len(train_history) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs_range, train_history, 'o-', color='#0284c7', linewidth=2.5, label='Training Reconstruction Loss (MSE)')
    plt.plot(epochs_range, val_history, 's--', color='#10b981', linewidth=2.5, label='Validation Reconstruction Loss (MSE)')

    num_epochs = len(train_history)
    if num_epochs <= 10:
        step = 1
    elif num_epochs <= 25:
        step = 2
    elif num_epochs <= 60:
        step = 5
    else:
        step = 10
    ticks = [1] + [e for e in range(step, num_epochs + 1, step) if e != 1]
    if num_epochs not in ticks:
        ticks.append(num_epochs)

    plt.title('Autoencoder Convergence Dynamics on MNIST (32-Dim Bottleneck)', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Reconstruction Loss (MSE)', fontsize=11)
    plt.yscale('log')
    plt.xticks(ticks)
    plt.xlim(0.5, num_epochs + 0.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    plt.show()

    print(f"📊 Final Training Loss:   {train_history[-1]:.5f}")
    print(f"📊 Final Validation Loss: {val_history[-1]:.5f}")
    print(f"📊 Generalization Gap:    {val_history[-1] - train_history[-1]:.5f} (Minimal overfitting!)")


def plot_reconstruction_audit(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_samples: int = 10
) -> None:
    """
    Renders side-by-side comparison across 3 rows:
    Row 1: Original image (x)
    Row 2: Reconstructed image (x̂)
    Row 3: Absolute pixel error heatmap (|x - x̂|)
    """
    model.eval()
    X_test_gpu, _ = get_gpu_tensor_data(test_loader, device)
    if X_test_gpu is not None:
        test_images = X_test_gpu[:num_samples]
    else:
        test_images, _ = next(iter(test_loader))
        test_images = test_images[:num_samples].to(device)

    with torch.no_grad():
        reconstructions = model(test_images)

    orig = test_images.cpu().numpy()
    recon = reconstructions.cpu().numpy()
    diff = np.abs(orig - recon)

    fig, axes = plt.subplots(3, num_samples, figsize=(18, 6))
    row_labels = ['Real (x)', 'Reconstruction (x̂)', 'Error (|x - x̂|)']

    for i in range(num_samples):
        # Row 1: Original / Real
        axes[0, i].imshow(orig[i, 0], cmap='gray', vmin=0.0, vmax=1.0)
        # Row 2: Reconstructed
        axes[1, i].imshow(recon[i, 0], cmap='gray', vmin=0.0, vmax=1.0)
        # Row 3: Absolute Error Heatmap
        axes[2, i].imshow(diff[i, 0], cmap='inferno', vmin=0.0, vmax=1.0)

        for row in range(3):
            if i == 0:
                axes[row, 0].set_ylabel(row_labels[row], fontsize=13, fontweight='bold', labelpad=8)
                axes[row, 0].set_xticks([])
                axes[row, 0].set_yticks([])
                for spine in axes[row, 0].spines.values():
                    spine.set_visible(False)
            else:
                axes[row, i].axis('off')

    plt.suptitle('Side-by-Side Visual Audit: Originals, Reconstructions & Error Heatmaps (32-Dim Bottleneck)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()


def benchmark_bottlenecks(
    model_cls: Type[nn.Module],
    train_loader: DataLoader,
    test_loader: DataLoader,
    bottleneck_dims: List[int] = [2, 8, 32, 128],
    epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None
) -> Tuple[Dict[int, nn.Module], Dict[int, float]]:
    """
    Trains separate Autoencoder architectures across different latent dimensions
    in GPU VRAM and computes test reconstruction MSE.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models_by_dim: Dict[int, nn.Module] = {}
    test_mse_by_dim: Dict[int, float] = {}

    X_train_gpu, _ = get_gpu_tensor_data(train_loader, device)
    X_test_gpu, _  = get_gpu_tensor_data(test_loader, device)
    use_vram = (X_train_gpu is not None and X_test_gpu is not None)

    N_train = len(train_loader.dataset)
    N_test  = len(test_loader.dataset)
    batch_size = 256
    crit = nn.MSELoss()

    print("🧪 Benchmarking Bottleneck Capacities...")
    start_all = time.time()

    for d in bottleneck_dims:
        t0 = time.time()
        m = model_cls(latent_dim=d).to(device)
        opt = optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)

        m.train()
        for ep in range(epochs):
            if use_vram:
                perm = torch.randperm(N_train, device=device)
                for i in range(0, N_train, batch_size):
                    idx = perm[i : i + batch_size]
                    imgs = X_train_gpu[idx]
                    opt.zero_grad(set_to_none=True)
                    loss = crit(m(imgs), imgs)
                    loss.backward()
                    opt.step()
            else:
                for imgs, _ in train_loader:
                    imgs = imgs.to(device)
                    opt.zero_grad(set_to_none=True)
                    loss = crit(m(imgs), imgs)
                    loss.backward()
                    opt.step()

        # Evaluate Test MSE
        m.eval()
        with torch.no_grad():
            if use_vram:
                test_mse = crit(m(X_test_gpu), X_test_gpu).item()
            else:
                tot_loss = 0.0
                for imgs, _ in test_loader:
                    imgs = imgs.to(device)
                    tot_loss += crit(m(imgs), imgs).item() * imgs.size(0)
                test_mse = tot_loss / N_test

        models_by_dim[d] = m
        test_mse_by_dim[d] = test_mse
        elapsed = time.time() - t0
        print(f"  • Latent Dim d={d:3d} (Compression {784/d:5.1f}x) -> Test MSE: {test_mse:.5f} ({elapsed:.2f}s)")

    total_time = time.time() - start_all
    print(f"✅ All {len(bottleneck_dims)} models benchmarked in {total_time:.2f}s!")
    return models_by_dim, test_mse_by_dim


def plot_bottleneck_comparison(
    models_by_dim: Dict[int, nn.Module],
    test_mse_by_dim: Dict[int, float],
    fixed_test_img: torch.Tensor,
    bottleneck_dims: List[int] = [2, 8, 32, 128],
    device: Optional[torch.device] = None
) -> None:
    """Plots visual reconstruction quality across latent dimensions and the Pareto Frontier curve."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fixed_test_tensor = fixed_test_img.unsqueeze(0).to(device)

    fig, axes = plt.subplots(1, len(bottleneck_dims) + 1, figsize=(14, 3.5))

    # Original Image
    axes[0].imshow(fixed_test_img.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Original\n(784 Pixels)", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Reconstructions across bottlenecks
    for i, d in enumerate(bottleneck_dims, start=1):
        m = models_by_dim[d]
        m.eval()
        with torch.no_grad():
            rec = m(fixed_test_tensor).cpu().squeeze().numpy()
        axes[i].imshow(rec, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f"d = {d} (Ratio: {784/d:.0f}x)\nMSE: {test_mse_by_dim[d]:.4f}", fontsize=10)
        axes[i].axis('off')

    plt.suptitle("Impact of Bottleneck Capacity on Reconstruction Quality (Fixed Test Digit '7')", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Plot Pareto Frontier
    plt.figure(figsize=(8, 4))
    all_mses = [test_mse_by_dim[d] for d in bottleneck_dims]
    plt.plot(bottleneck_dims, all_mses, 'o-', color='#8b5cf6', linewidth=2.5, markersize=8)
    for d in bottleneck_dims:
        plt.annotate(
            f"d={d}\nMSE={test_mse_by_dim[d]:.4f}",
            (d, test_mse_by_dim[d]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9
        )
    max_mse = max(all_mses)
    plt.ylim(0.0, max_mse * 1.25)
    plt.xlim(-5, max(bottleneck_dims) * 1.08)
    plt.title('Bottleneck Pareto Frontier: Dimension vs. Reconstruction Error', fontsize=12, fontweight='bold', pad=14)
    plt.xlabel('Bottleneck Latent Dimension (d)', fontsize=11)
    plt.ylabel('Test Reconstruction Loss (MSE)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_latent_activations(
    model: nn.Module,
    test_dataset: Dataset,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """Extracts and plots the latent code activation heatmap across digits 0 through 9."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    # Collect one sample image per digit class (0 through 9)
    digit_samples = []
    for target_digit in range(10):
        for img, lbl in test_dataset:
            if lbl == target_digit:
                digit_samples.append(img)
                break

    digit_batch = torch.stack(digit_samples).to(device)

    with torch.no_grad():
        x_flat = digit_batch.view(digit_batch.size(0), -1)
        z_codes = model.encoder(x_flat).cpu().numpy()

    plt.figure(figsize=(14, 6))
    plt.imshow(z_codes, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Activation Intensity')
    plt.title('Latent Feature Activations across Digits 0–9 (32 Bottleneck Channels)', fontsize=13, fontweight='bold')
    plt.xlabel('Latent Channel Index (0 to 31)', fontsize=11)
    plt.ylabel('Ground Truth Digit Class (0 to 9)', fontsize=11)
    plt.yticks(range(10), labels=[f"Digit {d}" for d in range(10)])
    plt.tight_layout()
    plt.show()

    print(f"🔍 Extracted Latent Matrix Shape: {z_codes.shape} -> (10 digits, 32 features)")
    print(f"🔍 Mean Latent Activation:        {z_codes.mean():.4f}")
    print(f"🔍 Sparsity (Fraction of Zeros):  {(z_codes == 0).mean()*100:.1f}% (Induced by ReLU!)")
    return z_codes


def run_anomaly_detection_experiment(
    model_cls: Type[nn.Module],
    train_dataset: Dataset,
    val_dataset: Dataset,
    latent_dim: int = 16,
    epochs: int = 5,
    batch_size: int = 64,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, float, float, float]:
    """
    Trains a nominal Autoencoder exclusively on healthy samples (Digit 0),
    and derives the calibrated 3-sigma anomaly threshold: tau = mu_val + 3*sigma_val.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("🏥 Setting Up In-VRAM GPU Anomaly Detection Experiment...")
    start_time = time.time()

    # Pre-filter Digit 0 directly
    # Check if cached in GPU
    from torch.utils.data import Subset
    train_0_indices = [i for i, (_, lbl) in enumerate(train_dataset) if lbl == 0]
    val_0_indices   = [i for i, (_, lbl) in enumerate(val_dataset) if lbl == 0]

    train_0_loader = DataLoader(Subset(train_dataset, train_0_indices), batch_size=batch_size, shuffle=True)
    val_0_loader   = DataLoader(Subset(val_dataset, val_0_indices), batch_size=batch_size, shuffle=False)

    X_train_0, _ = get_gpu_tensor_data(train_0_loader, device)
    X_val_0, _   = get_gpu_tensor_data(val_0_loader, device)
    use_vram = (X_train_0 is not None and X_val_0 is not None)

    anomaly_model = model_cls(latent_dim=latent_dim).to(device)
    anomaly_opt = optim.AdamW(anomaly_model.parameters(), lr=1e-3, weight_decay=1e-4)
    anomaly_crit = nn.MSELoss()

    N_train_0 = len(train_0_indices)
    anomaly_model.train()
    for ep in range(epochs):
        if use_vram:
            perm = torch.randperm(N_train_0, device=device)
            for i in range(0, N_train_0, batch_size):
                idx = perm[i : i + batch_size]
                imgs = X_train_0[idx]
                anomaly_opt.zero_grad(set_to_none=True)
                loss = anomaly_crit(anomaly_model(imgs), imgs)
                loss.backward()
                anomaly_opt.step()
        else:
            for imgs, _ in train_0_loader:
                imgs = imgs.to(device)
                anomaly_opt.zero_grad(set_to_none=True)
                loss = anomaly_crit(anomaly_model(imgs), imgs)
                loss.backward()
                anomaly_opt.step()

    print(f"✅ Nominal Autoencoder trained exclusively on healthy (Digit 0) samples in {time.time()-start_time:.2f}s!")

    # Calculate validation baseline statistics: mu_val and sigma_val
    anomaly_model.eval()
    with torch.no_grad():
        if use_vram:
            recon = anomaly_model(X_val_0)
            errs_tensor = ((X_val_0 - recon)**2).view(X_val_0.size(0), -1).mean(dim=1)
            mu_val = errs_tensor.mean().item()
            sigma_val = errs_tensor.std().item()
        else:
            val_errors = []
            for imgs, _ in val_0_loader:
                imgs = imgs.to(device)
                recon = anomaly_model(imgs)
                errs = ((imgs - recon)**2).view(imgs.size(0), -1).mean(dim=1).cpu().numpy()
                val_errors.extend(errs)
            val_errors = np.array(val_errors)
            mu_val = float(np.mean(val_errors))
            sigma_val = float(np.std(val_errors))

    tau = mu_val + 3.0 * sigma_val

    print(f"\n📊 Baseline Mean Error (μ_val):     {mu_val:.5f}")
    print(f"📊 Baseline Spread (σ_val):         {sigma_val:.5f}")
    print(f"🎯 Calibrated 3σ Threshold (τ):     {tau:.5f}")

    return anomaly_model, mu_val, sigma_val, tau


def plot_anomaly_detection_histogram(
    anomaly_model: nn.Module,
    test_dataset: Dataset,
    tau: float,
    device: Optional[torch.device] = None,
    num_samples_each: int = 200
) -> Tuple[float, float]:
    """Evaluates nominal and anomalous test samples, plots error histograms, and calculates sensitivity/specificity."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    anomaly_model.eval()

    test_normal_imgs = []
    test_anomaly_imgs = []

    for img, lbl in test_dataset:
        if lbl == 0 and len(test_normal_imgs) < num_samples_each:
            test_normal_imgs.append(img)
        elif lbl in [1, 7, 9] and len(test_anomaly_imgs) < num_samples_each:
            test_anomaly_imgs.append(img)
        if len(test_normal_imgs) >= num_samples_each and len(test_anomaly_imgs) >= num_samples_each:
            break

    t_norm = torch.stack(test_normal_imgs).to(device)
    t_anom = torch.stack(test_anomaly_imgs).to(device)

    with torch.no_grad():
        norm_rec = anomaly_model(t_norm)
        anom_rec = anomaly_model(t_anom)

        errs_norm_tensor = ((t_norm - norm_rec)**2).view(t_norm.size(0), -1).mean(dim=1)
        errs_anom_tensor = ((t_anom - anom_rec)**2).view(t_anom.size(0), -1).mean(dim=1)

        errs_norm = errs_norm_tensor.cpu().numpy()
        errs_anom = errs_anom_tensor.cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.hist(errs_norm, bins=30, alpha=0.7, color='#10b981', label='Healthy / Nominal (Digit 0)')
    plt.hist(errs_anom, bins=30, alpha=0.7, color='#ef4444', label='Anomalies / Defects (Digits 1, 7, 9)')
    plt.axvline(tau, color='#f59e0b', linestyle='--', linewidth=2.5, label=f'Threshold τ = μ + 3σ ({tau:.4f})')

    plt.title('Unsupervised Anomaly Detection: Reconstruction Error Distribution', fontsize=13, fontweight='bold')
    plt.xlabel('Reconstruction Error (MSE)', fontsize=11)
    plt.ylabel('Sample Count', fontsize=11)
    plt.legend(frameon=True, fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Calculate detection accuracy
    true_normal_detected = (errs_norm <= tau).mean() * 100
    anomalies_flagged = (errs_anom > tau).mean() * 100

    print(f"✅ Nominal Specificity (Normal classified as Normal): {true_normal_detected:.1f}%")
    print(f"🚨 Anomaly Sensitivity (Defects correctly flagged):   {anomalies_flagged:.1f}%")

    return float(true_normal_detected), float(anomalies_flagged)


# ============================================================================
# 5. Session 9 Latent Space Exploration & Visualization Utilities
# ============================================================================

@torch.no_grad()
def extract_latents(
    model: nn.Module, 
    loader: DataLoader, 
    dev: torch.device, 
    x_gpu: Optional[torch.Tensor] = None, 
    y_gpu: Optional[torch.Tensor] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts high-dimensional bottleneck representations (z) and class labels (y)
    for an entire dataset partition. Uses 1-shot in-VRAM tensor extraction when available.
    """
    model.eval()
    print("🔍 Extracting 32D representations for the dataset...")
    if x_gpu is not None and y_gpu is not None:
        x_flat = x_gpu.view(x_gpu.size(0), -1)
        z = model.encoder(x_flat).cpu().numpy()
        y = y_gpu.cpu().numpy()
        print(f"✅ Extraction complete. Shape: {z.shape}")
        return z, y
    zs, ys = [], []
    for x, y in loader:
        x = x.view(x.size(0), -1).to(dev)
        zs.append(model.encoder(x).cpu())
        ys.append(y)
    z_cat = torch.cat(zs).numpy()
    y_cat = torch.cat(ys).numpy()
    print(f"✅ Extraction complete. Shape: {z_cat.shape}")
    return z_cat, y_cat


def plot_umap_projection(z: np.ndarray, y: np.ndarray, seed: int = 42) -> None:
    """
    Projects high-dimensional latent vectors down to 2D using UMAP
    and renders a scatter plot colored by digit class.
    """
    import umap
    print("🌐 Running UMAP projection (this may take a few seconds)...")
    z_umap = umap.UMAP(n_components=2, random_state=seed).fit_transform(z)
    print("✅ UMAP projection complete.")

    plt.figure(figsize=(10, 7))
    sc = plt.scatter(
        z_umap[:, 0], z_umap[:, 1], 
        c=y, cmap='tab10', 
        alpha=0.6, s=5
    )
    plt.colorbar(sc, ticks=range(10), label='Digit Class')
    plt.title("UMAP Projection (d=32 → 2)", fontsize=14, fontweight='bold')
    plt.xlabel("UMAP 1", fontsize=11)
    plt.ylabel("UMAP 2", fontsize=11)
    plt.tight_layout()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()


@torch.no_grad()
def interpolate_slerp(
    model: nn.Module, 
    img_a: torch.Tensor, 
    img_b: torch.Tensor, 
    steps: int = 8, 
    dev: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Performs Spherical Linear Interpolation (SLERP) along the constant-norm
    great-circle geodesic between the latent representations of two images.
    """
    model.eval()
    za = model.encoder(img_a.view(1, -1).to(dev))
    zb = model.encoder(img_b.view(1, -1).to(dev))

    cos_theta = torch.clamp((za * zb).sum() / (torch.norm(za) * torch.norm(zb)), -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)

    a = torch.linspace(0, 1, steps, device=dev).unsqueeze(1)
    z_path = (torch.sin((1 - a) * theta) * za + torch.sin(a * theta) * zb) / torch.sin(theta)

    return model.decoder(z_path).view(-1, 28, 28).cpu()


def plot_slerp_metamorphosis(
    model: nn.Module, 
    img_a: torch.Tensor, 
    img_b: torch.Tensor, 
    steps: int = 8, 
    dev: torch.device = torch.device('cpu')
) -> None:
    """
    Executes and visualizes an 8-step SLERP geodesic path between two reference digits.
    """
    morph = interpolate_slerp(model, img_a, img_b, steps=steps, dev=dev)

    fig, axes = plt.subplots(1, steps, figsize=(14, 2.5))
    fig.suptitle("32D Vector Metamorphosis using SLERP: '1' -> '7'", fontsize=14, fontweight='bold')
    for i, ax in enumerate(axes):
        ax.imshow(morph[i].numpy(), cmap='magma')
        ax.axis('off')
        ax.set_title(rf"$\alpha={i / (steps - 1):.2f}$", fontsize=10)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_slant_attribute_manipulation(
    model: nn.Module, 
    upright_1_img: torch.Tensor, 
    slanted_1_img: torch.Tensor, 
    upright_7_img: torch.Tensor, 
    dev: torch.device = torch.device('cpu'), 
    betas: List[float] = [0.0, 0.5, 0.7, 1.0]
) -> None:
    """
    Isolates the latent slant direction vector d_slant from reference '1's
    and transfers it across varying injection intensities beta onto an upright host '7'.
    """
    model.eval()

    # 1. Encode reference images of '1' to isolate the slant attribute vector d_slant
    z_upright = model.encoder(upright_1_img.view(1, -1).to(dev))
    z_slanted = model.encoder(slanted_1_img.view(1, -1).to(dev))
    d_slant = z_slanted - z_upright

    # Pixel-space difference between slanted and upright reference
    pixel_diff = (slanted_1_img.squeeze() - upright_1_img.squeeze()).cpu()

    # 2. Encode host image (Upright '7')
    z_7 = model.encoder(upright_7_img.view(1, -1).to(dev))

    # 3. Inject slant attribute across varying strengths beta
    injected_images = []
    for beta in betas:
        z_mod = z_7 + beta * d_slant
        decoded_mod = model.decoder(z_mod).view(28, 28).cpu()
        injected_images.append(decoded_mod)

    # Plotting: Two-row diagnostic panel (4 columns)
    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.25)

    # Row 1: Isolating d_slant
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(upright_1_img.squeeze(), cmap='gray')
    ax0.set_title("1. Base: Upright '1'\n($z_{upright}$)", fontsize=11)
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(slanted_1_img.squeeze(), cmap='gray')
    ax1.set_title("2. Target: Slanted '1'\n($z_{slanted}$)", fontsize=11)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 2])
    im_diff = ax2.imshow(pixel_diff, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_title("Pixel Difference\n($x_{slanted} - x_{upright}$)", fontsize=11)
    ax2.axis('off')
    plt.colorbar(im_diff, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[0, 3])
    norm_val = d_slant.norm().item()
    ax3.text(0.5, 0.5, 
             r"$\vec{d}_{slant} = z_{slanted} - z_{upright}$" + "\n\n" +
             f"Latent Dim: d = 32\n" +
             f"Vector Norm: ||d_slant|| = {norm_val:.2f}\n\n" +
             "Acts as generator of\nstroke rotation / shear", 
             ha='center', va='center', fontsize=10.5, 
             bbox=dict(boxstyle='round,pad=0.7', facecolor='#f0f4f8', edgecolor='#1e3a8a', linewidth=1.5))
    ax3.axis('off')

    # Row 2: Applying d_slant to Upright '7' across beta intensities
    titles_map = {
        0.0: r"Original Host '7'" + "\n" + r"($\beta=0.0$)",
        0.5: r"$z_7 + 0.5\,\vec{d}_{\mathrm{slant}}$" + "\n" + r"($\beta=0.5$)",
        0.7: r"$z_7 + 0.7\,\vec{d}_{\mathrm{slant}}$" + "\n" + r"($\beta=0.7$)",
        1.0: r"$z_7 + 1.0\,\vec{d}_{\mathrm{slant}}$" + "\n" + r"($\beta=1.0$)"
    }

    for i, beta in enumerate(betas):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(injected_images[i].numpy(), cmap='magma')
        title = titles_map.get(beta, f"$z_7 + {beta:.1f}\\,\\vec{{d}}_{{\\mathrm{{slant}}}}$\n($\\beta={beta:.1f}$)")
        ax.set_title(title, fontsize=11, fontweight='bold' if beta in [0.0, 1.0] else 'normal')
        ax.axis('off')

    fig.suptitle(r"Theory Case 1: Attribute Manipulation on MNIST (Extracting $\vec{d}_{slant}$ from '1' $\to$ Applying to '7')", fontsize=13, fontweight='bold', y=0.98)
    plt.show()


@torch.no_grad()
def demonstrate_zero_to_eight_morph(
    model: nn.Module, 
    test_dataset: Dataset, 
    sample_indices: Optional[List[int]] = None, 
    style_labels: Optional[List[str]] = None, 
    alphas: List[float] = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25], 
    dev: torch.device = torch.device('cpu')
) -> None:
    """
    Isolates the topological 'central bridge' vector via cluster centroids
    and continuously injects it into distinct test zeros across varying alpha intensities.
    """
    model.eval()

    targets = test_dataset.targets
    zeros_idx = (targets == 0).nonzero(as_tuple=True)[0]
    eights_idx = (targets == 8).nonzero(as_tuple=True)[0]

    # 1. Compute Centroid Latent Vectors (N=100)
    x_zeros = torch.stack([test_dataset[i][0] for i in zeros_idx[:100]]).to(dev)
    x_eights = torch.stack([test_dataset[i][0] for i in eights_idx[:100]]).to(dev)

    z_zeros_mean = model.encoder(x_zeros.view(100, -1)).mean(dim=0)
    z_eights_mean = model.encoder(x_eights.view(100, -1)).mean(dim=0)

    # Central bridge direction vector
    d_bridge = z_eights_mean - z_zeros_mean

    # Default sample indices if not provided
    if sample_indices is None:
        sample_indices = [zeros_idx[429], zeros_idx[10], zeros_idx[12]]
    if style_labels is None:
        style_labels = ["Round '0'", "Oval '0'", "Thin '0'"]

    fig, axes = plt.subplots(len(sample_indices), len(alphas), figsize=(14, 6.5))

    for row_idx, s_idx in enumerate(sample_indices):
        test_img = test_dataset[s_idx][0].to(dev)
        z_orig = model.encoder(test_img.view(1, -1))

        for col_idx, a in enumerate(alphas):
            z_mod = z_orig + a * d_bridge
            reconstructed = model.decoder(z_mod).view(28, 28).cpu().numpy()

            ax = axes[row_idx, col_idx]
            cmap = 'gray' if a == 0.0 else 'magma'
            ax.imshow(reconstructed, cmap=cmap)
            ax.axis('off')

            if row_idx == 0:
                header = f"$\\alpha = {a:.2f}$"
                if a == 0.0:
                    header += "\n(Original 0)"
                elif a == 1.0:
                    header += "\n(Morphed 8)"
                elif a > 1.0:
                    header += "\n(Hyper-8)"
                ax.set_title(header, fontsize=11, fontweight='bold')

        axes[row_idx, 0].text(-0.25, 0.5, style_labels[row_idx], transform=axes[row_idx, 0].transAxes,
                              fontsize=11, fontweight='bold', va='center', ha='right', rotation=90)

    plt.suptitle(
        r"Topological Latent Arithmetic: Morphing $0 \to 8$ via Central Waist Injection ($\vec{d}_{\mathrm{bridge}} = \bar{z}_8 - \bar{z}_0$)",
        fontsize=13, fontweight='bold', y=0.99
    )
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def scan_pca_slice(
    model: nn.Module, 
    latents: np.ndarray, 
    grid_n: int = 20, 
    bound: float = 2.5, 
    dev: torch.device = torch.device('cpu')
) -> Tuple[np.ndarray, float, float]:
    """
    Calculates the 2D principal plane embedded in 32D and decodes a grid_n x grid_n
    matrix of synthetic digits across variance-scaled axes (sigma1, sigma2).
    """
    model.eval()

    # 1. Compute PCA on the 32D latents to find top 2 principal axes (v1, v2)
    latents_tensor = torch.as_tensor(latents, dtype=torch.float32)
    z_mean = latents_tensor.mean(dim=0)
    U, S, V = torch.pca_lowrank(latents_tensor - z_mean, q=2)
    v1, v2 = V[:, 0], V[:, 1]

    sigma1 = ((latents_tensor - z_mean) @ v1).std()
    sigma2 = ((latents_tensor - z_mean) @ v2).std()

    # 2. Create 2D grid scaled by empirical standard deviations
    u = torch.linspace(-bound, bound, grid_n, device=latents_tensor.device)
    v = torch.linspace(bound, -bound, grid_n, device=latents_tensor.device)
    gv, gu = torch.meshgrid(v, u, indexing='ij')

    # 3. Embed 2D grid back into 32D space
    z_grid = z_mean + \
             (gu.flatten().unsqueeze(1) * sigma1) * v1 + \
             (gv.flatten().unsqueeze(1) * sigma2) * v2

    # 4. Decode the coordinate vectors
    x_hat = model.decoder(z_grid.to(dev)).view(-1, 28, 28).cpu().numpy()

    # 5. Stitch into a master canvas montage
    canvas = np.zeros((grid_n * 28, grid_n * 28))
    for idx, img in enumerate(x_hat):
        r, c = idx // grid_n, idx % grid_n
        canvas[r * 28 : (r + 1) * 28, c * 28 : (c + 1) * 28] = img

    return canvas, sigma1.item(), sigma2.item()


def plot_pca_slice(
    model: nn.Module, 
    latents: np.ndarray, 
    grid_n: int = 20, 
    bound: float = 2.5, 
    dev: torch.device = torch.device('cpu')
) -> None:
    """
    Generates and renders the complete 2D PCA Slice of the 32D Manifold Landscape.
    """
    print(f"🗺️ Generating the {grid_n}x{grid_n} Principal Slice Atlas...")
    canvas, s1, s2 = scan_pca_slice(model, latents, grid_n=grid_n, bound=bound, dev=dev)
    print(f"📊 Principal Scales: σ₁ = {s1:.3f} (Horizontal PC1) | σ₂ = {s2:.3f} (Vertical PC2)")
    print(f"✅ Generated {grid_n * grid_n} synthetic digits across the 2D affine manifold cut.")

    plt.figure(figsize=(9.5, 9.5))
    plt.imshow(canvas, cmap='magma')
    plt.title("2D PCA Slice of 32D Manifold Landscape", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel(rf"Principal Component 1 Axis ($\pm {bound}\sigma_1$)", fontsize=11, labelpad=8)
    plt.ylabel(rf"Principal Component 2 Axis ($\pm {bound}\sigma_2$)", fontsize=11, labelpad=8)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()
