"""
Session 11 Lab Utilities: Training the Denoising Autoencoder (DAE) & Multi-Seed Calibration Lab
Course: Deep Learning: Foundations, Systems & Scientific AI Auditing
Authors: MSc. Antonio Aguilar & Dr. Luis Aguilar Ibáñez
National University of Piura (UNP) & Pontifical Catholic University of Chile (UC)

This module encapsulates boilerplate visualization suites, modular DAE architecture
components (Encoder, Decoder, BaselineDAE), high-throughput in-VRAM GPU training routines,
PSNR image quality metrics, multi-seed bottleneck calibration engines, Pareto knee detection
rules, and out-of-distribution (OOD) cross-noise robustness audit suites.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable, Union
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
# 2. Universal Noise Injection Suite (Functional & Vectorized)
# ============================================================================

def add_gaussian_noise(x: torch.Tensor, sigma: float = 0.40) -> torch.Tensor:
    """Zero-mean Additive White Gaussian Noise (AWGN) with boundary clamping to [0, 1]."""
    safe_sigma = max(0.0, float(sigma))
    noise = torch.randn_like(x) * safe_sigma
    return torch.clamp(x + noise, 0.0, 1.0)


def add_salt_and_pepper_noise(x: torch.Tensor, prob: float = 0.25) -> torch.Tensor:
    """Discrete Salt-and-Pepper impulse noise using uniform random partitioning."""
    prob = max(0.0, min(1.0, float(prob)))
    noisy = x.clone()
    rand = torch.rand_like(x)
    pepper_mask = rand < (prob / 2.0)
    salt_mask = (rand >= (prob / 2.0)) & (rand < prob)
    noisy[pepper_mask] = 0.0
    noisy[salt_mask] = 1.0
    return noisy


def add_masking_noise(x: torch.Tensor, drop_prob: float = 0.35, **kwargs) -> torch.Tensor:
    """Vectorized masking corruption using Bernoulli dropout."""
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
    """
    @staticmethod
    def gaussian(x: torch.Tensor, sigma: float = 0.40) -> torch.Tensor:
        return add_gaussian_noise(x, sigma=sigma)

    @staticmethod
    def salt_and_pepper(x: torch.Tensor, prob: float = 0.25) -> torch.Tensor:
        return add_salt_and_pepper_noise(x, prob=prob)

    @staticmethod
    def masking(x: torch.Tensor, drop_prob: float = 0.35, **kwargs) -> torch.Tensor:
        return add_masking_noise(x, drop_prob=drop_prob, **kwargs)

    @staticmethod
    def apply(x: torch.Tensor, noise_type: str = 'gaussian', noise_param: float = 0.40) -> torch.Tensor:
        """Applies the selected noise distribution directly to tensor x."""
        nt = noise_type.lower()
        if nt == 'gaussian':
            return NoiseInjector.gaussian(x, sigma=noise_param)
        elif nt in ['salt_and_pepper', 'sp', 'impulse']:
            return NoiseInjector.salt_and_pepper(x, prob=noise_param)
        elif nt in ['masking', 'dropout']:
            return NoiseInjector.masking(x, drop_prob=noise_param)
        else:
            raise ValueError(f"Unknown noise type: '{noise_type}'. Expected 'gaussian', 'salt_and_pepper', or 'masking'.")


def apply_noise(x: torch.Tensor, noise_type: str = 'gaussian', noise_param: float = 0.40) -> torch.Tensor:
    """Convenience wrapper for NoiseInjector.apply."""
    return NoiseInjector.apply(x, noise_type=noise_type, noise_param=noise_param)


# ============================================================================
# 3. DAE Architecture Components (Modular Encoder, Decoder, BaselineDAE)
# ============================================================================

class Encoder(nn.Module):
    """Compresses 784-dimensional or (B, 1, 28, 28) image tensors into a latent bottleneck code."""
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        h = self.relu1(self.fc1(x))
        z = self.relu2(self.fc2(h))
        return z


class Decoder(nn.Module):
    """Decompresses latent bottleneck codes back into the original 784-dimensional image canvas."""
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 784)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.relu1(self.fc1(z))
        x_recon = self.sigmoid(self.fc2(h))
        return x_recon


class BaselineDAE(nn.Module):
    """
    Symmetric Baseline Denoising Autoencoder (DAE) Architecture.
    Maps: 784 -> hidden_dim (128) -> latent_dim (d) -> hidden_dim (128) -> 784.
    Preserves input tensor dimensionality: (B, 1, 28, 28) -> (B, 1, 28, 28) and (B, 784) -> (B, 784).
    """
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_recon_flat = self.decoder(z)
        if x.dim() == 4:
            return x_recon_flat.view(x.size(0), 1, 28, 28)
        return x_recon_flat

    def count_parameters(self) -> Dict[str, int]:
        """Returns parameter counts for encoder, decoder, and total."""
        enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        dec = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        return {"encoder": enc, "decoder": dec, "total": enc + dec}


def build_dae_model(latent_dim: int = 32, hidden_dim: int = 128, device: Optional[torch.device] = None) -> nn.Module:
    """Constructs modular BaselineDAE with configurable latent dimension d on target device."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return BaselineDAE(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)


# ============================================================================
# 4. Quantitative Image Quality Metrics (MSE & PSNR)
# ============================================================================

def compute_mse(target: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Computes Mean Squared Error (MSE) between target and reconstructed tensors."""
    return torch.mean((target - reconstructed) ** 2).item()


def compute_psnr(target: torch.Tensor, reconstructed: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    PSNR = 10 * log10(max_val^2 / MSE).
    """
    mse = torch.mean((target - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10.0 * torch.log10((max_val ** 2) / mse)
    return psnr.item()


def compute_batch_psnr(target: torch.Tensor, reconstructed: torch.Tensor, max_val: float = 1.0) -> float:
    """Alias for compute_psnr across a batch."""
    return compute_psnr(target, reconstructed, max_val=max_val)


# ============================================================================
# 5. In-VRAM GPU Training Loops & Evaluation Engines
# ============================================================================

def train_epoch(
    model: nn.Module,
    data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    noise_type: str = 'gaussian',
    noise_param: float = 0.40,
    batch_size: int = 512,
    device: Optional[torch.device] = None
) -> float:
    """
    Executes one high-throughput training epoch directly in GPU VRAM.
    Fundamental DAE Asymmetry: Noisy inputs (x̃) are mapped to reconstructions (x̂),
    optimized strictly against clean ground-truth targets (x).
    """
    if device is None:
        device = next(model.parameters()).device
    model.train()
    total_loss = 0.0
    num_samples = data.size(0)
    perm = torch.randperm(num_samples, device=device)

    for i in range(0, num_samples, batch_size):
        idx = perm[i : i + batch_size]
        clean_batch = data[idx]
        noisy_batch = apply_noise(clean_batch, noise_type=noise_type, noise_param=noise_param)

        optimizer.zero_grad(set_to_none=True)
        recon_batch = model(noisy_batch)
        loss = criterion(recon_batch, clean_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * clean_batch.size(0)

    return total_loss / num_samples


def evaluate(
    model: nn.Module,
    data: torch.Tensor,
    criterion: nn.Module,
    noise_type: str = 'gaussian',
    noise_param: float = 0.40,
    batch_size: int = 512,
    device: Optional[torch.device] = None
) -> Tuple[float, float]:
    """Evaluates validation loss (MSE) and image recovery quality (PSNR) across fixed test data."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    total_mse = 0.0
    num_samples = data.size(0)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            clean_batch = data[i : i + batch_size]
            noisy_batch = apply_noise(clean_batch, noise_type=noise_type, noise_param=noise_param)
            recon_batch = model(noisy_batch)
            loss = criterion(recon_batch, clean_batch)
            total_mse += loss.item() * clean_batch.size(0)

    val_mse = total_mse / num_samples
    val_psnr = 10.0 * np.log10(1.0 / val_mse) if val_mse > 0 else 100.0
    return val_mse, val_psnr


def train_dae_model(
    model: nn.Module,
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    noise_type: str = 'gaussian',
    noise_param: float = 0.40,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 42,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Trains a BaselineDAE model for a specified number of epochs in GPU VRAM
    and records training MSE, validation MSE, and validation PSNR per epoch.
    """
    if device is None:
        device = next(model.parameters()).device

    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'val_psnr': []}

    print(f"🚀 Training Baseline DAE (d={model.latent_dim if hasattr(model, 'latent_dim') else 32}, Noise: {noise_type.upper()}={noise_param}, {epochs} Epochs)...\n")
    print("-" * 80)
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, X_train, optimizer, criterion,
                                 noise_type=noise_type, noise_param=noise_param,
                                 batch_size=batch_size, device=device)
        val_loss, val_psnr = evaluate(model, X_test, criterion,
                                      noise_type=noise_type, noise_param=noise_param,
                                      batch_size=batch_size, device=device)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_psnr'].append(val_psnr)

        epoch_time = time.time() - epoch_start
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"\rEpoch [{epoch:02d}/{epochs:02d}] | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | Val PSNR: {val_psnr:.2f} dB | Step: {epoch_time:.3f}s", end=" ")

    total_elapsed = time.time() - total_start
    print("-" * 80)
    print(f"\n✅ Baseline Model Training Complete in {total_elapsed:.2f}s! Final Val PSNR: {history['val_psnr'][-1]:.2f} dB")
    return model, history


# ============================================================================
# 6. High-Throughput Multi-Seed Bottleneck Calibration Engine
# ============================================================================

def run_multiseed_bottleneck_calibration(
    bottleneck_dims: list,
    seeds: list,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    noise_type: str = "gaussian",
    noise_param: float = 0.40,
    X_train: Optional[torch.Tensor] = None,
    X_test: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Executes a high-throughput multi-seed ablation sweep across candidate latent dimensions in GPU VRAM.
    Computes sample mean and sample standard deviation across seeds for each capacity d.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if X_train is None or X_test is None:
        raise ValueError("X_train and X_test tensors must be supplied to run_multiseed_bottleneck_calibration.")

    N_train = X_train.size(0)
    criterion = nn.MSELoss()

    results = {
        "dims": bottleneck_dims,
        "seeds": seeds,
        "raw_losses": {d: [] for d in bottleneck_dims},
        "raw_psnrs": {d: [] for d in bottleneck_dims},
        "mean_loss": {},
        "std_loss": {},
        "mean_psnr": {},
        "std_psnr": {}
    }

    total_runs = len(bottleneck_dims) * len(seeds)
    print(f"🧪 Starting Multi-Seed Calibration: {len(bottleneck_dims)} Dims x {len(seeds)} Seeds = {total_runs} Runs ({epochs} Epochs each)")
    print(f"📡 Noise Regime: {noise_type.upper()} (Parameter: {noise_param})")
    print("=" * 75)

    start_all = time.time()

    for d in bottleneck_dims:
        t_dim_start = time.time()
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = build_dae_model(latent_dim=d, device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

            model.train()
            for ep in range(epochs):
                perm = torch.randperm(N_train, device=device)
                for i in range(0, N_train, batch_size):
                    idx = perm[i : i + batch_size]
                    clean = X_train[idx]
                    noisy = apply_noise(clean, noise_type=noise_type, noise_param=noise_param)

                    optimizer.zero_grad(set_to_none=True)
                    recon = model(noisy)
                    loss = criterion(recon, clean)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                noisy_test = apply_noise(X_test, noise_type=noise_type, noise_param=noise_param)
                val_recon = model(noisy_test)
                val_mse = criterion(val_recon, X_test).item()
                val_psnr = 10.0 * np.log10(1.0 / val_mse) if val_mse > 0 else 100.0

            results["raw_losses"][d].append(val_mse)
            results["raw_psnrs"][d].append(val_psnr)

        losses_d = np.array(results["raw_losses"][d])
        psnrs_d  = np.array(results["raw_psnrs"][d])

        results["mean_loss"][d] = float(np.mean(losses_d))
        results["std_loss"][d]  = float(np.std(losses_d))
        results["mean_psnr"][d] = float(np.mean(psnrs_d))
        results["std_psnr"][d]  = float(np.std(psnrs_d))

        elapsed_dim = time.time() - t_dim_start
        m_l, s_l = results["mean_loss"][d], results["std_loss"][d]
        m_p, s_p = results["mean_psnr"][d], results["std_psnr"][d]
        print(f"📊 Dim d={d:2d} ({len(seeds)} Seeds) -> Val MSE: {m_l:.5f} ± {s_l:.5f} | PSNR: {m_p:.2f} ± {s_p:.2f} dB ({elapsed_dim:.1f}s)")

    total_time = time.time() - start_all
    print("=" * 75)
    print(f"✅ Completed all {total_runs} calibration runs in {total_time:.2f}s ({total_time/60:.2f} min)!")
    return results


def train_calibrated_model(
    d_star: int,
    X_train: torch.Tensor,
    noise_type: str = "gaussian",
    noise_param: float = 0.40,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 42,
    device: Optional[torch.device] = None
) -> nn.Module:
    """Trains the final calibrated Model m* with optimal bottleneck capacity d* for 50 epochs."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Final Calibrated Model m* (d*={d_star}) for {epochs} Epochs...")
    torch.manual_seed(seed)
    model_m_star = build_dae_model(latent_dim=d_star, device=device)
    opt_star = torch.optim.AdamW(model_m_star.parameters(), lr=lr, weight_decay=1e-4)
    crit_star = nn.MSELoss()

    N_train = X_train.size(0)
    model_m_star.train()
    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(N_train, device=device)
        for i in range(0, N_train, batch_size):
            idx = perm[i : i + batch_size]
            clean = X_train[idx]
            noisy = apply_noise(clean, noise_type=noise_type, noise_param=noise_param)

            opt_star.zero_grad(set_to_none=True)
            recon = model_m_star(noisy)
            loss = crit_star(recon, clean)
            loss.backward()
            opt_star.step()

    print(f"✅ Model m* trained in {time.time()-t0:.2f}s!\n")
    return model_m_star


# ============================================================================
# 7. Pedagogical Visualizations & Publication-Grade Diagnostic Suites
# ============================================================================

def plot_training_convergence(history: Dict[str, List[float]], noise_type: str = 'gaussian') -> None:
    """Plots dual learning dynamics curves: Logarithmic MSE loss & Decibel PSNR progression."""
    epochs = len(history['train_loss'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    # 1. MSE Loss Curve
    ax1.plot(range(1, epochs + 1), history['train_loss'], 'o-', color='#38bdf8', label='Train MSE Loss')
    ax1.plot(range(1, epochs + 1), history['val_loss'], 's--', color='#f43f5e', label='Val MSE Loss')
    ax1.set_title(f"Reconstruction Loss (d=32, {noise_type})", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=10)
    ax1.set_yscale('log')
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.25)

    # 2. PSNR Curve
    ax2.plot(range(1, epochs + 1), history['val_psnr'], '^-', color='#10b981', label='Validation PSNR (dB)')
    ax2.set_title(f"Fidelity Progression: PSNR (d=32, {noise_type})", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Peak Signal-to-Noise Ratio (dB)", fontsize=10)
    #ax2.axhline(20.0, color='#f59e0b', linestyle=':', label='Target Benchmark (20 dB)')
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.25)

    plt.suptitle("Baseline Denoising Autoencoder (DAE) Learning Dynamics", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_reconstruction_showcase(
    model: nn.Module,
    clean_samples: torch.Tensor,
    noise_type: str = 'gaussian',
    noise_param: float = 0.40,
    n_samples: int = 8
) -> None:
    """Generates a 4-row visual restoration showcase: Clean vs. Noisy vs. DAE Reconstructed vs. Absolute Residual."""
    model.eval()
    device = next(model.parameters()).device
    clean_samples = clean_samples.to(device)
    with torch.no_grad():
        samples = clean_samples[:n_samples]
        noisy_samples = apply_noise(samples, noise_type=noise_type, noise_param=noise_param)
        reconstructions = model(noisy_samples)
        residuals = torch.abs(samples - reconstructions)

    fig, axes = plt.subplots(4, n_samples, figsize=(n_samples * 1.8, 7.5))

    for i in range(n_samples):
        raw_psnr = compute_psnr(samples[i], noisy_samples[i])
        recon_psnr = compute_psnr(samples[i], reconstructions[i])

        axes[0, i].imshow(samples[i].view(28, 28).cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"Target #{i+1}", fontsize=9, fontweight='bold')

        axes[1, i].imshow(noisy_samples[i].view(28, 28).cpu(), cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f"{raw_psnr:.1f} dB", fontsize=9, color='#f59e0b')

        axes[2, i].imshow(reconstructions[i].view(28, 28).cpu(), cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f"{recon_psnr:.1f} dB", fontsize=9, color='#10b981', fontweight='bold')

        axes[3, i].imshow(residuals[i].view(28, 28).cpu(), cmap='hot', vmin=0, vmax=0.8)

        for r in range(4):
            axes[r, i].axis('off')

    axes[0, 0].text(-0.35, 0.5, "1. Clean Target (x)", transform=axes[0, 0].transAxes, 
                    fontsize=10, fontweight='bold', color='#10b981', va='center', rotation=90)
    axes[1, 0].text(-0.35, 0.5, f"2. Noisy ({noise_type[:4]})", transform=axes[1, 0].transAxes, 
                    fontsize=10, fontweight='bold', color='#f59e0b', va='center', rotation=90)
    axes[2, 0].text(-0.35, 0.5, "3. DAE Reconstructed", transform=axes[2, 0].transAxes, 
                    fontsize=10, fontweight='bold', color='#38bdf8', va='center', rotation=90)
    axes[3, 0].text(-0.35, 0.5, "4. Absolute Error", transform=axes[3, 0].transAxes, 
                    fontsize=10, fontweight='bold', color='#f43f5e', va='center', rotation=90)

    d_val = getattr(model, 'latent_dim', 32)
    plt.suptitle(f"Baseline DAE (d={d_val}) Restoration Showcase (Noise: {noise_type.upper()} = {noise_param})", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def plot_calibration_curves(
    calibration_results: Dict[str, Any],
    noise_type: str = "gaussian",
    tau_jnd: float = 0.5
) -> Dict[str, float]:
    """
    Renders publication-grade Pareto frontier curves (MSE & PSNR) with shaded ±1σ ribbons.
    No indicator lines are drawn on the plot, allowing students to inspect the curve and
    determine the optimal capacity d* visually using the Elbow Rule.
    """
    dims = calibration_results['dims']
    mean_losses = [calibration_results['mean_loss'][d] for d in dims]
    std_losses  = [calibration_results['std_loss'][d] for d in dims]
    mean_psnrs  = [calibration_results['mean_psnr'][d] for d in dims]
    std_psnrs   = [calibration_results['std_psnr'][d] for d in dims]

    marginal_gains = {}
    suggested_optimal_d = dims[-1]
    for i in range(1, len(dims)):
        gain = mean_psnrs[i] - mean_psnrs[i-1]
        marginal_gains[f"{dims[i-1]} -> {dims[i]}"] = gain
        if gain < tau_jnd and suggested_optimal_d == dims[-1]:
            suggested_optimal_d = dims[i-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Validation Reconstruction Loss (MSE) with Shaded Error Ribbon
    ax1.errorbar(dims, mean_losses, yerr=std_losses, fmt='o-', color='#0284c7', ecolor='#38bdf8', 
                 elinewidth=2, capsize=5, capthick=1.5, linewidth=2.5, markersize=7, label=r'Mean MSE $\pm 1\sigma$')
    ax1.fill_between(dims, np.array(mean_losses) - np.array(std_losses), np.array(mean_losses) + np.array(std_losses),
                     color='#0284c7', alpha=0.15)
    ax1.set_title("Bottleneck Capacity vs. Reconstruction Loss (MSE)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Bottleneck Dimension ($d$)", fontsize=11)
    ax1.set_ylabel("Validation Reconstruction Loss (MSE)", fontsize=11)
    ax1.set_xticks(dims)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True, fontsize=10)

    # Subplot 2: Validation PSNR (dB) with Shaded Error Ribbon
    ax2.errorbar(dims, mean_psnrs, yerr=std_psnrs, fmt='s-', color='#10b981', ecolor='#6ee7b7', 
                 elinewidth=2, capsize=5, capthick=1.5, linewidth=2.5, markersize=7, label=r'Mean PSNR $\pm 1\sigma$')
    ax2.fill_between(dims, np.array(mean_psnrs) - np.array(std_psnrs), np.array(mean_psnrs) + np.array(std_psnrs),
                     color='#10b981', alpha=0.15)
    ax2.set_title("Bottleneck Capacity vs. Image Recovery Fidelity (PSNR)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Bottleneck Dimension ($d$)", fontsize=11)
    ax2.set_ylabel("Peak Signal-to-Noise Ratio (dB)", fontsize=11)
    ax2.set_xticks(dims)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(frameon=True, fontsize=10)

    plt.suptitle(f"Multi-Seed Capacity Sweep & Pareto Frontier (Noise: {noise_type.upper()})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return marginal_gains


def plot_calibrated_showcase(
    model: nn.Module,
    test_subset: torch.Tensor,
    noise_type: str = "gaussian",
    noise_param: float = 0.40,
    d_star: int = 64
) -> None:
    """Renders a 4-row diagnostic showcase for the calibrated Model m* across test digits."""
    model.eval()
    device = next(model.parameters()).device
    test_subset = test_subset.to(device)
    n_digits = min(8, test_subset.size(0))
    with torch.no_grad():
        sub = test_subset[:n_digits]
        noisy_subset = apply_noise(sub, noise_type=noise_type, noise_param=noise_param)
        reconstructions = model(noisy_subset)
        residuals = torch.abs(sub - reconstructions)

    fig, axes = plt.subplots(4, n_digits, figsize=(n_digits * 2.0, 8))
    row_titles = ["Pristine (x)", f"Noisy ({noise_type})", f"Model m* (d={d_star})", "Residual (|x - x̂|)"]

    for col in range(n_digits):
        axes[0, col].imshow(sub[col].view(28, 28).cpu(), cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(noisy_subset[col].view(28, 28).cpu(), cmap="gray", vmin=0, vmax=1)
        axes[2, col].imshow(reconstructions[col].view(28, 28).cpu(), cmap="gray", vmin=0, vmax=1)
        axes[3, col].imshow(residuals[col].view(28, 28).cpu(), cmap="inferno", vmin=0, vmax=1)
        for row in range(4):
            if col == 0:
                axes[row, col].set_ylabel(row_titles[row], fontsize=11, fontweight="bold", labelpad=6)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    plt.suptitle(f"Calibrated Model m* (d*={d_star}) Visual Restoration Showcase Across {n_digits} Test Digits", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_ood_showcase(
    model: nn.Module,
    clean_samples: torch.Tensor,
    noise: Optional[Union[Dict[str, Any], str]] = None,
    trained_noise: Optional[Union[Dict[str, Any], str]] = None,
    trained_noise_type: Optional[str] = None,
    eval_noise_params: Optional[Dict[str, float]] = None,
    n_samples: int = 6,
    device: Optional[torch.device] = None,
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates and visualizes Cross-Distribution / Out-of-Distribution (OOD) restoration
    robustness across all three core noise families: Gaussian, Salt & Pepper, and Masking.

    Parameters
    ----------
    model : nn.Module
        Trained DAE model instance (e.g. baseline or calibrated model m*).
    clean_samples : torch.Tensor
        Batch of clean reference images (e.g. fixed_clean_batch[:6]).
    noise : Optional[Union[Dict, str]]
        Dictionary specifying the noise family and parameter/threshold, e.g.:
            noise = {"salt_and_pepper": 0.25}
            noise = {CHOSEN_NOISE_TYPE: CHOSEN_NOISE_PARAM}
            noise = {"family": "gaussian", "value": 0.40}
        Can also specify custom thresholds for multiple or all noise families:
            noise = {"salt_and_pepper": 0.25, "gaussian": 0.50, "masking": 0.40}
    trained_noise : Optional[Union[Dict, str]]
        Alternative alias for noise dictionary / string.
    trained_noise_type : Optional[str]
        Backward-compatible string specifying trained noise type (e.g. 'gaussian').
    eval_noise_params : Optional[Dict[str, float]]
        Optional explicit overrides for evaluation thresholds across noise families.
    n_samples : int
        Number of test digits to showcase across columns (default: 6).
    device : Optional[torch.device]
        Target compute device. If None, derived from model parameters.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Benchmark summary dictionary containing mean raw PSNR, restored PSNR, and gain (dB)
        for each evaluated noise family.
    """
    default_noise_params: Dict[str, float] = {
        "gaussian": 0.40,
        "salt_and_pepper": 0.25,
        "masking": 0.35,
    }

    noise_meta: Dict[str, Dict[str, Any]] = {
        "gaussian": {
            "display_name": "Gaussian",
            "symbol": "σ",
            "default": 0.40,
        },
        "salt_and_pepper": {
            "display_name": "Salt & Pepper",
            "symbol": "p",
            "default": 0.25,
        },
        "masking": {
            "display_name": "Masking",
            "symbol": "p",
            "default": 0.35,
        },
    }

    def _normalize(name: str) -> str:
        n = str(name).lower().strip()
        if n in ["gaussian", "awgn", "gauss"]:
            return "gaussian"
        elif n in ["salt_and_pepper", "sp", "salt_pepper", "impulse", "s&p"]:
            return "salt_and_pepper"
        elif n in ["masking", "mask", "dropout"]:
            return "masking"
        return n

    # 1. Resolve noise specification
    target_noise = noise if noise is not None else (trained_noise if trained_noise is not None else trained_noise_type)
    resolved_trained_type = None
    resolved_trained_param = None
    active_eval_params = dict(default_noise_params)

    if isinstance(target_noise, dict):
        if "family" in target_noise:
            fam = _normalize(target_noise["family"])
            val = target_noise.get("value", target_noise.get("param", target_noise.get("threshold", default_noise_params.get(fam, 0.25))))
            resolved_trained_type = fam
            resolved_trained_param = float(val)
            active_eval_params[fam] = float(val)
        else:
            first_key = True
            for k, v in target_noise.items():
                norm_k = _normalize(k)
                if norm_k in default_noise_params:
                    active_eval_params[norm_k] = float(v)
                    if first_key:
                        resolved_trained_type = norm_k
                        resolved_trained_param = float(v)
                        first_key = False
    elif isinstance(target_noise, str):
        resolved_trained_type = _normalize(target_noise)

    # Legacy kwargs support
    if "ood_noise_type" in kwargs and kwargs["ood_noise_type"]:
        ood_t = _normalize(kwargs["ood_noise_type"])
        if "ood_noise_param" in kwargs and kwargs["ood_noise_param"] is not None:
            active_eval_params[ood_t] = float(kwargs["ood_noise_param"])

    if eval_noise_params:
        for k, v in eval_noise_params.items():
            norm_k = _normalize(k)
            if norm_k in active_eval_params:
                active_eval_params[norm_k] = float(v)

    if resolved_trained_type is None:
        resolved_trained_type = "salt_and_pepper"

    if resolved_trained_param is None:
        resolved_trained_param = active_eval_params.get(resolved_trained_type, default_noise_params[resolved_trained_type])
    else:
        active_eval_params[resolved_trained_type] = resolved_trained_param

    # Put trained noise first (In-Distribution reference), followed by the remaining two (OOD)
    all_families = ["gaussian", "salt_and_pepper", "masking"]
    ordered_families = [resolved_trained_type] + [f for f in all_families if f != resolved_trained_type]

    # Prepare model and device
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    clean_sub = clean_samples[:n_samples].to(device)

    # Collect predictions and PSNR metrics across all 3 noise families
    results = {}
    benchmark_metrics = {}

    with torch.no_grad():
        for fam in ordered_families:
            param = active_eval_params[fam]
            noisy = apply_noise(clean_sub, noise_type=fam, noise_param=param)
            restored = model(noisy)

            raw_psnrs = [compute_psnr(clean_sub[i], noisy[i]) for i in range(n_samples)]
            out_psnrs = [compute_psnr(clean_sub[i], restored[i]) for i in range(n_samples)]
            gains = [out_psnrs[i] - raw_psnrs[i] for i in range(n_samples)]

            results[fam] = {
                "noisy": noisy.cpu(),
                "restored": restored.cpu(),
                "raw_psnrs": raw_psnrs,
                "out_psnrs": out_psnrs,
                "gains": gains,
                "param": param,
                "is_trained": (fam == resolved_trained_type),
            }
            benchmark_metrics[fam] = {
                "raw_psnr_mean": float(np.mean(raw_psnrs)),
                "out_psnr_mean": float(np.mean(out_psnrs)),
                "gain_mean": float(np.mean(gains)),
                "param": param,
                "is_trained": (fam == resolved_trained_type),
            }

    # Render 7-row diagnostic showcase
    fig, axes = plt.subplots(7, n_samples, figsize=(n_samples * 2.2, 14.5))

    # Row 0: Clean Pristine Ground Truth (x)
    for col in range(n_samples):
        axes[0, col].imshow(clean_sub[col].view(28, 28).cpu(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, col].set_title(f"Sample #{col + 1}", fontsize=10, fontweight="bold", pad=4)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        for spine in axes[0, col].spines.values():
            spine.set_visible(True)
            spine.set_color("#cbd5e1")
            spine.set_linewidth(0.8)
    axes[0, 0].set_ylabel("Clean (x)", fontsize=10, fontweight="bold", color="#10b981", labelpad=8)

    # Rows 1..6: Noisy & Restored for all 3 noise families
    for idx, fam in enumerate(ordered_families):
        res = results[fam]
        meta = noise_meta[fam]
        is_trained = res["is_trained"]
        regime_badge = "TRAINED (ID)" if is_trained else "OOD"
        tag_color = "#16a34a" if is_trained else "#ea580c"
        recon_tag_color = "#0284c7" if is_trained else "#7c3aed"

        noisy_row = 1 + 2 * idx
        recon_row = 2 + 2 * idx

        for col in range(n_samples):
            raw_p = res["raw_psnrs"][col]
            out_p = res["out_psnrs"][col]
            gain = res["gains"][col]
            gain_sign = "+" if gain >= 0 else ""

            # Noisy image
            axes[noisy_row, col].imshow(res["noisy"][col].view(28, 28), cmap="gray", vmin=0.0, vmax=1.0)
            axes[noisy_row, col].set_title(f"Raw: {raw_p:.1f} dB", fontsize=9, color="#9a3412", pad=3)
            axes[noisy_row, col].set_xticks([])
            axes[noisy_row, col].set_yticks([])
            for spine in axes[noisy_row, col].spines.values():
                spine.set_visible(True)
                spine.set_color("#cbd5e1")
                spine.set_linewidth(0.8)

            # Restored image
            axes[recon_row, col].imshow(res["restored"][col].view(28, 28), cmap="gray", vmin=0.0, vmax=1.0)
            axes[recon_row, col].set_title(f"DAE: {out_p:.1f} dB ({gain_sign}{gain:.1f})", fontsize=8.5, fontweight="bold", color=recon_tag_color, pad=3)
            axes[recon_row, col].set_xticks([])
            axes[recon_row, col].set_yticks([])
            for spine in axes[recon_row, col].spines.values():
                spine.set_visible(True)
                spine.set_color("#cbd5e1")
                spine.set_linewidth(0.8)

        param_str = f"{meta['symbol']}={res['param']:.2f}"
        axes[noisy_row, 0].set_ylabel(
            f"Noisy: {meta['display_name']}\n{param_str}\n[{regime_badge}]",
            fontsize=8.5, fontweight="bold", color=tag_color, labelpad=8
        )
        axes[recon_row, 0].set_ylabel(
            f"DAE Output (x̂)\n[{'IN-DIST' if is_trained else 'OOD'}]",
            fontsize=8.5, fontweight="bold", color=recon_tag_color, labelpad=8
        )

    trained_meta = noise_meta[resolved_trained_type]
    plt.suptitle(
        f"Cross-Noise Robustness Evaluation Across All 3 Noise Families\n"
        f"Model Trained on: {trained_meta['display_name']} ({trained_meta['symbol']} = {resolved_trained_param:.2f}) [IN-DISTRIBUTION]",
        fontsize=13, fontweight="bold", y=0.99
    )
    plt.subplots_adjust(top=0.94, bottom=0.02, hspace=0.35, wspace=0.15)
    plt.show()

    # Console Telemetry / Summary Table
    print("=" * 82)
    print("🛡️  CROSS-NOISE RESTORATION BENCHMARK (ALL 3 NOISE FAMILIES)")
    print("=" * 82)
    print(f"🎯 Reference Trained Regime : {trained_meta['display_name']} ({trained_meta['symbol']} = {resolved_trained_param:.2f}) [IN-DISTRIBUTION BASELINE]")
    print("-" * 82)
    print(f"{'Noise Family':<18} | {'Regime':<14} | {'Threshold':<11} | {'Input PSNR':<11} | {'DAE PSNR':<11} | {'Gain (Δ)':<9}")
    print("-" * 82)
    for fam in ordered_families:
        m = benchmark_metrics[fam]
        f_meta = noise_meta[fam]
        reg_str = "TRAINED (ID)" if m["is_trained"] else "OOD"
        thresh_str = f"{f_meta['symbol']} = {m['param']:.2f}"
        gain_s = "+" if m["gain_mean"] >= 0 else ""
        print(f"{f_meta['display_name']:<18} | {reg_str:<14} | {thresh_str:<11} | {m['raw_psnr_mean']:>6.2f} dB    | {m['out_psnr_mean']:>6.2f} dB   | {gain_s}{m['gain_mean']:>5.2f} dB")
    print("=" * 82)

    return benchmark_metrics
