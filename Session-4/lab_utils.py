"""
lab_utils.py - Helper Utilities for Session 4 Ingestion Pipelines & DataLoader Lab
===================================================================================
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# 1. Section 1.2: Sample Digits Visualization Helper
# ============================================================================
def plot_mnist_samples(
    dataset: Dataset,
    num_samples: int = 16,
    suptitle: str = "Sample Handwritten Digits from MNIST Collection D"
) -> None:
    """
    Plots a clean 2-row grid of empirical samples from an MNIST dataset
    with non-overlapping titles.

    Parameters:
        dataset: PyTorch Dataset supporting indexed evaluation dataset[i] -> (image, label).
        num_samples: Number of samples to display (default: 16 -> 2 rows of 8).
        suptitle: Main title for the figure.
    """
    cols = min(8, num_samples)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for idx, ax in enumerate(axes_flat):
        if idx < num_samples:
            img, lbl = dataset[idx]
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Digit: {lbl}", fontsize=11, fontweight="bold", color="#0284c7")
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88, hspace=0.4)
    plt.show()


# ============================================================================
# 2. Section 2.1: Pixel Distribution Histogram Helper
# ============================================================================
def plot_pixel_distribution(
    pixels: np.ndarray,
    mean: Optional[float] = None,
    bins: int = 30,
    title: str = "Histogram of Unstandardized Scaled Pixels [0.0, 1.0]"
) -> None:
    """
    Plots a stylized histogram of raw/unstandardized scaled pixel intensities,
    optionally marking the empirical dataset mean.

    Parameters:
        pixels: Flattened numpy array of pixel intensities.
        mean: Optional empirical mean value μ to display as a vertical dashed line.
        bins: Number of histogram bins (default: 30).
        title: Title of the histogram plot.
    """
    plt.figure(figsize=(8, 3.5))
    plt.hist(pixels, bins=bins, color="#f43f5e", alpha=0.75, edgecolor="black")

    if mean is not None:
        plt.axvline(
            mean,
            color="#38bdf8",
            linestyle="--",
            linewidth=2,
            label=f"Mean μ = {mean:.3f}"
        )
        plt.legend()

    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.show()


# ============================================================================
# 3. Task 1: Side-by-Side Distribution Comparison Helper
# ============================================================================
def plot_distribution_comparison(
    unstandardized_pixels: np.ndarray,
    standardized_pixels: np.ndarray,
    mean_unstandardized: Optional[float] = None,
    mean_standardized: float = 0.0,
    bins: int = 30
) -> None:
    """
    Renders a side-by-side comparison of unstandardized vs standardized pixel
    distributions to demonstrate the centering and scaling effect of z = (x - μ) / σ.

    Parameters:
        unstandardized_pixels: Pixel values scaled to [0.0, 1.0].
        standardized_pixels: Z-score standardized pixel values.
        mean_unstandardized: Empirical mean of unstandardized distribution.
        mean_standardized: Target mean of standardized distribution (default: 0.0).
        bins: Number of histogram bins (default: 30).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))

    # 1. Unstandardized
    ax1.hist(unstandardized_pixels, bins=bins, color="#f43f5e", alpha=0.7, edgecolor="black")
    if mean_unstandardized is not None:
        ax1.axvline(
            mean_unstandardized,
            color="#38bdf8",
            linestyle="--",
            linewidth=2,
            label=f"Mean μ = {mean_unstandardized:.3f}"
        )
        ax1.legend()
    ax1.set_title("Unstandardized [0.0, 1.0]\n(All positive, steep Hessian ravine)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Pixel Value")
    ax1.grid(alpha=0.3)

    # 2. Standardized
    ax2.hist(standardized_pixels, bins=bins, color="#10b981", alpha=0.7, edgecolor="black")
    ax2.axvline(
        mean_standardized,
        color="#38bdf8",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_standardized:.1f}"
    )
    ax2.set_title("Standardized z-score\n(Zero mean, unit variance)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Standardized Value (z-score)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================================
# 4. Section 4.2: Mini-Batch Visualization Helper
# ============================================================================
def plot_batch_samples(
    batch_images: torch.Tensor,
    batch_labels: torch.Tensor,
    num_samples: int = 16,
    batch_size: Optional[int] = None
) -> None:
    """
    Plots a sample of images from a DataLoader mini-batch, squeezing channel
    dimensions and displaying class label annotations.

    Parameters:
        batch_images: Tensor of shape (B, 1, 28, 28) or (B, 28, 28).
        batch_labels: 1D Tensor of shape (B,) containing class indices.
        num_samples: Number of images from the batch to render (default: 16).
        batch_size: Optional total batch size for figure title.
    """
    cols = min(8, num_samples)
    rows = (num_samples + cols - 1) // cols
    b_size = batch_size if batch_size is not None else batch_images.size(0)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for idx, ax in enumerate(axes_flat):
        if idx < num_samples:
            img = batch_images[idx]
            if img.dim() == 3 and img.size(0) == 1:
                img = img.squeeze(0)
            img_np = img.detach().cpu().numpy()
            lbl = batch_labels[idx].item()
            ax.imshow(img_np, cmap="gray")
            ax.set_title(f"Label: {lbl}", fontsize=11, fontweight="bold", color="#10b981")
        ax.axis("off")

    plt.suptitle(f"Sample Batch of B={b_size} Images from train_loader", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.4)
    plt.show()


# ============================================================================
# 5. Section 6.3: Untrained Model Accuracy Evaluation Helper
# ============================================================================
def evaluate_untrained_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    batch_images: Optional[torch.Tensor] = None,
    batch_labels: Optional[torch.Tensor] = None,
    device: str = "cpu"
) -> float:
    """
    Computes and displays model accuracy on both an immediate mini-batch
    and across the entire dataset without training, followed by a conceptual
    discussion prompt for students.

    Parameters:
        model: PyTorch neural network model.
        data_loader: DataLoader providing batches for whole-dataset evaluation.
        batch_images: Optional single batch input tensor (B, C, H, W).
        batch_labels: Optional ground-truth labels for the single batch.
        device: Device to run evaluation on (default: 'cpu').

    Returns:
        float: Overall accuracy on the full data_loader (between 0.0 and 1.0).
    """
    model.eval()

    # 1. Mini-batch evaluation if provided
    if batch_images is not None and batch_labels is not None:
        with torch.no_grad():
            logits = model(batch_images.to(device))
            batch_preds = logits.argmax(dim=-1).cpu()
            b_labels = batch_labels.cpu()
            batch_correct = (batch_preds == b_labels).sum().item()
            b_size = b_labels.size(0)
            batch_acc = batch_correct / b_size

        print(f"📦 Mini-Batch Evaluation (B={b_size}):")
        print(f"   Predicted classes: {batch_preds[:16].tolist()} ...")
        print(f"   Ground-truth:     {b_labels[:16].tolist()} ...")
        print(f"   Batch Accuracy:   {batch_correct}/{b_size} = {batch_acc * 100:.2f}%\n")

    # 2. Evaluation across the entire DataLoader
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"🌐 Whole Dataset Evaluation (N={total_samples:,}):")
    print(f"   Total Correct:    {total_correct:,} / {total_samples:,}")
    print(f"   Overall Accuracy: {overall_acc * 100:.2f}%\n")


# ============================================================================
# 6. Optimization Geometry: 3D Loss Landscape Visualization Helper
# ============================================================================
def _prepare_input_variants(images: torch.Tensor):
    """
    Given a batch of images, robustly derives:
      1. X_raw [0, 255]
      2. X_scaled [0.0, 1.0]
      3. X_standardized (zero-mean, unit-variance)
    """
    img = images.float()
    if img.min() < -0.1:
        # Input is standardized (background shifted negative)
        X_std = img
        X_scaled = img * 0.3081 + 0.1307
        X_raw = torch.clamp(X_scaled * 255.0, 0.0, 255.0)
    elif img.max() > 1.5:
        # Input is raw [0, 255]
        X_raw = img
        X_scaled = img / 255.0
        X_std = (X_scaled - 0.1307) / 0.3081
    else:
        # Input is scaled [0.0, 1.0]
        X_scaled = img
        X_raw = img * 255.0
        X_std = (X_scaled - 0.1307) / 0.3081

    return X_raw, X_scaled, X_std


def plot_3d_loss_landscapes(
    model: nn.Module,
    batch_images: torch.Tensor,
    batch_labels: torch.Tensor,
    grid_res: int = 25,
    grid_range: float = 1.0,
    seed: int = 100,
    elev: int = 26,
    azim: int = 45,
    start_pt: tuple = (-0.85, -0.85)
) -> None:
    """
    Computes and plots 3D loss landscape slices (Li et al., 2018 filter normalization)
    comparing Raw [0, 255] vs Scaled [0.0, 1.0] vs Standardized (z-score) inputs.
    Includes simulated gradient descent trajectories with directional quiver arrows
    to visually demonstrate how gradients oscillate across canyon walls on raw inputs
    versus descending directly into the minimum on standardized inputs.

    Parameters:
        model: PyTorch neural network model (e.g. nn.Sequential).
        batch_images: Batch of input images (B, 1, 28, 28).
        batch_labels: Ground truth labels (B,).
        grid_res: Resolution of the 2D mesh grid (default: 25x25 = 625 evaluations).
        grid_range: Range of directional perturbations [-grid_range, grid_range].
        seed: Random seed for direction vector generation.
        elev: Matplotlib 3D elevation view angle.
        azim: Matplotlib 3D azimuth view angle.
        start_pt: Starting coordinate (alpha, beta) for gradient descent trajectory.
    """
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.interpolate import RegularGridInterpolator

    # 1. Prepare input representations
    X_raw, X_scaled, X_std = _prepare_input_variants(batch_images)
    criterion = nn.CrossEntropyLoss()

    # 2. Sample two random direction vectors in parameter space with filter normalization
    torch.manual_seed(seed)
    base_params = [p.clone().detach() for p in model.parameters()]
    d1 = [torch.randn_like(p) for p in base_params]
    d2 = [torch.randn_like(p) for p in base_params]

    for p, v1, v2_dir in zip(base_params, d1, d2):
        v1.mul_(p.norm() / (v1.norm() + 1e-7))
        v2_dir.mul_(p.norm() / (v2_dir.norm() + 1e-7))

    # 3. Build 2D evaluation grid
    alphas = np.linspace(-grid_range, grid_range, grid_res)
    betas = np.linspace(-grid_range, grid_range, grid_res)
    A, B = np.meshgrid(alphas, betas)

    loss_raw = np.zeros((grid_res, grid_res))
    loss_scaled = np.zeros((grid_res, grid_res))
    loss_std = np.zeros((grid_res, grid_res))

    model.eval()
    try:
        with torch.no_grad():
            for i in range(grid_res):
                for j in range(grid_res):
                    a = A[i, j]
                    b = B[i, j]
                    for p, p0, v1, v2_dir in zip(model.parameters(), base_params, d1, d2):
                        p.copy_(p0 + a * v1 + b * v2_dir)

                    loss_raw[i, j] = criterion(model(X_raw), batch_labels).item()
                    loss_scaled[i, j] = criterion(model(X_scaled), batch_labels).item()
                    loss_std[i, j] = criterion(model(X_std), batch_labels).item()
    finally:
        with torch.no_grad():
            for p, p0 in zip(model.parameters(), base_params):
                p.copy_(p0)

    # 4. Simulate 2D Gradient Descent Trajectories
    def _simulate_trajectory(Z_grid, lr, steps=8, start=start_pt):
        interp = RegularGridInterpolator(
            (betas, alphas), Z_grid, method="cubic", bounds_error=False, fill_value=None
        )
        eps = 1e-3
        curr_a, curr_b = float(start[0]), float(start[1])
        traj_a = [curr_a]
        traj_b = [curr_b]
        traj_z = [float(interp((curr_b, curr_a)))]

        for _ in range(steps):
            grad_a = (interp((curr_b, curr_a + eps)) - interp((curr_b, curr_a - eps))) / (2 * eps)
            grad_b = (interp((curr_b + eps, curr_a)) - interp((curr_b - eps, curr_a))) / (2 * eps)

            curr_a = float(np.clip(curr_a - lr * grad_a, -0.95 * grid_range, 0.95 * grid_range))
            curr_b = float(np.clip(curr_b - lr * grad_b, -0.95 * grid_range, 0.95 * grid_range))

            traj_a.append(curr_a)
            traj_b.append(curr_b)
            traj_z.append(float(interp((curr_b, curr_a))))

        return np.array(traj_a), np.array(traj_b), np.array(traj_z)

    traj_raw_a, traj_raw_b, traj_raw_z = _simulate_trajectory(loss_raw, lr=0.038, steps=9)
    traj_scaled_a, traj_scaled_b, traj_scaled_z = _simulate_trajectory(loss_scaled, lr=1.8, steps=7)
    traj_std_a, traj_std_b, traj_std_z = _simulate_trajectory(loss_std, lr=1.8, steps=6)

    # 5. Render 3-panel 3D visualization with trajectory arrows
    fig = plt.figure(figsize=(18, 5.6))

    def _plot_panel(ax, Z, cmap, title, color_title, t_a, t_b, t_z, traj_label):
        ax.plot_surface(A, B, Z, cmap=cmap, edgecolor="none", alpha=0.75)

        # Lift trajectory slightly above the surface to prevent 3D clipping
        z_offset = (Z.max() - Z.min()) * 0.04

        # Plot trajectory line path
        ax.plot(t_a, t_b, t_z + z_offset, color="#e11d48", linewidth=2.5, zorder=10, label=traj_label)

        # Plot start and current step markers
        ax.scatter([t_a[0]], [t_b[0]], [t_z[0] + z_offset], color="#38bdf8", s=90, marker="o", edgecolor="black", linewidth=1.5, zorder=12, label="Start $\\theta_0$")
        ax.scatter([t_a[-1]], [t_b[-1]], [t_z[-1] + z_offset], color="#22c55e", s=160, marker="*", edgecolor="black", linewidth=1.5, zorder=12, label="Current Step")

        # Draw 3D arrows along the trajectory segments
        for k in range(len(t_a) - 1):
            da = t_a[k + 1] - t_a[k]
            db = t_b[k + 1] - t_b[k]
            dz = t_z[k + 1] - t_z[k]
            if np.sqrt(da**2 + db**2) > 0.02:
                ax.quiver(
                    t_a[k], t_b[k], t_z[k] + z_offset,
                    da, db, dz,
                    color="#991b1b",
                    arrow_length_ratio=0.35,
                    linewidth=1.8,
                    pivot="tail"
                )

        ax.set_title(title, fontsize=11, fontweight="bold", color=color_title)
        ax.set_xlabel(r"$\alpha$ ($d_1$)", labelpad=2)
        ax.set_ylabel(r"$\beta$ ($d_2$)", labelpad=2)
        ax.set_zlabel("Loss", labelpad=2)
        ax.view_init(elev=elev, azim=azim)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    _plot_panel(
        ax1, loss_raw, "coolwarm",
        f"1. Raw [0, 255]\n[Canyon Oscillation: Violent Zig-Zag Across Walls]\nLoss: {loss_raw.min():.1f} → {loss_raw.max():.1f}",
        "#f43f5e",
        traj_raw_a, traj_raw_b, traj_raw_z, "Oscillating Path"
    )

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    _plot_panel(
        ax2, loss_scaled, "viridis",
        f"2. Scaled [0.0, 1.0]\n[Asymmetric Ravine: Curved Descent Path]\nLoss: {loss_scaled.min():.2f} → {loss_scaled.max():.2f}",
        "#f59e0b",
        traj_scaled_a, traj_scaled_b, traj_scaled_z, "Curved Path"
    )

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    _plot_panel(
        ax3, loss_std, "plasma",
        f"3. Standardized (z-score)\n[Isotropic Basin: Direct Descent to Minimum]\nLoss: {loss_std.min():.2f} → {loss_std.max():.2f}",
        "#10b981",
        traj_std_a, traj_std_b, traj_std_z, "Direct Path"
    )

    fig.suptitle(
        "Optimization Trajectories: Gradient Descent Path Towards the Minimum",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout()
    plt.show()

    # Print summary interpretation
    print("📊 3D Loss Curvature & Optimization Trajectory Summary:")
    print(f"   • Raw [0, 255]:         Loss range = [{loss_raw.min():.2f}, {loss_raw.max():.2f}] (Δ = {loss_raw.max() - loss_raw.min():.2f})")
    print("     ⚠️  Trajectory: Arrows violently bounce back and forth across the steep canyon walls.")
    print(f"   • Scaled [0.0, 1.0]:    Loss range = [{loss_scaled.min():.4f}, {loss_scaled.max():.4f}] (Δ = {loss_scaled.max() - loss_scaled.min():.4f})")
    print("     ⚠️  Trajectory: Descent path curves along an asymmetric ravine due to non-zero positive mean.")
    print(f"   • Standardized z-score: Loss range = [{loss_std.min():.4f}, {loss_std.max():.4f}] (Δ = {loss_std.max() - loss_std.min():.4f})")
    print("     ✅  Trajectory: Arrows point directly straight down the isotropic basin toward the global minimum!")

