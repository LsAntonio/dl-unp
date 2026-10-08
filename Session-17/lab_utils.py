"""
Session 17 Lab Utilities: CIFAR-10 Scaling & Normalization Dynamics (BatchNorm vs. LayerNorm)
Course: UNP Deep Learning Foundations — Unit II (Week 6, Session 17)
Instructors: MSc. Antonio Aguilar & Dr. Luis Aguilar Ibáñez

This helper module encapsulates:
1. CIFAR-10 dataset acquisition, preprocessing, and optimized DataLoader configuration.
2. Automated layer-by-layer tensor dimension tracking via PyTorch forward hooks.
3. Visualization suites for training dynamics (Loss & Accuracy curves).
4. Normalization parameter inspection: learned affine scaling (gamma, beta) & running statistics.
5. The Socratic Ablation Suite: Automated demonstration of the eval() vs. train() deployment bug.
"""

import os
import sys
import time
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


# ==============================================================================
# 1. Dataset & DataLoader Management
# ==============================================================================

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# High-speed CDN mirror to replace the slow University of Toronto server
CIFAR10_MIRROR_URL = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"

def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    download: bool = True
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Constructs standardized CIFAR-10 train and test DataLoaders with canonical normalization.
    Uses high-speed CDN mirror and searches local course directories to prevent slow downloads.
    """
    # 1. Override torchvision URL with high-speed CDN mirror
    torchvision.datasets.CIFAR10.url = CIFAR10_MIRROR_URL

    # 2. Intelligent local directory resolution across course workspaces
    resolved_dir = data_dir
    candidate_paths = [
        data_dir,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Week-1/1_Monday/NotebookLab/data_cifar10")),
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Week-1/2_Wednesday/data")),
        "./data_cifar10",
        "../data_cifar10",
        "/content/data_cifar10"
    ]
    for c_path in candidate_paths:
        if os.path.exists(os.path.join(c_path, "cifar-10-batches-py")):
            resolved_dir = c_path
            download = False
            break

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    train_set = CIFAR10(root=resolved_dir, train=True, download=download, transform=train_transform)
    test_set  = CIFAR10(root=resolved_dir, train=False, download=download, transform=test_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    return train_loader, test_loader, classes


# ==============================================================================
# 2. Forward Hook Dimension & Parameter Tracing
# ==============================================================================

def trace_model_geometry(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 32, 32),
    device: str = "cpu"
) -> List[Dict[str, Any]]:
    """
    Attaches forward hooks to all submodules in the model, pushes a dummy tensor,
    and displays a formatted architectural blueprint table.
    """
    records: List[Dict[str, Any]] = []
    hooks = []

    def hook_fn(module, input_t, output_t, name):
        if len(list(module.children())) == 0:  # Leaf module only
            in_shape = list(input_t[0].shape) if isinstance(input_t, tuple) and len(input_t) > 0 else []
            out_shape = list(output_t.shape) if isinstance(output_t, torch.Tensor) else []
            params = sum(p.numel() for p in module.parameters())
            records.append({
                "name": name,
                "type": module.__class__.__name__,
                "in_shape": in_shape,
                "out_shape": out_shape,
                "params": params
            })

    for name, mod in model.named_modules():
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, n=name: hook_fn(m, i, o, n)
        ))

    model.eval()
    model.to(device)
    dummy_input = torch.zeros(input_size, device=device)
    with torch.no_grad():
        _ = model(dummy_input)

    for h in hooks:
        h.remove()

    # Pretty-print architectural blueprint
    print("=" * 95)
    print(f"{'Layer / Module Name':<30} | {'Type':<18} | {'Input Shape':<16} | {'Output Shape':<16} | {'Params'}")
    print("=" * 95)
    total_params = 0
    for r in records:
        in_s = str(r["in_shape"])
        out_s = str(r["out_shape"])
        print(f"{r['name']:<30} | {r['type']:<18} | {in_s:<16} | {out_s:<16} | {r['params']:>8,}")
        total_params += r["params"]
    print("=" * 95)
    print(f"Total Architecture Parameters: {total_params:,} ({total_params/1e6:.2f} Million)")
    print("=" * 95)

    return records


# ==============================================================================
# 3. Training & Evaluation Engine
# ==============================================================================

# Universal compatibility helper for Mixed Precision
def make_grad_scaler(enabled: bool = True):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def get_autocast_ctx():
    try:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(dtype=torch.float16)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[Any] = None
) -> Tuple[float, float]:
    """Runs a single training epoch with optional Automatic Mixed Precision (AMP)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    use_amp = scaler is not None and device.type == "cuda"

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with get_autocast_ctx():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluates the model on the test partition strictly under model.eval()."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
    scheduler: Optional[Any] = None,
    use_amp: bool = True
) -> Dict[str, List[float]]:
    """Executes a full multi-epoch training pipeline with metric tracking."""
    scaler = make_grad_scaler(enabled=(use_amp and device.type == "cuda"))
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "epoch_time": []
    }

    print(f"\n🚀 Initiating Training: {epochs} Epochs on {device} (AMP={use_amp and device.type == 'cuda'})")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(elapsed)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}%")

    return history


# ==============================================================================
# 4. Visualization Suites
# ==============================================================================

def plot_training_curves(history: Dict[str, List[float]], title_suffix: str = "") -> None:
    """Renders professional 2-panel loss and accuracy progression curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=120)

    # Loss Curve
    axes[0].plot(epochs, history["train_loss"], "o-", color="#1f77b4", lw=2, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "s--", color="#d62728", lw=2, label="Validation Loss")
    axes[0].set_title(f"Cross-Entropy Loss Dynamics {title_suffix}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Accuracy Curve
    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], "o-", color="#2ca02c", lw=2, label="Train Acc")
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], "s--", color="#ff7f0e", lw=2, label="Validation Acc")
    axes[1].set_title(f"Top-1 Accuracy Progression {title_suffix}", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_affine_parameters(model: nn.Module) -> None:
    """
    Extracts and visualizes learned affine parameters (gamma and beta)
    across all normalization modules in the network.
    """
    gammas, betas, layer_names = [], [], []

    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)) or module.__class__.__name__ == "LayerNorm2d":
            if module.weight is not None:
                gammas.append(module.weight.detach().cpu().numpy())
                betas.append(module.bias.detach().cpu().numpy())
                layer_names.append(name.split(".")[-1])

    if not gammas:
        print("⚠️ No normalization layers with affine parameters found.")
        return

    n_layers = len(gammas)
    fig, axes = plt.subplots(2, n_layers, figsize=(3.8 * n_layers, 5.5), dpi=120)

    for i in range(n_layers):
        # Gamma histogram
        axes[0, i].hist(gammas[i], bins=25, color="#1f77b4", alpha=0.7, edgecolor="black")
        axes[0, i].axvline(1.0, color="red", linestyle="--", label="Init (1.0)")
        axes[0, i].set_title(f"{layer_names[i]}: Learned $\\gamma$ (Scale)", fontsize=11)
        axes[0, i].set_xlabel("Value")
        if i == 0:
            axes[0, i].set_ylabel("Frequency")
        axes[0, i].legend(fontsize=8)

        # Beta histogram
        axes[1, i].hist(betas[i], bins=25, color="#2ca02c", alpha=0.7, edgecolor="black")
        axes[1, i].axvline(0.0, color="red", linestyle="--", label="Init (0.0)")
        axes[1, i].set_title(f"{layer_names[i]}: Learned $\\beta$ (Shift)", fontsize=11)
        axes[1, i].set_xlabel("Value")
        if i == 0:
            axes[1, i].set_ylabel("Frequency")
        axes[1, i].legend(fontsize=8)

    plt.suptitle("Distribution of Learned Affine Parameters across Normalization Stages", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 5. The Eval Bug Demonstration Suite
# ==============================================================================

def demonstrate_eval_bug(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    batch_sizes: List[int] = [1, 2, 4, 16, 64, 128]
) -> None:
    """
    Rigorously illustrates the #1 production deployment trap:
    Evaluating under model.train() vs. model.eval() across varying mini-batch sizes.
    """
    eval_accs, train_mode_accs = [], []
    criterion = nn.CrossEntropyLoss()

    print("\n" + "=" * 75)
    print("DEMONSTRATING THE EVAL BUG: model.eval() vs. model.train() AT TEST TIME")
    print("=" * 75)
    print(f"{'Batch Size (N)':<16} | {'model.eval() Acc':<20} | {'model.train() Acc':<20} | {'Discrepancy'}")
    print("-" * 75)

    # Subsample 500 test images for rapid demonstration across different batch sizes
    test_dataset = test_loader.dataset
    subset_indices = list(range(min(500, len(test_dataset))))
    subset = torch.utils.data.Subset(test_dataset, subset_indices)

    for bs in batch_sizes:
        sub_loader = DataLoader(subset, batch_size=bs, shuffle=False)

        # 1. Correct Evaluation Mode
        model.eval()
        _, acc_eval = evaluate_model(model, sub_loader, criterion, device)
        eval_accs.append(acc_eval * 100)

        # 2. Buggy Training Mode Inference
        model.train()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in sub_loader:
                bx, by = bx.to(device), by.to(device)
                if bs == 1 and isinstance(list(model.modules())[1], nn.BatchNorm2d):
                    # PyTorch throws ValueError or NaNs if batch_size=1 in train mode with BatchNorm2d
                    preds = torch.zeros_like(by)
                else:
                    try:
                        out = model(bx)
                        preds = out.argmax(dim=1)
                    except Exception:
                        preds = torch.zeros_like(by)
                correct += preds.eq(by).sum().item()
                total += by.size(0)
        acc_train_mode = correct / total
        train_mode_accs.append(acc_train_mode * 100)

        delta = (acc_eval - acc_train_mode) * 100
        print(f"{bs:<16} | {acc_eval*100:>18.2f}% | {acc_train_mode*100:>18.2f}% | {delta:>12.2f}%")

    print("=" * 75)

    # Plot comparative results
    plt.figure(figsize=(8.5, 4.5), dpi=120)
    plt.plot(batch_sizes, eval_accs, "o-", color="#2ca02c", lw=2.5, label="Legitimate: model.eval()")
    plt.plot(batch_sizes, train_mode_accs, "s--", color="#d62728", lw=2.5, label="Buggy: model.train() at test time")
    plt.xscale("log", base=2)
    plt.title("The Deployment Trap: Test Accuracy vs. Batch Size under Different Modes", fontsize=12, fontweight="bold")
    plt.xlabel("Evaluation Mini-Batch Size (log2 scale)", fontsize=11)
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 6. Environment & Architectural Validation Helpers
# ==============================================================================

def init_environment(seed: int = 42) -> torch.device:
    """
    Initializes reproducible pseudorandom seeds across PyTorch and NumPy,
    resolves the compute accelerator (CUDA GPU or CPU), and configures torchvision settings.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure high-speed CDN mirror for CIFAR-10
    torchvision.datasets.CIFAR10.url = CIFAR10_MIRROR_URL

    print("✓ PyTorch Execution Environment Initialized.")
    print(f"  PyTorch Version: {torch.__version__} | Torchvision: {torchvision.__version__}")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"  Target Accelerator: {device} ({device_name})")
    return device


def verify_canonical_alexnet_collapse() -> None:
    """
    Empirically demonstrates the spatial collapse runtime exception when passing
    a 32x32 CIFAR-10 input into torchvision's canonical ImageNet AlexNet.
    """
    cifar_dummy = torch.randn(1, 3, 32, 32)
    canonical_alexnet = torchvision.models.alexnet(weights=None)

    try:
        print("Passing CIFAR-10 tensor [1, 3, 32, 32] through canonical torchvision AlexNet...")
        _ = canonical_alexnet(cifar_dummy)
    except RuntimeError as e:
        print(f"\n❌ Expected Architectural Failure Observed:\n   {e}")
        print("\n✓ Mathematical Proof Confirmed: Canonical AlexNet cannot process 32x32 inputs.")


# ==============================================================================
# 7. Parameter Accounting & Budget Visualizer
# ==============================================================================

def split_params(model: nn.Module) -> Tuple[float, float, float]:
    """Computes convolution, normalization, and linear parameter totals in millions."""
    conv = sum(p.numel() for m in model.modules() if isinstance(m, nn.Conv2d) for p in m.parameters())
    norm = sum(p.numel() for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)) or type(m).__name__ == "LayerNorm2d" for p in m.parameters())
    dense = sum(p.numel() for m in model.modules() if isinstance(m, nn.Linear) for p in m.parameters())
    return conv / 1e6, norm / 1e6, dense / 1e6


def plot_parameter_budget(models_dict: Dict[str, nn.Module]) -> None:
    """
    Renders professional 2-panel parameter breakdown:
    Left: Component distribution (Conv vs. Norm vs. Linear) across models.
    Right: Layer-by-layer parameter budget on log scale for adapted AlexNet-CIFAR.
    """
    stats = {k: split_params(m) for k, m in models_dict.items()}
    names = list(stats.keys())
    conv = np.array([stats[n][0] for n in names])
    norm = np.array([stats[n][1] for n in names])
    dense = np.array([stats[n][2] for n in names])
    y = np.arange(len(names))

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1.3, 1]), dpi=120)
    color_conv = "#0284c7"
    color_dense = "#f43f5e"
    color_norm = "#10b981"

    # Panel 1: Parameter distribution by structural component
    ax[0].barh(y, conv, color=color_conv, label="Conv (feature extractor)")
    ax[0].barh(y, norm, left=conv, color=color_norm, label="Normalization (BN/LN)")
    ax[0].barh(y, dense, left=conv + norm, color=color_dense, label="Linear (dense head)")
    for i in y:
        tot = conv[i] + norm[i] + dense[i]
        pct = (dense[i] / tot) * 100 if tot > 0 else 0
        ax[0].text(tot + 0.8, i, f"{tot:.2f} M (dense {pct:.0f}%)", va="center", fontsize=9, fontweight="bold")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(names, fontsize=10)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("Parameters (Millions)", fontsize=10)
    ax[0].set_xlim(0, max(conv + norm + dense) * 1.3)
    ax[0].set_title("Architecture Parameter Breakdown: ImageNet vs. CIFAR-10", fontsize=11, fontweight="bold")
    ax[0].legend(loc="lower right", fontsize=9)
    ax[0].grid(True, linestyle=":", alpha=0.4)

    # Panel 2: Per-layer parameter budget for adapted AlexNet-CIFAR (BN reference)
    ref_model = next((m for k, m in models_dict.items() if "BN" in k or "BatchNorm" in k), list(models_dict.values())[-1])
    rows = []
    for l in ref_model.modules():
        if isinstance(l, (nn.Conv2d, nn.Linear)):
            name = type(l).__name__.replace("Conv2d", "Conv").replace("Linear", "FC")
            label = f"{name} {l.weight.shape[0]}"
            count = sum(p.numel() for p in l.parameters())
            color = color_conv if isinstance(l, nn.Conv2d) else color_dense
            rows.append((label, count, color))

    ax[1].bar(range(len(rows)), [r[1] for r in rows], color=[r[2] for r in rows], edgecolor="#334155", lw=1.2)
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(rows)))
    ax[1].set_xticklabels([r[0] for r in rows], rotation=35, fontsize=9)
    for i, r in enumerate(rows):
        ax[1].text(i, r[1] * 1.15, f"{r[1] / 1e3:,.0f}k", ha="center", fontsize=8.5, fontweight="bold")
    ax[1].set_ylabel("Parameters (Log Scale)", fontsize=10)
    ax[1].set_title("AlexNet-CIFAR: Layer-by-Layer Budget", fontsize=11, fontweight="bold")
    ax[1].grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 8. Landscape Smoothing & Optimization Geometry Suites (Santurkar et al. 2018)
# ==============================================================================

def train_and_audit_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    lr: float = 0.03,
    max_steps: int = 260,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Trains a model on CIFAR-10 while periodically computing the Santurkar et al. (2018)
    loss range metric: variation in loss when stepping alpha * lr along -grad L for alpha in [0.25, 4.0].
    """
    if device is None:
        device = next(model.parameters()).device

    torch.manual_seed(42)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    min_loss_range = []
    max_loss_range = []
    audit_steps = []
    step_alphas = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

    model.train()
    step = 0
    t0 = time.time()

    for epoch in range(5):
        for images, targets in train_loader:
            step += 1
            if step > max_steps:
                break

            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            loss_history.append(loss.item())

            # Evaluate loss range along -grad L
            if step % 2 == 1:
                orig_params = [p.data.clone() for p in model.parameters()]
                step_losses = []

                with torch.no_grad():
                    for alpha in step_alphas:
                        for p, p_orig in zip(model.parameters(), orig_params):
                            if p.grad is not None:
                                p.data = p_orig - (alpha * lr) * p.grad

                        test_out = model(images)
                        step_losses.append(criterion(test_out, targets).item())

                        for p, p_orig in zip(model.parameters(), orig_params):
                            p.data.copy_(p_orig)

                audit_steps.append(step)
                min_loss_range.append(min(step_losses))
                max_loss_range.append(max(step_losses))

            optimizer.step()

        if step > max_steps:
            break

    elapsed = time.time() - t0
    return {
        "loss_history": np.array(loss_history),
        "steps": np.array(audit_steps),
        "min_range": np.array(min_loss_range),
        "max_range": np.array(max_loss_range),
        "elapsed": elapsed
    }


def run_landscape_audit(
    models: Dict[str, nn.Module],
    train_loader: DataLoader,
    lr: float = 0.03,
    max_steps: int = 260,
    device: torch.device = None
) -> Dict[str, Dict[str, Any]]:
    """
    Executes comparative landscape training and gradient line-search audit across multiple models.
    """
    print(f"\n🚀 Initiating Comparative Landscape Audit on {device} ({len(models)} models)...")
    results = {}
    total_models = len(models)

    for i, (name, model) in enumerate(models.items(), 1):
        print(f"[{i}/{total_models}] Training {name}...")
        res = train_and_audit_single_model(model, train_loader, lr=lr, max_steps=max_steps, device=device)
        print(f"    ✓ {name} completed in {res['elapsed']:.1f}s | Final Loss: {res['loss_history'][-1]:.4f}")
        results[name] = res

    return results


def plot_landscape_smoothing_benchmark(
    audit_results: Dict[str, Dict[str, Any]],
    lr: float = 0.03
) -> None:
    """
    Renders the exact 3-panel Santurkar et al. (NeurIPS 2018, Figure 9) Landscape Smoothing benchmark:
    Panel 1: Theoretical 2D contour trajectory (kappa=30 ill-conditioned vs. kappa=2 conditioned).
    Panel 2: Empirical CIFAR-10 training loss trajectory (10-step rolling average).
    Panel 3: Loss range when stepping 0.25*lr ... 4.0*lr along -grad L (shaded envelope, log scale).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0), dpi=140)

    # --------------------------------------------------------------------------
    # PANEL 1: Theoretical Curvature Intuition
    # --------------------------------------------------------------------------
    ax1 = axes[0]
    x_grid = np.linspace(-2.0, 2.0, 300)
    y_grid = np.linspace(-2.0, 2.0, 300)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z_ill = 0.5 * (15.0 * X**2 + 1.0 * (Y - 0.3)**2)
    Z_well = 0.5 * (1.5 * X**2 + 1.0 * (Y - 0.3)**2)

    ax1.contour(X, Y, Z_ill, levels=[0.1, 0.3, 0.8, 1.8, 3.5, 6.0, 10.0], colors='#e74c3c', alpha=0.35, linewidths=0.9)
    ax1.contour(X, Y, Z_well, levels=[0.1, 0.3, 0.8, 1.8, 3.5], colors='#3498db', alpha=0.35, linewidths=0.9)

    n_osc = 18
    y_traj_ill = np.linspace(1.6, 0.32, n_osc)
    x_sign = np.array([(-1)**i for i in range(n_osc)])
    x_damp = 1.6 * (y_traj_ill / 1.6)**0.65
    x_traj_ill = np.append(x_sign * x_damp, [0.0])
    y_traj_ill = np.append(y_traj_ill, [0.3])

    n_smooth = 16
    y_traj_well = np.linspace(1.6, 0.3, n_smooth)
    x_traj_well = -1.6 * (y_traj_well - 0.3) / 1.3

    ax1.plot(x_traj_ill, y_traj_ill, '.-', color='#d62728', lw=1.2, ms=5, label=r'$\kappa = 30$ (no norm)')
    ax1.plot(x_traj_well, y_traj_well, 's-', color='#1f77b4', lw=2.0, ms=4, label=r'$\kappa = 2$ (normalised)')
    ax1.plot(0.0, 0.3, 'k*', ms=9, label='Local Min')
    ax1.set_title(r"$\mathbf{Intuition:}$ same $\eta$, different curvature $\beta$", fontsize=12)
    ax1.set_xlim(-2.0, 2.0)
    ax1.set_ylim(-2.0, 2.0)
    ax1.set_xlabel(r"$w_1$", fontsize=11)
    ax1.set_ylabel(r"$w_2$", fontsize=11)
    ax1.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # --------------------------------------------------------------------------
    # PANEL 2: CIFAR-10 Training Loss Trajectory
    # --------------------------------------------------------------------------
    ax2 = axes[1]
    def rolling_mean(arr, window=10):
        res = np.empty_like(arr)
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            res[i] = np.mean(arr[start:i+1])
        return res

    colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e']
    for idx, (name, res) in enumerate(audit_results.items()):
        steps = np.arange(1, len(res["loss_history"]) + 1)
        c = colors[idx % len(colors)]
        ax2.plot(steps, rolling_mean(res["loss_history"]), color=c, lw=2.2, label=name)

    ax2.set_title(rf"$\mathbf{{Training\ loss,}}$ SGD $\eta = {lr}$ (10-step avg)", fontsize=12)
    ax2.set_xlabel("step", fontsize=11)
    ax2.set_ylabel("cross-entropy", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # --------------------------------------------------------------------------
    # PANEL 3: Loss Range Along -grad L (Santurkar et al. 2018, Fig. 4a)
    # --------------------------------------------------------------------------
    ax3 = axes[2]
    alphas = [0.35, 0.50, 0.45, 0.40]
    for idx, (name, res) in enumerate(audit_results.items()):
        c = colors[idx % len(colors)]
        a = alphas[idx % len(alphas)]
        ax3.fill_between(res["steps"], res["min_range"], res["max_range"],
                         color=c, alpha=a, label=name)

    ax3.set_yscale("log")
    ax3.set_title(r"$\mathbf{Loss\ range\ when\ stepping\ 0.25\eta \dots 4\eta\ along\ -\nabla \mathcal{L}}$" + "\n(Santurkar et al. 2018, Fig. 4a)", fontsize=11)
    ax3.set_xlabel("step", fontsize=11)
    ax3.set_ylabel("loss (log)", fontsize=11)
    ax3.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
    ax3.grid(True, which="both", linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.show()


def compare_test_accuracies(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    device: torch.device = None
) -> None:
    """
    Evaluates top-1 test accuracy across multiple models and prints a clean benchmark table.
    """
    criterion = nn.CrossEntropyLoss()
    print("=" * 65)
    print("CIFAR-10 TOP-1 ACCURACY BENCHMARK ON TEST SET")
    print("=" * 65)
    for idx, (name, model) in enumerate(models.items(), 1):
        dev = device if device is not None else next(model.parameters()).device
        _, acc = evaluate_model(model, test_loader, criterion, dev)
        print(f"  {idx}. {name:<32}: {acc * 100:6.2f}%")
    print("=" * 65)

