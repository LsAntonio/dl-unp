import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris, load_digits


# ============================================================================
# Section 1.3: XOR Decision Boundary Visualization Helper
# ============================================================================
def plot_xor_decision_boundary(lin_model: nn.Module = None, mlp_model: nn.Module = None,
                               X_xor: torch.Tensor = None, y_xor: torch.Tensor = None,
                               epochs: int = 500, lr: float = 0.1, verbose: bool = True):
    """
    Trains (if models not provided) and plots side-by-side 2D decision boundaries
    for Linear vs Non-Linear models on the classic XOR problem.
    """
    # 1. Define synthetic XOR dataset (4 ground-truth coordinates) if not provided
    if X_xor is None:
        X_xor = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    if y_xor is None:
        y_xor = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    def _quick_train(model, X, y, n_epochs=epochs, step_lr=lr):
        opt = torch.optim.Adam(model.parameters(), lr=step_lr)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(n_epochs):
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()

    # 2. Multi-Layer Deep Linear Model (No Activation Functions)
    if lin_model is None:
        lin_model = nn.Sequential(
            nn.Linear(2, 8),
            nn.Linear(8, 8),
            nn.Linear(8, 1)
        )
        _quick_train(lin_model, X_xor, y_xor)

    # 3. Non-Linear Model with ReLU activations
    if mlp_model is None:
        mlp_model = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        _quick_train(mlp_model, X_xor, y_xor)

    # 4. Compare predictions on XOR coordinates
    with torch.no_grad():
        pred_lin = torch.sigmoid(lin_model(X_xor)).round().numpy().flatten()
        pred_nonlin = torch.sigmoid(mlp_model(X_xor)).round().numpy().flatten()

    if verbose:
        print("--- Predictions on XOR Data ---")
        print("Target XOR Labels:        \n", y_xor.numpy().flatten())
        print("Linear Model Predictions: \n", pred_lin)
        print("Non-Linear Model Predictions:\n", pred_nonlin)

    # 5. Plot decision boundaries
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        zz_lin = torch.sigmoid(lin_model(grid)).reshape(xx.shape).numpy()
        zz_mlp = torch.sigmoid(mlp_model(grid)).reshape(xx.shape).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Linear Model Boundary
    axes[0].scatter(X_xor[y_xor.squeeze() == 0, 0], X_xor[y_xor.squeeze() == 0, 1],
                    color="royalblue", s=100, edgecolors="k", label="Class 0 (0,0 / 1,1)")
    axes[0].scatter(X_xor[y_xor.squeeze() == 1, 0], X_xor[y_xor.squeeze() == 1, 1],
                    color="crimson", s=100, edgecolors="k", label="Class 1 (0,1 / 1,0)")
    
    if (zz_lin.max() - zz_lin.min()) > 0.05:
        axes[0].contourf(xx, yy, zz_lin, levels=np.linspace(0.0, 1.0, 50), cmap="coolwarm", alpha=0.6, vmin=0.0, vmax=1.0)
        axes[0].contour(xx, yy, zz_lin, levels=[0.5], colors="black", linewidths=2, linestyles="--")
    else:
        axes[0].contourf(xx, yy, zz_lin, levels=np.linspace(0.0, 1.0, 50), cmap="coolwarm", alpha=0.6, vmin=0.0, vmax=1.0)

    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    axes[0].set_title("Deep Linear Model: Cannot Solve XOR\n(Decision surface is completely flat)",
                      fontsize=11, fontweight="bold", color="crimson")
    axes[0].set_xlabel("Feature $x_1$")
    axes[0].set_ylabel("Feature $x_2$")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Non-Linear Model Boundary
    axes[1].scatter(X_xor[y_xor.squeeze() == 0, 0], X_xor[y_xor.squeeze() == 0, 1],
                    color="royalblue", s=100, edgecolors="k", label="Class 0 (0,0 / 1,1)")
    axes[1].scatter(X_xor[y_xor.squeeze() == 1, 0], X_xor[y_xor.squeeze() == 1, 1],
                    color="crimson", s=100, edgecolors="k", label="Class 1 (0,1 / 1,0)")
    axes[1].contourf(xx, yy, zz_mlp, levels=np.linspace(0.0, 1.0, 50), cmap="coolwarm", alpha=0.6, vmin=0.0, vmax=1.0)
    axes[1].contour(xx, yy, zz_mlp, levels=[0.5], colors="black", linewidths=2, linestyles="--")
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)
    axes[1].set_title("Non-Linear MLP (with ReLU): Solves XOR Perfectly\n(Non-linear decision boundary)",
                      fontsize=11, fontweight="bold", color="forestgreen")
    axes[1].set_xlabel("Feature $x_1$")
    axes[1].set_ylabel("Feature $x_2$")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


# ============================================================================
# Section 2.1: Activation & Gradient Curves Plotting Helper
# ============================================================================
def plot_activation_curves(x: torch.Tensor = None, activations: dict = None, gradients: dict = None):
    """Plots 4 activation functions and their corresponding analytical/autograd derivatives."""
    if x is None:
        x = torch.linspace(-4.0, 4.0, 500, requires_grad=True)

    if activations is None:
        activations = {
            "Sigmoid":   torch.sigmoid(x),
            "Tanh":      torch.tanh(x),
            "ReLU":      F.relu(x),
            "LeakyReLU": F.leaky_relu(x, negative_slope=0.1)
        }

    if gradients is None:
        gradients = {}
        for name, act_val in activations.items():
            if x.grad is not None:
                x.grad.zero_()
            act_val.sum().backward(retain_graph=True)
            gradients[name] = x.grad.clone()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    colors = ["royalblue", "darkorange", "forestgreen", "crimson"]

    x_np = x.detach().numpy()
    for idx, (name, act_val) in enumerate(activations.items()):
        ax = axes[idx]
        grad_val = gradients[name]
        
        ax.plot(x_np, act_val.detach().numpy(), label=f"{name} Activation $\\sigma(x)$",
                color=colors[idx], linewidth=2.5)
        ax.plot(x_np, grad_val.detach().numpy(), label=f"{name} Derivative $\\sigma'(x)$",
                color="black", linestyle="--", linewidth=2.0)
        
        ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.7)
        ax.set_title(f"{name} Dynamics", fontsize=12, fontweight="bold")
        ax.set_xlabel("Input $x$")
        ax.set_ylabel("Value / Gradient")
        ax.set_ylim(-1.2, 2.2)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.show()


# ============================================================================
# Section 2.2: Vanishing Gradient Flow Simulation & Plotting Helper
# ============================================================================
def plot_vanishing_gradient_dynamics(depth: int = 12, dim: int = 64, verbose: bool = True):
    """
    Simulates backpropagation through deep Sigmoid vs. ReLU networks
    and plots layer-wise gradient magnitudes and signal retention.
    """
    torch.manual_seed(42)
    sigmoid_layers = []
    relu_layers = []

    for _ in range(depth):
        sigmoid_layers.extend([nn.Linear(dim, dim), nn.Sigmoid()])
        relu_layers.extend([nn.Linear(dim, dim), nn.ReLU()])

    sigmoid_net = nn.Sequential(*sigmoid_layers)
    relu_net = nn.Sequential(*relu_layers)

    for s_m, r_m in zip(sigmoid_net, relu_net):
        if isinstance(s_m, nn.Linear):
            nn.init.xavier_uniform_(s_m.weight)
            nn.init.zeros_(s_m.bias)
            r_m.weight.data.copy_(s_m.weight.data)
            r_m.bias.data.copy_(s_m.bias.data)

    x_deep = torch.randn(32, dim)
    out_sig = sigmoid_net(x_deep).sum()
    out_relu = relu_net(x_deep).sum()

    out_sig.backward()
    out_relu.backward()

    sig_grads = []
    relu_grads = []

    for s_m, r_m in zip(sigmoid_net, relu_net):
        if isinstance(s_m, nn.Linear):
            sig_grads.append(s_m.weight.grad.norm().item())
            relu_grads.append(r_m.weight.grad.norm().item())

    if verbose:
        print(f"Layer 1 Gradient Norm ({depth}-layer Sigmoid Net): {sig_grads[0]:.2e}  <-- VANISHED!")
        print(f"Layer 1 Gradient Norm ({depth}-layer ReLU Net):    {relu_grads[0]:.2e}  <-- HEALTHY!")
        ratio = relu_grads[0] / (sig_grads[0] + 1e-15)
        print(f"Gradient flow is {ratio:.0f}x stronger with ReLU!\n")

    # Trace layers in backward direction: from output (Layer 12) down to input (Layer 1)
    layers = np.arange(depth, 0, -1)
    sig_flipped = list(reversed(sig_grads))
    relu_flipped = list(reversed(relu_grads))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(layers, sig_flipped, "o--", color="crimson", linewidth=2.5, markersize=7,
            label=f"{depth}-Layer Sigmoid Net (Vanishing)")
    ax.plot(layers, relu_flipped, "s-", color="forestgreen", linewidth=2.5, markersize=7,
            label=f"{depth}-Layer ReLU Net (Stable Flow)")

    ax.set_yscale("log")
    ax.set_xlim(depth + 0.5, 0.5)  # Invert x-axis to flow left-to-right from output to input
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}\n(Output)" if i == depth else (f"L{i}\n(Input)" if i == 1 else f"L{i}") for i in layers])

    ax.set_xlabel(f"Layer Index (Tracing Backward: Output Layer {depth} $\\to$ Input Layer 1)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Weight Gradient Norm $\|\nabla_W \mathcal{L}\|_2$ (Log Scale)", fontsize=11, fontweight="bold")
    ax.set_title("Backpropagation Flow: Gradient Magnitude from Output to Input", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Shaded vanished region for earliest layers (e.g. Layers 3, 2, 1)
    ax.axvspan(3.5, 0.5, color="crimson", alpha=0.12, label="Vanished Region (Near-Zero Update)")

    # Directional arrow indicating backpropagation flow
    ax.annotate("", xy=(1.5, 1200), xytext=(11.5, 1200),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="navy"))
    ax.text(6.5, 1800, "Direction of Backpropagation Flow (Chain Rule: Loss $\\to$ Input)",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="navy")

    ax.set_ylim(1e-6, 5000)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Section 6.1: Multi-Layer Activation Variance Plotting Helper
# ============================================================================
def plot_layer_activation_dynamics(std_bad: list, std_xavier: list, std_kaiming: list, n_layers: int = 10):
    """Plots activation standard deviation across deep layers for different initializations."""
    plt.figure(figsize=(9, 5))
    layers = list(range(n_layers + 1))
    plt.plot(layers, std_bad, 'o--', color="crimson", label="Bad Init: N(0, 1) + Tanh (Signal Saturated)", linewidth=2)
    plt.plot(layers, std_xavier, 's-', color="royalblue", label="Xavier Init + Tanh (Stable)", linewidth=2)
    plt.plot(layers, std_kaiming, '^-.', color="forestgreen", label="Kaiming Init + ReLU (Stable)", linewidth=2)
    plt.yscale("log")
    plt.title("Activation Standard Deviation Across 10 Layers", fontsize=12, fontweight="bold")
    plt.xlabel("Layer Index")
    plt.ylabel("Activation Std (Log Scale)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.show()


# ============================================================================
# Section 7: Challenge 1 (Iris Flower Dataset) Helpers with Decision Surface
# ============================================================================
def load_iris_data():
    """Loads the Iris dataset and returns (X_iris, y_iris, feature_names, target_names)."""
    iris = load_iris()
    X = torch.tensor(iris.data, dtype=torch.float32)
    y = torch.tensor(iris.target, dtype=torch.long)
    return X, y, iris.feature_names, iris.target_names


def train_and_evaluate_iris(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int = 150, lr: float = 0.03):
    """Trains an Iris model, verifies accuracy, and renders 2D class decision boundary contour."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(X).argmax(dim=-1)
        acc = (preds == y).float().mean().item()

    # 2D Decision Boundary Contour across Petal Length & Petal Width
    mean_feat0 = X[:, 0].mean().item()
    mean_feat1 = X[:, 1].mean().item()

    x_min, x_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
    y_min, y_max = X[:, 3].min() - 0.5, X[:, 3].max() + 0.5
    xm, ym = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    grid_f2 = xm.ravel()
    grid_f3 = ym.ravel()
    grid_f0 = np.full_like(grid_f2, mean_feat0)
    grid_f1 = np.full_like(grid_f2, mean_feat1)
    grid_pts = torch.tensor(np.column_stack([grid_f0, grid_f1, grid_f2, grid_f3]), dtype=torch.float32)

    with torch.no_grad():
        grid_preds = model(grid_pts).argmax(dim=-1).reshape(xm.shape).numpy()

    plt.figure(figsize=(7, 5))
    cmap_bg = ListedColormap(["#dbeafe", "#dcfce7", "#fee2e2"])
    plt.contourf(xm, ym, grid_preds, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap_bg, alpha=0.8)
    plt.contour(xm, ym, grid_preds, levels=[0.5, 1.5], colors="black", linewidths=1.5, linestyles="--")

    colors = ["royalblue", "forestgreen", "crimson"]
    species = ["setosa", "versicolor", "virginica"]
    for i, sp in enumerate(species):
        mask = (y.numpy() == i)
        plt.scatter(X[mask, 2], X[mask, 3], label=f"Class {i}: {sp.capitalize()}", color=colors[i], edgecolors="k", s=50)

    plt.title(f"Iris Decision Boundaries (Accuracy: {acc*100:.1f}%)", fontsize=12, fontweight="bold")
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

    return acc


# ============================================================================
# Section 7: Challenge 2 (2D Crescent Moon Dataset) Helpers
# ============================================================================
def generate_moon_data(n_samples: int = 400, noise: float = 0.08):
    """Generates synthetic 2D interlocking crescent moon dataset."""
    t = np.linspace(0, np.pi, n_samples // 2)
    x1 = np.cos(t) + np.random.normal(0, noise, len(t))
    y1 = np.sin(t) + np.random.normal(0, noise, len(t))
    x2 = 1 - np.cos(t) + np.random.normal(0, noise, len(t))
    y2 = 0.5 - np.sin(t) + np.random.normal(0, noise, len(t))

    X_np = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y_np = np.hstack([np.zeros(len(t)), np.ones(len(t))])

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    return X, y


def train_and_evaluate_moon(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int = 300, lr: float = 0.03):
    """Trains a Moon classifier, computes accuracy, and renders 2D decision boundary contour."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(X).argmax(dim=-1)
        acc = (preds == y).float().mean().item()

    # Visualizing decision surface
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xm, ym = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_pts = torch.tensor(np.c_[xm.ravel(), ym.ravel()], dtype=torch.float32)

    with torch.no_grad():
        probs = F.softmax(model(grid_pts), dim=-1)[:, 1].reshape(xm.shape).numpy()

    plt.figure(figsize=(7, 5))
    plt.contourf(xm, ym, probs, levels=50, cmap="coolwarm", alpha=0.7)
    plt.contour(xm, ym, probs, levels=[0.5], colors="black", linewidths=2, linestyles="--")
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="royalblue", edgecolors="k", label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="crimson", edgecolors="k", label="Class 1")
    plt.title(f"Moon Dataset Classification (Accuracy: {acc*100:.1f}%)", fontsize=12, fontweight="bold")
    plt.xlabel("Feature $x_1$")
    plt.ylabel("Feature $x_2$")
    plt.colorbar(label="Predicted $P(y=1)$")
    plt.legend()
    plt.show()

    return acc


# ============================================================================
# Section 7: Challenge 3 (8x8 Digits Dataset) Helpers
# ============================================================================
def load_digits_data():
    """Loads 8x8 handwritten digits dataset normalized to [0, 1]."""
    digits = load_digits()
    X = torch.tensor(digits.data / 16.0, dtype=torch.float32)
    y = torch.tensor(digits.target, dtype=torch.long)
    return X, y, digits.images


def train_and_evaluate_digits(model: nn.Module, X: torch.Tensor, y: torch.Tensor, images: np.ndarray, epochs: int = 150, lr: float = 0.01):
    """Trains Digits classifier, verifies accuracy, and renders 2x5 sample digit gallery."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(X).argmax(dim=-1)
        acc = (preds == y).float().mean().item()

    # Visualizing 2x5 Gallery of Predictions
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        img = images[i]
        true_label = y[i].item()
        pred_label = preds[i].item()
        
        ax.imshow(img, cmap="gray_r", interpolation="nearest")
        is_correct = (true_label == pred_label)
        ax.set_title(f"True: {true_label} | Pred: {pred_label}",
                     color="forestgreen" if is_correct else "crimson",
                     fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    plt.suptitle(f"8x8 Handwritten Digits Model Predictions (Accuracy: {acc*100:.1f}%)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.subplots_adjust(top=0.88, bottom=0.08, hspace=0.4, wspace=0.3)
    plt.show()

    return acc
