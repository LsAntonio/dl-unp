"""
Autograd Visualizer & Experiment Helpers for Deep Learning Foundations
Zero-dependency PyTorch dynamic computational graph visualizer using native Mermaid.
Works out-of-the-box on Windows, macOS, Linux, and Google Colab with no C++ Graphviz binaries needed.
"""

import torch
import matplotlib.pyplot as plt
from IPython.display import display, Markdown


def show_autograd_graph(root, params=None, orientation="TD"):
    """
    Renders an unfolded dynamic computational graph in Jupyter Notebook / Google Colab
    using native Mermaid flowchart syntax (Zero external binary dependencies).

    Parameters
    ----------
    root : torch.Tensor
        The output tensor (e.g. loss or prediction) whose grad_fn graph should be traced.
    params : dict, optional
        A dictionary mapping parameter and input names to tensors, e.g.
        {'w': w, 'b': b, 'x': x, 'y': y}.
    orientation : str, default 'TD'
        Orientation of the flowchart. 'TD' (top-down vertical) or 'LR' (left-to-right horizontal).
    """
    lines = [
        "```mermaid",
        "%%{init: {'theme': 'dark', 'themeVariables': {'fontSize': '14px', 'lineColor': '#94a3b8', 'edgeLabelBackground': '#0f172a'}, 'flowchart': {'nodeSpacing': 22, 'rankSpacing': 36}}}%%",
        f"flowchart {orientation}",
        "    classDef param fill:#7c3aed,stroke:#a78bfa,stroke-width:2.5px,color:#fff;",
        "    classDef data fill:#0284c7,stroke:#38bdf8,stroke-width:2.5px,color:#fff;",
        "    classDef op fill:#1e293b,stroke:#64748b,stroke-width:2.5px,color:#fff;",
        "    classDef loss fill:#e11d48,stroke:#fb7185,stroke-width:2.5px,color:#fff;"
    ]

    seen = set()
    param_lookup = {}
    non_grad_inputs = {}

    if params:
        for name, t in params.items():
            if isinstance(t, torch.Tensor):
                if t.requires_grad:
                    param_lookup[t.data_ptr()] = (name, t)
                else:
                    non_grad_inputs[name] = t

    def trace(fn):
        if fn in seen or fn is None:
            return
        seen.add(fn)
        fn_id = f"fn_{abs(id(fn))}"
        raw_name = type(fn).__name__.replace('Backward0', '').replace('Backward', '')

        if 'AccumulateGrad' in type(fn).__name__:
            var = getattr(fn, 'variable', None)
            name = 'Parameter'
            val_str = ''
            if var is not None:
                val_str = f" = {var.item():.2f}" if var.numel() == 1 else f" {list(var.shape)}"
                if var.data_ptr() in param_lookup:
                    name = param_lookup[var.data_ptr()][0]
            label = f"{name}{val_str}<br/>(Learnable Parameter)"
            lines.append(f'    {fn_id}["{label}"]:::param')
            return

        label_map = {
            'Add': 'Add (+)<br/>z = u + b',
            'Mul': 'Multiply (×)<br/>u = w · x',
            'Pow': 'Power (^2)<br/>loss = error²',
            'Sub': 'Subtract (-)<br/>error = ŷ - y',
            'Div': 'Divide (/)',
            'Exp': 'Exp (e^z)',
            'Relu': 'ReLU',
            'Sigmoid': 'Sigmoid (σ)'
        }
        label = label_map.get(raw_name, raw_name)
        lines.append(f'    {fn_id}["{label}"]:::op')

        for idx, (next_fn, _) in enumerate(fn.next_functions):
            if next_fn is not None:
                next_id = f"fn_{abs(id(next_fn))}"
                lines.append(f"    {next_id} --> {fn_id}")
                trace(next_fn)

    if hasattr(root, 'grad_fn') and root.grad_fn is not None:
        # Add non-grad input boxes
        if 'x' in non_grad_inputs:
            lines.append(f'    inp_x["x = {non_grad_inputs["x"].item():.2f}<br/>(Input Feature)"]:::data')
        if 'y' in non_grad_inputs or 'target' in non_grad_inputs:
            y_val = (non_grad_inputs.get('y') or non_grad_inputs.get('target')).item()
            lines.append(f'    inp_y["y = {y_val:.2f}<br/>(Target Label)"]:::data')

        trace(root.grad_fn)
        root_id = f"fn_{abs(id(root.grad_fn))}"

        # Connect non-grad inputs to their operations
        for fn in seen:
            if 'Mul' in type(fn).__name__ and 'x' in non_grad_inputs:
                lines.append(f'    inp_x --> fn_{abs(id(fn))}')
            if 'Sub' in type(fn).__name__ and ('y' in non_grad_inputs or 'target' in non_grad_inputs):
                lines.append(f'    inp_y --> fn_{abs(id(fn))}')

        lines.append(f'    loss_out["Loss = {root.item():.2f}<br/><b>Total Scalar Error</b>"]:::loss')
        lines.append(f"    {root_id} --> loss_out")

    lines.append("```")
    display(Markdown("\n".join(lines)))


def plot_learning_rate_experiments(x, y_true, learning_rates=None, epochs=40):
    """
    Renders a 1x4 multi-panel dashboard comparing Loss curves and prediction line
    movements across different learning rate regimes.

    Parameters
    ----------
    x : torch.Tensor
        Input feature vector.
    y_true : torch.Tensor
        Ground truth target vector.
    learning_rates : list of float, optional
        Learning rates to compare. Defaults to [0.005, 0.1, 1.8].
    epochs : int, default 40
        Number of training epochs.
    """
    if learning_rates is None:
        learning_rates = [0.005, 0.1, 1.8]

    labels = ["Too Small (lr=0.005)", "Just Right (lr=0.1)", "Too Large (lr=1.8)"]
    colors = ["#fb7185", "#10b981", "#f43f5e"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))

    # 1. Loss Curves across Epochs
    for lr, label, color in zip(learning_rates, labels, colors):
        w_test = torch.tensor([0.0], requires_grad=True)
        b_test = torch.tensor([0.0], requires_grad=True)
        history = []

        for epoch in range(epochs):
            y_hat = w_test * x + b_test
            loss = torch.mean((y_hat - y_true) ** 2)
            history.append(loss.item())

            loss.backward()
            with torch.no_grad():
                w_test -= lr * w_test.grad
                b_test -= lr * b_test.grad
            w_test.grad = None
            b_test.grad = None

        axes[0].plot(history, label=label, color=color, linewidth=2.0)

    axes[0].set_title("1. Loss vs. Epochs", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_ylim(0, 15)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    # 2. Prediction Line Progressions for each Learning Rate
    track_epochs = [1, max(2, epochs // 8), max(3, epochs // 3), epochs]
    line_styles = [':', '--', '-.', '-']
    alphas = [0.45, 0.65, 0.85, 1.0]

    for idx, (lr, label, color) in enumerate(zip(learning_rates, labels, colors), start=1):
        w_test = torch.tensor([0.0], requires_grad=True)
        b_test = torch.tensor([0.0], requires_grad=True)

        # Scatter data cloud
        axes[idx].scatter(x.numpy(), y_true.numpy(), color="#38bdf8", alpha=0.35, s=15, label="Data")

        for epoch in range(1, epochs + 1):
            y_hat = w_test * x + b_test
            loss = torch.mean((y_hat - y_true) ** 2)

            if epoch in track_epochs:
                ep_idx = track_epochs.index(epoch)
                axes[idx].plot(
                    x.numpy(),
                    y_hat.detach().numpy(),
                    color=color,
                    linestyle=line_styles[ep_idx],
                    alpha=alphas[ep_idx],
                    linewidth=2.0 if epoch == epochs else 1.2,
                    label=f"Epoch {epoch}"
                )

            loss.backward()
            with torch.no_grad():
                w_test -= lr * w_test.grad
                b_test -= lr * b_test.grad
            w_test.grad = None
            b_test.grad = None

        axes[idx].set_title(f"{label}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("x")
        axes[idx].set_ylabel("y")
        axes[idx].set_ylim(-6, 8)
        axes[idx].legend(fontsize=7.5, loc="upper left")
        axes[idx].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()
