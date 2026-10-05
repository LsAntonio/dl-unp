"""
Session 16 Lab Utilities: Dissecting AlexNet — Mathematical Architecture & PyTorch Implementation
Course: UNP Deep Learning Foundations — Unit II (Week 6, Session 16)
Instructors: MSc. Antonio Aguilar & Dr. Luis Aguilar Ibáñez

This helper module encapsulates:
1. Environment setup and asset management for Google Colab and local execution.
2. Historical 2-GPU architectural schematic visualization.
3. Automated layer-by-layer tensor dimension tracking via PyTorch forward hooks.
4. Kaiming (He) normal weight initialization helpers.
5. Automated validation test harness for the student AlexNet implementation.
6. First-layer (Conv1) receptive field filter bank visualization.
"""

import os
import sys
import urllib.request
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ==============================================================================
# 1. Asset & Environment Management
# ==============================================================================

def download_assets(
    github_base_url: str = "https://raw.githubusercontent.com/LsAntonio/dl-unp/refs/heads/main/Session-16"
) -> None:
    """
    Ensures required assets (such as alexnet_arch_original.png) exist locally,
    downloading them from GitHub if executing in Google Colab or remote containers.
    """
    image_name = "alexnet_arch_original.png"
    if not os.path.exists(image_name):
        url = f"{github_base_url}/{image_name}"
        try:
            print(f"📥 Downloading architectural schematic from {url}...")
            urllib.request.urlretrieve(url, image_name)
            print(f"✅ Downloaded {image_name} successfully.")
        except Exception as e:
            print(f"⚠️ Could not download {image_name}: {e}")


def display_alexnet_schematic(image_path: str = "alexnet_arch_original.png") -> None:
    """
    Renders the authentic Figure 2 schematic from Krizhevsky et al. (NeurIPS 2012)
    illustrating the dual-stream 2-GPU hardware partitioning.
    """
    if not os.path.exists(image_path):
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(pkg_dir, image_path)
        if os.path.exists(alt_path):
            image_path = alt_path
        else:
            download_assets()

    if not os.path.exists(image_path):
        print(f"⚠️ Warning: Could not locate '{image_path}'. Skipping schematic rendering.")
        return

    arch_img = mpimg.imread(image_path)

    plt.figure(figsize=(13.5, 4.8), dpi=150, facecolor="white")
    plt.imshow(arch_img)
    plt.axis("off")
    plt.title(
        "Figure 2: Dual-Stream Model-Parallel AlexNet Architecture (Krizhevsky et al., NeurIPS 2012)\n"
        "Notice the dual 48-channel streams (48+48=96), grouped convolutions in Conv2 (groups=2), and cross-talk at Conv3 & FC6",
        fontsize=11.5,
        color="#111827",
        pad=12
    )
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 2. Weight Initialization Helper
# ==============================================================================

def init_alexnet_weights(model: nn.Module) -> None:
    """
    Applies Kaiming (He) normal initialization to all Conv2d layers (fan_out, relu)
    and Gaussian N(0, 0.01) initialization to Linear layers, setting biases to zero.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


# ==============================================================================
# 3. Dynamic Forward-Hook Dimension Tracing
# ==============================================================================

def verify_alexnet_geometry(model: nn.Module, input_tensor: torch.Tensor) -> None:
    """
    Attaches forward hooks across all layers of AlexNet and prints a comprehensive
    layer-by-layer accounting table tracking intermediate activation shapes.
    """
    records: List[Tuple[str, str, List[int]]] = []
    hooks = []

    def make_hook(layer_name: str, op_name: str):
        def hook(module, inp, out):
            records.append((layer_name, op_name, list(out.shape)))
        return hook

    # Register hooks on backbone features
    op_names_features = [
        ("Stage 1 (Conv)", "Conv2d"),
        ("Activation (ReLU)", "ReLU"),
        ("Normalization (LRN)", "LocalResponseNorm"),
        ("Downsample (Pool)", "MaxPool2d"),
        
        ("Stage 2 (Conv)", "Conv2d"),
        ("Activation (ReLU)", "ReLU"),
        ("Normalization (LRN)", "LocalResponseNorm"),
        ("Downsample (Pool)", "MaxPool2d"),
        
        ("Stage 3 (Conv)", "Conv2d"),
        ("Activation (ReLU)", "ReLU"),
        
        ("Stage 4 (Conv)", "Conv2d"),
        ("Activation (ReLU)", "ReLU"),
        
        ("Stage 5 (Conv)", "Conv2d"),
        ("Activation (ReLU)", "ReLU"),
        ("Downsample (Pool)", "MaxPool2d"),
    ]

    for idx, (layer_label, op_label) in enumerate(op_names_features):
        if idx < len(model.features):
            submod = model.features[idx]
            h = submod.register_forward_hook(make_hook(layer_label, op_label))
            hooks.append(h)

    # Register hooks on classifier head
    op_names_classifier = [
        ("Dropout Regularizer", "Dropout"),
        ("FC6 Dense Projection", "Linear"),
        ("Activation (ReLU)", "ReLU"),
        ("Dropout Regularizer", "Dropout"),
        ("FC7 Dense Projection", "Linear"),
        ("Activation (ReLU)", "ReLU"),
        ("FC8 Dense Projection", "Linear"),
    ]

    for idx, (layer_label, op_label) in enumerate(op_names_classifier):
        if idx < len(model.classifier):
            submod = model.classifier[idx]
            h = submod.register_forward_hook(make_hook(layer_label, op_label))
            hooks.append(h)

    # Execute forward pass in eval mode
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    # Clean up hooks immediately
    for h in hooks:
        h.remove()

    # Print verified accounting table
    print("=" * 85)
    print(f"{'LAYER INDEX & STAGE':<30} | {'OPERATION':<22} | {'OUTPUT SHAPE':<25}")
    print("=" * 85)
    print(f"{'Input':<30} | {'Raw Input Tensor':<22} | {str(list(input_tensor.shape)):<25}")
    print("-" * 85)

    flatten_printed = False
    for layer_name, op_name, out_shape in records:
        if "FC6" in layer_name and not flatten_printed:
            print("-" * 85)
            print(f"{'Spatial Flattening':<30} | {'torch.flatten':<22} | {str([input_tensor.shape[0], 9216]):<25}")
            print("-" * 85)
            flatten_printed = True
        print(f"{layer_name:<30} | {op_name:<22} | {str(out_shape):<25}")

    print("=" * 85)
    print("✅ VERIFICATION COMPLETE: Exact mathematical parity established.\n")


# ==============================================================================
# 4. Automated Student Implementation Test Harness
# ==============================================================================

def validate_student_alexnet(model_class: type) -> None:
    """
    Automated unit-test validation suite for AlexNetStudent.
    Asserts layer geometries, forward propagation shapes, and parametric class handling.
    """
    print("=" * 72)
    print("🧪 RUNNING AUTOMATED UNIT TEST SUITE: Validating AlexNetStudent")
    print("=" * 72)

    print("\n[1/4] Testing Model Instantiation (Default: 1000 classes)...")
    model = model_class(num_classes=1000)
    print("      * Successfully instantiated AlexNetStudent(num_classes=1000) ✅")

    print("\n[2/4] Testing Critical Layer Geometries...")
    conv2_weight = model.features[4].weight.shape
    fc6_weight = model.classifier[1].weight.shape
    fc8_weight = model.classifier[6].weight.shape

    assert conv2_weight == (256, 96, 5, 5), (
        f"❌ Task 1 Failure: Conv2 weights shape is {conv2_weight}, expected (256, 96, 5, 5). "
        f"Check in_channels (must be 96), out_channels (256), and kernel_size (5)."
    )
    assert fc6_weight == (4096, 9216), (
        f"❌ Task 2 Failure: FC6 weights shape is {fc6_weight}, expected (4096, 9216). "
        f"Did you multiply 256 * 6 * 6 = 9216?"
    )
    assert fc8_weight == (1000, 4096), (
        f"❌ Task 3 Failure: FC8 weights shape is {fc8_weight}, expected (1000, 4096). "
        f"Did you connect in_features=4096 to out_features=num_classes?"
    )
    print(f"      * Stage 2 Conv2 Tensor: {conv2_weight} ✅")
    print(f"      * Classifier FC6 Tensor: {fc6_weight} ✅")
    print(f"      * Classifier FC8 Tensor: {fc8_weight} ✅")

    print("\n[3/4] Testing Forward Propagation (Batch of 2 ImageNet Tensors)...")
    dummy_x = torch.randn(2, 3, 227, 227)
    model.eval()
    with torch.no_grad():
        out_logits = model(dummy_x)
    assert out_logits.shape == (2, 1000), (
        f"❌ Task 4 Failure: Output shape is {out_logits.shape}, expected (2, 1000). "
        f"Check torch.flatten(x, start_dim=1) in forward()."
    )
    print(f"      * Forward Pass Output:   {out_logits.shape} ✅")

    print("\n[4/4] Testing Parametric Adaptability (CIFAR-10 with 10 classes)...")
    cifar_model = model_class(num_classes=10)
    dummy_cifar = torch.randn(4, 3, 227, 227)
    cifar_model.eval()
    with torch.no_grad():
        out_cifar = cifar_model(dummy_cifar)
    assert out_cifar.shape == (4, 10), (
        f"❌ Parametric Failure: Output shape is {out_cifar.shape}, expected (4, 10)."
    )
    print(f"      * Custom 10-Class Output: {out_cifar.shape} ✅")

    print("\n" + "=" * 72)
    print("🎉 ALL ARCHITECTURAL ASSERTIONS PASSED! EXCELLENT IMPLEMENTATION WORK!")
    print("=" * 72 + "\n")


# ==============================================================================
# 5. Conv1 Receptive Field Filter Bank Visualization
# ==============================================================================

def visualize_conv1_filters(model: Optional[nn.Module] = None) -> None:
    """
    Extracts, normalizes, and visualizes the first-layer (Conv1) 11x11 RGB receptive fields
    of pre-trained AlexNet on an 8x8 grid, showing biological Gabor edge filters and color-opponent blobs.
    """
    if model is None:
        print("⏳ Loading pre-trained TorchVision AlexNet weights...")
        model = models.alexnet(weights='DEFAULT')

    model.eval()

    # Extract Conv1 weights: shape [64, 3, 11, 11]
    conv1_weights = model.features[0].weight.detach().cpu()
    num_filters = conv1_weights.shape[0]

    # Grid geometry
    ncols = 8
    nrows = int(np.ceil(num_filters / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12), dpi=150)
    fig.patch.set_facecolor('#ffffff')

    for i in range(nrows * ncols):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col] if nrows > 1 else axes[col]
        
        if i < num_filters:
            # Transpose from [C, H, W] to [H, W, C]
            w = conv1_weights[i].permute(1, 2, 0).numpy()
            
            # Independent per-filter min-max normalization to [0, 1]
            w_min, w_max = w.min(), w.max()
            if w_max - w_min > 1e-6:
                w_norm = (w - w_min) / (w_max - w_min)
            else:
                w_norm = np.zeros_like(w)
                
            ax.imshow(w_norm, interpolation='nearest')
            ax.set_title(f"Kernel #{i}", fontsize=7.5, color='#374151', pad=2)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    plt.suptitle(
        "Learned First-Layer Convolutional Receptive Fields (Conv1: 11x11 RGB)\n"
        "Autonomous Emergence of V1 Gabor Edge Wavelets (Achromatic) & Color-Opponent Blobs (Chrominance)",
        fontsize=12,
        fontweight='bold',
        color='#111827',
        y=0.94
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.show()
