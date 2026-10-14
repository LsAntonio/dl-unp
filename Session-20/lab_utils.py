"""
Session 20 Lab Utilities: Modern Data Augmentation & CNN Model Architecture
Course: UNP Deep Learning Foundations — Unit II (Week 7, Session 20)
Institutional Standard: MIT EECS / ETH Zürich D-INFK
Instructors: MSc. Antonio Aguilar & Dr. Luis Aguilar Ibáñez

This module encapsulates all auxiliary visualization, validation assertions,
and architectural diagram rendering for Session 20:
  1. plot_baseline_summary: Preprocessing visual anchor (Raw vs Scaled + Batch).
  2. Augmentation effect plots:
       - plot_crop_demonstration
       - plot_flip_demonstration
       - plot_rotation_demonstration
       - plot_color_jitter_demonstration
       - plot_normalize_demonstration
       - plot_augmented_realizations
  3. Transformation pipeline validation: validate_transforms
  4. CNN Layer verification & assertions:
       - check_layer: Shape contract verification for individual layers
       - verify_pipeline_shapes: Forward pass verification across layers 1-4 and classifier
       - verify_real_image_forward: Real image end-to-end forward test
  5. High-fidelity architectural visualization: visualize_catdog_architecture
  6. Final model forward pass validation: test_model_forward
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import visualtorch


# ==============================================================================
# 1. Baseline Preprocessing Visual Anchor
# ==============================================================================
def plot_baseline_summary(
    raw_cat_img: Image.Image,
    raw_dog_img: Image.Image,
    base_transform: Callable,
    train_loader: Any,
    mean_cpu: Sequence[float],
    std_cpu: Sequence[float],
    figsize: Tuple[int, int] = (15, 7)
) -> None:
    """
    Renders 2-row visual inspection of baseline preprocessed samples:
      Row 1: Raw PIL inputs vs. Scaled [0, 1] & Resized (224x224) tensors.
      Row 2: Standardized batch samples from DataLoader (denormalized to RGB).
    """
    cat_base = base_transform(raw_cat_img)
    dog_base = base_transform(raw_dog_img)
    preview_images, preview_labels = next(iter(train_loader))

    fig, axes = plt.subplots(2, 4, figsize=figsize, dpi=120)
    plt.suptitle("Baseline Pre-processing Anchor (Before Data Augmentation)", fontsize=13, fontweight="bold", y=0.98)

    # Row 1: Raw vs Base Transform
    axes[0, 0].imshow(raw_cat_img)
    axes[0, 0].set_title(f"1. Raw Cat Input\nResolution: {raw_cat_img.size} ({raw_cat_img.mode})", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cat_base.permute(1, 2, 0).numpy())
    axes[0, 1].set_title(f"2. Scaled Cat (224x224)\nRange: [{cat_base.min():.2f}, {cat_base.max():.2f}] | μ={cat_base.mean():.2f}", fontsize=10)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(raw_dog_img)
    axes[0, 2].set_title(f"3. Raw Dog Input\nResolution: {raw_dog_img.size} ({raw_dog_img.mode})", fontsize=10)
    axes[0, 2].axis("off")

    axes[0, 3].imshow(dog_base.permute(1, 2, 0).numpy())
    axes[0, 3].set_title(f"4. Scaled Dog (224x224)\nRange: [{dog_base.min():.2f}, {dog_base.max():.2f}] | μ={dog_base.mean():.2f}", fontsize=10)
    axes[0, 3].axis("off")

    # Row 2: Standardized Training Batch Samples
    mean_t = torch.tensor(mean_cpu).view(3, 1, 1)
    std_t = torch.tensor(std_cpu).view(3, 1, 1)
    class_names = ["Cat", "Dog"]

    for idx in range(min(4, len(preview_images))):
        img = preview_images[idx]
        label = preview_labels[idx].item()
        denorm_img = (img * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[1, idx].imshow(denorm_img)
        axes[1, idx].set_title(
            f"Batch [{idx}]: {class_names[label]} (Class {label})\n"
            f"Tensor: {tuple(img.shape)} | μ={img.mean():.2f}, σ={img.std():.2f}",
            fontsize=10
        )
        axes[1, idx].axis("off")

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 2. Augmentation Demonstration Utilities (Section 1.7)
# ==============================================================================
def plot_crop_demonstration(
    raw_cat_img: Image.Image,
    base_224: torch.Tensor,
    crop_transform: Callable,
    save_path: Optional[str] = "figures/aug_1_resized_crop.png"
) -> None:
    """Visualizes canonical 224x224 vs 3 stochastic RandomResizedCrop realizations."""
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), dpi=150)
    axes[0].imshow(base_224.permute(1, 2, 0).numpy())
    axes[0].set_title("Canonical (224x224)\nNo Crop", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    for i in range(1, 4):
        crop_t = crop_transform(raw_cat_img)
        axes[i].imshow(crop_t.permute(1, 2, 0).numpy())
        axes[i].set_title(f"Stochastic Crop #{i}\nScale: [0.8, 1.0]", fontsize=10)
        axes[i].axis("off")

    plt.suptitle("Impact of v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_flip_demonstration(
    raw_cat_img: Image.Image,
    base_224: torch.Tensor,
    flip_fn: Callable,
    save_path: Optional[str] = "figures/aug_2_horizontal_flip.png"
) -> None:
    """Visualizes original vs horizontally flipped image."""
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    axes[0].imshow(base_224.permute(1, 2, 0).numpy())
    axes[0].set_title("Original (Left-to-Right)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    flipped_t = flip_fn(base_224)
    axes[1].imshow(flipped_t.permute(1, 2, 0).numpy())
    axes[1].set_title("Horizontally Inverted ($p=1.0$)\nReflection Symmetry", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    plt.suptitle("Impact of v2.RandomHorizontalFlip(p=0.5)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_rotation_demonstration(
    raw_cat_img: Image.Image,
    base_224: torch.Tensor,
    save_path: Optional[str] = "figures/aug_3_rotation.png"
) -> None:
    """Visualizes canonical vs negative, positive, and random rotations."""
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), dpi=150)
    axes[0].imshow(base_224.permute(1, 2, 0).numpy())
    axes[0].set_title("Canonical (0°)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    rot_neg15 = v2.functional.rotate(base_224, angle=-15)
    rot_pos15 = v2.functional.rotate(base_224, angle=15)
    rot_rand = v2.RandomRotation(degrees=15)(base_224)

    axes[1].imshow(rot_neg15.permute(1, 2, 0).numpy())
    axes[1].set_title("Deterministic -15°\n(Tilt Left)", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(rot_pos15.permute(1, 2, 0).numpy())
    axes[2].set_title("Deterministic +15°\n(Tilt Right)", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(rot_rand.permute(1, 2, 0).numpy())
    axes[3].set_title("Stochastic Sample\n$\\theta \\sim \\mathcal{U}(-15^\\circ, 15^\\circ)$", fontsize=10)
    axes[3].axis("off")

    plt.suptitle("Impact of v2.RandomRotation(degrees=15)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_color_jitter_demonstration(
    raw_cat_img: Image.Image,
    base_224: torch.Tensor,
    jitter_fn: Callable,
    save_path: Optional[str] = "figures/aug_4_color_jitter.png"
) -> None:
    """Visualizes canonical vs 3 stochastic color jitter realizations."""
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), dpi=150)
    axes[0].imshow(base_224.permute(1, 2, 0).numpy())
    axes[0].set_title("Pristine RGB\n(No Jitter)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    for i in range(1, 4):
        jit_t = jitter_fn(base_224)
        axes[i].imshow(jit_t.permute(1, 2, 0).numpy())
        axes[i].set_title(f"Jitter Realization #{i}\nPhotometric Perturbation", fontsize=10)
        axes[i].axis("off")

    plt.suptitle("Impact of v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_normalize_demonstration(
    raw_cat_img: Image.Image,
    base_224: torch.Tensor,
    norm_fn: Callable,
    mean_cpu: Sequence[float],
    std_cpu: Sequence[float],
    save_path: Optional[str] = "figures/aug_5_normalize.png"
) -> None:
    """Visualizes raw scaled [0, 1] vs standardized Z-score vs histogram."""
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    norm_cat = norm_fn(base_224)
    # Scaled to [0, 1] for visualization of contrast
    norm_vis = (norm_cat - norm_cat.min()) / (norm_cat.max() - norm_cat.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    axes[0].imshow(base_224.permute(1, 2, 0).numpy())
    axes[0].set_title(f"1. Continuous Float [0, 1]\nMin: {base_224.min():.2f}, Max: {base_224.max():.2f}\nMean: {base_224.mean():.2f}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(norm_vis.permute(1, 2, 0).numpy())
    axes[1].set_title(f"2. Standardized Z-Score Tensor\nMin: {norm_cat.min():.2f}, Max: {norm_cat.max():.2f}\nMean: {norm_cat.mean():.2f} (Zero-Centered)", fontsize=10)
    axes[1].axis("off")

    # Histogram of standardized distribution
    axes[2].hist(norm_cat[0].flatten().numpy(), bins=40, color="crimson", alpha=0.5, label="Red Channel (Z)")
    axes[2].hist(norm_cat[1].flatten().numpy(), bins=40, color="forestgreen", alpha=0.5, label="Green Channel (Z)")
    axes[2].hist(norm_cat[2].flatten().numpy(), bins=40, color="royalblue", alpha=0.5, label="Blue Channel (Z)")
    axes[2].axvline(0, color="black", linestyle="--", linewidth=1.5, label="Zero-Mean Target")
    axes[2].set_title("3. Standardized Intensity Distribution", fontsize=10)
    axes[2].set_xlabel("Z-Score ($z = \\frac{x - \\mu}{\\sigma}$)")
    axes[2].set_ylabel("Pixel Frequency")
    axes[2].legend(fontsize=8)

    plt.suptitle("Impact of Terminal v2.Normalize (Empirical Dataset Z-Score Standardization)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_augmented_realizations(
    train_transforms: Callable,
    raw_cat_img: Image.Image,
    mean_cpu: Sequence[float],
    std_cpu: Sequence[float],
    num_realizations: int = 4
) -> None:
    """Visualizes multi-sample stochastic realizations of train_transforms."""
    mean_tensor = torch.tensor(mean_cpu).view(3, 1, 1)
    std_tensor = torch.tensor(std_cpu).view(3, 1, 1)

    fig, axes = plt.subplots(1, num_realizations, figsize=(16, 4), dpi=150)
    for idx in range(num_realizations):
        aug_t = train_transforms(raw_cat_img)
        aug_vis = (aug_t * std_tensor + mean_tensor).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[idx].imshow(aug_vis)
        axes[idx].set_title(f"Augmented Sample #{idx+1}\nRange: [{aug_t.min():.2f}, {aug_t.max():.2f}]", fontsize=10)
        axes[idx].axis("off")

    plt.suptitle("Stochastic train_transforms Verification (4 Dynamic Realizations)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def validate_transforms(
    train_transforms: Callable,
    eval_transforms: Callable,
    raw_cat_img: Image.Image
) -> None:
    """Automated assertions on transformation pipeline outputs."""
    test_train_out = train_transforms(raw_cat_img)
    test_eval_out = eval_transforms(raw_cat_img)

    assert test_train_out.shape == (3, 224, 224), f"Expected train shape (3, 224, 224), got {test_train_out.shape}"
    assert test_eval_out.shape == (3, 224, 224), f"Expected eval shape (3, 224, 224), got {test_eval_out.shape}"
    assert test_train_out.dtype == torch.float32, f"Expected float32, got {test_train_out.dtype}"
    assert test_eval_out.dtype == torch.float32, f"Expected float32, got {test_eval_out.dtype}"
    print("✓ Defensive assertions passed: Both pipelines adhere to the [3, 224, 224] float32 tensor contract.")


# ==============================================================================
# 3. CNN Layer Construction & Verification Helpers (Section 2)
# ==============================================================================
def check_layer(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
    layer_name: str = "Layer"
) -> None:
    """
    Defensive shape assertion helper for CNN layer construction.
    Validates tensor dimensions and prints formatted confirmation.
    """
    assert tensor.shape == expected_shape, (
        f"❌ {layer_name} shape mismatch: got {tuple(tensor.shape)}, expected {expected_shape}."
    )
    print(f"✓ {layer_name} assertion passed: output shape = {tuple(tensor.shape)}")


def verify_pipeline_shapes(
    h1: torch.Tensor,
    h2: torch.Tensor,
    h3: torch.Tensor,
    h4: torch.Tensor,
    logits_test: torch.Tensor
) -> None:
    """Verifies sequential forward pass dimensions across all four layers and classifier."""
    print(f"✓ After Layer 1 Output Shape:    {h1.shape} (Resolution halved: 224 -> 112)")
    print(f"✓ After Layer 2 Output Shape:    {h2.shape} (Resolution halved: 112 -> 56)")
    print(f"✓ After Layer 3 Output Shape:    {h3.shape} (Resolution halved: 56 -> 28)")
    print(f"✓ After Layer 4 Output Shape:    {h4.shape} (Resolution halved: 28 -> 14)")
    print(f"✓ After Classifier Logits Shape: {logits_test.shape} (Binary logits: [B, 2])")

    assert h1.shape == (4, 32, 112, 112), f"Layer 1 shape mismatch: got {h1.shape}, expected (4, 32, 112, 112)"
    assert h2.shape == (4, 64, 56, 56),   f"Layer 2 shape mismatch: got {h2.shape}, expected (4, 64, 56, 56)"
    assert h3.shape == (4, 128, 28, 28),  f"Layer 3 shape mismatch: got {h3.shape}, expected (4, 128, 28, 28)"
    assert h4.shape == (4, 256, 14, 14),  f"Layer 4 shape mismatch: got {h4.shape}, expected (4, 256, 14, 14)"
    assert logits_test.shape == (4, 2),   f"Classifier shape mismatch: got {logits_test.shape}, expected (4, 2)"


def verify_real_image_forward(
    raw_cat_img: Image.Image,
    base_transform: Callable,
    layer1: nn.Module,
    layer2: nn.Module,
    layer3: nn.Module,
    layer4: nn.Module,
    classifier: nn.Module,
    device: torch.device
) -> None:
    """Performs real image sample forward test through all modular layers."""
    cat_tensor_gpu = base_transform(raw_cat_img).unsqueeze(0).to(device)
    with torch.no_grad():
        cat_h1 = layer1(cat_tensor_gpu)
        cat_h2 = layer2(cat_h1)
        cat_h3 = layer3(cat_h2)
        cat_h4 = layer4(cat_h3)
        cat_logits = classifier(cat_h4)
        cat_probs = F.softmax(cat_logits, dim=-1)

    print(f"\nReal Image Test (raw_cat_img):")
    print(f"  Input Tensor:   {cat_tensor_gpu.shape} | Range: [{cat_tensor_gpu.min():.2f}, {cat_tensor_gpu.max():.2f}]")
    print(f"  Layer 4 Feat:   {cat_h4.shape} | Mean Activation: {cat_h4.mean():.4f}")
    print(f"  Output Logits:  {cat_logits.shape} -> [{cat_logits[0, 0].item():.4f}, {cat_logits[0, 1].item():.4f}]")
    print(f"  Probabilities:  P(Cat)={cat_probs[0, 0].item():.4f}, P(Dog)={cat_probs[0, 1].item():.4f}")

    assert cat_logits.shape == (1, 2), f"Real image logits mismatch: got {cat_logits.shape}"
    print("\n✓ Forward pass test verified: complete pipeline (Layers 1-4 + Classifier) is mathematically consistent!")


# ==============================================================================
# 4. High-Fidelity Architectural Visualization
# ==============================================================================
def find_perspective_coeffs(src_coords, dst_coords):
    matrix = []
    for (x, y), (u, v) in zip(dst_coords, src_coords):
        matrix.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        matrix.append([0, 0, 0, x, y, 1, -v*x, -v*y])
    A = np.array(matrix, dtype=float)
    B = np.array(src_coords).reshape(8)
    return tuple(np.linalg.solve(A, B))


def visualize_catdog_architecture(
    model: nn.Module,
    raw_cat_path: Union[str, Path] = "archive_v2/training_set/cat/cat.1.jpg",
    save_path: str = "figures/cat_dog_cnn_architecture.png",
    dpi: int = 150
) -> None:
    """
    Renders volumetric tensor flow diagram with inclined photograph input modality
    and saves publication asset to figures directory.
    """
    input_shape = (1, 3, 224, 224)
    orig_device = next(model.parameters()).device
    model.to("cpu")

    img_flow = visualtorch.render(
        model,
        input_shape=input_shape,
        style="flow",
        legend=True,
        legend_position="top-right",
        show_dimension=True,
        draw_volume=True,
        scale_xy=1.4,
        scale_z=0.22,
        spacing=18,
        draw_funnel=True,
        shade_step=10,
        outline_width=1,
        palette="material",
        show_input=False
    )

    sample_path = Path(raw_cat_path)
    if not sample_path.exists():
        fallback = list(Path("archive_v2").glob("**/cat.1.jpg"))
        if fallback:
            sample_path = fallback[0]

    if sample_path.exists():
        raw_img = Image.open(sample_path).convert("RGBA")
        h_target = 314
        w_target = 314
        raw_resized = raw_img.resize((w_target, h_target), Image.Resampling.LANCZOS)

        incline_dx = 35
        incline_dy = 24
        src = [(0, 0), (w_target, 0), (w_target, h_target), (0, h_target)]
        dst = [
            (10, incline_dy),
            (w_target - incline_dx, incline_dy * 2),
            (w_target - incline_dx, h_target - incline_dy * 2),
            (10, h_target - incline_dy)
        ]
        coeffs = find_perspective_coeffs(src, dst)
        warped = raw_resized.transform((w_target, h_target), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)

        draw_warped = ImageDraw.Draw(warped)
        draw_warped.polygon(dst, outline="#2D3748", width=2)

        gap = 70
        pad_left = 35
        total_w = pad_left + w_target + gap + img_flow.width
        total_h = img_flow.height

        composite_img = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(composite_img)

        img_x = pad_left
        img_y = 104

        composite_img.paste(warped, (img_x, img_y), mask=warped)

        flow_x = img_x + w_target + gap
        composite_img.paste(img_flow, (flow_x, 0))

        right_top = (img_x + dst[1][0], img_y + dst[1][1])
        right_bot = (img_x + dst[2][0], img_y + dst[2][1])
        right_mid = (right_top[0], (right_top[1] + right_bot[1]) // 2)

        layer1_x = flow_x + 37
        layer1_top = (layer1_x, 104)
        layer1_bot = (layer1_x, 418)
        layer1_mid = (layer1_x, (104 + 418) // 2)

        funnel_pen = "#2B6CB0"
        draw.line([right_top, layer1_top], fill=funnel_pen, width=1)
        draw.line([right_bot, layer1_bot], fill=funnel_pen, width=1)

        draw.line([(right_mid[0] + 6, right_mid[1]), (layer1_x - 6, layer1_mid[1])], fill=funnel_pen, width=3)
        draw.polygon([
            (layer1_x - 4, layer1_mid[1]),
            (layer1_x - 16, layer1_mid[1] - 7),
            (layer1_x - 16, layer1_mid[1] + 7)
        ], fill=funnel_pen)

        try:
            font_bold = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Bold.ttf", 13)
            font_reg = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 12)
        except Exception:
            font_bold = ImageFont.load_default()
            font_reg = font_bold

        t1 = "Input Image (Cat)"
        t2 = "(1, 3, 224, 224)"
        w1 = font_bold.getbbox(t1)[2] - font_bold.getbbox(t1)[0]
        w2 = font_reg.getbbox(t2)[2] - font_reg.getbbox(t2)[0]

        text_y = img_y + h_target + 12
        draw.text((img_x + (w_target - w1) // 2, text_y), t1, fill="#1A202C", font=font_bold)
        draw.text((img_x + (w_target - w2) // 2, text_y + 18), t2, fill="#4A5568", font=font_reg)

        final_render = composite_img
    else:
        final_render = img_flow

    fig, ax = plt.subplots(figsize=(final_render.width / dpi, final_render.height / dpi), dpi=dpi)
    ax.imshow(final_render)
    ax.axis("off")
    plt.title("CatDogCNN Macro-Architecture with Input Modality", fontsize=12, pad=15, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    # Restore model to its active computation device
    model.to(orig_device)


# ==============================================================================
# 5. Final Model Forward Pass Test Helper
# ==============================================================================
def test_model_forward(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 8
) -> None:
    """Verifies CatDogCNN forward propagation with dummy tensor on active compute device."""
    model.to(device)
    model.eval()
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)

    with torch.no_grad():
        dummy_logits = model(dummy_input)
        dummy_probs = F.softmax(dummy_logits, dim=-1)

    print(f"Dummy Input Tensor Shape:  {dummy_input.shape} on {dummy_input.device}")
    print(f"Dummy Output Logits Shape: {dummy_logits.shape} on {dummy_logits.device}")
    print(f"Dummy Probabilities Shape: {dummy_probs.shape}")

    assert dummy_logits.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {dummy_logits.shape}"
    assert torch.allclose(dummy_probs.sum(dim=-1), torch.ones(batch_size, device=device), atol=1e-5), \
        "Probabilities must sum to 1.0 along class dimension"

    print("\nSample Output Logits & Softmax Probabilities:")
    for i in range(min(4, batch_size)):
        cat_p = dummy_probs[i, 0].item()
        dog_p = dummy_probs[i, 1].item()
        print(f"  Sample [{i}]: Logits=[{dummy_logits[i, 0].item():7.4f}, {dummy_logits[i, 1].item():7.4f}] | "
              f"P(Cat)={cat_p:.4f}, P(Dog)={dog_p:.4f} (Sum: {cat_p + dog_p:.4f})")

    print("\n✓ All architectural layer assertions and forward dummy activation tests passed successfully on GPU!")
