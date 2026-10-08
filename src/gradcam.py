"""
Grad-CAM visualization for chest X-ray predictions.
Implements manual Grad-CAM using PyTorch hooks (no external library needed).
"""
import io
import logging
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from src.config import IMAGE_SIZE, MEAN, STD, CLASS_NAMES, get_device

logger = logging.getLogger(__name__)

# ─── Inference transform ──────────────────────────────────────────────────────
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017)."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """Generate Grad-CAM heatmap for the given input."""
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)

        # Normalise
        cam -= cam.min()
        denom = cam.max()
        if denom > 0:
            cam /= denom

        cam_np = cam.squeeze().cpu().numpy()
        return cam_np, class_idx


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess a PIL image for inference."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    return INFER_TRANSFORM(image).unsqueeze(0)


def overlay_heatmap(original_image: Image.Image, cam: np.ndarray,
                    alpha: float = 0.45) -> Image.Image:
    """Overlay Grad-CAM heatmap on the original image."""
    # Resize CAM to match image
    img_arr = np.array(original_image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS
    ))
    cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap     = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * img_arr).astype(np.uint8)
    return Image.fromarray(overlay)


def predict_with_gradcam(model, image: Image.Image, device=None):
    """
    Run inference + Grad-CAM on a PIL image.

    Returns:
        dict with keys: class_name, confidence, gradcam_image (PIL)
    """
    if device is None:
        device = get_device()

    model = model.to(device).eval()

    # Hook into last conv layer of ResNet-18
    target_layer = model.layer4[-1].conv2
    gradcam      = GradCAM(model, target_layer)

    input_tensor = preprocess_image(image).to(device)

    with torch.enable_grad():
        cam, class_idx = gradcam.generate(input_tensor)

    # Confidence
    with torch.no_grad():
        output     = model(input_tensor)
        probs      = torch.softmax(output, dim=1)[0]
        confidence = probs[class_idx].item()

    class_name  = CLASS_NAMES[class_idx]
    gradcam_img = overlay_heatmap(image, cam)

    return {
        "class_name":  class_name,
        "class_idx":   class_idx,
        "confidence":  round(confidence * 100, 2),
        "probabilities": {
            CLASS_NAMES[0]: round(probs[0].item() * 100, 2),
            CLASS_NAMES[1]: round(probs[1].item() * 100, 2),
        },
        "gradcam_image": gradcam_img,
    }


def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    """Convert PIL image to bytes."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()
