"""
ResNet-18 model with transfer learning for chest X-ray classification.
"""
import torch
import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES, CHECKPOINT, get_device


def build_model(pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """
    Build a ResNet-18 model fine-tuned for binary classification.

    Args:
        pretrained:      Load ImageNet pretrained weights.
        freeze_backbone: Freeze all layers except the final FC layer.

    Returns:
        model (nn.Module): The modified ResNet-18.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final FC layer
    num_ftrs    = model.fc.in_features
    model.fc    = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )

    return model


def load_checkpoint(model: nn.Module, path: str = CHECKPOINT,
                    device: torch.device = None) -> nn.Module:
    """Load model weights from checkpoint."""
    if device is None:
        device = get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def save_checkpoint(model: nn.Module, optimizer, epoch: int,
                    val_acc: float, path: str = CHECKPOINT):
    """Save model checkpoint."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "val_acc":          val_acc,
    }, path)
