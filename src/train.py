"""
Training loop with early stopping, LR scheduling, and TensorBoard logging.
"""
import os
import time
import logging
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.config import (
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE,
    OUTPUT_DIR, LOG_DIR, CHECKPOINT, CLASS_NAMES
)
from src.model import save_checkpoint

logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)
    return running_loss / total, correct / total


def train(model, train_loader, val_loader):
    """Full training pipeline with early stopping."""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")
    model = model.to(device)

    # Class weights for imbalanced data
    class_counts = torch.tensor([1.0, 3.0])   # PNEUMONIA is ~3x more
    class_weights = (1.0 / class_counts).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    writer     = SummaryWriter(log_dir=LOG_DIR)
    best_acc   = 0.0
    no_improve = 0
    history    = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        elapsed = time.time() - t0

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Acc",  {"train": train_acc,  "val": val_acc},  epoch)

        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,    4),
        }
        history.append(rec)
        logger.info(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_acc:
            best_acc   = val_acc
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_acc, CHECKPOINT)
            logger.info(f"  ✓ Best model saved (val_acc={best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    writer.close()

    # Save history
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training done. Best val_acc={best_acc:.4f}")
    return history, best_acc
