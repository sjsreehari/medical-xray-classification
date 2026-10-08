"""
Model evaluation: metrics, confusion matrix, ROC/PR curves.
"""
import os
import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, classification_report
)

from src.config import OUTPUT_DIR, CLASS_NAMES, get_device

logger = logging.getLogger(__name__)


def evaluate(model, test_loader, device=None):
    """Run inference on test_loader and return all predictions."""
    if device is None:
        device = get_device()

    model.eval().to(device)
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)[:, 1]   # P(PNEUMONIA)
            preds   = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_prob):
    """Compute and return all evaluation metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score":    float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "roc_auc":     float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "true_negatives":  int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives":  int(tp),
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_dir=OUTPUT_DIR):
    """Plot and save confusion matrix."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {path}")
    return path


def plot_roc_curve(y_true, y_prob, save_dir=OUTPUT_DIR):
    """Plot and save ROC curve."""
    os.makedirs(save_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved → {path}")
    return path


def plot_pr_curve(y_true, y_prob, save_dir=OUTPUT_DIR):
    """Plot and save Precision-Recall curve."""
    os.makedirs(save_dir, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#DD8452", lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "pr_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"PR curve saved → {path}")
    return path


def plot_training_history(history, save_dir=OUTPUT_DIR):
    """Plot training/validation loss and accuracy curves."""
    os.makedirs(save_dir, exist_ok=True)
    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    train_acc  = [h["train_acc"]  for h in history]
    val_acc    = [h["val_acc"]    for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, train_loss, label="Train", color="#4C72B0")
    ax1.plot(epochs, val_loss,   label="Val",   color="#DD8452")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_acc, label="Train", color="#4C72B0")
    ax2.plot(epochs, val_acc,   label="Val",   color="#DD8452")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def full_evaluation(model, test_loader, history=None):
    """Run complete evaluation and save all plots/metrics."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()

    y_true, y_pred, y_prob = evaluate(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)

    # Save metrics JSON
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Plots
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_prob)
    plot_pr_curve(y_true, y_prob)
    if history:
        plot_training_history(history)

    logger.info("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:20s}: {v:.4f}")
        else:
            logger.info(f"  {k:20s}: {v}")

    return metrics
