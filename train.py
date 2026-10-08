"""
Main training script. Run this to:
  1. Download / generate the dataset
  2. Train the ResNet-18 model
  3. Evaluate on the test set
  4. Save plots and metrics to outputs/
"""
import os
import sys
import logging
import torch

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import SEED, LOG_DIR, OUTPUT_DIR, CHECKPOINT, get_device
from src.dataset import download_kaggle_dataset, get_dataloaders
from src.model import build_model, load_checkpoint
from src.train import train
from src.evaluate import full_evaluation

# ─── Logging ──────────────────────────────────────────────────────────────────
os.makedirs(LOG_DIR,    exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "training.log")),
    ]
)
logger = logging.getLogger(__name__)


def main():
    torch.manual_seed(SEED)
    logger.info("=" * 60)
    logger.info("  Chest X-Ray Disease Detection — Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"PyTorch  : {torch.__version__}")
    logger.info(f"Device   : {get_device()}")

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    logger.info("\n[1/3] Preparing dataset…")
    download_kaggle_dataset()
    train_loader, val_loader, test_loader = get_dataloaders()

    # ── 2. Train ──────────────────────────────────────────────────────────────
    logger.info("\n[2/3] Training model…")
    model   = build_model(pretrained=True, freeze_backbone=False)
    history, best_acc = train(model, train_loader, val_loader)

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    logger.info("\n[3/3] Evaluating on test set…")
    test_model = build_model(pretrained=False)
    test_model = load_checkpoint(test_model, CHECKPOINT)
    metrics    = full_evaluation(test_model, test_loader, history)

    logger.info("\n" + "=" * 60)
    logger.info("  Training complete!")
    logger.info(f"  Best val_acc : {best_acc:.4f}")
    logger.info(f"  Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Test AUC-ROC : {metrics['roc_auc']:.4f}")
    logger.info(f"  Test F1      : {metrics['f1_score']:.4f}")
    logger.info(f"  Results saved to: {OUTPUT_DIR}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
