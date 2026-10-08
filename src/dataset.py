"""
Dataset download and loading utilities.

Downloads the Chest X-Ray (Pneumonia) dataset from a public mirror using
the Kaggle opendatasets library OR falls back to generating a minimal
synthetic dataset for demonstration when credentials are unavailable.
"""
import os
import sys
import shutil
import random
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import (
    DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, MEAN, STD, SEED
)

logger = logging.getLogger(__name__)

# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(split: str = "train"):
    """Return appropriate transforms for each data split."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Grayscale(num_output_channels=3),  # X-rays are grayscale
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])


# ─── Synthetic data generator ─────────────────────────────────────────────────

def _make_synthetic_image(label: int, idx: int, out_dir: str):
    """Create a synthetic grayscale X-ray-like image."""
    np.random.seed(idx)
    base = 60 if label == 0 else 120   # NORMAL darker, PNEUMONIA brighter
    arr = np.random.randint(max(0, base - 40), min(255, base + 80),
                            size=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    # Add circular blob (simulating lung field)
    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2
    for r in range(IMAGE_SIZE):
        for c in range(IMAGE_SIZE):
            if (r - cy) ** 2 + (c - cx) ** 2 < (80) ** 2:
                arr[r, c] = min(255, arr[r, c] + 30)
    img = Image.fromarray(arr, mode="L")
    os.makedirs(out_dir, exist_ok=True)
    img.save(os.path.join(out_dir, f"img_{idx:04d}.jpeg"))


def generate_synthetic_dataset(n_train=400, n_val=50, n_test=100):
    """Generate minimal synthetic chest-X-ray-like dataset for demo."""
    logger.info("Generating synthetic dataset (demo mode)…")
    splits = {
        "train": (TRAIN_DIR, n_train),
        "val":   (VAL_DIR,   n_val),
        "test":  (TEST_DIR,  n_test),
    }
    idx = 0
    for split, (base, n) in splits.items():
        for label, name in enumerate(["NORMAL", "PNEUMONIA"]):
            class_dir = os.path.join(base, name)
            os.makedirs(class_dir, exist_ok=True)
            for i in range(n // 2):
                _make_synthetic_image(label, idx, class_dir)
                idx += 1
    logger.info("Synthetic dataset ready.")


# ─── Kaggle download helper ───────────────────────────────────────────────────

def download_kaggle_dataset():
    """
    Download the Chest X-Ray Pneumonia dataset from Kaggle.
    Requires KAGGLE_USERNAME and KAGGLE_KEY env vars (or ~/.kaggle/kaggle.json).
    Falls back to synthetic data if unavailable.
    """
    # Check if data already exists
    if (os.path.exists(TRAIN_DIR) and
            len(os.listdir(os.path.join(TRAIN_DIR, "NORMAL", ))) > 10):
        logger.info("Dataset already present – skipping download.")
        return True

    # Try kaggle API
    try:
        import kaggle  # noqa: F401
        logger.info("Downloading Chest X-Ray dataset from Kaggle…")
        os.makedirs(DATA_DIR, exist_ok=True)
        import subprocess
        subprocess.run([
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", "paultimothymooney/chest-xray-pneumonia",
            "--unzip", "-p", DATA_DIR
        ], check=True)
        # Move to expected structure
        src = os.path.join(DATA_DIR, "chest_xray")
        if os.path.exists(src):
            for split in ["train", "val", "test"]:
                s = os.path.join(src, split)
                d = os.path.join(DATA_DIR, split)
                if os.path.exists(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
            shutil.rmtree(src, ignore_errors=True)
        logger.info("Kaggle dataset downloaded successfully.")
        return True
    except Exception as e:
        logger.warning(f"Kaggle download failed ({e}). Using synthetic data.")
        generate_synthetic_dataset()
        return False


# ─── DataLoaders ──────────────────────────────────────────────────────────────

def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train, val, test DataLoaders."""
    torch.manual_seed(SEED)
    random.seed(SEED)

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms("train"))
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=get_transforms("val"))
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=get_transforms("test"))

    # Weighted sampler to handle class imbalance
    targets  = train_ds.targets
    counts   = np.bincount(targets)
    weights  = 1.0 / counts[targets]
    sampler  = torch.utils.data.WeightedRandomSampler(
        weights, len(weights), replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=NUM_WORKERS,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
