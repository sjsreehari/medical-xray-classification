"""
Configuration settings for Chest X-Ray Disease Detection System
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")
TRAIN_DIR     = os.path.join(DATA_DIR, "train")
VAL_DIR       = os.path.join(DATA_DIR, "val")
TEST_DIR      = os.path.join(DATA_DIR, "test")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")
LOG_DIR       = os.path.join(BASE_DIR, "logs")
CHECKPOINT    = os.path.join(MODEL_DIR, "best_model.pth")

# ─── Model / Training ─────────────────────────────────────────────────────────
IMAGE_SIZE    = 224
BATCH_SIZE    = 32
NUM_WORKERS   = 0          # 0 for Windows compatibility
NUM_EPOCHS    = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 5          # Early stopping patience
NUM_CLASSES   = 2
CLASS_NAMES   = ["NORMAL", "PNEUMONIA"]

# ─── Normalisation (ImageNet stats) ──────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ─── API ─────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ─── Misc ─────────────────────────────────────────────────────────────────────
SEED = 42
