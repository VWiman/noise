from pathlib import Path


# ============================================================
# Dataset
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
VALID_EXTENSIONS = (".jpg", ".jpeg")

# Procentuell uppdelning på originalbildsnivå.
TRAIN_PERCENT = 80
VAL_PERCENT = 10
TEST_PERCENT = 10


# ============================================================
# Träningsinställningar
# ============================================================

RANDOM_SEED = 42
TILE_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 4
NOISE_FACTOR = 0.25
LEARNING_RATE = 0.0001


# ============================================================
# Output-mappar
# ============================================================

OUTPUT_DIR = PROJECT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoint"
EDA_DIR = OUTPUT_DIR / "eda"
RESULTS_DIR = OUTPUT_DIR / "results"

CHECKPOINT_PATH = CHECKPOINT_DIR / "best_denoising_unet.keras"
EDA_OVERVIEW_PATH = EDA_DIR / "dataset_overview.png"
EDA_SAMPLES_PATH = EDA_DIR / "sample_train_tiles.png"
EDA_SUMMARY_PATH = EDA_DIR / "eda_summary.txt"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.png"
TILE_RECONSTRUCTIONS_PATH = RESULTS_DIR / "reconstructions.png"
WHOLE_IMAGE_RECONSTRUCTIONS_PATH = (
    RESULTS_DIR / "whole_image_reconstructions.png"
)
