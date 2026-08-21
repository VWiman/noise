from pathlib import Path


# ============================================================
# Dataset
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
STAGE_TWO_DATASET_DIR = PROJECT_DIR / "dataset_stage_two"
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
BATCH_SIZE = 20
STAGE_ONE_EPOCHS = 200
STAGE_TWO_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 5
MIN_NOISE = 0.10
MAX_NOISE = 0.30
VAL_NOISE = 0.20
LEARNING_RATE = 0.0001


# ============================================================
# Output-mappar
# ============================================================

OUTPUT_DIR = PROJECT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoint"
EDA_DIR = OUTPUT_DIR / "eda"
RESULTS_DIR = OUTPUT_DIR / "results"

STAGE_ONE_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_stage_one_unet.keras"
STAGE_TWO_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_stage_two_residual_unet.keras"
STAGE_TWO_MANIFEST_PATH = STAGE_TWO_DATASET_DIR / "manifest.csv"
EDA_OVERVIEW_PATH = EDA_DIR / "dataset_overview.png"
EDA_SAMPLES_PATH = EDA_DIR / "sample_train_tiles.png"
EDA_SUMMARY_PATH = EDA_DIR / "eda_summary.txt"
EVALUATION_SUMMARY_PATH = RESULTS_DIR / "evaluation_summary.txt"
METRIC_COMPARISON_PATH = RESULTS_DIR / "metric_comparison.png"
TILE_METRICS_PATH = RESULTS_DIR / "tile_metrics.csv"
WHOLE_IMAGE_METRICS_PATH = RESULTS_DIR / "whole_image_metrics.csv"
STAGE_ONE_HISTORY_DATA_PATH = RESULTS_DIR / "stage_one_training_history.csv"
STAGE_TWO_HISTORY_DATA_PATH = RESULTS_DIR / "stage_two_training_history.csv"
STAGE_ONE_HISTORY_PATH = RESULTS_DIR / "stage_one_training_history.png"
STAGE_TWO_HISTORY_PATH = RESULTS_DIR / "stage_two_training_history.png"
TILE_RECONSTRUCTIONS_PATH = RESULTS_DIR / "two_stage_tile_reconstructions.png"
WHOLE_IMAGE_RECONSTRUCTIONS_PATH = (
    RESULTS_DIR / "two_stage_whole_image_reconstructions.png"
)
