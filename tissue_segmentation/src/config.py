# --- Data ---
TRAIN_DATA_PATH = "annotations/big_tile/tiles/tiles_train.npz"
VAL_DATA_PATH   = "annotations/big_tile/tiles/tiles_val.npz"

# --- Model ---
TILE_SIZE = 256
IN_CHANNELS = 3
NUM_CLASSES = 10
CONVNEXT_VARIANT = "tiny"  # "tiny", "base", or "large"
PRETRAINED = True

# --- Training ---
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 100
BATCH_SIZE = 4
NUM_WORKERS = 4

# --- Scheduler ---
WARMUP_EPOCHS = 5
MIN_LR = 1e-6

# --- Augmentation ---
RANDOM_FLIP = True
RANDOM_ROTATE = True
COLOR_JITTER_BRIGHTNESS = 0.3
COLOR_JITTER_CONTRAST = 0.3
COLOR_JITTER_SATURATION = 0.3
COLOR_JITTER_HUE = 0.1
GAUSSIAN_BLUR_PROB = 0.3
ELASTIC_TRANSFORM_PROB = 0.2

# --- Early Stopping ---
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 1e-4

# --- Misc ---
SEED = 67
MIXED_PRECISION = True
GRADIENT_CLIP_VAL = 1.0
LOG_INTERVAL = 10
SAVE_DIR = "checkpoints"
