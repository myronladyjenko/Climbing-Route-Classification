import os

# ----------------------------- #
#          DATA PATHS           #
# ----------------------------- # 
ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_DATA_YAML = os.path.join(ROOT_DIRECTORY, "datasets_combined", "data.yaml")
RAW_IMAGES_DIR = os.path.join(ROOT_DIRECTORY, "raw_images")
PREPROCESSED_IMAGES_DIR = os.path.join(ROOT_DIRECTORY, "preprocessed_images")
MODELS_DIR = os.path.join(ROOT_DIRECTORY, "models")
DETECTION_OUTPUT_DIR = os.path.join(ROOT_DIRECTORY, "detection_outputs")
COMBINED_DATASET_ROOT = os.path.join(ROOT_DIRECTORY, "datasets_combined")

# ----------------------------- #
#            TRAINING           #
# ----------------------------- # 
# Overwrite number of  in the code (detection.py) when passing to a function
NUM_EPOCHS = 20
IMAGE_SIZE = 640
# Value of 0 - GPU
DEVICE = 'cpu' 
BASE_MODEL_NAME = "yolo_hold_detector_"

# Different YOLO models to finetune 
BASE_YOLO_MODEL_n = "yolo11n.pt"
BASE_YOLO_MODEL_s = "yolo11s.pt"
BASE_YOLO_MODEL_m = "yolo11m.pt"

# ----------------------------- #
# FUTURE WORK: ROUTE GENERATION #
# ----------------------------- # 
