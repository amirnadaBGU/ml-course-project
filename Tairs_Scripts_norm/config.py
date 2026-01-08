from pathlib import Path

# Paths to body-frame normalized keypoints
TRAIN_XLSX = Path("/train_on_all/keypoints_body_frame.xlsx")
TEST_XLSX = Path("/Users/taircarmon/Desktop/ml-course-project/prawn_2025_circ_small_v1/keypoints_body_frame.xlsx")

# Base output directory for all model runs in this normalized-keypoint workspace
OUTPUT_BASE = Path(__file__).resolve().parent / "models_csv" / "models"

# Keypoint indices (should match how the body-frame Excel encodes keypoint_index)
KP_CARAPACE = 0
KP_EYES = 1
KP_ROSTRUM = 2
KP_TAIL = 3

NUM_KPTS = 4
MIN_VIS = 1

