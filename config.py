import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models trained on the "basic" 9-column feature set -- these are the
# ONLY NSL-KDD features that can realistically be recomputed from a raw
# live packet capture without a proper flow-meter tool. This is the
# model used for real-time scoring.
MODEL_DIR = os.path.join(BASE_DIR, "network_models_basic")

# Models trained on the full 41-column NSL-KDD feature set. Useful as an
# offline/legacy comparison in the "model performance" tab, but NOT used
# for live traffic since most of these features (num_failed_logins,
# dst_host_srv_rerror_rate, etc.) require a dedicated flow-extraction
# tool (e.g. CICFlowMeter) to compute from raw packets.
MODEL_DIR_FULL = os.path.join(BASE_DIR, "network_models_full")

BASIC_FEATURE_COLS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
NUMERIC_COLS = ["duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent"]

# Live pipeline directories
STREAM_INPUT_DIR = os.path.join(BASE_DIR, "stream_input")         # producer drops flow batches here
STREAM_CHECKPOINT_DIR = os.path.join(BASE_DIR, "stream_checkpoint")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "predictions.jsonl")  # Streamlit reads this
MAX_PREDICTIONS_KEPT = 5000

# NSL-KDD training/test files -- put these next to the scripts,
# download from https://www.unb.ca/cic/datasets/nsl.html
TRAIN_FILE = os.path.join(BASE_DIR, "KDDTrain+.txt")
TEST_FILE = os.path.join(BASE_DIR, "KDDTest+.txt")
