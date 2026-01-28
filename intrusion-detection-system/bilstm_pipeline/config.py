# Configuration for BiLSTM training and transfer learning

SEED = 42

# Base training
TARGET_K = 25
BATCH_SIZE = 1024
MAX_EPOCHS = 100
DROP_IPS = True

# Data
DATA_CSV_RELATIVE = "data/train_test_network.csv"
DROP_TYPES = ["mitm", "ransomware"]

# Artifacts
BASE_ARTIFACT_SUBDIR = "bilstm_smote_base"

# Feature scaling
LOG_COLS = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "src_pkts",
    "dst_pkts",
    "http_request_body_len",
    "http_response_body_len",
    "missed_bytes",
]

# SMOTE
SMOTE_K_NEIGHBORS_MAX = 5

# Transfer learning
TRANSFER_BATCH_SIZE = 1024
TRANSFER_MAX_EPOCHS = 50
TRANSFER_FREEZE_EPOCHS = 5
TRANSFER_LR_FROZEN = 1e-3
TRANSFER_LR_FINETUNE = 1e-4
TRANSFER_ARTIFACT_SUBDIR = "bilstm_transfer_8class"
