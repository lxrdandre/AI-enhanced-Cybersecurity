# Configuration for base training and transfer learning

SEED = 42

# Base training
TARGET_K = 25
BATCH_SIZE = 1024
MAX_EPOCHS = 100
DROP_IPS = True
BASE_MODEL_TYPE = "se_dwnet"  # options: "se_dwnet", "resnet_mlp"

# Data
DATA_CSV_RELATIVE = "data/train_test_network.csv"
DROP_TYPES = ["mitm", "ransomware"]

# Artifacts
BASE_ARTIFACT_SUBDIR = "se_dwnet_base"

# Loss settings
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA_CLIP = (0.5, 5.0)

# Feature self-attention
USE_FEATURE_MHA = False
MHA_HEADS = 4
MHA_KEY_DIM = 8
MHA_DROPOUT = 0.0

LOG_COLS = [
    "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
    "http_request_body_len", "http_response_body_len", "missed_bytes",
]

CAT_COLS = [
    "proto", "service", "conn_state", "dns_query", "dns_qclass",
    "dns_qtype", "dns_rcode", "http_user_agent", "ssl_version",
    "ssl_cipher", "http_method", "http_version", "src_port", "dst_port",
]

# Transfer learning (all classes)
TRANSFER_FREEZE_EPOCHS = 5
TRANSFER_MAX_EPOCHS = 50
TRANSFER_LR_FROZEN = 1e-3
TRANSFER_LR_FINETUNE = 1e-4
TRANSFER_BATCH_SIZE = 1024
