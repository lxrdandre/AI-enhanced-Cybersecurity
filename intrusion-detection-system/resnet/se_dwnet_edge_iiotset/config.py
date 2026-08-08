"""Configuration constants for the Edge-IIoTset SE-DWNet training package."""

from __future__ import annotations

import os

TARGET_CLASSES = ("backdoor", "dos_ddos", "injection", "normal", "password", "scanning")

DEFAULT_ARTIFACT_DIR_NAME = "se_dwnet_edge_iiotset_random_holdout"
MODEL_FILENAME = "se_dwnet_model.keras"
LEGACY_MODEL_FILENAME = "resnet_model.keras"
PIPELINE_FILENAME = "preprocessing_pipeline.pkl"
FEATURES_FILENAME = "final_features.txt"


def project_root() -> str:
    env_root = os.environ.get("TON_IOT_PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    cwd = os.path.abspath(os.getcwd())
    if os.path.isdir(os.path.join(cwd, "data")) and os.path.isdir(os.path.join(cwd, "artifacts")):
        return cwd
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_output_dir() -> str:
    return os.path.join(project_root(), "artifacts", DEFAULT_ARTIFACT_DIR_NAME)
