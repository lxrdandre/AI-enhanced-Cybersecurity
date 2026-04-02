from __future__ import annotations

import os
import pickle
import sys

import tensorflow as tf

from app.preprocessing import SafeLabelEncoder


def _register_pickle_compat_aliases() -> None:
    """Register legacy class symbols needed for loading old pickle artifacts."""
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "SafeLabelEncoder"):
        setattr(main_module, "SafeLabelEncoder", SafeLabelEncoder)

    uvicorn_main = sys.modules.get("uvicorn.__main__")
    if uvicorn_main is not None and not hasattr(uvicorn_main, "SafeLabelEncoder"):
        setattr(uvicorn_main, "SafeLabelEncoder", SafeLabelEncoder)


def _load_features_from_text(path: str) -> list[str]:
    with open(path, "r") as f:
        features = [line.strip() for line in f.readlines()]
    return [feature for feature in features if feature]


def load_artifacts(
    *,
    artifact_dir: str,
    model_filename: str,
    pipeline_filename: str,
    features_filename: str,
    calibration_filename: str = "calibration.pkl",
) -> tuple[tf.keras.Model, dict, list[str], dict | None]:
    model_path = os.path.join(artifact_dir, model_filename)
    pipeline_path = os.path.join(artifact_dir, pipeline_filename)
    features_path = os.path.join(artifact_dir, features_filename)
    calibration_path = os.path.join(artifact_dir, calibration_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

    _register_pickle_compat_aliases()

    model = tf.keras.models.load_model(model_path, compile=False)
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    if os.path.exists(features_path):
        final_features = _load_features_from_text(features_path)
    else:
        final_features = [str(col).strip() for col in pipeline.get("features", []) if str(col).strip()]

    if not final_features:
        raise RuntimeError("Final feature list is empty. Expected final_features.txt or pipeline['features'].")

    calibration = None
    if os.path.exists(calibration_path):
        with open(calibration_path, "rb") as f:
            calibration = pickle.load(f)

    return model, pipeline, final_features, calibration
