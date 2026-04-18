from __future__ import annotations

import os
import pickle
import sys

import tensorflow as tf

from app.preprocessing import SafeLabelEncoder

_TF_GPU_CONFIGURED = False


def configure_tensorflow_gpu() -> None:
    """Avoid TensorFlow preallocating the whole GPU and starving Ollama."""
    global _TF_GPU_CONFIGURED
    if _TF_GPU_CONFIGURED:
        return

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        _TF_GPU_CONFIGURED = True
        return

    limit_mb = os.environ.get("TON_IOT_TF_GPU_MEMORY_LIMIT_MB", "").strip()
    try:
        if limit_mb:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=int(limit_mb))],
            )
        elif os.environ.get("TON_IOT_TF_MEMORY_GROWTH", "1").lower() not in {"0", "false", "no"}:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        # TensorFlow was already initialized; the setting only applies on restart.
        pass

    _TF_GPU_CONFIGURED = True


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

    configure_tensorflow_gpu()
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


def load_domain_router(*, artifact_dir: str, router_filename: str = "domain_router.pkl") -> dict | None:
    router_path = os.path.join(artifact_dir, router_filename)
    if not os.path.exists(router_path):
        return None
    _register_pickle_compat_aliases()
    with open(router_path, "rb") as f:
        return pickle.load(f)


def load_model_file(path: str) -> tf.keras.Model:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    configure_tensorflow_gpu()
    return tf.keras.models.load_model(path, compile=False)
