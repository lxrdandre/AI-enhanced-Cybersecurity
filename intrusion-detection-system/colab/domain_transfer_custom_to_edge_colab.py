"""Evaluate custom-Zeek artifacts on an Edge-IIoTset Zeek-feature CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from zeek_colab_utils import (
    evaluate_predictions,
    load_pickle,
    save_json,
    softmax_scores,
    transform_external_csv,
)


DEFAULT_TARGET_CSV = "/content/drive/MyDrive/thesis_ids/data/edge_public_zeek_same_features_100k.csv"
DEFAULT_ARTIFACT_DIR = "/content/drive/MyDrive/thesis_ids/artifacts/se_dwnet_zeek_crossval_colab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test domain transfer from custom Zeek training artifacts to Edge-IIoTset Zeek CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR, help="SE-DWNet artifact directory.")
    parser.add_argument("--target-csv", default=DEFAULT_TARGET_CSV, help="External Edge-IIoTset Zeek-feature CSV.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--sample-per-class", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baseline-artifact-dir",
        default=None,
        help="Optional baseline artifact directory from train_custom_baselines_colab.py.",
    )
    return parser.parse_args()


def find_model_path(artifact_dir: Path) -> Path:
    candidates = [
        artifact_dir / "se_dwnet_zeek_model.keras",
        artifact_dir / "se_dwnet_model.keras",
        artifact_dir / "resnet_model.keras",
        artifact_dir / "model.keras",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    keras_files = sorted(artifact_dir.glob("*.keras"))
    if keras_files:
        return keras_files[0]
    raise FileNotFoundError(f"No Keras model found in {artifact_dir}")


def predict_baseline(model, x: np.ndarray, class_names: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        pred_idx = np.argmax(probs, axis=1)
        return np.asarray(class_names, dtype=object)[pred_idx].astype(str), probs
    pred_idx = model.predict(x)
    if np.issubdtype(np.asarray(pred_idx).dtype, np.integer):
        pred = np.asarray(class_names, dtype=object)[pred_idx].astype(str)
    else:
        pred = np.asarray(pred_idx).astype(str)
    if hasattr(model, "decision_function"):
        return pred, softmax_scores(model.decision_function(x))
    return pred, None


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir) if args.output_dir else artifact_dir / "domain_transfer_edge_iiotset"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = artifact_dir / "preprocessing_pipeline.pkl"
    if not pipeline_path.exists():
        raise FileNotFoundError(pipeline_path)
    pipeline = load_pickle(pipeline_path)
    class_names = list(pipeline["target_encoder"].classes_)

    x_external, y_external, external_info = transform_external_csv(
        csv_path=args.target_csv,
        preprocess_bundle=pipeline,
        label_col=args.label_col,
        sample_per_class=args.sample_per_class,
        seed=args.seed,
    )
    save_json(output_dir / "external_feature_coverage.json", external_info)
    print("External feature coverage:")
    print(external_info["feature_coverage"])

    model_path = find_model_path(artifact_dir)
    print(f"Loading SE-DWNet model: {model_path}")
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    probs = model.predict(x_external, batch_size=args.batch_size)
    pred = np.asarray(class_names, dtype=object)[np.argmax(probs, axis=1)].astype(str)
    metrics = {
        "se_dwnet": evaluate_predictions(
            y_true=y_external,
            y_pred=pred,
            class_names=class_names,
            output_dir=str(output_dir / "se_dwnet"),
            prefix="se_dwnet_domain_transfer",
            probabilities=probs,
        )
    }

    if args.baseline_artifact_dir:
        baseline_dir = Path(args.baseline_artifact_dir)
        baseline_pipeline_path = baseline_dir / "baseline_preprocessing_bundle.pkl"
        if not baseline_pipeline_path.exists():
            raise FileNotFoundError(baseline_pipeline_path)
        baseline_pipeline = load_pickle(baseline_pipeline_path)
        baseline_class_names = list(baseline_pipeline["target_encoder"].classes_)
        x_baseline, y_baseline, baseline_info = transform_external_csv(
            csv_path=args.target_csv,
            preprocess_bundle=baseline_pipeline,
            label_col=args.label_col,
            sample_per_class=args.sample_per_class,
            seed=args.seed,
        )
        save_json(output_dir / "baseline_external_feature_coverage.json", baseline_info)
        for model_path in sorted((baseline_dir / "models").glob("*.joblib")):
            name = model_path.stem
            print(f"Loading baseline: {model_path}")
            model = joblib.load(model_path)
            pred, probs = predict_baseline(model, x_baseline, baseline_class_names)
            metrics[name] = evaluate_predictions(
                y_true=y_baseline,
                y_pred=pred,
                class_names=baseline_class_names,
                output_dir=str(output_dir / name),
                prefix=f"{name}_domain_transfer",
                probabilities=probs,
            )

    save_json(output_dir / "domain_transfer_summary.json", metrics)
    print(f"DONE. Domain-transfer reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
