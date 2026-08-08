"""Train classical ML baselines on the custom Zeek dataset in Colab."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from zeek_colab_utils import (
    build_custom_zeek_matrices,
    evaluate_predictions,
    maybe_sample_training_rows,
    save_json,
    save_pickle,
    softmax_scores,
)


DEFAULT_CSV = "/content/drive/MyDrive/thesis_ids/data/zeek_crossval.csv"
DEFAULT_OUTPUT_DIR = "/content/drive/MyDrive/thesis_ids/artifacts/custom_zeek_baselines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RF/ExtraTrees/HistGB/linear baselines on the custom Zeek dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--split", choices=("random", "temporal", "source"), default="random")
    parser.add_argument("--source-group-mode", choices=("family", "label"), default="family")
    parser.add_argument("--time-col", default="ts")
    parser.add_argument("--temporal-fallback", choices=("error", "random"), default="random")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--final-holdout-size", type=float, default=0.05)
    parser.add_argument("--final-holdout-mode", choices=("random", "temporal", "source"), default="random")
    parser.add_argument("--target-k", type=int, default=192)
    parser.add_argument("--smote", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--smote-imbalance-ratio", type=float, default=1.10)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--expected-per-class", type=int, default=60_000)
    parser.add_argument("--feature-inference-sample-size", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        default="rf,extratrees,histgb,sgd_logreg,linear_svm",
        help="Comma-separated: rf, extratrees, histgb, sgd_logreg, linear_svm, xgboost",
    )
    parser.add_argument("--n-estimators", type=int, default=180)
    parser.add_argument("--max-depth", type=int, default=0, help="0 means unlimited for tree models.")
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--histgb-max-iter", type=int, default=300)
    parser.add_argument(
        "--max-train-rows-per-class",
        type=int,
        default=0,
        help="Use a smaller value, e.g. 20000, if Colab free RAM is tight. 0 uses full training set.",
    )
    return parser.parse_args()


def build_model(name: str, args: argparse.Namespace):
    max_depth = None if args.max_depth <= 0 else args.max_depth
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.seed,
            verbose=1,
        )
    if name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed,
            verbose=1,
        )
    if name == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=args.histgb_max_iter,
            learning_rate=0.06,
            l2_regularization=0.01,
            random_state=args.seed,
            verbose=1,
        )
    if name == "sgd_logreg":
        return SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=1500,
            tol=1e-4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed,
        )
    if name == "linear_svm":
        return LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=args.seed,
            max_iter=5000,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install xgboost or remove xgboost from --models.") from exc
        return XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=8 if max_depth is None else max_depth,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=args.seed,
        )
    raise ValueError(f"Unknown model: {name}")


def predict_with_probabilities(model, x: np.ndarray, class_names: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
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
        probs = softmax_scores(model.decision_function(x))
        return pred, probs
    return pred, None


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    data = build_custom_zeek_matrices(
        csv_path=args.csv,
        label_col=args.label_col,
        split=args.split,
        source_group_mode=args.source_group_mode,
        time_col=args.time_col,
        temporal_fallback=args.temporal_fallback,
        val_size=args.val_size,
        test_size=args.test_size,
        final_holdout_size=args.final_holdout_size,
        final_holdout_mode=args.final_holdout_mode,
        target_k=args.target_k,
        smote_mode=args.smote,
        smote_imbalance_ratio=args.smote_imbalance_ratio,
        dedupe=args.dedupe,
        expected_per_class=args.expected_per_class,
        feature_inference_sample_size=args.feature_inference_sample_size,
        seed=args.seed,
    )

    class_names = data["class_names"]
    target_encoder = data["target_encoder"]
    y_test_readable = target_encoder.inverse_transform(data["y_test"])

    x_train, y_train = maybe_sample_training_rows(
        data["x_train"],
        data["y_train"],
        max_rows_per_class=args.max_train_rows_per_class,
        seed=args.seed,
    )
    print(f"Baseline fit matrix: {x_train.shape}")

    save_pickle(output_dir / "baseline_preprocessing_bundle.pkl", data["preprocess_bundle"])
    save_json(output_dir / "baseline_metadata.json", data["metadata"])
    with open(output_dir / "final_features.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(data["metadata"]["final_features"]) + "\n")

    selected_models = [model.strip().lower() for model in args.models.split(",") if model.strip()]
    summary = []
    for name in selected_models:
        print(f"\n=== Training baseline: {name} ===")
        model = build_model(name, args)
        start = time.perf_counter()
        model.fit(x_train, y_train)
        train_seconds = time.perf_counter() - start
        print(f"{name} fit time: {train_seconds:.1f}s")
        joblib.dump(model, output_dir / "models" / f"{name}.joblib")

        model_dir = output_dir / name
        model_dir.mkdir(exist_ok=True)
        pred_test, probs_test = predict_with_probabilities(model, data["x_test"], class_names)
        metrics = evaluate_predictions(
            y_true=y_test_readable,
            y_pred=pred_test,
            class_names=class_names,
            output_dir=str(model_dir),
            prefix=f"{name}_test",
            probabilities=probs_test,
        )
        metrics["model"] = name
        metrics["train_seconds"] = float(train_seconds)

        if len(data["y_holdout"]):
            pred_holdout, probs_holdout = predict_with_probabilities(model, data["x_holdout"], class_names)
            holdout_metrics = evaluate_predictions(
                y_true=data["y_holdout"],
                y_pred=pred_holdout,
                class_names=class_names,
                output_dir=str(model_dir),
                prefix=f"{name}_final_holdout",
                probabilities=probs_holdout,
            )
            for key, value in holdout_metrics.items():
                metrics[f"holdout_{key}"] = value
        summary.append(metrics)

    summary_path = output_dir / "baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDONE. Baseline artifacts saved to: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
