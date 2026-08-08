"""Validate a labelled CSV with the saved Edge-IIoTset 6-class SE-DWNet artifact.

The expected target taxonomy is:

    backdoor, dos_ddos, injection, normal, password, scanning

This validator is intended for datasets with Edge-IIoTset/TShark-style feature
columns such as dns_qry_name, mqtt_topic, tcp_flags, udp_time_delta, etc. It
uses the saved preprocessing_pipeline.pkl and final_features.txt from training,
so the validation CSV may contain extra columns and may omit non-selected
pipeline columns; missing columns are filled by the saved preprocessing logic.

Example
-------
    python resnet/validate_edge_iiotset_dataset.py \
        --csv /data/datasets/my_new_dataset.csv \
        --model-dir /data/ton-iot-project/fresh_start/artifacts/se_dwnet_edge_iiotset_random_holdout \
        --dataset-name my_new_dataset
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import sys
import unicodedata
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

from app.preprocessing import transform_with_pipeline  # noqa: E402


class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    """Compatibility shim for older pipeline pickles that reference __main__."""

    def __init__(self):
        """Initialize the safe label encoder instance."""
        self.mapper = {}
        self.unknown_token = 0

    def fit(self, y):
        """Fit the compatibility encoder and return self."""
        y_series = pd.Series(y).astype(str)
        unique_labels = np.unique(y_series.values)
        unique_labels = np.sort(unique_labels)
        self.mapper = {label: idx + 1 for idx, label in enumerate(unique_labels)}
        return self

    def transform(self, y):
        """Transform labels while mapping unknown values to the fallback class."""
        return pd.Series(y).astype(str).map(self.mapper).fillna(self.unknown_token).astype(np.int32).values


TARGET_CLASSES = [
    "backdoor",
    "dos_ddos",
    "injection",
    "normal",
    "password",
    "scanning",
]

DROP_LABELS = {"mitm", "ransomware"}
LABEL_CANDIDATES = ("type", "attack", "attack_type", "category", "class", "label", "Label")
METADATA_COLUMNS = {
    "split_time",
    "source_label",
    "ts",
    "timestamp",
    "datetime",
    "date",
    "time",
    "frame_time_epoch",
    "frame_time",
    "src_ip",
    "dst_ip",
    "srcip",
    "dstip",
    "src_mac",
    "dst_mac",
}

EXPECTED_DATASET_FEATURES = [
    "type",
    "arp_hw_size",
    "arp_opcode",
    "dns_qry_name",
    "dns_qry_name_len",
    "dns_qry_qu",
    "dns_qry_type",
    "dns_retransmission",
    "dns_retransmit_request",
    "dns_retransmit_request_in",
    "http_content_length",
    "http_file_data",
    "http_referer",
    "http_request_full_uri",
    "http_request_method",
    "http_request_uri_query",
    "http_request_version",
    "http_response",
    "http_tls_port",
    "icmp_checksum",
    "icmp_seq_le",
    "icmp_transmit_timestamp",
    "icmp_unused",
    "mbtcp_len",
    "mbtcp_trans_id",
    "mbtcp_unit_id",
    "mqtt_conack_flags",
    "mqtt_conflag_cleansess",
    "mqtt_conflags",
    "mqtt_hdrflags",
    "mqtt_len",
    "mqtt_msg",
    "mqtt_msg_decoded_as",
    "mqtt_msgtype",
    "mqtt_proto_len",
    "mqtt_protoname",
    "mqtt_topic",
    "mqtt_topic_len",
    "mqtt_ver",
    "tcp_ack",
    "tcp_ack_raw",
    "tcp_checksum",
    "tcp_connection_fin",
    "tcp_connection_rst",
    "tcp_connection_syn",
    "tcp_connection_synack",
    "tcp_dstport",
    "tcp_flags",
    "tcp_flags_ack",
    "tcp_len",
    "tcp_options",
    "tcp_payload",
    "tcp_seq",
    "tcp_srcport",
    "udp_port",
    "udp_time_delta",
]


def _select_probs(raw_output) -> np.ndarray:
    """Normalize Keras outputs into a probability array."""
    if isinstance(raw_output, dict):
        return np.asarray(next(iter(raw_output.values())))
    if isinstance(raw_output, (list, tuple)):
        return np.asarray(raw_output[0])
    return np.asarray(raw_output)


def _register_pickle_compat_aliases() -> None:
    """Register legacy pickle aliases needed by saved preprocessing artifacts."""
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "SafeLabelEncoder"):
        setattr(main_module, "SafeLabelEncoder", SafeLabelEncoder)


def _detect_project_root() -> str:
    """Detect the project root used for default artifact discovery."""
    env_root = os.environ.get("TON_IOT_PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    cwd = os.path.abspath(os.getcwd())
    if os.path.isdir(os.path.join(cwd, "data")) and os.path.isdir(os.path.join(cwd, "artifacts")):
        return cwd
    return os.path.normpath(os.path.join(SCRIPT_DIR, ".."))


def _find_model_dir(project_root: str, explicit: str | None) -> str:
    """Find an Edge-IIoTSet model directory containing model and pipeline files."""
    if explicit and os.path.isdir(explicit):
        return os.path.abspath(explicit)

    candidates = [
        os.path.join(project_root, "artifacts", "se_dwnet_edge_iiotset_random_holdout"),
        os.path.join(project_root, "artifacts", "resnet_edge_iiotset_random_holdout"),
        os.path.join(project_root, "artifacts", "resnet_edge_iiotset_sensor_temporal"),
        os.path.join(project_root, "artifacts", "resnet_edge_iiotset_temporal"),
        os.path.join(project_root, "artifacts", "resnet_edge_iiotset"),
    ]
    for path in candidates:
        model_files = glob.glob(os.path.join(path, "*.keras"))
        pipeline_files = glob.glob(os.path.join(path, "*pipeline*.pkl"))
        if model_files and pipeline_files:
            return path

    raise FileNotFoundError(
        "Model directory not found. Tried:\n"
        + "\n".join(f"  - {path}" for path in candidates)
        + "\nUse --model-dir to specify explicitly."
    )


def _load_final_features(model_dir: str, pipeline: dict) -> list[str]:
    """Load the final ordered feature list from artifacts or pipeline metadata."""
    txt_path = os.path.join(model_dir, "final_features.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as handle:
            features = [line.strip() for line in handle if line.strip()]
        if features:
            return features

    features = pipeline.get("features")
    if not features:
        raise RuntimeError("Cannot determine final feature list.")
    return [str(feature).strip() for feature in features if str(feature).strip()]


def _pick_model_file(model_dir: str) -> str:
    """Select the Keras model file to validate."""
    preferred = os.path.join(model_dir, "se_dwnet_model.keras")
    if os.path.exists(preferred):
        return preferred
    legacy = os.path.join(model_dir, "resnet_model.keras")
    if os.path.exists(legacy):
        return legacy
    model_files = sorted(glob.glob(os.path.join(model_dir, "*.keras")))
    if not model_files:
        raise FileNotFoundError(f"No .keras model found in {model_dir}")
    return model_files[-1]


def _pick_pipeline_file(model_dir: str) -> str:
    """Select the preprocessing pipeline pickle to validate with."""
    preferred = os.path.join(model_dir, "preprocessing_pipeline.pkl")
    if os.path.exists(preferred):
        return preferred
    pipeline_files = sorted(glob.glob(os.path.join(model_dir, "*pipeline*.pkl")))
    if not pipeline_files:
        raise FileNotFoundError(f"No pipeline pickle found in {model_dir}")
    return pipeline_files[-1]


def _load_label_encoder(model_dir: str, pipeline: dict) -> tuple[object, list[str]]:
    """Load the label encoder and class order used during training."""
    meta_path = os.path.join(model_dir, "training_metadata.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as handle:
            meta = pickle.load(handle)
        classes = [str(cls) for cls in meta.get("classes", [])]
        label_encoder = meta.get("label_encoder")
        if label_encoder is not None and classes:
            return label_encoder, classes
        if classes:
            encoder = LabelEncoder()
            encoder.fit(classes)
            return encoder, classes

    pipeline_encoder = pipeline.get("target_encoder")
    if pipeline_encoder is not None and hasattr(pipeline_encoder, "classes_"):
        return pipeline_encoder, [str(cls) for cls in pipeline_encoder.classes_]

    encoder = LabelEncoder()
    encoder.fit(TARGET_CLASSES)
    return encoder, TARGET_CLASSES


def _normalize_column(column: object) -> str:
    """Normalize a raw validation column name to artifact-compatible form."""
    value = unicodedata.normalize("NFKD", str(column)).encode("ascii", "ignore").decode("ascii")
    value = value.strip().lower()
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate validation columns by first non-null value."""
    if not df.columns.duplicated().any():
        return df

    result = pd.DataFrame(index=df.index)
    for col in dict.fromkeys(df.columns):
        same = df.loc[:, df.columns == col]
        if same.shape[1] == 1:
            result[col] = same.iloc[:, 0]
        else:
            result[col] = same.bfill(axis=1).iloc[:, 0]
    return result


def _normalize_columns(df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    """Normalize validation dataframe columns when requested."""
    df = df.copy()
    if enabled:
        df.columns = [_normalize_column(col) for col in df.columns]
        df = _coalesce_duplicate_columns(df)
    else:
        df.columns = df.columns.str.strip()
    return df


def _safe_name(value: str) -> str:
    """Return a filesystem-safe dataset or report name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip().lower())
    return cleaned.strip("_") or "dataset"


def _infer_dataset_name(csv_path: str, explicit: str | None) -> str:
    """Infer the validation dataset name from CLI input or CSV path."""
    if explicit:
        return _safe_name(explicit)
    return _safe_name(os.path.splitext(os.path.basename(csv_path))[0])


def _canon_label(label: object) -> str:
    """Return the canonical validation label for Edge-IIoTSet classes."""
    value = _normalize_column(label)
    value = value.removesuffix("_attack")
    value = value.removesuffix("_attacks")

    if value in {"dos", "ddos", "dos_ddos", "ddos_dos"}:
        return "dos_ddos"
    if value.startswith("ddos_") or value.startswith("dos_"):
        return "dos_ddos"
    if value in {"backdoor", "backdoor_attack"}:
        return "backdoor"
    if value in {"normal", "benign", "normal_traffic"} or value.startswith("normal_"):
        return "normal"
    if value in {"password", "passwords", "password_attack", "password_attacks"}:
        return "password"
    if value in {"sql_injection", "xss", "uploading", "injection"}:
        return "injection"
    if value in {"port_scanning", "port_scan", "scanning", "os_fingerprinting", "vulnerability_scanner"}:
        return "scanning"
    if value in {"mitm", "ransomware"}:
        return value
    return value


def _find_label_column(df: pd.DataFrame, explicit: str | None, *, allow_unlabeled: bool) -> str | None:
    """Find the validation label column or return None for unlabeled data."""
    if explicit:
        normalized = _normalize_column(explicit)
        for candidate in (explicit, normalized):
            if candidate in df.columns:
                return candidate
        if allow_unlabeled:
            return None
        raise RuntimeError(f"Label column '{explicit}' not found. Columns: {list(df.columns[:40])}")

    for col in LABEL_CANDIDATES:
        candidates = [col, _normalize_column(col)]
        for candidate in candidates:
            if candidate not in df.columns:
                continue
            values = set(df[candidate].dropna().astype(str).head(50_000).map(_canon_label).unique())
            if values.intersection(TARGET_CLASSES) or values.intersection(DROP_LABELS):
                return candidate

    if allow_unlabeled:
        return None

    raise RuntimeError(
        "Could not identify a label column. "
        f"Tried {LABEL_CANDIDATES}. Columns: {list(df.columns[:40])}"
    )


def _clean_validation_frame(
    df: pd.DataFrame,
    *,
    class_names: list[str],
    label_col: str | None,
    allow_unlabeled: bool,
) -> tuple[pd.DataFrame, np.ndarray | None, dict]:
    """Filter labels, drop metadata, and return validation features."""
    target_col = _find_label_column(df, label_col, allow_unlabeled=allow_unlabeled)
    diagnostics: dict[str, object] = {"label_column": target_col}

    y_true = None
    if target_col is not None:
        labels = df[target_col].map(_canon_label)
        before = len(df)
        keep = ~labels.isin(DROP_LABELS)
        df = df.loc[keep].copy()
        labels = labels.loc[keep].copy()

        keep = labels.isin(class_names)
        dropped_unknown = int((~keep).sum())
        df = df.loc[keep].copy()
        labels = labels.loc[keep].to_numpy()
        if df.empty:
            raise RuntimeError(f"No validation rows remain after filtering to classes: {class_names}")

        y_true = labels
        diagnostics["rows_before_label_filter"] = int(before)
        diagnostics["rows_after_label_filter"] = int(len(df))
        diagnostics["dropped_unsupported_labels"] = dropped_unknown
        diagnostics["label_counts"] = dict(Counter(y_true))

    drop_cols = set(LABEL_CANDIDATES) | METADATA_COLUMNS
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors="ignore")
    return df.reset_index(drop=True), y_true, diagnostics


def _feature_diagnostics(
    *,
    df: pd.DataFrame,
    pipeline: dict,
    final_features: list[str],
) -> dict[str, list[str]]:
    """Summarize feature coverage against the saved preprocessing pipeline."""
    valid_cat_cols = [str(col) for col in pipeline.get("valid_cat_cols", [])]
    num_cols = [str(col) for col in pipeline.get("num_cols", [])]
    required_pipeline_cols = list(dict.fromkeys(valid_cat_cols + num_cols + final_features))

    return {
        "final_features_present": [col for col in final_features if col in df.columns],
        "final_features_missing": [col for col in final_features if col not in df.columns],
        "pipeline_columns_present": [col for col in required_pipeline_cols if col in df.columns],
        "pipeline_columns_missing": [col for col in required_pipeline_cols if col not in df.columns],
        "expected_dataset_features_missing": [col for col in EXPECTED_DATASET_FEATURES if col not in df.columns],
    }


def validate(
    *,
    csv_path: str,
    model_dir: str,
    output_dir: str | None,
    dataset_name: str,
    label_col: str | None,
    batch_size: int,
    chunk_size: int,
    max_samples: int | None,
    normalize_columns: bool,
    allow_unlabeled: bool,
    strict_features: bool,
) -> dict[str, str]:
    """Run external validation and write metrics, reports, and diagnostics."""
    if output_dir is None:
        output_dir = os.path.join(model_dir, f"{dataset_name}_validation")
    os.makedirs(output_dir, exist_ok=True)

    model_path = _pick_model_file(model_dir)
    pipeline_path = _pick_pipeline_file(model_dir)

    print("=== SE-DWNet Edge-IIoTset 6-Class External Validation ===")
    print(f"Dataset:    {dataset_name}")
    print(f"CSV:        {csv_path}")
    print(f"Model dir:  {model_dir}")
    print(f"Model:      {model_path}")
    print(f"Pipeline:   {pipeline_path}")
    print(f"Output dir: {output_dir}")

    _register_pickle_compat_aliases()
    model = tf.keras.models.load_model(model_path, compile=False)
    with open(pipeline_path, "rb") as handle:
        pipeline = pickle.load(handle)

    final_features = _load_final_features(model_dir, pipeline)
    label_encoder, class_names = _load_label_encoder(model_dir, pipeline)
    class_names = [str(cls) for cls in class_names]

    print(f"Features:   {len(final_features)} -> {final_features}")
    print(f"Classes:    {class_names}")

    print("Loading validation CSV...")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False, on_bad_lines="skip")
    df = _normalize_columns(df, enabled=normalize_columns)
    df, y_true, label_diagnostics = _clean_validation_frame(
        df,
        class_names=class_names,
        label_col=label_col,
        allow_unlabeled=allow_unlabeled,
    )

    if max_samples and len(df) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(df))[:max_samples]
        df = df.iloc[idx].reset_index(drop=True)
        if y_true is not None:
            y_true = y_true[idx]
        print(f"Subsampled to {len(df):,} rows")

    feature_info = _feature_diagnostics(df=df, pipeline=pipeline, final_features=final_features)
    missing_final = feature_info["final_features_missing"]
    if strict_features and missing_final:
        raise RuntimeError("Missing final model features: " + ", ".join(missing_final))

    print(f"Rows:       {len(df):,}")
    if y_true is not None:
        print(f"Labels:     {dict(Counter(y_true))}")
    else:
        print("Labels:     none; prediction-only mode")
    print(f"Columns:    {len(df.columns)}")
    print(f"Final features present: {len(feature_info['final_features_present'])}/{len(final_features)}")
    if missing_final:
        print(f"Final features filled as missing/zero: {missing_final}")
    expected_missing = feature_info["expected_dataset_features_missing"]
    if expected_missing:
        print(f"Expected dataset columns not found: {expected_missing}")

    all_probs = []
    records = df.to_dict(orient="records")
    for start in range(0, len(records), chunk_size):
        stop = min(start + chunk_size, len(records))
        x_chunk = transform_with_pipeline(
            records[start:stop],
            pipeline=pipeline,
            final_features=final_features,
        )
        probs = _select_probs(model.predict(x_chunk, batch_size=batch_size, verbose=0))
        all_probs.append(probs)
        print(f"  Processed {stop:,}/{len(records):,}")

    probs = np.vstack(all_probs)
    pred_int = np.argmax(probs, axis=1)
    y_pred = np.asarray(label_encoder.inverse_transform(pred_int)).astype(str)
    confidence = np.max(probs, axis=1)

    report_path = os.path.join(output_dir, f"{dataset_name}_classification_report.txt")
    cm_path = os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png")
    pred_path = os.path.join(output_dir, f"{dataset_name}_predictions.csv")

    if y_true is not None:
        report = classification_report(
            y_true,
            y_pred,
            labels=class_names,
            target_names=class_names,
            zero_division=0,
            digits=4,
        )

        print("\n" + "=" * 70)
        print(f"{dataset_name.upper()} VALIDATION")
        print("=" * 70)
        print(report)

        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(f"=== {dataset_name} Validation ===\n")
            handle.write(f"CSV: {csv_path}\n")
            handle.write(f"Model: {model_path}\n")
            handle.write(f"Pipeline: {pipeline_path}\n")
            handle.write(f"Rows: {len(y_true)}\n")
            handle.write(f"Labels: {dict(Counter(y_true))}\n")
            handle.write(f"Label diagnostics: {label_diagnostics}\n")
            handle.write(f"Final features present: {len(feature_info['final_features_present'])}/{len(final_features)}\n")
            handle.write(f"Final features missing: {missing_final}\n\n")
            handle.write(report)

        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
        plt.title(f"{dataset_name} Validation")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"Report saved: {report_path}")
        print(f"Confusion matrix saved: {cm_path}")
    else:
        report_path = ""
        cm_path = ""

    pred_data = {
        "predicted_class": y_pred,
        "confidence": confidence,
    }
    if y_true is not None:
        pred_data = {
            "true_class": y_true,
            "predicted_class": y_pred,
            "confidence": confidence,
            "correct": y_true == y_pred,
        }
    for index, cls in enumerate(class_names):
        pred_data[f"prob_{cls}"] = probs[:, index]
    pd.DataFrame(pred_data).to_csv(pred_path, index=False)
    print(f"Predictions saved: {pred_path}")

    print(
        "Confidence: "
        f"mean={confidence.mean():.4f}, "
        f"median={np.median(confidence):.4f}, "
        f"min={confidence.min():.4f}"
    )

    if y_true is not None:
        for cls in class_names:
            mask = y_true == cls
            if not mask.any():
                continue
            correct = y_pred[mask] == cls
            print(
                f"  {cls:>12s}: n={int(mask.sum()):>7d}  "
                f"conf={confidence[mask].mean():.3f}  "
                f"acc={correct.mean():.3f}"
            )

    return {
        "report_path": report_path,
        "confusion_matrix_path": cm_path,
        "predictions_path": pred_path,
        "output_dir": output_dir,
    }


def main() -> None:
    """Run the command-line entry point."""
    root = _detect_project_root()
    parser = argparse.ArgumentParser(
        description="Validate a labelled Edge-IIoTset-style CSV with a saved Edge 6-class model artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True, help="Labelled validation CSV.")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Model artifact directory. Defaults to the Edge random-holdout artifact when available.",
    )
    parser.add_argument("--dataset-name", default=None, help="Name used in output files. Auto-derived from CSV if omitted.")
    parser.add_argument("--label-col", default=None, help="Override target label column. Auto-detected if omitted.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--no-normalize-columns",
        action="store_true",
        help="Do not normalize column names. By default names are lowercased and punctuation becomes underscores.",
    )
    parser.add_argument("--allow-unlabeled", action="store_true", help="Run prediction-only mode if no label column is found.")
    parser.add_argument("--strict-features", action="store_true", help="Fail if any final model feature is missing from the CSV.")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    model_dir = _find_model_dir(root, args.model_dir)
    dataset_name = _infer_dataset_name(csv_path, args.dataset_name)
    validate(
        csv_path=csv_path,
        model_dir=model_dir,
        output_dir=args.output_dir,
        dataset_name=dataset_name,
        label_col=args.label_col,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
        normalize_columns=not args.no_normalize_columns,
        allow_unlabeled=args.allow_unlabeled,
        strict_features=args.strict_features,
    )


if __name__ == "__main__":
    main()
