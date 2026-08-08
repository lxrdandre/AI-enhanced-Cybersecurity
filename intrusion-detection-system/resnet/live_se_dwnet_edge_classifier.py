"""Standalone live SE-DWNet Edge-IIoTset classifier.

This script is for live testing the Edge-IIoTset random-holdout SE-DWNet model.
It does not use the FastAPI app, ClawdBot agent, Telegram, firewall actions, or
LLM triage. It captures short TShark windows, extracts Edge-IIoTset-style packet
fields, applies the saved preprocessing pipeline, and prints model predictions.

Requirements on the capture host:

    tshark
    python packages used by the training/inference environment

Example:

    sudo -E python3 -u resnet/live_se_dwnet_edge_classifier.py \
      --interface eth0 \
      --artifact-dir /data/ton-iot-project/fresh_start/artifacts/se_dwnet_edge_iiotset_random_holdout \
      --interval 5 \
      --show-normal \
      --csv /data/ton-iot-project/fresh_start/artifacts/live_edge_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

from app.preprocessing import SafeLabelEncoder, transform_with_pipeline  # noqa: E402


DEFAULT_ARTIFACT_DIR = "artifacts/se_dwnet_edge_iiotset_random_holdout"
DEFAULT_MODEL_FILENAME = "se_dwnet_model.keras"
DEFAULT_PIPELINE_FILENAME = "preprocessing_pipeline.pkl"
DEFAULT_FEATURES_FILENAME = "final_features.txt"

MISSING_TOKENS = {"", "-", "nan", "none", "null", "missing", "<na>"}

TSHARK_FIELDS: list[tuple[str, str]] = [
    ("frame.time_epoch", "frame_time_epoch"),
    ("ip.src", "src_ip"),
    ("ip.dst", "dst_ip"),
    ("arp.hw.size", "arp_hw_size"),
    ("arp.opcode", "arp_opcode"),
    ("dns.qry.name", "dns_qry_name"),
    ("dns.qry.name.len", "dns_qry_name_len"),
    ("dns.qry.qu", "dns_qry_qu"),
    ("dns.qry.type", "dns_qry_type"),
    ("dns.retransmission", "dns_retransmission"),
    ("dns.retransmit_request", "dns_retransmit_request"),
    ("dns.retransmit_request_in", "dns_retransmit_request_in"),
    ("http.content_length", "http_content_length"),
    ("http.file_data", "http_file_data"),
    ("http.referer", "http_referer"),
    ("http.request.full_uri", "http_request_full_uri"),
    ("http.request.method", "http_request_method"),
    ("http.request.uri.query", "http_request_uri_query"),
    ("http.request.version", "http_request_version"),
    ("http.response", "http_response"),
    ("http.tls_port", "http_tls_port"),
    ("icmp.checksum", "icmp_checksum"),
    ("icmp.seq_le", "icmp_seq_le"),
    ("icmp.transmit_timestamp", "icmp_transmit_timestamp"),
    ("icmp.unused", "icmp_unused"),
    ("mbtcp.len", "mbtcp_len"),
    ("mbtcp.trans_id", "mbtcp_trans_id"),
    ("mbtcp.unit_id", "mbtcp_unit_id"),
    ("mqtt.conack.flags", "mqtt_conack_flags"),
    ("mqtt.conflag.cleansess", "mqtt_conflag_cleansess"),
    ("mqtt.conflags", "mqtt_conflags"),
    ("mqtt.hdrflags", "mqtt_hdrflags"),
    ("mqtt.len", "mqtt_len"),
    ("mqtt.msg", "mqtt_msg"),
    ("mqtt.msg_decoded_as", "mqtt_msg_decoded_as"),
    ("mqtt.msgtype", "mqtt_msgtype"),
    ("mqtt.proto_len", "mqtt_proto_len"),
    ("mqtt.protoname", "mqtt_protoname"),
    ("mqtt.topic", "mqtt_topic"),
    ("mqtt.topic_len", "mqtt_topic_len"),
    ("mqtt.ver", "mqtt_ver"),
    ("tcp.ack", "tcp_ack"),
    ("tcp.ack_raw", "tcp_ack_raw"),
    ("tcp.checksum", "tcp_checksum"),
    ("tcp.connection.fin", "tcp_connection_fin"),
    ("tcp.connection.rst", "tcp_connection_rst"),
    ("tcp.connection.syn", "tcp_connection_syn"),
    ("tcp.connection.synack", "tcp_connection_synack"),
    ("tcp.dstport", "tcp_dstport"),
    ("tcp.flags", "tcp_flags"),
    ("tcp.flags.ack", "tcp_flags_ack"),
    ("tcp.len", "tcp_len"),
    ("tcp.options", "tcp_options"),
    ("tcp.payload", "tcp_payload"),
    ("tcp.seq", "tcp_seq"),
    ("tcp.srcport", "tcp_srcport"),
    ("udp.port", "udp_port"),
    ("udp.time_delta", "udp_time_delta"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def register_pickle_compat_aliases() -> None:
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "SafeLabelEncoder"):
        setattr(main_module, "SafeLabelEncoder", SafeLabelEncoder)


def configure_tensorflow() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


def load_features(path: Path, pipeline: dict) -> list[str]:
    if path.exists():
        features = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if features:
            return features
    features = pipeline.get("features")
    if not features:
        raise RuntimeError(f"Could not load final features from {path} or pipeline['features'].")
    return [str(feature).strip() for feature in features if str(feature).strip()]


def pick_model_path(artifact_dir: Path, model_filename: str) -> Path:
    model_path = artifact_dir / model_filename
    if model_path.exists():
        return model_path
    fallback = artifact_dir / "resnet_model.keras"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Model not found: {model_path} or {fallback}")


def load_artifacts(artifact_dir: Path, model_filename: str, pipeline_filename: str, features_filename: str):
    register_pickle_compat_aliases()
    configure_tensorflow()

    model_path = pick_model_path(artifact_dir, model_filename)
    pipeline_path = artifact_dir / pipeline_filename
    features_path = artifact_dir / features_filename
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")

    model = tf.keras.models.load_model(str(model_path), compile=False)
    with pipeline_path.open("rb") as handle:
        pipeline = pickle.load(handle)
    final_features = load_features(features_path, pipeline)
    target_encoder = pipeline.get("target_encoder")
    if target_encoder is None or not hasattr(target_encoder, "classes_"):
        raise RuntimeError("Pipeline does not include target_encoder.classes_.")
    class_names = [str(name) for name in target_encoder.classes_.tolist()]
    return model, pipeline, final_features, target_encoder, class_names, model_path, pipeline_path


def supported_tshark_fields(tshark_path: str) -> set[str] | None:
    try:
        result = subprocess.run(
            [tshark_path, "-G", "fields"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        print(f"Warning: could not query TShark field registry ({exc}); trying configured fields as-is.", flush=True)
        return None

    fields: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] == "F":
            fields.add(parts[2])
    return fields


def selected_tshark_fields(tshark_path: str) -> list[tuple[str, str]]:
    supported = supported_tshark_fields(tshark_path)
    if supported is None:
        return TSHARK_FIELDS

    selected = [(field, col) for field, col in TSHARK_FIELDS if field in supported]
    missing = [field for field, _ in TSHARK_FIELDS if field not in supported]
    if missing:
        print(f"TShark does not support {len(missing)} configured fields; they will be filled as missing/zero.", flush=True)
        print("Missing fields: " + ", ".join(missing[:30]) + (" ..." if len(missing) > 30 else ""), flush=True)
    return selected


def capture_window(
    *,
    tshark_path: str,
    interface: str,
    bpf_filter: str,
    interval: float,
    fields: list[tuple[str, str]],
) -> pd.DataFrame:
    if not fields:
        raise RuntimeError("No usable TShark fields are available.")

    cmd = [
        tshark_path,
        "-i",
        interface,
        "-a",
        f"duration:{max(interval, 0.1)}",
    ]
    if bpf_filter:
        cmd.extend(["-f", bpf_filter])
    cmd.extend(["-T", "fields", "-E", "header=y", "-E", "separator=,", "-E", "quote=d", "-E", "occurrence=f"])
    for field, _ in fields:
        cmd.extend(["-e", field])

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"TShark failed with exit={result.returncode}: {result.stderr.strip()}")
    if not result.stdout.strip():
        return pd.DataFrame(columns=[col for _, col in TSHARK_FIELDS])

    reader = csv.DictReader(io.StringIO(result.stdout))
    rows = []
    field_to_col = dict(fields)
    for raw in reader:
        row = {col: "" for _, col in TSHARK_FIELDS}
        for field, value in raw.items():
            col = field_to_col.get(field)
            if col is not None:
                row[col] = value
        rows.append(row)
    return pd.DataFrame(rows)


def nonempty_final_feature_count(record: dict, final_features: list[str]) -> int:
    count = 0
    for col in final_features:
        value = str(record.get(col, "")).strip().lower()
        if value not in MISSING_TOKENS:
            count += 1
    return count


def select_probs(raw_output) -> np.ndarray:
    if isinstance(raw_output, dict):
        return np.asarray(next(iter(raw_output.values())))
    if isinstance(raw_output, (list, tuple)):
        return np.asarray(raw_output[0])
    return np.asarray(raw_output)


def predict_records(model, pipeline: dict, final_features: list[str], target_encoder, class_names: list[str], records: list[dict], batch_size: int):
    x = transform_with_pipeline(records, pipeline=pipeline, final_features=final_features)
    probs = select_probs(model.predict(x, batch_size=batch_size, verbose=0))
    pred_idx = np.argmax(probs, axis=1)
    raw_labels = np.asarray(target_encoder.inverse_transform(pred_idx)).astype(str)
    confidence = np.max(probs, axis=1)
    results = []
    for idx, record in enumerate(records):
        probabilities = {name: float(probs[idx, class_idx]) for class_idx, name in enumerate(class_names)}
        results.append(
            {
                "timestamp": utc_now(),
                "predicted_label": str(raw_labels[idx]),
                "confidence": float(confidence[idx]),
                "probabilities": probabilities,
                "record": record,
            }
        )
    return results


def flow_text(record: dict) -> str:
    src = record.get("src_ip") or "?"
    dst = record.get("dst_ip") or "?"
    src_port = record.get("tcp_srcport") or ""
    dst_port = record.get("tcp_dstport") or ""
    udp_port = record.get("udp_port") or ""
    if src_port or dst_port:
        return f"{src}:{src_port or '?'} -> {dst}:{dst_port or '?'} TCP"
    if udp_port:
        return f"{src} -> {dst} UDP:{udp_port}"
    return f"{src} -> {dst}"


def print_cycle(results: list[dict], *, show_normal: bool, top: int) -> None:
    counts = Counter(row["predicted_label"] for row in results)
    print(f"[{utc_now()}] packets={len(results)} predictions={dict(counts)}", flush=True)
    visible = [row for row in results if show_normal or row["predicted_label"] != "normal"]
    visible.sort(key=lambda row: row["confidence"], reverse=True)
    if top > 0:
        visible = visible[:top]
    for row in visible:
        print(
            f"  {row['predicted_label']:>12s} "
            f"conf={row['confidence']:.4f} "
            f"{flow_text(row['record'])}",
            flush=True,
        )


def open_csv_writer(path: str | None, class_names: list[str]):
    if not path:
        return None, None
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exists = out_path.exists() and out_path.stat().st_size > 0
    handle = out_path.open("a", newline="", encoding="utf-8")
    fields = [
        "timestamp",
        "predicted_label",
        "confidence",
        "src_ip",
        "dst_ip",
        "tcp_srcport",
        "tcp_dstport",
        "udp_port",
        "tcp_flags",
        "http_request_method",
        "dns_qry_name",
    ] + [f"prob_{name}" for name in class_names]
    writer = csv.DictWriter(handle, fieldnames=fields)
    if not exists:
        writer.writeheader()
    return handle, writer


def write_csv_rows(writer, results: list[dict], class_names: list[str]) -> None:
    if writer is None:
        return
    for row in results:
        record = row["record"]
        out = {
            "timestamp": row["timestamp"],
            "predicted_label": row["predicted_label"],
            "confidence": row["confidence"],
            "src_ip": record.get("src_ip", ""),
            "dst_ip": record.get("dst_ip", ""),
            "tcp_srcport": record.get("tcp_srcport", ""),
            "tcp_dstport": record.get("tcp_dstport", ""),
            "udp_port": record.get("udp_port", ""),
            "tcp_flags": record.get("tcp_flags", ""),
            "http_request_method": record.get("http_request_method", ""),
            "dns_qry_name": record.get("dns_qry_name", ""),
        }
        for name in class_names:
            out[f"prob_{name}"] = row["probabilities"].get(name, 0.0)
        writer.writerow(out)


def append_jsonl(path: str | None, results: list[dict]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SE-DWNet Edge-IIoTset classifier on live TShark-extracted traffic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--interface", required=True)
    parser.add_argument("--bpf-filter", default="ip", help="Capture filter passed to TShark.")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    parser.add_argument("--pipeline-filename", default=DEFAULT_PIPELINE_FILENAME)
    parser.add_argument("--features-filename", default=DEFAULT_FEATURES_FILENAME)
    parser.add_argument("--tshark", default=shutil.which("tshark") or "tshark")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--show-normal", action="store_true")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--min-nonempty-final-features", type=int, default=1)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--cycles", type=int, default=0, help="0 means run until interrupted.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    model, pipeline, final_features, target_encoder, class_names, model_path, pipeline_path = load_artifacts(
        artifact_dir,
        args.model_filename,
        args.pipeline_filename,
        args.features_filename,
    )
    fields = selected_tshark_fields(args.tshark)

    print("=== Live SE-DWNet Edge-IIoTset Classifier ===", flush=True)
    print(f"Interface:  {args.interface}", flush=True)
    print(f"BPF filter: {args.bpf_filter}", flush=True)
    print(f"TShark:     {args.tshark}", flush=True)
    print(f"Artifact:   {artifact_dir}", flush=True)
    print(f"Model:      {model_path}", flush=True)
    print(f"Pipeline:   {pipeline_path}", flush=True)
    print(f"Classes:    {class_names}", flush=True)
    print(f"Features:   {len(final_features)}", flush=True)
    print("Mode:       Edge-IIoTset/TShark features; no API, no LLM, no ClawdBot agent", flush=True)

    stop = False

    def handle_signal(signum, _frame):
        nonlocal stop
        print(f"Received {signal.Signals(signum).name}; stopping after current cycle.", flush=True)
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    csv_handle, csv_writer = open_csv_writer(args.csv, class_names)
    cycle = 0
    try:
        while not stop:
            cycle += 1
            frame = capture_window(
                tshark_path=args.tshark,
                interface=args.interface,
                bpf_filter=args.bpf_filter,
                interval=args.interval,
                fields=fields,
            )
            if frame.empty:
                print(f"[{utc_now()}] packets=0", flush=True)
            else:
                records = frame.to_dict(orient="records")
                if args.min_nonempty_final_features > 0:
                    records = [
                        record for record in records
                        if nonempty_final_feature_count(record, final_features) >= args.min_nonempty_final_features
                    ]
                if not records:
                    print(f"[{utc_now()}] packets=0 after feature filter", flush=True)
                else:
                    results = predict_records(
                        model,
                        pipeline,
                        final_features,
                        target_encoder,
                        class_names,
                        records,
                        args.batch_size,
                    )
                    print_cycle(results, show_normal=args.show_normal, top=args.top)
                    write_csv_rows(csv_writer, results, class_names)
                    append_jsonl(args.jsonl, results)
                    if csv_handle is not None:
                        csv_handle.flush()
            if args.once or (args.cycles > 0 and cycle >= args.cycles):
                break
    finally:
        if csv_handle is not None:
            csv_handle.close()
        print("Stopped.", flush=True)


if __name__ == "__main__":
    main()
