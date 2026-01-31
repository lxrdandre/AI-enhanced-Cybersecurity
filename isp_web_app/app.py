from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, render_template

app = Flask(__name__)

DATA_DIR = Path("/ton-iot-project/fedlearning/data/processed")
DEVICE_FILES = [
    DATA_DIR / "device_1.csv",
    DATA_DIR / "device_2.csv",
    DATA_DIR / "device_3.csv",
]
HEARTBEAT_FILES = [
    DATA_DIR / "device_1.heartbeat",
    DATA_DIR / "device_2.heartbeat",
    DATA_DIR / "device_3.heartbeat",
]
TEST_FILE = DATA_DIR / "test_supervised.csv"
WEIGHTS_DIR = DATA_DIR / "global_weights"
LABEL_COL = "label"
TS_COL = "ts"
THRESHOLD = 0.5
CUMULATIVE_FILE = DATA_DIR / "anomaly_cumulative.json"
APP_START = datetime.utcnow()
STARTUP_GRACE_SEC = 60
MAX_EVAL_SAMPLES = 20000


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def _latest_weights() -> Path | None:
    if not WEIGHTS_DIR.exists():
        return None
    weights = sorted(WEIGHTS_DIR.glob("round_*.pt"))
    return weights[-1] if weights else None


def _latest_metrics() -> dict | None:
    if not WEIGHTS_DIR.exists():
        return None
    metrics_files = sorted(WEIGHTS_DIR.glob("round_*_metrics.json"))
    if not metrics_files:
        return None
    try:
        return json.loads(metrics_files[-1].read_text())
    except Exception:
        return None


def _load_cumulative() -> dict:
    if not CUMULATIVE_FILE.exists():
        return {}
    try:
        return json.loads(CUMULATIVE_FILE.read_text())
    except Exception:
        return {}


def _save_cumulative(data: dict) -> None:
    CUMULATIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CUMULATIVE_FILE.write_text(json.dumps(data))


def _load_device_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def _load_model(input_dim: int) -> nn.Module | None:
    weights = _latest_weights()
    if not weights:
        return None
    model = MLPClassifier(input_dim=input_dim)
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _global_accuracy() -> float | None:
    if not TEST_FILE.exists():
        return None
    df = pd.read_csv(TEST_FILE)
    if LABEL_COL not in df.columns:
        return None
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    if not feature_cols:
        return None
    if MAX_EVAL_SAMPLES and len(df) > MAX_EVAL_SAMPLES:
        df = df.sample(n=MAX_EVAL_SAMPLES, random_state=42)
    x = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].astype(int).to_numpy()
    model = _load_model(input_dim=x.shape[1])
    if model is None:
        return None
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(x))).cpu().numpy()
    preds = (probs > THRESHOLD).astype(int)
    return float((preds == y).mean())


def _global_recall_fpr() -> tuple[float, float] | None:
    if not TEST_FILE.exists():
        return None
    df = pd.read_csv(TEST_FILE)
    if LABEL_COL not in df.columns:
        return None
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    if not feature_cols:
        return None
    if MAX_EVAL_SAMPLES and len(df) > MAX_EVAL_SAMPLES:
        df = df.sample(n=MAX_EVAL_SAMPLES, random_state=42)
    x = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].astype(int).to_numpy()
    model = _load_model(input_dim=x.shape[1])
    if model is None:
        return None
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(x))).cpu().numpy()
    preds = (probs > THRESHOLD).astype(int)
    tp = int(((preds == 1) & (y == 1)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    recall = tp / (tp + fn + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    return float(recall), float(fpr)


def _device_anomalies(df: pd.DataFrame) -> dict:
    if LABEL_COL not in df.columns:
        return {"count": 0, "rate": 0.0, "window": "N/A"}

    if TS_COL in df.columns:
        ts = pd.to_datetime(df[TS_COL], errors="coerce", utc=True)
        cutoff = datetime.utcnow().replace(tzinfo=ts.dt.tz) - timedelta(hours=24)
        recent = df[ts >= cutoff]
        count = int((recent[LABEL_COL] == 1).sum())
        total = max(int(len(recent)), 1)
        rate = count / total
        return {"count": count, "rate": rate, "window": "24h"}

    count = int((df[LABEL_COL] == 1).sum())
    total = max(int(len(df)), 1)
    rate = count / total
    return {"count": count, "rate": rate, "window": "all"}


def _status_from_last_seen(last_seen: datetime) -> str:
    now = datetime.utcnow()
    if (now - APP_START) < timedelta(seconds=STARTUP_GRACE_SEC):
        return "online"
    delta = now - last_seen
    if delta < timedelta(seconds=30):
        return "online"
    if delta < timedelta(minutes=1):
        return "degraded"
    return "offline"


def _device_records() -> list[dict]:
    regions = ["North", "East", "South"]
    firmware = ["1.4.2", "1.4.1", "1.3.9"]
    model_version = "v2.1" if _latest_weights() else "v2.0"
    metrics = _latest_metrics() or {}
    anomalies_by_router = metrics.get("anomalies_by_router", {})
    cumulative = _load_cumulative()

    devices = []
    for idx, path in enumerate(DEVICE_FILES, start=1):
        if not path.exists():
            continue
        df = _load_device_df(path)
        anomalies = _device_anomalies(df)
        router_id = f"router-00{idx}"
        if router_id in anomalies_by_router:
            incoming = int(anomalies_by_router[router_id])
            prev = int(cumulative.get(router_id, 0))
            if incoming > prev:
                cumulative[router_id] = incoming
            anomalies["count"] = int(cumulative.get(router_id, incoming))
        hb_path = HEARTBEAT_FILES[idx - 1]
        if hb_path.exists():
            last_seen = datetime.utcfromtimestamp(hb_path.stat().st_mtime)
            status = _status_from_last_seen(last_seen)
        else:
            last_seen = datetime.utcfromtimestamp(path.stat().st_mtime)
            status = "online"
        devices.append(
            {
                "id": router_id,
                "region": regions[idx - 1],
                "status": status,
                "last_seen": last_seen,
                "firmware": firmware[idx - 1],
                "anomalies_24h": anomalies["count"],
                "anomaly_rate": anomalies["rate"],
                "anomaly_window": anomalies["window"],
                "model_version": model_version,
            }
        )

    _save_cumulative(cumulative)
    return devices


def _summary(devices: list[dict]) -> dict:
    total = len(devices)
    online = sum(1 for d in devices if d["status"] == "online")
    degraded = sum(1 for d in devices if d["status"] == "degraded")
    offline = sum(1 for d in devices if d["status"] == "offline")
    anomalies = sum(d["anomalies_24h"] for d in devices)
    return {
        "total": total,
        "online": online,
        "degraded": degraded,
        "offline": offline,
        "anomalies": anomalies,
    }


def _alerts(devices: list[dict]) -> list[dict]:
    alerts = []
    for d in devices:
        if d["anomaly_rate"] >= 0.2:
            alerts.append(
                {
                    "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                    "router_id": d["id"],
                    "severity": "high",
                    "message": f"High anomaly rate ({d['anomaly_rate']:.1%}) in {d['anomaly_window']} window.",
                }
            )
        elif d["anomaly_rate"] > 0:
            alerts.append(
                {
                    "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                    "router_id": d["id"],
                    "severity": "medium",
                    "message": f"Anomalies detected ({d['anomaly_rate']:.1%}) in {d['anomaly_window']} window.",
                }
            )
        if d["status"] == "offline":
            alerts.append(
                {
                    "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                    "router_id": d["id"],
                    "severity": "low",
                    "message": "Device offline.",
                }
            )
    return alerts


@app.route("/")
def index():
    devices = _device_records()
    summary = _summary(devices)
    chart_data = {
        "labels": [d["id"] for d in devices],
        "anomalies": [d["anomalies_24h"] for d in devices],
        "participation": [100 if d["status"] != "offline" else 0 for d in devices],
    }
    return render_template(
        "index.html",
        summary=summary,
        chart_data=chart_data,
        devices=devices,
    )


@app.route("/devices")
def devices():
    return render_template("devices.html", devices=_device_records())


@app.route("/model")
def model():
    weights = _latest_weights()
    last_round = int(weights.stem.split("_")[-1]) if weights else 0
    last_update = (
        datetime.utcfromtimestamp(weights.stat().st_mtime).strftime("%Y-%m-%d %H:%M UTC")
        if weights
        else "N/A"
    )
    devices = _device_records()
    total_clients = len(DEVICE_FILES)
    online_clients = sum(1 for d in devices if d["status"] != "offline")
    participation = int((online_clients / max(total_clients, 1)) * 100)

    metrics = _latest_metrics() or {}
    reported = len(metrics.get("anomalies_by_router", {}))
    update_success = int((reported / max(total_clients, 1)) * 100)

    accuracy = _global_accuracy()
    recall_fpr = _global_recall_fpr()

    model_info = {
        "version": "v2.1" if weights else "v2.0",
        "last_round": last_round,
        "threshold": THRESHOLD,
        "recall": f"{recall_fpr[0]:.4f}" if recall_fpr else "N/A",
        "fpr": f"{recall_fpr[1]:.4f}" if recall_fpr else "N/A",
        "accuracy": f"{accuracy:.4f}" if accuracy is not None else "N/A",
        "participation": participation,
        "update_success": update_success,
        "last_update": last_update,
    }
    return render_template("model.html", model_info=model_info)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
