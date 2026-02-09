from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
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
    DATA_DIR / "device_4.csv",
    DATA_DIR / "device_5.csv",
]
HEARTBEAT_FILES = [
    DATA_DIR / "device_1.heartbeat",
    DATA_DIR / "device_2.heartbeat",
    DATA_DIR / "device_3.heartbeat",
    DATA_DIR / "device_4.heartbeat",
    DATA_DIR / "device_5.heartbeat",
]
TEST_FILE = DATA_DIR / "test_supervised.csv"
WEIGHTS_DIR = DATA_DIR / "global_weights"
LABEL_COL = "label"
TS_COL = "ts"
THRESHOLD = 0.5
CUMULATIVE_FILE = DATA_DIR / "anomaly_cumulative.json"
EET = ZoneInfo("Europe/Bucharest")
APP_START = datetime.now(tz=EET)
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


def _now_eet() -> datetime:
    return datetime.now(tz=EET)


def _format_eet(dt: datetime) -> str:
    return dt.astimezone(EET).strftime("%Y-%m-%d %H:%M EET")


def _device_anomalies(df: pd.DataFrame) -> dict:
    if LABEL_COL not in df.columns:
        return {"count": 0, "rate": 0.0, "window": "N/A"}

    if TS_COL in df.columns:
        ts = pd.to_datetime(df[TS_COL], errors="coerce", utc=True).dt.tz_convert(EET)
        cutoff = _now_eet() - timedelta(hours=24)
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
    now = _now_eet()
    if (now - APP_START) < timedelta(seconds=STARTUP_GRACE_SEC):
        return "online"
    delta = now - last_seen
    if delta < timedelta(seconds=30):
        return "online"
    if delta < timedelta(minutes=10):
        return "degraded"
    return "offline"


def _device_records() -> list[dict]:
    regions = ["Timisoara", "Baia Mare", "Ortisoara", "Cluj", "Arad"]
    model_name = _latest_model_name()
    metrics = _latest_metrics() or {}
    anomalies_by_router = metrics.get("anomalies_by_router", {})
    latest_round = metrics.get("round")
    cumulative = _load_cumulative()

    devices = []
    now = _now_eet()
    for idx, path in enumerate(DEVICE_FILES, start=1):
        if not path.exists():
            continue
        df = _load_device_df(path)
        anomalies = _device_anomalies(df)
        router_id = f"router-00{idx}"
        if router_id in anomalies_by_router:
            incoming = int(anomalies_by_router[router_id])
            prev_entry = cumulative.get(router_id, {})
            if isinstance(prev_entry, dict):
                prev_count = int(prev_entry.get("count", 0))
                prev_round = prev_entry.get("round")
            else:
                prev_count = int(prev_entry or 0)
                prev_round = None
            if latest_round is None or latest_round != prev_round:
                new_total = prev_count + incoming
                cumulative[router_id] = {
                    "count": new_total,
                    "ts": now.isoformat(),
                    "round": latest_round,
                }
                anomalies["count"] = new_total
            else:
                anomalies["count"] = prev_count
        hb_path = HEARTBEAT_FILES[idx - 1]
        if hb_path.exists():
            last_seen = datetime.fromtimestamp(hb_path.stat().st_mtime, tz=EET)
        else:
            last_seen = datetime.fromtimestamp(path.stat().st_mtime, tz=EET)
        status = _status_from_last_seen(last_seen)
        status_overview = "degraded" if status == "offline" else status
        if anomalies["count"] == 0:
            entry = cumulative.get(router_id)
            if isinstance(entry, dict):
                last_count = int(entry.get("count", 0))
                last_ts_raw = entry.get("ts")
                last_ts = (
                    datetime.fromisoformat(last_ts_raw)
                    if isinstance(last_ts_raw, str)
                    else None
                )
            else:
                last_count = int(entry or 0)
                last_ts = None
            last_age_ok = True
            if last_ts is not None:
                last_age_ok = (now - last_ts) < timedelta(hours=24)
            if last_count > 0 and last_age_ok:
                anomalies["count"] = last_count
        devices.append(
            {
                "id": router_id,
                "region": regions[idx - 1],
                "status": status,
                "status_overview": status_overview,
                "last_seen": last_seen,
                "last_seen_str": _format_eet(last_seen),
                "anomalies_24h": anomalies["count"],
                "anomaly_rate": anomalies["rate"],
                "anomaly_window": anomalies["window"],
                "model_name": model_name,
            }
        )

    _save_cumulative(cumulative)
    return devices


def _summary(devices: list[dict]) -> dict:
    total = len(devices)
    online = sum(1 for d in devices if d.get("status_overview", d["status"]) == "online")
    degraded = sum(1 for d in devices if d.get("status_overview", d["status"]) == "degraded")
    offline = sum(1 for d in devices if d.get("status_overview", d["status"]) == "offline")
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
                    "time": _format_eet(_now_eet()),
                    "router_id": d["id"],
                    "severity": "high",
                    "message": f"High anomaly rate ({d['anomaly_rate']:.1%}) in {d['anomaly_window']} window.",
                }
            )
        elif d["anomaly_rate"] > 0:
            alerts.append(
                {
                    "time": _format_eet(_now_eet()),
                    "router_id": d["id"],
                    "severity": "medium",
                    "message": f"Anomalies detected ({d['anomaly_rate']:.1%}) in {d['anomaly_window']} window.",
                }
            )
        if d["status"] == "offline":
            alerts.append(
                {
                    "time": _format_eet(_now_eet()),
                    "router_id": d["id"],
                    "severity": "low",
                    "message": "Device offline.",
                }
            )
    return alerts


def _latest_model_name() -> str:
    return "MLPClassifier"


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


@app.route("/devices/<router_id>")
def device_detail(router_id: str):
    devices = _device_records()
    device = next((d for d in devices if d["id"] == router_id), None)
    if device is None:
        return render_template("device_detail.html", device=None), 404
    return render_template("device_detail.html", device=device)


@app.route("/model")
def model():
    weights = _latest_weights()
    last_round = int(weights.stem.split("_")[-1]) if weights else 0
    last_update = (
        _format_eet(datetime.fromtimestamp(weights.stat().st_mtime, tz=EET))
        if weights
        else "N/A"
    )
    total_clients = len(DEVICE_FILES)
    metrics = _latest_metrics() or {}
    reported = len(metrics.get("anomalies_by_router", {}))
    participation = int((reported / max(total_clients, 1)) * 100)
    update_success = participation

    accuracy = _global_accuracy()
    recall_fpr = _global_recall_fpr()

    model_info = {
        "version": "alpha" if weights else "beta",
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
