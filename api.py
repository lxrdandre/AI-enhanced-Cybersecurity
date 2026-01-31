from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def _resolve_data_dir() -> Path:
    env_dir = os.getenv("FEDLEARNING_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()

    default_posix = Path("/ton-iot-project/fedlearning/data/processed")
    if default_posix.exists():
        return default_posix

    return Path(__file__).resolve().parents[1] / "data" / "processed"


def _latest_metrics_files(weights_dir: Path, limit: int = 6) -> List[Path]:
    files = sorted(weights_dir.glob("round_*_metrics.json"))
    return files[-limit:]


def _load_latest_metrics(weights_dir: Path) -> Dict[str, Any] | None:
    files = sorted(weights_dir.glob("round_*_metrics.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def _router_index(router_id: str) -> int | None:
    digits = "".join(ch for ch in router_id if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def _heartbeat_status(data_dir: Path, router_id: str) -> str:
    router_idx = _router_index(router_id)
    if router_idx is None:
        return "online"
    heartbeat = data_dir / f"device_{router_idx}.heartbeat"
    if not heartbeat.exists():
        return "online"
    try:
        last_seen = heartbeat.stat().st_mtime
    except OSError:
        return "online"
    age = datetime.now(timezone.utc).timestamp() - last_seen
    if age >= 60:
        return "offline"
    if age >= 30:
        return "degraded"
    return "online"


def _build_devices(metrics: Dict[str, Any] | None, data_dir: Path) -> List[Dict[str, Any]]:
    anomalies = (metrics or {}).get("anomalies_by_router", {})
    devices: List[Dict[str, Any]] = []

    for router_id, count in anomalies.items():
        status = _heartbeat_status(data_dir, router_id)
        devices.append(
            {
                "name": router_id,
                "room": "Client site",
                "status": status,
                "last_seen": (metrics or {}).get("timestamp", "-"),
                "firmware": "-",
                "usage_24h": f"{count} alerts",
                "model": "Federated-IDS",
            }
        )

    if devices:
        return devices

    heartbeat_files = sorted(data_dir.glob("device_*.heartbeat"))
    for heartbeat in heartbeat_files:
        router_id = heartbeat.stem.replace("device_", "router-")
        try:
            last_seen = datetime.fromtimestamp(
                heartbeat.stat().st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC")
        except OSError:
            last_seen = "-"
        devices.append(
            {
                "name": router_id,
                "room": "Client site",
                "status": _heartbeat_status(data_dir, router_id),
                "last_seen": last_seen,
                "firmware": "-",
                "usage_24h": "-",
                "model": "Federated-IDS",
            }
        )

    return devices


def _build_alerts(metrics: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    anomalies = (metrics or {}).get("anomalies_by_router", {})
    for router_id, count in anomalies.items():
        if count <= 0:
            continue
        severity = "low"
        if count >= 10:
            severity = "high"
        elif count >= 5:
            severity = "medium"
        alerts.append(
            {
                "title": router_id,
                "message": f"{count} anomalies detected in the last round.",
                "time": metrics.get("timestamp", "-"),
                "severity": severity,
            }
        )

    return alerts


def _load_profile(data_dir: Path) -> Dict[str, Any]:
    profile_path = data_dir / "client_profile.json"
    if not profile_path.exists():
        return {}
    try:
        return json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_chart(weights_dir: Path) -> Dict[str, Any]:
    files = _latest_metrics_files(weights_dir)
    if not files:
        return {"labels": ["00", "04", "08", "12"], "usage": [0, 0, 0, 0], "speed": [0, 0, 0, 0]}

    labels: List[str] = []
    usage: List[int] = []
    speed: List[int] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        round_id = payload.get("round", 0)
        labels.append(str(round_id))
        total = int(payload.get("total_anomalies", 0))
        usage.append(total)
        speed.append(max(10, 50 - min(total, 40)))

    if not labels:
        return {"labels": ["00", "04", "08", "12"], "usage": [0, 0, 0, 0], "speed": [0, 0, 0, 0]}

    return {"labels": labels, "usage": usage, "speed": speed}


def _summary(devices: List[Dict[str, Any]], alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(devices)
    online = sum(1 for d in devices if d.get("status") == "online")
    degraded = sum(1 for d in devices if d.get("status") == "degraded")
    return {
        "total": total,
        "online": online,
        "degraded": degraded,
        "alerts_24h": len(alerts),
    }


app = FastAPI(title="FedLearning Client API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/api/dashboard")
async def dashboard() -> Dict[str, Any]:
    data_dir = _resolve_data_dir()
    weights_dir = data_dir / "global_weights"
    metrics = _load_latest_metrics(weights_dir)

    devices = _build_devices(metrics, data_dir)
    alerts = _build_alerts(metrics)
    chart = _build_chart(weights_dir)

    profile = _load_profile(data_dir)
    service = {
        "plan": profile.get("plan", "Unknown"),
        "next_bill": profile.get("next_bill", "Unknown"),
        "due_date": profile.get("due_date", "Unknown"),
        "billing_cycle": profile.get("billing_cycle", "Unknown"),
        "router_model": profile.get("router_model", "Unknown"),
        "uptime": float(profile.get("uptime", 0.0)),
        "speed_consistency": float(profile.get("speed_consistency", 0.0)),
    }

    return {
        "summary": _summary(devices, alerts),
        "top_devices": [
            {"name": d.get("name", "Device"), "usage": d.get("usage_24h", "0 GB")}
            for d in devices[:4]
        ],
        "chart": chart,
        "devices": devices,
        "alerts": alerts,
        "service": service,
    }
