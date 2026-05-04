"""Dashboard control-plane files shared with the ClawdBot agent."""

from __future__ import annotations

import json
import os
import time
from uuid import uuid4

from clawdbot.flow_roles import parse_ip_csv


def _read_json(path: str, default):
    """Read json."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: str, payload) -> None:
    """Write json."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def load_protected_ips(path: str, defaults=None) -> frozenset[str]:
    """Load dashboard-managed protected IPs, falling back to defaults."""
    default_set = parse_ip_csv(defaults)
    if not path or not os.path.exists(path):
        return default_set
    payload = _read_json(path, {})
    if isinstance(payload, dict):
        return parse_ip_csv(payload.get("ips", []))
    if isinstance(payload, list):
        return parse_ip_csv(payload)
    return default_set


def save_protected_ips(path: str, ips) -> frozenset[str]:
    """Persist the exact protected IP set and return it."""
    clean = sorted(parse_ip_csv(ips))
    _write_json(path, {"ips": clean, "updated_at": time.time()})
    return frozenset(clean)


def load_firewall_requests(path: str) -> list[dict]:
    """Load firewall requests data."""
    payload = _read_json(path, [])
    return payload if isinstance(payload, list) else []


def save_firewall_requests(path: str, requests: list[dict]) -> None:
    """Persist firewall requests data."""
    _write_json(path, requests)


def queue_firewall_request(
    path: str,
    *,
    action: str,
    ip: str,
    ttl: int | None = None,
    reason: str = "dashboard",
) -> dict:
    """Append a pending firewall request for the agent to process."""
    requests = load_firewall_requests(path)
    now = time.time()
    req = {
        "id": uuid4().hex,
        "action": action,
        "ip": str(ip).strip(),
        "ttl": int(ttl) if ttl else None,
        "reason": str(reason or "dashboard").strip() or "dashboard",
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "result": None,
    }
    requests.append(req)
    save_firewall_requests(path, requests)
    return req


def pending_firewall_requests(path: str) -> list[dict]:
    """Return firewall requests that are still pending."""
    return [req for req in load_firewall_requests(path) if req.get("status") == "pending"]


def complete_firewall_request(path: str, request_id: str, result: dict) -> None:
    """Mark a firewall request as completed."""
    requests = load_firewall_requests(path)
    now = time.time()
    for req in requests:
        if req.get("id") == request_id:
            req["status"] = "applied" if result.get("applied") else "skipped"
            req["updated_at"] = now
            req["result"] = result
            break
    save_firewall_requests(path, requests)
