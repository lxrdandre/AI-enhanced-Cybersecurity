from __future__ import annotations

import csv
import ipaddress
import io
import json
import math
import os
import sqlite3
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from flask import Flask, Response, jsonify, redirect, render_template, request, url_for

from clawdbot.control_plane import (
    load_firewall_requests,
    load_protected_ips,
    queue_firewall_request,
    save_protected_ips,
)
from clawdbot.flow_roles import normalize_flow_roles, parse_ip_csv
from app.triage import response_actions_for_label


@dataclass(frozen=True)
class DashboardSettings:
    """Container for dashboard settings configuration values."""
    project_root: str
    log_dir: str
    audit_log: str
    api_url: str
    threat_db: str = ""
    protected_ips_file: str = ""
    firewall_queue: str = ""
    max_lines: int = 8000
    refresh_seconds: int = 5
    ignored_ports: frozenset[int] = frozenset({22, 64295, 5000, 8000})
    protected_ips: frozenset[str] = frozenset()
    health_services: tuple[str, ...] = (
        "ids-api.service",
        "clawdbot-agent.service",
        "ids-dashboard.service",
        "ollama.service",
    )
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    ollama_base_url: str = "http://127.0.0.1:11434"
    triage_backend: str = "ollama"
    health_error_window_seconds: int = 3600


def _parse_ports(value: str) -> frozenset[int]:
    """Parse ports."""
    ports = {22, 64295, 5000, 8000}
    if value.strip().lower() in {"none", "off", "false", "0"}:
        return frozenset()
    for item in value.split(","):
        item = item.strip()
        if item.isdigit():
            ports.add(int(item))
    return frozenset(ports)


def _parse_csv_items(value: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Parse csv items."""
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or default


def settings_from_env() -> DashboardSettings:
    """Build dashboard settings from environment variables."""
    root = os.path.abspath(os.environ.get("TON_IOT_PROJECT_ROOT") or os.getcwd())
    log_dir = (
        os.environ.get("TON_IOT_DASHBOARD_LOG_DIR")
        or os.environ.get("CLAWDBOT_LOG_DIR")
        or os.path.join(root, "logs")
    )
    return DashboardSettings(
        project_root=root,
        log_dir=log_dir,
        audit_log=os.environ.get(
            "TON_IOT_DASHBOARD_AUDIT_LOG",
            os.path.join(root, "artifacts", "audit", "analyze_events.jsonl"),
        ),
        api_url=os.environ.get("TON_IOT_DASHBOARD_API_URL", "http://127.0.0.1:8000").rstrip("/"),
        threat_db=os.environ.get(
            "TON_IOT_DASHBOARD_THREAT_DB",
            os.environ.get("THREAT_CACHE_DB", os.path.join(root, "data", "threat_cache.db")),
        ),
        protected_ips_file=os.environ.get(
            "TON_IOT_DASHBOARD_PROTECTED_IPS_FILE",
            os.path.join(root, "data", "protected_ips.json"),
        ),
        firewall_queue=os.environ.get(
            "TON_IOT_DASHBOARD_FIREWALL_QUEUE",
            os.path.join(root, "data", "firewall_requests.json"),
        ),
        max_lines=int(os.environ.get("TON_IOT_DASHBOARD_MAX_LINES", "8000")),
        refresh_seconds=int(os.environ.get("TON_IOT_DASHBOARD_REFRESH_SECONDS", "5")),
        ignored_ports=_parse_ports(os.environ.get("TON_IOT_DASHBOARD_IGNORE_PORTS", "")),
        protected_ips=parse_ip_csv(
            os.environ.get("TON_IOT_DASHBOARD_PROTECTED_IPS")
            or os.environ.get("CLAWDBOT_PROTECTED_IPS", "")
        ),
        health_services=_parse_csv_items(
            os.environ.get("TON_IOT_DASHBOARD_HEALTH_SERVICES", ""),
            (
                "ids-api.service",
                "clawdbot-agent.service",
                "ids-dashboard.service",
                "ollama.service",
            ),
        ),
        telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        triage_backend=os.environ.get("TON_IOT_TRIAGE_BACKEND", "ollama").lower(),
        health_error_window_seconds=int(os.environ.get("TON_IOT_DASHBOARD_HEALTH_ERROR_WINDOW_SECONDS", "3600")),
    )


def effective_protected_ips(settings: DashboardSettings) -> frozenset[str]:
    """Return protected IPs after merging configured and persisted values."""
    return load_protected_ips(settings.protected_ips_file, settings.protected_ips)


def _valid_ip(value: str) -> bool:
    """Return True when a value parses as an IP address."""
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def read_jsonl_tail(path: str, max_lines: int) -> list[dict[str, Any]]:
    """Read jsonl tail."""
    if not os.path.exists(path):
        return []
    rows: deque[dict[str, Any]] = deque(maxlen=max_lines)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return list(rows)


def _epoch(row: dict[str, Any]) -> float:
    """Convert a timestamp-like value to epoch seconds."""
    for key in ("epoch", "timestamp"):
        value = row.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return 0.0


def _iso(ts: float) -> str:
    """Format epoch seconds for dashboard display."""
    if not ts:
        return "unknown"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float, returning the default on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct(part: int | float, total: int | float) -> float:
    """Format a numerator and denominator as a percentage."""
    return round((float(part) / float(total) * 100.0), 1) if total else 0.0


def _fetch_json(url: str, timeout: float = 0.7) -> dict[str, Any] | None:
    """Fetch JSON from an HTTP endpoint with timeout handling."""
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _counter_items(counter: Counter, limit: int = 8) -> list[dict[str, Any]]:
    """Return sorted name/value rows from a counter."""
    return [{"name": str(name), "value": int(value)} for name, value in counter.most_common(limit)]


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert a value to int, returning the default on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _labels_from_json(value: Any) -> list[str]:
    """Decode a JSON label list into normalized strings."""
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return [str(value)]
    if not isinstance(parsed, list):
        return [str(parsed)]
    return [str(item) for item in parsed if str(item)]


def _reputation_badge(row: dict[str, Any]) -> tuple[str, str]:
    """Return the dashboard reputation badge and CSS class."""
    cumulative = _safe_int(row.get("cumulative_severity"))
    abuse = max(_safe_int(row.get("abuseipdb_score"), -1), 0)
    vt = max(_safe_int(row.get("vt_malicious"), -1), 0)
    otx = max(_safe_int(row.get("otx_pulse_count"), -1), 0)
    if abuse >= 80 or vt >= 3 or cumulative >= 5:
        return "Known-bad", "known"
    if abuse >= 40 or vt >= 1 or otx >= 3 or cumulative >= 2:
        return "Suspicious", "suspicious"
    return "Unknown", "unknown"


def _score_value(value: Any) -> str:
    """Return a display-safe external reputation score."""
    int_value = _safe_int(value, -1)
    return "-" if int_value < 0 else str(int_value)


def load_ip_intel(db_path: str, protected_ips: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Load the SQLite IP intelligence cache for the dashboard page."""
    result: dict[str, Any] = {
        "db_path": db_path,
        "db_exists": os.path.exists(db_path),
        "error": "",
        "rows": [],
        "total_ips": 0,
        "total_hits": 0,
        "known_bad": 0,
        "suspicious": 0,
        "external_checked": 0,
        "top_labels": [],
        "mitre_count": 0,
        "mitre_updated": "never",
    }
    if not result["db_exists"]:
        return result

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        result["error"] = f"Could not open threat cache: {exc}"
        return result

    try:
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ip_intel'"
        ).fetchone()
        if table is None:
            result["error"] = "Threat cache does not contain an ip_intel table yet."
            return result

        rows = conn.execute(
            """SELECT *
               FROM ip_intel
               ORDER BY cumulative_severity DESC, hit_count DESC, last_seen DESC"""
        ).fetchall()
        label_counts: Counter[str] = Counter()
        display_rows = []
        for raw in rows:
            row = dict(raw)
            if str(row.get("ip", "")).strip() in protected_ips:
                continue
            labels = _labels_from_json(row.get("labels"))
            label_counts.update(labels)
            badge, badge_class = _reputation_badge(row)
            first_seen = _safe_float(row.get("first_seen"))
            last_seen = _safe_float(row.get("last_seen"))
            api_checked_at = _safe_float(row.get("api_checked_at"))
            display = {
                "ip": row.get("ip", "-"),
                "first_seen": _iso(first_seen),
                "last_seen": _iso(last_seen),
                "api_checked_at": _iso(api_checked_at) if api_checked_at else "never",
                "hit_count": _safe_int(row.get("hit_count")),
                "cumulative_severity": _safe_int(row.get("cumulative_severity")),
                "labels": labels,
                "labels_text": ", ".join(labels) if labels else "-",
                "abuseipdb_score": _score_value(row.get("abuseipdb_score")),
                "vt_malicious": _score_value(row.get("vt_malicious")),
                "otx_pulse_count": _score_value(row.get("otx_pulse_count")),
                "badge": badge,
                "badge_class": badge_class,
            }
            display_rows.append(display)

        mitre_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='mitre_attack'"
        ).fetchone()
        if mitre_table is not None:
            mitre_count = conn.execute("SELECT COUNT(*) AS c FROM mitre_attack").fetchone()
            result["mitre_count"] = int(mitre_count["c"])

        meta_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if meta_table is not None:
            meta = conn.execute(
                "SELECT value FROM meta WHERE key = 'mitre_updated_at'"
            ).fetchone()
            if meta:
                result["mitre_updated"] = _iso(_safe_float(meta["value"]))

        result.update({
            "rows": display_rows,
            "total_ips": len(display_rows),
            "total_hits": sum(row["hit_count"] for row in display_rows),
            "known_bad": sum(1 for row in display_rows if row["badge_class"] == "known"),
            "suspicious": sum(1 for row in display_rows if row["badge_class"] == "suspicious"),
            "external_checked": sum(1 for row in display_rows if row["api_checked_at"] != "never"),
            "top_labels": _counter_items(label_counts, 10),
        })
    except sqlite3.Error as exc:
        result["error"] = f"Could not read threat cache: {exc}"
    finally:
        conn.close()

    return result


def _attack_label(row: dict[str, Any]) -> str:
    """Return the primary attack label for an event row."""
    pred = row.get("prediction") or {}
    triage = row.get("triage") or {}
    return str(triage.get("label") or pred.get("label") or pred.get("predicted_label") or "unknown")


def _attack_severity(row: dict[str, Any]) -> str:
    """Return normalized triage severity for an event row."""
    return str((row.get("triage") or {}).get("severity") or "unknown").lower()


def _attack_role(row: dict[str, Any]) -> str:
    """Return the incident role for an event row."""
    return str((row.get("triage") or {}).get("incident_role") or "primary").lower()


def _flow_roles(row: dict[str, Any], protected_ips: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Return normalized source and target fields for an event row."""
    return normalize_flow_roles(row.get("flow") or {}, protected_ips)


def _incident_key(row: dict[str, Any], protected_ips: frozenset[str] = frozenset()) -> str:
    """Build a stable incident key from row metadata."""
    incident_id = row.get("incident_id")
    if incident_id:
        return f"incident:{incident_id}"

    triage = row.get("triage") or {}
    roles = _flow_roles(row, protected_ips)
    primary = triage.get("incident_primary_label") or _attack_label(row)
    src = roles.get("originator_ip") or "-"
    dst = roles.get("target_ip") or "-"
    port = roles.get("target_port") or "-"
    time_bucket = int(_epoch(row) // 300) if _epoch(row) else 0
    if primary == "scanning":
        return f"campaign:scanning:{src}:{time_bucket}"
    if primary == "ddos_dos":
        return f"campaign:ddos_dos:{dst}:{port}:{time_bucket}"
    return f"campaign:{primary}:{src}:{dst}:{port}:{time_bucket}"


def _incident_counts(
    rows: list[dict[str, Any]],
    protected_ips: frozenset[str] = frozenset(),
) -> dict[str, int]:
    """Build primary and secondary incident counts."""
    primary_keys: set[str] = set()
    primary_by_key: dict[str, str] = {}
    possible_labels: set[tuple[str, str]] = set()
    for row in rows:
        key = _incident_key(row, protected_ips)
        label = _attack_label(row)
        triage_primary = (row.get("triage") or {}).get("incident_primary_label")
        if triage_primary:
            primary_by_key.setdefault(key, str(triage_primary))
        if _attack_role(row) != "secondary":
            primary_keys.add(key)
            primary_by_key.setdefault(key, label)

    for row in rows:
        if _attack_role(row) != "secondary":
            continue
        key = _incident_key(row, protected_ips)
        label = _attack_label(row)
        primary_label = primary_by_key.get(key, (row.get("triage") or {}).get("incident_primary_label") or label)
        if label != primary_label:
            possible_labels.add((key, label))

    return {
        "threats": len(primary_keys),
        "possible": len(possible_labels),
    }


def _fallback_actions(label: str) -> list[str]:
    """Return fallback response actions for a label."""
    return response_actions_for_label(label)


def _response_actions(label: str, triage: dict[str, Any]) -> list[str]:
    """Return triage response actions with dashboard defaults applied."""
    generic_fragments = (
        "Review correlated logs for same source/destination context.",
        "Validate whether this pattern matches expected baseline behavior.",
        "Review correlated logs for the same source and destination.",
        "Validate whether this traffic is expected for the asset and time window.",
    )
    actions = [
        str(action).strip()
        for action in triage.get("next_actions", [])
        if str(action).strip()
        and str(action).strip() not in generic_fragments
    ]
    return actions[:5] if len(actions) >= 3 else _fallback_actions(label)


def _is_ignored_port_attack(row: dict[str, Any], ignored_ports: frozenset[int]) -> bool:
    """Return whether ignored port attack."""
    flow = row.get("flow") or {}
    for key in ("src_port", "dst_port"):
        try:
            if int(flow.get(key)) in ignored_ports:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _timeline(
    attacks: list[dict[str, Any]],
    now: float,
    protected_ips: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Build timeline buckets for recent detections."""
    window_seconds = 6 * 60 * 60
    bucket_seconds = 15 * 60
    bucket_count = window_seconds // bucket_seconds
    start = now - window_seconds
    buckets: dict[int, dict[str, set[str]]] = defaultdict(lambda: {"total": set(), "high": set()})
    for row in attacks:
        if _attack_role(row) == "secondary":
            continue
        ts = _epoch(row)
        if ts < start:
            continue
        bucket = min(bucket_count - 1, max(0, int((ts - start) // bucket_seconds)))
        key = _incident_key(row, protected_ips)
        severity = _attack_severity(row)
        buckets[bucket]["total"].add(key)
        if severity in {"high", "critical"}:
            buckets[bucket]["high"].add(key)

    points = []
    for idx in range(bucket_count):
        ts = start + idx * bucket_seconds
        label = datetime.fromtimestamp(ts).strftime("%H:%M")
        points.append({
            "label": label,
            "total": len(buckets[idx]["total"]),
            "high": len(buckets[idx]["high"]),
        })
    return points


def _severity_rank(value: str) -> int:
    """Return numeric sort order for a severity value."""
    return {"unknown": 0, "low": 1, "review": 2, "medium": 3, "high": 4, "critical": 5}.get(value, 0)


def _display_value(values: list[Any], suffix: str) -> str:
    """Return a compact placeholder-safe display value."""
    clean = sorted({str(value) for value in values if value not in {None, "", "-"}}, key=str)
    if not clean:
        return "-"
    return clean[0] if len(clean) == 1 else f"{len(clean)} {suffix}"


def _techniques_from_rows(rows: list[dict[str, Any]], primary_label: str) -> list[dict[str, str]]:
    """Collect MITRE techniques from primary incident rows."""
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        if _attack_role(row) == "secondary" or _attack_label(row) != primary_label:
            continue
        for tech in (row.get("triage") or {}).get("mitre_techniques", []):
            if not isinstance(tech, dict):
                continue
            tid = str(tech.get("id", "?"))
            by_id[tid] = {
                "id": tid,
                "name": str(tech.get("name", "?")),
                "confidence": str(tech.get("confidence", "")),
                "reason": str(tech.get("reason", "")),
            }
    return list(by_id.values())


def _top_probabilities(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the highest model probabilities for display."""
    probabilities = (row.get("prediction") or {}).get("probabilities") or {}
    return sorted(
        (
            {"label": str(name), "value": round(_safe_float(value), 4)}
            for name, value in probabilities.items()
        ),
        key=lambda item: item["value"],
        reverse=True,
    )[:5]


def _recent_attacks(
    attacks: list[dict[str, Any]],
    limit: int = 18,
    protected_ips: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Build recent attack rows for the dashboard."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attacks:
        grouped[_incident_key(row, protected_ips)].append(row)

    incidents = []
    for key, rows in grouped.items():
        rows = sorted(rows, key=_epoch, reverse=True)
        primary_rows = [row for row in rows if _attack_role(row) != "secondary"]
        main = primary_rows[0] if primary_rows else rows[0]
        main_pred = main.get("prediction") or {}
        main_triage = main.get("triage") or {}
        primary_label = str(main_triage.get("incident_primary_label") or _attack_label(main))
        flow_rows = [row.get("flow") or {} for row in rows]
        role_rows = [normalize_flow_roles(flow, protected_ips) for flow in flow_rows]
        secondary_rows = [
            row for row in rows
            if _attack_role(row) == "secondary"
            and _attack_label(row) != primary_label
        ]
        secondary_counts = Counter(_attack_label(row) for row in secondary_rows)
        possible_count = len(secondary_counts)
        latest_epoch = max(_epoch(row) for row in rows)
        severity = max((_attack_severity(row) for row in (primary_rows or rows)), key=_severity_rank)
        confidence = max(_safe_float((row.get("prediction") or {}).get("confidence")) for row in (primary_rows or rows))
        route_counts = Counter(str((row.get("prediction") or {}).get("route", "-")) for row in rows)
        sample_flows = []
        for row in rows[:10]:
            flow = row.get("flow") or {}
            roles = normalize_flow_roles(flow, protected_ips)
            sample_flows.append({
                "time": _iso(_epoch(row)),
                "label": _attack_label(row),
                "role": _attack_role(row),
                "src": roles.get("originator_ip") or "-",
                "dst": roles.get("target_ip") or "-",
                "src_port": roles.get("originator_port") or "-",
                "dst_port": roles.get("target_port") or "-",
                "proto": flow.get("proto", "-"),
            })

        tactics = []
        for row in primary_rows or [main]:
            for tactic in (row.get("triage") or {}).get("mitre_tactics", []):
                if str(tactic) not in tactics:
                    tactics.append(str(tactic))

        block_result = next(((row.get("actions") or {}).get("block_result") for row in rows if (row.get("actions") or {}).get("block_result")), None)
        reputation = next((row.get("reputation") for row in rows if row.get("reputation")), None)
        summary = (
            f"1 primary {primary_label} incident across {len(rows)} flow(s)"
            f" with {possible_count} possible secondary attack label(s)."
        )
        if main_triage.get("summary"):
            summary = f"{summary} {main_triage['summary']}"

        incidents.append({
            "event_id": key,
            "epoch": round(latest_epoch, 3),
            "time": _iso(latest_epoch),
            "label": primary_label,
            "severity": severity,
            "role": "primary",
            "primary_label": primary_label,
            "route": route_counts.most_common(1)[0][0] if route_counts else "-",
            "router_confidence": round(_safe_float(main_pred.get("router_confidence")), 3)
            if main_pred.get("router_confidence") is not None else None,
            "confidence": round(confidence, 3),
            "confidence_note": main_triage.get("confidence_note", ""),
            "src": _display_value([roles.get("originator_ip") for roles in role_rows], "sources"),
            "dst": _display_value([roles.get("target_ip") for roles in role_rows], "targets"),
            "src_port": "-",
            "port": _display_value([roles.get("target_port") for roles in role_rows], "ports"),
            "proto": _display_value([flow.get("proto") for flow in flow_rows], "protocols"),
            "summary": summary,
            "source": main_triage.get("source", ""),
            "telegram_sent": any(bool((row.get("actions") or {}).get("telegram_sent")) for row in rows),
            "secondary_reason": "",
            "llm_reclassified": any(bool((row.get("triage") or {}).get("llm_reclassified")) for row in rows),
            "mitre_tactics": tactics,
            "mitre_techniques": _techniques_from_rows(rows, primary_label),
            "next_actions": _response_actions(primary_label, main_triage),
            "top_probabilities": _top_probabilities(main),
            "block_result": block_result,
            "reputation": reputation,
            "audit_id": main.get("audit_id", ""),
            "incident_id": key,
            "incident_summary": {
                "threat_count": 1,
                "primary_label": primary_label,
                "possible_count": possible_count,
                "secondary_labels": [
                    {"name": str(label), "value": int(count)}
                    for label, count in secondary_counts.most_common()
                ],
                "flow_count": len(rows),
            },
            "sample_flows": sample_flows,
            "flow_count": len(rows),
            "possible_count": possible_count,
        })

    return sorted(incidents, key=lambda row: row["epoch"], reverse=True)[:limit]


def _recent_actions(actions: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    """Build recent firewall/action rows for the dashboard."""
    rows = sorted(actions, key=_epoch, reverse=True)[:limit]
    return [
        {
            "time": _iso(_epoch(row)),
            "event": row.get("event", "action"),
            "detail": row.get("detail", {}),
        }
        for row in rows
    ]


def _audit_metrics(audits: list[dict[str, Any]], cutoff: float) -> dict[str, Any]:
    """Summarize API audit log activity."""
    recent = [row for row in audits if _epoch(row) >= cutoff]
    predictions = [
        pred
        for row in recent
        for pred in row.get("predictions", [])
        if isinstance(pred, dict)
    ]
    label_counts = Counter(str(pred.get("predicted_label", "unknown")) for pred in predictions)
    route_counts = Counter(str(pred.get("route", "single")) for pred in predictions)
    unknown = label_counts.get("unknown", 0)
    total_predictions = len(predictions)
    return {
        "batches": len(recent),
        "records": int(sum(int(row.get("record_count", 0) or 0) for row in recent)),
        "llm_errors": int(sum(1 for row in recent if row.get("llm_error"))),
        "unknown_rate": _pct(unknown, total_predictions),
        "labels": _counter_items(label_counts, 9),
        "routes": _counter_items(route_counts, 5),
    }


def _api_status(api_url: str) -> dict[str, Any]:
    """Check and summarize IDS API availability."""
    health = _fetch_json(f"{api_url}/health") if api_url else None
    metadata = _fetch_json(f"{api_url}/metadata") if health and health.get("status") == "ok" else None
    online = bool(health and health.get("status") == "ok")
    return {
        "online": online,
        "status": "online" if online else "offline",
        "model": (metadata or {}).get("model_name", "unavailable"),
        "feature_count": (metadata or {}).get("feature_count"),
        "routing_enabled": bool((metadata or {}).get("routing_enabled")),
        "classes": (metadata or {}).get("class_names", []),
        "artifact_dir": (health or {}).get("artifact_dir", ""),
        "startup_error": (health or {}).get("startup_error") or "",
    }


def _run_readonly_command(args: list[str], timeout: float = 1.5) -> tuple[int, str, str]:
    """Run a read-only health command with timeout protection."""
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return 127, "", f"{args[0]} is not installed"
    except subprocess.TimeoutExpired:
        return 124, "", f"{args[0]} timed out"
    except OSError as exc:
        return 1, "", str(exc)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _health_item(
    *,
    name: str,
    category: str,
    status: str,
    summary: str,
    detail: str = "",
    value: str = "",
) -> dict[str, str]:
    """Build one dashboard health item payload."""
    return {
        "name": name,
        "category": category,
        "status": status,
        "summary": summary,
        "detail": detail,
        "value": value,
    }


def _systemd_service_status(service: str) -> dict[str, str]:
    """Read systemd status for a configured service."""
    code, stdout, stderr = _run_readonly_command([
        "systemctl",
        "show",
        service,
        "--property=LoadState,ActiveState,SubState,Result,NRestarts,ExecMainStatus",
        "--no-pager",
    ])
    if code != 0:
        return _health_item(
            name=service,
            category="Services",
            status="warning",
            summary="Status unavailable",
            detail=stderr or stdout or "systemctl could not read this unit.",
            value="unknown",
        )

    props: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            props[key] = value

    active = props.get("ActiveState", "unknown")
    sub = props.get("SubState", "")
    load = props.get("LoadState", "unknown")
    result = props.get("Result", "")
    restarts = props.get("NRestarts", "")
    exec_status = props.get("ExecMainStatus", "")

    if load == "not-found":
        status = "critical"
        summary = "Service unit missing"
    elif active == "active":
        status = "ok"
        summary = "Online"
    elif active in {"failed", "inactive", "deactivating"}:
        status = "critical"
        summary = "Offline"
    else:
        status = "warning"
        summary = f"State: {active}"

    detail_parts = [part for part in (sub, f"result={result}" if result else "", f"restarts={restarts}" if restarts else "") if part]
    if exec_status and exec_status != "0":
        detail_parts.append(f"exit={exec_status}")
        if status == "ok":
            status = "warning"
            summary = "Online with previous error"
    return _health_item(
        name=service,
        category="Services",
        status=status,
        summary=summary,
        detail=", ".join(detail_parts),
        value=active,
    )


def _ids_api_health_item(api: dict[str, Any]) -> dict[str, str]:
    """Build the IDS API health item."""
    if api.get("online"):
        detail = api.get("model") or "model metadata unavailable"
        return _health_item(
            name="IDS API endpoint",
            category="Services",
            status="ok",
            summary="Online",
            detail=str(detail),
            value="online",
        )
    startup_error = str(api.get("startup_error") or "").strip()
    return _health_item(
        name="IDS API endpoint",
        category="Services",
        status="critical",
        summary="Offline or unhealthy",
        detail=startup_error or "GET /health did not return status=ok.",
        value="offline",
    )


def _ollama_health_item(settings: DashboardSettings) -> dict[str, str]:
    """Build the Ollama health item."""
    if settings.triage_backend != "ollama":
        return _health_item(
            name="Ollama API",
            category="Integrations",
            status="warning",
            summary="Not the active triage backend",
            detail=f"TON_IOT_TRIAGE_BACKEND={settings.triage_backend}",
            value="skipped",
        )
    payload = _fetch_json(f"{settings.ollama_base_url}/api/tags", timeout=0.8)
    if payload is None:
        return _health_item(
            name="Ollama API",
            category="Integrations",
            status="critical",
            summary="Unreachable",
            detail=f"No response from {settings.ollama_base_url}/api/tags.",
            value="offline",
        )
    models = payload.get("models") if isinstance(payload, dict) else None
    count = len(models) if isinstance(models, list) else 0
    return _health_item(
        name="Ollama API",
        category="Integrations",
        status="ok",
        summary="Online",
        detail=f"{count} local model(s) visible.",
        value="online",
    )


def _telegram_health_item(settings: DashboardSettings) -> dict[str, str]:
    """Build the Telegram health item."""
    token = settings.telegram_bot_token.strip()
    chat_id = settings.telegram_chat_id.strip()
    if not token or not chat_id:
        missing = []
        if not token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not chat_id:
            missing.append("TELEGRAM_CHAT_ID")
        return _health_item(
            name="Telegram bot",
            category="Integrations",
            status="warning",
            summary="Not configured",
            detail=", ".join(missing),
            value="disabled",
        )

    base = f"https://api.telegram.org/bot{token}"
    me = _fetch_json(f"{base}/getMe", timeout=1.0)
    if not me or not me.get("ok"):
        return _health_item(
            name="Telegram bot",
            category="Integrations",
            status="critical",
            summary="Bot token failed",
            detail="Telegram getMe did not return ok=true.",
            value="offline",
        )

    query = urllib.parse.urlencode({"chat_id": chat_id})
    chat = _fetch_json(f"{base}/getChat?{query}", timeout=1.0)
    if not chat or not chat.get("ok"):
        username = ((me.get("result") or {}).get("username") or "bot")
        return _health_item(
            name="Telegram bot",
            category="Integrations",
            status="critical",
            summary="Chat unreachable",
            detail=f"@{username} is reachable, but the configured chat is not.",
            value="offline",
        )

    username = ((me.get("result") or {}).get("username") or "bot")
    chat_type = ((chat.get("result") or {}).get("type") or "chat")
    return _health_item(
        name="Telegram bot",
        category="Integrations",
        status="ok",
        summary="Online",
        detail=f"@{username} can access configured {chat_type}.",
        value="online",
    )


def _path_status(
    name: str,
    path: str,
    *,
    required: bool,
    kind: str = "file",
    missing_status: str | None = None,
) -> dict[str, str]:
    """Summarize filesystem path existence and permissions."""
    if not path:
        return _health_item(
            name=name,
            category="Storage",
            status="warning" if required else "ok",
            summary="Not configured",
            value="missing",
        )

    exists = os.path.isdir(path) if kind == "dir" else os.path.isfile(path)
    if not exists:
        status = missing_status or ("critical" if required else "ok")
        return _health_item(
            name=name,
            category="Storage",
            status=status,
            summary="Missing" if status != "ok" else "Not created yet",
            detail=path,
            value="missing",
        )

    readable = os.access(path, os.R_OK)
    writable = os.access(path, os.W_OK)
    if not readable:
        return _health_item(
            name=name,
            category="Storage",
            status="critical",
            summary="Not readable",
            detail=path,
            value="blocked",
        )

    return _health_item(
        name=name,
        category="Storage",
        status="ok",
        summary="Ready",
        detail=f"{path} ({'writable' if writable else 'read-only'})",
        value="ready",
    )


def _sqlite_health_item(name: str, path: str, *, required: bool) -> dict[str, str]:
    """Build a health item for a SQLite database."""
    if not os.path.exists(path):
        return _health_item(
            name=name,
            category="Storage",
            status="critical" if required else "warning",
            summary="Missing",
            detail=path,
            value="missing",
        )
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0.8)
        try:
            result = conn.execute("PRAGMA quick_check").fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return _health_item(
            name=name,
            category="Storage",
            status="critical",
            summary="SQLite error",
            detail=str(exc),
            value="error",
        )
    ok = bool(result and str(result[0]).lower() == "ok")
    return _health_item(
        name=name,
        category="Storage",
        status="ok" if ok else "critical",
        summary="Integrity OK" if ok else "Integrity check failed",
        detail=path if ok else str(result[0] if result else "no result"),
        value="ready" if ok else "error",
    )


def _storage_health_items(settings: DashboardSettings) -> list[dict[str, str]]:
    """Build filesystem and database health items."""
    return [
        _path_status("ClawdBot log directory", settings.log_dir, required=True, kind="dir"),
        _path_status("IDS audit log", settings.audit_log, required=False, missing_status="warning"),
        _sqlite_health_item("Threat intel database", settings.threat_db, required=False),
        _path_status("Firewall request queue", settings.firewall_queue, required=False),
        _path_status("Protected IP list", settings.protected_ips_file, required=False),
    ]


def _service_is_online(items: list[dict[str, Any]], service: str) -> bool:
    """Return True when a health item reports online state."""
    return any(
        item.get("name") == service
        and item.get("category") == "Services"
        and item.get("status") == "ok"
        and item.get("value") == "active"
        for item in items
    )


def _agent_lifecycle_item(actions: list[dict[str, Any]], *, agent_service_online: bool) -> dict[str, str]:
    """Build the capture-agent lifecycle health item."""
    starts = [_epoch(row) for row in actions if row.get("action") == "agent_start"]
    stops = [_epoch(row) for row in actions if row.get("action") == "agent_stop"]
    last_start = max(starts, default=0.0)
    last_stop = max(stops, default=0.0)
    if agent_service_online:
        if last_start and last_start >= last_stop:
            detail = f"Last start: {_iso(last_start)}"
        elif last_stop:
            detail = f"clawdbot-agent.service is active. Last loaded lifecycle event is stop at {_iso(last_stop)}."
        else:
            detail = "clawdbot-agent.service is active. No agent_start event exists in the loaded actions.jsonl history."
        return _health_item(
            name="ClawdBot lifecycle",
            category="Runtime",
            status="ok",
            summary="Online by service state",
            detail=detail,
            value="running",
        )
    if last_start and last_start >= last_stop:
        return _health_item(
            name="ClawdBot lifecycle",
            category="Runtime",
            status="ok",
            summary="Started",
            detail=f"Last start: {_iso(last_start)}",
            value="running",
        )
    if last_stop:
        return _health_item(
            name="ClawdBot lifecycle",
            category="Runtime",
            status="critical",
            summary="Stopped",
            detail=f"Last stop: {_iso(last_stop)}",
            value="stopped",
        )
    return _health_item(
        name="ClawdBot lifecycle",
        category="Runtime",
        status="warning",
        summary="No lifecycle event",
        detail="actions.jsonl has no agent_start event in the loaded history.",
        value="unknown",
    )


def _firewall_queue_item(settings: DashboardSettings) -> dict[str, str]:
    """Build the firewall queue health item."""
    pending = sum(1 for item in load_firewall_requests(settings.firewall_queue) if item.get("status") == "pending")
    if pending:
        return _health_item(
            name="Firewall queue",
            category="Runtime",
            status="warning",
            summary="Pending actions",
            detail=f"{pending} block/unblock request(s) waiting for the agent.",
            value=str(pending),
        )
    return _health_item(
        name="Firewall queue",
        category="Runtime",
        status="ok",
        summary="Clear",
        detail="No pending dashboard firewall requests.",
        value="0",
    )


def _audit_error_item(audits: list[dict[str, Any]], cutoff: float) -> dict[str, str]:
    """Build the audit error health item."""
    recent = [row for row in audits if _epoch(row) >= cutoff and row.get("llm_error")]
    if recent:
        latest = max((_epoch(row) for row in recent), default=0.0)
        return _health_item(
            name="LLM triage errors",
            category="Errors",
            status="warning",
            summary="Errors detected",
            detail=f"{len(recent)} LLM error(s) since {_iso(cutoff)}. Latest: {_iso(latest)}",
            value=str(len(recent)),
        )
    return _health_item(
        name="LLM triage errors",
        category="Errors",
        status="ok",
        summary="None",
        detail=f"No audit LLM errors since {_iso(cutoff)}.",
        value="0",
    )


def _journal_epoch(row: dict[str, Any]) -> float:
    """Extract epoch seconds from a journal row."""
    for key in ("__REALTIME_TIMESTAMP", "_SOURCE_REALTIME_TIMESTAMP"):
        value = row.get(key)
        try:
            return float(value) / 1_000_000.0
        except (TypeError, ValueError):
            pass
    return 0.0


def _journal_priority(row: dict[str, Any]) -> int:
    """Extract numeric priority from a journal row."""
    try:
        return int(row.get("PRIORITY", 6))
    except (TypeError, ValueError):
        return 6


def _journal_line(row: dict[str, Any]) -> str:
    """Format one journal row for display."""
    epoch = _journal_epoch(row)
    timestamp = datetime.fromtimestamp(epoch).astimezone().isoformat(timespec="seconds") if epoch else "unknown"
    host = str(row.get("_HOSTNAME") or row.get("HOSTNAME") or "-")
    ident = str(row.get("SYSLOG_IDENTIFIER") or row.get("_COMM") or "-")
    message = str(row.get("MESSAGE") or "").strip()
    return f"{timestamp} {host} {ident}: {message}"


def _is_successful_service_start(row: dict[str, Any], service: str) -> bool:
    """Return whether successful service start."""
    if str(row.get("SYSLOG_IDENTIFIER", "")).lower() != "systemd":
        return False
    message = str(row.get("MESSAGE") or "").strip().lower()
    service = service.lower()
    return message.startswith("started ") and (service in message or service.removesuffix(".service") in message)


def _journal_rows_for_service(service: str, cutoff: float) -> tuple[int, list[dict[str, Any]], str]:
    """Return recent journal rows for a systemd service."""
    args = [
        "journalctl",
        "--no-pager",
        "-o",
        "json",
        "--since",
        _iso(cutoff),
        "-n",
        "500",
        "-u",
        service,
    ]
    code, stdout, stderr = _run_readonly_command(args, timeout=2.0)
    rows: list[dict[str, Any]] = []
    if code != 0:
        return code, rows, stderr or stdout
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return code, rows, ""


def _journal_error_item(services: tuple[str, ...], cutoff: float) -> dict[str, Any]:
    """Build a health item from recent journal errors."""
    lines: list[str] = []
    unavailable: list[str] = []

    for service in services:
        code, rows, error = _journal_rows_for_service(service, cutoff)
        if code != 0:
            unavailable.append(f"{service}: {error or 'journalctl could not read service logs'}")
            continue

        latest_successful_start = max(
            (_journal_epoch(row) for row in rows if _is_successful_service_start(row, service)),
            default=0.0,
        )
        active_cutoff = max(cutoff, latest_successful_start)
        for row in rows:
            epoch = _journal_epoch(row)
            if epoch and epoch < active_cutoff:
                continue
            if _journal_priority(row) <= 3:
                lines.append(_journal_line(row))

    if unavailable and not lines:
        return {
            **_health_item(
                name="System journal errors",
                category="Errors",
                status="warning",
                summary="Journal unavailable",
                detail="; ".join(unavailable)[:500],
                value="unknown",
            ),
            "lines": [],
        }

    lines = sorted(lines)
    if lines:
        latest = lines[-1]
        return {
            **_health_item(
                name="System journal errors",
                category="Errors",
                status="critical",
                summary="Errors detected",
                detail=f"{len(lines)} unresolved error line(s). Latest: {latest[-220:]}",
                value=str(len(lines)),
            ),
            "lines": lines[-8:],
        }
    detail = f"No unresolved systemd error logs since {_iso(cutoff)}."
    if unavailable:
        detail += f" Journal unavailable for {len(unavailable)} service(s)."
    return {
        **_health_item(
            name="System journal errors",
            category="Errors",
            status="ok" if not unavailable else "warning",
            summary="None" if not unavailable else "Partial",
            detail=detail,
            value="0",
        ),
        "lines": [],
    }


def _local_log_error_item(log_dir: str, cutoff: float) -> dict[str, Any]:
    """Build a health item from local log errors."""
    patterns = (" error ", " critical ", "traceback", "exception", "failed to")
    matches: list[str] = []
    if not os.path.isdir(log_dir):
        return {
            **_health_item(
                name="Local log errors",
                category="Errors",
                status="warning",
                summary="Log directory missing",
                detail=log_dir,
                value="unknown",
            ),
            "lines": [],
        }

    for filename in sorted(os.listdir(log_dir)):
        if not filename.endswith((".log", ".err", ".out", ".txt")):
            continue
        path = os.path.join(log_dir, filename)
        try:
            if os.path.getmtime(path) < cutoff:
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                tail = deque(f, maxlen=200)
        except OSError:
            continue
        for line in tail:
            normalized = f" {line.strip().lower()} "
            if any(pattern in normalized for pattern in patterns):
                matches.append(f"{filename}: {line.strip()}")
    if matches:
        return {
            **_health_item(
                name="Local log errors",
                category="Errors",
                status="critical",
                summary="Errors detected",
                detail=f"{len(matches)} matching log line(s).",
                value=str(len(matches)),
            ),
            "lines": matches[-8:],
        }
    return {
        **_health_item(
            name="Local log errors",
            category="Errors",
            status="ok",
            summary="None",
            detail=f"No local log error patterns since {_iso(cutoff)}.",
            value="0",
        ),
        "lines": [],
    }


def _overall_health(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce grouped health checks into one status."""
    counts = Counter(str(item.get("status", "unknown")) for item in items)
    critical = counts.get("critical", 0)
    warning = counts.get("warning", 0) + counts.get("unknown", 0)
    ok = counts.get("ok", 0)
    if critical:
        status = "critical"
        headline = "Action required"
        detail = f"{critical} critical check(s), {warning} warning check(s), {ok} healthy check(s)."
    elif warning:
        status = "warning"
        headline = "Degraded"
        detail = f"{warning} warning check(s), {ok} healthy check(s), no critical failures."
    else:
        status = "ok"
        headline = "All monitored systems healthy"
        detail = f"{ok} check(s) passed. Services, bots, storage, and recent error scans are clean."
    return {
        "status": status,
        "headline": headline,
        "detail": detail,
        "counts": {
            "ok": ok,
            "warning": warning,
            "critical": critical,
        },
    }


def _group_health_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group health items by dashboard category."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item.get("category", "Other"))].append(item)
    order = ["Services", "Integrations", "Runtime", "Storage", "Errors"]
    return [
        {"name": category, "checks": grouped[category]}
        for category in order
        if grouped.get(category)
    ]


def _latest_epoch(rows: list[dict[str, Any]]) -> float:
    """Return the newest epoch from rows containing timestamps."""
    return max((_epoch(row) for row in rows), default=0.0)


def _activity_metrics(
    api: dict[str, Any],
    attacks: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    audit: dict[str, Any],
    now: float,
) -> dict[str, Any]:
    """Build dashboard activity and pulse metrics."""
    last_attack = _latest_epoch(attacks)
    last_action = _latest_epoch(actions)
    last_analysis = _latest_epoch(audits)
    last_llm_error = max((_epoch(row) for row in audits if row.get("llm_error")), default=0.0)
    recent_attack = last_attack >= now - 90
    recent_analysis = last_analysis >= now - 45

    if not api.get("online"):
        state = "dead"
        label = "dead"
        headline = "IoT VLAN telemetry link down"
        detail = "Dashboard is online, but the IPS API is not responding."
    elif recent_attack or recent_analysis:
        state = "thinking"
        label = "thinking"
        headline = "Inline IPS analyzing flows"
        detail = "Recent IoT VLAN telemetry is being classified, triaged, and correlated."
    else:
        state = "breathing"
        label = "breathing"
        headline = "IoT VLAN monitor online"
        detail = "IPS API is healthy and waiting for the next device flow batch."

    return {
        "state": state,
        "label": label,
        "headline": headline,
        "detail": detail,
        "last_attack_epoch": round(last_attack, 3),
        "last_analysis_epoch": round(last_analysis, 3),
        "last_action_epoch": round(last_action, 3),
        "last_llm_error_epoch": round(last_llm_error, 3),
        "recent_attack": recent_attack,
        "recent_analysis": recent_analysis,
        "signals": [
            {"name": "IPS API", "value": api.get("status", "offline")},
            {"name": "Router", "value": "enabled" if api.get("routing_enabled") else "single"},
            {"name": "Flow records", "value": int(audit["records"])},
            {"name": "LLM errors", "value": int(audit["llm_errors"])},
        ],
    }


def build_metrics(settings: DashboardSettings | None = None) -> dict[str, Any]:
    """Build metrics."""
    settings = settings or settings_from_env()
    protected_ips = effective_protected_ips(settings)
    now = time.time()
    attacks = [
        row for row in read_jsonl_tail(os.path.join(settings.log_dir, "attacks.jsonl"), settings.max_lines)
        if not _is_ignored_port_attack(row, settings.ignored_ports)
    ]
    actions = read_jsonl_tail(os.path.join(settings.log_dir, "actions.jsonl"), settings.max_lines)
    audits = read_jsonl_tail(settings.audit_log, settings.max_lines)

    cutoff_24h = now - 24 * 60 * 60
    cutoff_1h = now - 60 * 60
    attacks_24h = [row for row in attacks if _epoch(row) >= cutoff_24h]
    attacks_1h = [row for row in attacks if _epoch(row) >= cutoff_1h]

    labels = Counter(_attack_label(row) for row in attacks_24h)
    severities = Counter(_attack_severity(row) for row in attacks_24h)
    role_rows_24h = [_flow_roles(row, protected_ips) for row in attacks_24h]
    sources = Counter(
        roles.get("originator_ip") for roles in role_rows_24h
        if roles.get("originator_ip")
    )
    destinations = Counter(
        roles.get("target_ip") for roles in role_rows_24h
        if roles.get("target_ip")
    )
    ports = Counter(
        str(roles.get("target_port")) for roles in role_rows_24h
        if roles.get("target_port") not in {None, ""}
    )
    counts_24h = _incident_counts(attacks_24h, protected_ips)
    counts_1h = _incident_counts(attacks_1h, protected_ips)
    high_critical = len({
        _incident_key(row, protected_ips)
        for row in attacks_24h
        if _attack_role(row) != "secondary"
        and _attack_severity(row) in {"high", "critical"}
    })
    critical = len({
        _incident_key(row, protected_ips)
        for row in attacks_24h
        if _attack_role(row) != "secondary"
        and _attack_severity(row) == "critical"
    })

    audit = _audit_metrics(audits, cutoff_24h)
    api = _api_status(settings.api_url)
    risk_score = min(100, int(high_critical * 12 + counts_1h["threats"] * 4 + critical * 20))

    return {
        "generated_at": _iso(now),
        "paths": {
            "log_dir": settings.log_dir,
            "audit_log": settings.audit_log,
            "threat_db": settings.threat_db,
            "firewall_queue": settings.firewall_queue,
        },
        "api": api,
        "deployment": {
            "vlan": "IoT VLAN",
            "ips_placement": "inline edge",
            "protected_devices": len(protected_ips),
        },
        "activity": _activity_metrics(api, attacks, actions, audits, audit, now),
        "kpis": {
            "risk_score": risk_score,
            "threats_24h": counts_24h["threats"],
            "threats_1h": counts_1h["threats"],
            "possible_threats_24h": counts_24h["possible"],
            "possible_threats_1h": counts_1h["possible"],
            "high_critical_24h": high_critical,
            "analyzed_records_24h": audit["records"],
            "unknown_rate_24h": audit["unknown_rate"],
            "llm_errors_24h": audit["llm_errors"],
            "top_source": sources.most_common(1)[0][0] if sources else "-",
        },
        "series": {
            "timeline": _timeline(attacks_24h, now, protected_ips),
            "labels": _counter_items(labels, 9),
            "severities": _counter_items(severities, 6),
            "sources": _counter_items(sources, 8),
            "destinations": _counter_items(destinations, 8),
            "ports": _counter_items(ports, 8),
            "routes": audit["routes"],
            "prediction_labels": audit["labels"],
        },
        "recent": {
            "attacks": _recent_attacks(attacks, limit=200, protected_ips=protected_ips),
            "actions": _recent_actions(actions, limit=10),
        },
        "audit": audit,
    }


def _attack_rows(settings: DashboardSettings) -> list[dict[str, Any]]:
    """Return cached attack rows for export."""
    return [
        row for row in read_jsonl_tail(os.path.join(settings.log_dir, "attacks.jsonl"), settings.max_lines)
        if not _is_ignored_port_attack(row, settings.ignored_ports)
    ]


def _action_rows(settings: DashboardSettings) -> list[dict[str, Any]]:
    """Return cached action rows for export."""
    return read_jsonl_tail(os.path.join(settings.log_dir, "actions.jsonl"), settings.max_lines)


def load_ip_detail(ip: str, settings: DashboardSettings) -> dict[str, Any]:
    """Load ip detail data."""
    protected_ips = effective_protected_ips(settings)
    intel = load_ip_intel(settings.threat_db, protected_ips)
    row = next((item for item in intel["rows"] if item["ip"] == ip), None)
    attacks = _attack_rows(settings)
    related_raw = []
    role_counts: Counter[str] = Counter()
    for attack in attacks:
        roles = _flow_roles(attack, protected_ips)
        role = ""
        if roles.get("originator_ip") == ip:
            role = "originator"
        elif roles.get("target_ip") == ip:
            role = "target"
        if role:
            role_counts[role] += 1
            related_raw.append(attack)
    related = _recent_attacks(related_raw, limit=100, protected_ips=protected_ips)
    return {
        "ip": ip,
        "intel": row,
        "is_protected": ip in protected_ips,
        "role_counts": dict(role_counts),
        "incidents": related,
        "labels": _counter_items(Counter(_attack_label(row) for row in related_raw), 12),
    }


def _derive_active_blocks(actions: list[dict[str, Any]], now: float) -> dict[str, dict[str, Any]]:
    """Derive active firewall blocks from action history."""
    active: dict[str, dict[str, Any]] = {}
    for row in sorted(actions, key=_epoch):
        detail = row.get("detail") or {}
        result = detail.get("result") if isinstance(detail.get("result"), dict) else detail
        if not isinstance(result, dict):
            continue
        ip = result.get("ip")
        if not ip:
            continue
        action = result.get("action")
        if action == "unblock" and result.get("applied"):
            active.pop(ip, None)
            continue
        if action != "block" or not result.get("applied"):
            continue
        ttl = _safe_int(result.get("ttl"), 0)
        expires_at = _epoch(row) + ttl if ttl else 0
        if expires_at and expires_at <= now:
            active.pop(ip, None)
            continue
        active[ip] = {
            "ip": ip,
            "reason": result.get("reason", ""),
            "ttl": ttl,
            "expires_at": _iso(expires_at) if expires_at else "unknown",
            "seconds_remaining": max(0, int(expires_at - now)) if expires_at else 0,
            "dry_run": bool(result.get("dry_run")),
            "source": row.get("event", "firewall"),
        }
    return active


def load_firewall_panel(settings: DashboardSettings) -> dict[str, Any]:
    """Load firewall panel data."""
    now = time.time()
    actions = _action_rows(settings)
    requests = load_firewall_requests(settings.firewall_queue)
    active = sorted(_derive_active_blocks(actions, now).values(), key=lambda row: row["seconds_remaining"], reverse=True)
    pending = [req for req in requests if req.get("status") == "pending"]
    recent = sorted(requests, key=lambda req: float(req.get("updated_at") or req.get("created_at") or 0), reverse=True)[:25]
    return {
        "active": active,
        "pending": pending,
        "recent": recent,
        "queue_path": settings.firewall_queue,
        "actuator_events": [
            row for row in sorted(actions, key=_epoch, reverse=True)
            if str(row.get("event", "")).startswith("firewall")
        ][:25],
    }


def load_protected_assets(settings: DashboardSettings) -> dict[str, Any]:
    """Load protected assets data."""
    protected_ips = effective_protected_ips(settings)
    attacks = _attack_rows(settings)
    assets = []
    for ip in sorted(protected_ips):
        targeted = []
        originator_counts: Counter[str] = Counter()
        port_counts: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        last_seen = 0.0
        for row in attacks:
            roles = _flow_roles(row, protected_ips)
            if roles.get("target_ip") != ip:
                continue
            targeted.append(row)
            if roles.get("originator_ip"):
                originator_counts[str(roles["originator_ip"])] += 1
            if roles.get("target_port") not in {None, ""}:
                port_counts[str(roles["target_port"])] += 1
            label_counts[_attack_label(row)] += 1
            last_seen = max(last_seen, _epoch(row))
        assets.append({
            "ip": ip,
            "targeted_count": len(targeted),
            "unique_originators": len(originator_counts),
            "top_originators": _counter_items(originator_counts, 5),
            "top_ports": _counter_items(port_counts, 5),
            "labels": _counter_items(label_counts, 5),
            "last_seen": _iso(last_seen),
        })
    return {
        "ips": sorted(protected_ips),
        "assets": assets,
        "path": settings.protected_ips_file,
    }


MITRE_TACTIC_ORDER = [
    "Reconnaissance",
    "Resource Development",
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command and Control",
    "Exfiltration",
    "Impact",
]

MITRE_TACTIC_BY_KEY = {tactic.lower().replace(" ", "-"): tactic for tactic in MITRE_TACTIC_ORDER}


def _display_tactic(value: str) -> str:
    """Return the canonical display name for a MITRE tactic."""
    key = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    return MITRE_TACTIC_BY_KEY.get(key, str(value).replace("-", " ").title())


def _load_mitre_catalog_counts(settings: DashboardSettings) -> tuple[Counter[str], int]:
    """Load mitre catalog counts data."""
    counts: Counter[str] = Counter()
    if not settings.threat_db or not os.path.exists(settings.threat_db):
        return counts, 0
    try:
        with sqlite3.connect(settings.threat_db) as conn:
            conn.row_factory = sqlite3.Row
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='mitre_attack'"
            ).fetchone()
            if table is None:
                return counts, 0
            rows = conn.execute("SELECT tactics FROM mitre_attack").fetchall()
    except sqlite3.Error:
        return counts, 0
    for row in rows:
        try:
            tactics = json.loads(row["tactics"] or "[]")
        except (TypeError, json.JSONDecodeError):
            tactics = []
        for tactic in tactics:
            counts[_display_tactic(tactic)] += 1
    return counts, len(rows)


def _radar_label_lines(label: str, limit: int = 14) -> list[str]:
    """Split a MITRE tactic label for radar display."""
    words = label.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > limit:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [label]


def _radar_point(center: float, radius: float, angle: float, factor: float) -> tuple[float, float]:
    """Return one polar-coordinate radar point."""
    bounded = max(0.0, min(1.0, factor))
    return (
        round(center + math.cos(angle) * radius * bounded, 2),
        round(center + math.sin(angle) * radius * bounded, 2),
    )


def _build_mitre_radar(tactic_counts: Counter[str], catalog_counts: Counter[str]) -> dict[str, Any]:
    """Build mitre radar."""
    size = 620
    center = size / 2
    radius = 190
    label_radius = 260
    tactics = list(MITRE_TACTIC_ORDER)
    extras = sorted((set(tactic_counts) | set(catalog_counts)) - set(tactics))
    tactics.extend(extras)

    if not catalog_counts:
        catalog_counts = Counter({tactic: 1 for tactic in tactics})

    catalog_max = max(catalog_counts.values() or [1])
    observed_max = max(tactic_counts.values() or [1])
    total = len(tactics) or 1
    catalog_points: list[str] = []
    observed_points: list[str] = []
    catalog_markers: list[dict[str, float]] = []
    observed_markers: list[dict[str, float]] = []
    axes = []

    for index, tactic in enumerate(tactics):
        angle = -math.pi / 2 + (2 * math.pi * index / total)
        axis_x, axis_y = _radar_point(center, radius, angle, 1.0)
        label_x, label_y = _radar_point(center, label_radius, angle, 1.0)
        catalog_value = catalog_counts.get(tactic, 0)
        observed_value = tactic_counts.get(tactic, 0)
        catalog_factor = catalog_value / catalog_max if catalog_max else 0.0
        observed_factor = observed_value / observed_max if observed_max else 0.0
        catalog_x, catalog_y = _radar_point(center, radius, angle, catalog_factor)
        observed_x, observed_y = _radar_point(center, radius, angle, observed_factor)
        anchor = "middle"
        if math.cos(angle) > 0.25:
            anchor = "start"
        elif math.cos(angle) < -0.25:
            anchor = "end"
        label_lines = _radar_label_lines(tactic)
        axes.append({
            "tactic": tactic,
            "x": axis_x,
            "y": axis_y,
            "label_x": label_x,
            "label_y": label_y,
            "anchor": anchor,
            "label_lines": label_lines,
            "first_dy": -7 * (len(label_lines) - 1),
            "catalog_count": catalog_value,
            "observed_count": observed_value,
            "catalog_score": round(catalog_factor, 2),
            "observed_score": round(observed_factor, 2),
        })
        catalog_points.append(f"{catalog_x},{catalog_y}")
        observed_points.append(f"{observed_x},{observed_y}")
        catalog_markers.append({"x": catalog_x, "y": catalog_y})
        observed_markers.append({"x": observed_x, "y": observed_y})

    rings = []
    for step in range(1, 6):
        factor = step / 5
        points = []
        for index in range(total):
            angle = -math.pi / 2 + (2 * math.pi * index / total)
            x, y = _radar_point(center, radius, angle, factor)
            points.append(f"{x},{y}")
        rings.append({
            "points": " ".join(points),
            "label": f"{factor:.1f}",
            "x": round(center + 8, 2),
            "y": round(center - radius * factor + 4, 2),
        })

    return {
        "size": size,
        "center": center,
        "rings": rings,
        "axes": axes,
        "catalog_points": " ".join(catalog_points),
        "observed_points": " ".join(observed_points),
        "catalog_markers": catalog_markers,
        "observed_markers": observed_markers,
        "catalog_max": catalog_max,
        "observed_max": observed_max,
        "catalog_total": sum(catalog_counts.values()),
        "observed_total": sum(tactic_counts.values()),
    }


def load_mitre_matrix(settings: DashboardSettings) -> dict[str, Any]:
    """Load mitre matrix data."""
    attacks = _attack_rows(settings)
    cells: dict[str, dict[str, Any]] = {}
    tactic_counts: Counter[str] = Counter()
    for row in attacks:
        if _attack_role(row) == "secondary":
            continue
        triage = row.get("triage") or {}
        tactics = [_display_tactic(tactic) for tactic in triage.get("mitre_tactics", [])]
        techniques = triage.get("mitre_techniques", []) or []
        if not tactics and techniques:
            tactics = ["Unmapped"]
        for tactic in tactics:
            tactic_counts[tactic] += 1
            for tech in techniques or [{"id": "unmapped", "name": _attack_label(row)}]:
                key = f"{tactic}|{tech.get('id', 'unknown')}"
                cell = cells.setdefault(key, {
                    "tactic": tactic,
                    "id": str(tech.get("id", "unknown")),
                    "name": str(tech.get("name", _attack_label(row))),
                    "count": 0,
                    "severity": "low",
                    "last_seen": 0.0,
                })
                cell["count"] += 1
                if _severity_rank(_attack_severity(row)) > _severity_rank(cell["severity"]):
                    cell["severity"] = _attack_severity(row)
                cell["last_seen"] = max(cell["last_seen"], _epoch(row))
    tactics = [t for t in MITRE_TACTIC_ORDER if tactic_counts.get(t)]
    tactics.extend(sorted(t for t in tactic_counts if t not in tactics))
    matrix = []
    for tactic in tactics:
        items = [cell for cell in cells.values() if cell["tactic"] == tactic]
        for item in items:
            item["last_seen"] = _iso(item["last_seen"])
        matrix.append({
            "tactic": tactic,
            "count": tactic_counts[tactic],
            "techniques": sorted(items, key=lambda item: item["count"], reverse=True),
        })
    catalog_counts, catalog_technique_count = _load_mitre_catalog_counts(settings)
    return {
        "matrix": matrix,
        "radar": _build_mitre_radar(tactic_counts, catalog_counts),
        "total_techniques": len(cells),
        "total_mappings": sum(tactic_counts.values()),
        "catalog_techniques": catalog_technique_count,
    }


def load_model_quality(settings: DashboardSettings) -> dict[str, Any]:
    """Load model quality data."""
    audits = read_jsonl_tail(settings.audit_log, settings.max_lines)
    predictions = [
        pred for row in audits
        for pred in row.get("predictions", [])
        if isinstance(pred, dict)
    ]
    confidences = [_safe_float(pred.get("confidence")) for pred in predictions]
    label_counts = Counter(str(pred.get("predicted_label", "unknown")) for pred in predictions)
    route_counts = Counter(str(pred.get("route", "single")) for pred in predictions)
    bins = Counter()
    for conf in confidences:
        if conf < 0.4:
            bins["<0.40"] += 1
        elif conf < 0.6:
            bins["0.40-0.59"] += 1
        elif conf < 0.8:
            bins["0.60-0.79"] += 1
        else:
            bins["0.80-1.00"] += 1
    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    unknown = label_counts.get("unknown", 0)
    return {
        "audit_log": settings.audit_log,
        "batches": len(audits),
        "predictions": len(predictions),
        "avg_confidence": avg_conf,
        "unknown_rate": _pct(unknown, len(predictions)),
        "llm_errors": int(sum(1 for row in audits if row.get("llm_error"))),
        "labels": _counter_items(label_counts, 12),
        "routes": _counter_items(route_counts, 8),
        "confidence_bins": _counter_items(bins, 8),
    }


def load_health_panel(settings: DashboardSettings) -> dict[str, Any]:
    """Load health panel data."""
    now = time.time()
    actions = _action_rows(settings)
    audits = read_jsonl_tail(settings.audit_log, settings.max_lines)
    api = _api_status(settings.api_url)
    cutoff = now - settings.health_error_window_seconds
    journal_item = _journal_error_item(settings.health_services, cutoff)
    local_log_item = _local_log_error_item(settings.log_dir, cutoff)
    service_items = [_systemd_service_status(service) for service in settings.health_services]
    items: list[dict[str, Any]] = [
        _ids_api_health_item(api),
        *service_items,
        _telegram_health_item(settings),
        _ollama_health_item(settings),
        _agent_lifecycle_item(actions, agent_service_online=_service_is_online(service_items, "clawdbot-agent.service")),
        _firewall_queue_item(settings),
        *_storage_health_items(settings),
        _audit_error_item(audits, cutoff),
        journal_item,
        local_log_item,
    ]
    return {
        "api": api,
        "checked_at": _iso(now),
        "window_seconds": settings.health_error_window_seconds,
        "overall": _overall_health(items),
        "groups": _group_health_items(items),
        "journal_lines": journal_item.get("lines", []),
        "local_log_lines": local_log_item.get("lines", []),
    }


def _csv_response(filename: str, rows: list[dict[str, Any]], fields: list[str]) -> Response:
    """Build a CSV download response."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _truncate_file(path: str) -> None:
    """Safely truncate a dashboard log file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8"):
        pass


def create_app(settings: DashboardSettings | None = None) -> Flask:
    """Create and configure the Flask dashboard application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["DASHBOARD_SETTINGS"] = settings or settings_from_env()

    @app.get("/")
    def index():
        """Render the dashboard overview page."""
        cfg = app.config["DASHBOARD_SETTINGS"]
        return render_template(
            "index.html",
            active_page="index",
            refresh_seconds=cfg.refresh_seconds,
            api_url=cfg.api_url,
            rail_card_small="Loading paths...",
            rail_card_small_id="pathText",
        )

    @app.get("/ip-intel")
    def ip_intel():
        """Render the IP intelligence page."""
        cfg = app.config["DASHBOARD_SETTINGS"]
        intel = load_ip_intel(cfg.threat_db, effective_protected_ips(cfg))
        return render_template(
            "ip_intel.html",
            active_page="ip_intel",
            refresh_seconds=cfg.refresh_seconds,
            intel=intel,
        )

    @app.get("/ip-intel/<ip>")
    def ip_detail(ip: str):
        """Render one IP intelligence detail response."""
        cfg = app.config["DASHBOARD_SETTINGS"]
        return render_template("ip_detail.html", active_page="ip_intel", detail=load_ip_detail(ip, cfg))

    @app.get("/firewall")
    def firewall():
        """Render the firewall operations page."""
        cfg = app.config["DASHBOARD_SETTINGS"]
        return render_template("firewall.html", active_page="firewall", firewall=load_firewall_panel(cfg))

    @app.post("/firewall/block")
    def firewall_block():
        """Queue a dashboard-requested firewall block."""
        cfg = app.config["DASHBOARD_SETTINGS"]
        ip = str(request.form.get("ip", "")).strip()
        ttl = _safe_int(request.form.get("ttl"), 3600)
        reason = str(request.form.get("reason") or "dashboard manual block")
        if _valid_ip(ip):
            queue_firewall_request(cfg.firewall_queue, action="block", ip=ip, ttl=ttl, reason=reason)
        return redirect(url_for("firewall"))

    @app.post("/firewall/unblock")
    def firewall_unblock():
        """Queue a dashboard-requested firewall unblock."""
        cfg = app.config["DASHBOARD_SETTINGS"]
        ip = str(request.form.get("ip", "")).strip()
        reason = str(request.form.get("reason") or "dashboard manual unblock")
        if _valid_ip(ip):
            queue_firewall_request(cfg.firewall_queue, action="unblock", ip=ip, reason=reason)
        return redirect(url_for("firewall"))

    @app.get("/protected-assets")
    def protected_assets():
        """Render the protected-assets management page."""
        cfg = app.config["DASHBOARD_SETTINGS"]
        return render_template("protected_assets.html", active_page="protected_assets", protected=load_protected_assets(cfg))

    @app.post("/protected-assets/add")
    def protected_assets_add():
        """Add a protected asset from dashboard form input."""
        cfg = app.config["DASHBOARD_SETTINGS"]
        ip = str(request.form.get("ip", "")).strip()
        current = set(effective_protected_ips(cfg))
        if _valid_ip(ip):
            current.add(ip)
            save_protected_ips(cfg.protected_ips_file, current)
        return redirect(url_for("protected_assets"))

    @app.post("/protected-assets/remove")
    def protected_assets_remove():
        """Remove a protected asset from dashboard form input."""
        cfg = app.config["DASHBOARD_SETTINGS"]
        ip = str(request.form.get("ip", "")).strip()
        current = set(effective_protected_ips(cfg))
        current.discard(ip)
        save_protected_ips(cfg.protected_ips_file, current)
        return redirect(url_for("protected_assets"))

    @app.get("/mitre")
    def mitre():
        """Render the MITRE coverage page."""
        return render_template("mitre.html", active_page="mitre", mitre=load_mitre_matrix(app.config["DASHBOARD_SETTINGS"]))

    @app.get("/model-quality")
    def model_quality():
        """Render the model-quality page."""
        return render_template("model_quality.html", active_page="model_quality", quality=load_model_quality(app.config["DASHBOARD_SETTINGS"]))

    @app.get("/health-panel")
    def health_panel():
        """Render the health panel page."""
        return render_template("health_panel.html", active_page="health_panel", health=load_health_panel(app.config["DASHBOARD_SETTINGS"]))

    @app.get("/api/ip-intel")
    def api_ip_intel():
        """Return IP intelligence data for the dashboard API."""
        cfg = app.config["DASHBOARD_SETTINGS"]
        return jsonify(load_ip_intel(cfg.threat_db, effective_protected_ips(cfg)))

    @app.get("/api/firewall")
    def api_firewall():
        """Return firewall panel data for the dashboard API."""
        return jsonify(load_firewall_panel(app.config["DASHBOARD_SETTINGS"]))

    @app.get("/api/metrics")
    def api_metrics():
        """Return dashboard metrics as JSON."""
        return jsonify(build_metrics(app.config["DASHBOARD_SETTINGS"]))

    @app.get("/api/events")
    def api_events():
        """Return recent event rows as JSON."""
        metrics = build_metrics(app.config["DASHBOARD_SETTINGS"])
        return jsonify(metrics["recent"])

    @app.get("/export/ip-intel.csv")
    def export_ip_intel():
        """Export ip intel."""
        cfg = app.config["DASHBOARD_SETTINGS"]
        rows = load_ip_intel(cfg.threat_db, effective_protected_ips(cfg))["rows"]
        return _csv_response(
            "ip-intel.csv",
            rows,
            ["ip", "badge", "hit_count", "cumulative_severity", "labels_text", "abuseipdb_score", "vt_malicious", "otx_pulse_count", "first_seen", "last_seen", "api_checked_at"],
        )

    @app.get("/export/incidents.csv")
    def export_incidents():
        """Export incidents."""
        cfg = app.config["DASHBOARD_SETTINGS"]
        rows = _recent_attacks(_attack_rows(cfg), limit=10000, protected_ips=effective_protected_ips(cfg))
        return _csv_response(
            "incidents.csv",
            rows,
            ["time", "label", "severity", "route", "confidence", "src", "dst", "port", "summary", "telegram_sent", "flow_count", "possible_count"],
        )

    @app.get("/health")
    def health():
        """Return service health information."""
        return jsonify({"status": "ok", "service": "ids-dashboard"})

    @app.post("/logs/clear")
    def clear_logs():
        """Clear logs."""
        cfg = app.config["DASHBOARD_SETTINGS"]
        for path in (
            os.path.join(cfg.log_dir, "attacks.jsonl"),
            os.path.join(cfg.log_dir, "actions.jsonl"),
            cfg.audit_log,
        ):
            _truncate_file(path)

        next_path = str(request.form.get("next", "")).strip()
        if next_path.startswith("/"):
            return redirect(next_path)
        return redirect(url_for("index"))

    return app


app = create_app()


def main() -> None:
    """Run the command-line entry point."""
    settings = settings_from_env()
    host = os.environ.get("TON_IOT_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("TON_IOT_DASHBOARD_PORT", "5000"))
    debug = os.environ.get("TON_IOT_DASHBOARD_DEBUG", "").lower() in {"1", "true", "yes"}
    create_app(settings).run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
