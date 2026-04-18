from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from flask import Flask, jsonify, render_template

from app.triage import response_actions_for_label


@dataclass(frozen=True)
class DashboardSettings:
    project_root: str
    log_dir: str
    audit_log: str
    api_url: str
    max_lines: int = 8000
    refresh_seconds: int = 5
    ignored_ports: frozenset[int] = frozenset({22, 64295, 5000, 8000})


def _parse_ports(value: str) -> frozenset[int]:
    ports = {22, 64295, 5000, 8000}
    if value.strip().lower() in {"none", "off", "false", "0"}:
        return frozenset()
    for item in value.split(","):
        item = item.strip()
        if item.isdigit():
            ports.add(int(item))
    return frozenset(ports)


def settings_from_env() -> DashboardSettings:
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
        max_lines=int(os.environ.get("TON_IOT_DASHBOARD_MAX_LINES", "8000")),
        refresh_seconds=int(os.environ.get("TON_IOT_DASHBOARD_REFRESH_SECONDS", "5")),
        ignored_ports=_parse_ports(os.environ.get("TON_IOT_DASHBOARD_IGNORE_PORTS", "")),
    )


def read_jsonl_tail(path: str, max_lines: int) -> list[dict[str, Any]]:
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
    for key in ("epoch", "timestamp"):
        value = row.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return 0.0


def _iso(ts: float) -> str:
    if not ts:
        return "unknown"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct(part: int | float, total: int | float) -> float:
    return round((float(part) / float(total) * 100.0), 1) if total else 0.0


def _fetch_json(url: str, timeout: float = 0.7) -> dict[str, Any] | None:
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _counter_items(counter: Counter, limit: int = 8) -> list[dict[str, Any]]:
    return [{"name": str(name), "value": int(value)} for name, value in counter.most_common(limit)]


def _attack_label(row: dict[str, Any]) -> str:
    pred = row.get("prediction") or {}
    triage = row.get("triage") or {}
    return str(triage.get("label") or pred.get("label") or pred.get("predicted_label") or "unknown")


def _attack_severity(row: dict[str, Any]) -> str:
    return str((row.get("triage") or {}).get("severity") or "unknown").lower()


def _attack_role(row: dict[str, Any]) -> str:
    return str((row.get("triage") or {}).get("incident_role") or "primary").lower()


def _incident_key(row: dict[str, Any]) -> str:
    incident_id = row.get("incident_id")
    if incident_id:
        return f"incident:{incident_id}"

    triage = row.get("triage") or {}
    flow = row.get("flow") or {}
    primary = triage.get("incident_primary_label") or _attack_label(row)
    src = flow.get("src_ip", "-")
    dst = flow.get("dst_ip", "-")
    port = flow.get("dst_port", "-")
    time_bucket = int(_epoch(row) // 300) if _epoch(row) else 0
    if primary == "scanning":
        return f"campaign:scanning:{src}:{time_bucket}"
    if primary == "ddos_dos":
        return f"campaign:ddos_dos:{dst}:{port}:{time_bucket}"
    return f"campaign:{primary}:{src}:{dst}:{port}:{time_bucket}"


def _incident_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    primary_keys: set[str] = set()
    primary_by_key: dict[str, str] = {}
    possible_labels: set[tuple[str, str]] = set()
    for row in rows:
        key = _incident_key(row)
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
        key = _incident_key(row)
        label = _attack_label(row)
        primary_label = primary_by_key.get(key, (row.get("triage") or {}).get("incident_primary_label") or label)
        if label != primary_label:
            possible_labels.add((key, label))

    return {
        "threats": len(primary_keys),
        "possible": len(possible_labels),
    }


def _fallback_actions(label: str) -> list[str]:
    return response_actions_for_label(label)


def _response_actions(label: str, triage: dict[str, Any]) -> list[str]:
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
    flow = row.get("flow") or {}
    for key in ("src_port", "dst_port"):
        try:
            if int(flow.get(key)) in ignored_ports:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _timeline(attacks: list[dict[str, Any]], now: float) -> list[dict[str, Any]]:
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
        key = _incident_key(row)
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
    return {"unknown": 0, "low": 1, "review": 2, "medium": 3, "high": 4, "critical": 5}.get(value, 0)


def _display_value(values: list[Any], suffix: str) -> str:
    clean = sorted({str(value) for value in values if value not in {None, "", "-"}}, key=str)
    if not clean:
        return "-"
    return clean[0] if len(clean) == 1 else f"{len(clean)} {suffix}"


def _techniques_from_rows(rows: list[dict[str, Any]], primary_label: str) -> list[dict[str, str]]:
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
    probabilities = (row.get("prediction") or {}).get("probabilities") or {}
    return sorted(
        (
            {"label": str(name), "value": round(_safe_float(value), 4)}
            for name, value in probabilities.items()
        ),
        key=lambda item: item["value"],
        reverse=True,
    )[:5]


def _recent_attacks(attacks: list[dict[str, Any]], limit: int = 18) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attacks:
        grouped[_incident_key(row)].append(row)

    incidents = []
    for key, rows in grouped.items():
        rows = sorted(rows, key=_epoch, reverse=True)
        primary_rows = [row for row in rows if _attack_role(row) != "secondary"]
        main = primary_rows[0] if primary_rows else rows[0]
        main_pred = main.get("prediction") or {}
        main_triage = main.get("triage") or {}
        primary_label = str(main_triage.get("incident_primary_label") or _attack_label(main))
        flow_rows = [row.get("flow") or {} for row in rows]
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
            sample_flows.append({
                "time": _iso(_epoch(row)),
                "label": _attack_label(row),
                "role": _attack_role(row),
                "src": flow.get("src_ip", "-"),
                "dst": flow.get("dst_ip", "-"),
                "src_port": flow.get("src_port", "-"),
                "dst_port": flow.get("dst_port", "-"),
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
            "src": _display_value([flow.get("src_ip") for flow in flow_rows], "sources"),
            "dst": _display_value([flow.get("dst_ip") for flow in flow_rows], "targets"),
            "src_port": "-",
            "port": _display_value([flow.get("dst_port") for flow in flow_rows], "ports"),
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
    health = _fetch_json(f"{api_url}/health") if api_url else None
    metadata = _fetch_json(f"{api_url}/metadata") if health else None
    online = bool(health and health.get("status") == "ok")
    return {
        "online": online,
        "status": "online" if online else "offline",
        "model": (metadata or {}).get("model_name", "unavailable"),
        "feature_count": (metadata or {}).get("feature_count"),
        "routing_enabled": bool((metadata or {}).get("routing_enabled")),
        "classes": (metadata or {}).get("class_names", []),
    }


def _latest_epoch(rows: list[dict[str, Any]]) -> float:
    return max((_epoch(row) for row in rows), default=0.0)


def _activity_metrics(
    api: dict[str, Any],
    attacks: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    audit: dict[str, Any],
    now: float,
) -> dict[str, Any]:
    last_attack = _latest_epoch(attacks)
    last_action = _latest_epoch(actions)
    last_analysis = _latest_epoch(audits)
    last_llm_error = max((_epoch(row) for row in audits if row.get("llm_error")), default=0.0)
    recent_attack = last_attack >= now - 90
    recent_analysis = last_analysis >= now - 45

    if not api.get("online"):
        state = "dead"
        label = "dead"
        headline = "No signal"
        detail = "Dashboard is online, but the IDS API is not responding."
    elif recent_attack or recent_analysis:
        state = "thinking"
        label = "thinking"
        headline = "Cognition active"
        detail = "Recent telemetry is being analyzed and triaged."
    else:
        state = "breathing"
        label = "breathing"
        headline = "Neural core online"
        detail = "IDS API is healthy and waiting for the next flow batch."

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
            {"name": "IDS API", "value": api.get("status", "offline")},
            {"name": "Router", "value": "enabled" if api.get("routing_enabled") else "single"},
            {"name": "Analyzed", "value": int(audit["records"])},
            {"name": "LLM errors", "value": int(audit["llm_errors"])},
        ],
    }


def build_metrics(settings: DashboardSettings | None = None) -> dict[str, Any]:
    settings = settings or settings_from_env()
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
    sources = Counter((row.get("flow") or {}).get("src_ip", "-") for row in attacks_24h)
    destinations = Counter((row.get("flow") or {}).get("dst_ip", "-") for row in attacks_24h)
    ports = Counter(str((row.get("flow") or {}).get("dst_port", "-")) for row in attacks_24h)
    counts_24h = _incident_counts(attacks_24h)
    counts_1h = _incident_counts(attacks_1h)
    high_critical = len({
        _incident_key(row)
        for row in attacks_24h
        if _attack_role(row) != "secondary"
        and _attack_severity(row) in {"high", "critical"}
    })
    critical = len({
        _incident_key(row)
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
        },
        "api": api,
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
            "timeline": _timeline(attacks_24h, now),
            "labels": _counter_items(labels, 9),
            "severities": _counter_items(severities, 6),
            "sources": _counter_items(sources, 8),
            "destinations": _counter_items(destinations, 8),
            "ports": _counter_items(ports, 8),
            "routes": audit["routes"],
            "prediction_labels": audit["labels"],
        },
        "recent": {
            "attacks": _recent_attacks(attacks, limit=200),
            "actions": _recent_actions(actions, limit=10),
        },
        "audit": audit,
    }


def create_app(settings: DashboardSettings | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["DASHBOARD_SETTINGS"] = settings or settings_from_env()

    @app.get("/")
    def index():
        cfg = app.config["DASHBOARD_SETTINGS"]
        return render_template(
            "index.html",
            refresh_seconds=cfg.refresh_seconds,
            api_url=cfg.api_url,
        )

    @app.get("/api/metrics")
    def api_metrics():
        return jsonify(build_metrics(app.config["DASHBOARD_SETTINGS"]))

    @app.get("/api/events")
    def api_events():
        metrics = build_metrics(app.config["DASHBOARD_SETTINGS"])
        return jsonify(metrics["recent"])

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "ids-dashboard"})

    return app


app = create_app()


def main() -> None:
    settings = settings_from_env()
    host = os.environ.get("TON_IOT_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("TON_IOT_DASHBOARD_PORT", "5000"))
    debug = os.environ.get("TON_IOT_DASHBOARD_DEBUG", "").lower() in {"1", "true", "yes"}
    create_app(settings).run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
