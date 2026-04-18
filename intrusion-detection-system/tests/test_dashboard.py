from __future__ import annotations

import json
import time

import dashboard.app as dashboard_app
from dashboard.app import DashboardSettings, build_metrics, create_app, read_jsonl_tail


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_read_jsonl_tail_ignores_bad_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"ok": 1}\nnot-json\n{"ok": 2}\n', encoding="utf-8")

    assert read_jsonl_tail(str(path), 10) == [{"ok": 1}, {"ok": 2}]


def test_build_metrics_counts_attack_and_audit_rows(tmp_path):
    now = time.time()
    log_dir = tmp_path / "logs"
    audit_log = tmp_path / "audit" / "analyze_events.jsonl"
    _write_jsonl(log_dir / "attacks.jsonl", [
        {
            "epoch": now,
            "prediction": {"label": "scanning", "confidence": 0.91},
            "triage": {"severity": "high", "summary": "scan fan-out"},
            "flow": {"src_ip": "10.0.0.4", "dst_ip": "10.0.0.8", "dst_port": 8081},
        }
    ])
    _write_jsonl(log_dir / "actions.jsonl", [
        {"epoch": now, "event": "agent_start", "detail": {"interface": "wt0"}}
    ])
    _write_jsonl(audit_log, [
        {
            "timestamp": now,
            "record_count": 3,
            "llm_error": None,
            "predictions": [
                {"predicted_label": "scanning", "route": "original"},
                {"predicted_label": "unknown", "route": "custom"},
            ],
        }
    ])
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(log_dir),
        audit_log=str(audit_log),
        api_url="",
    )

    metrics = build_metrics(settings)

    assert metrics["kpis"]["threats_24h"] == 1
    assert metrics["kpis"]["high_critical_24h"] == 1
    assert metrics["kpis"]["analyzed_records_24h"] == 3
    assert metrics["kpis"]["unknown_rate_24h"] == 50.0
    assert metrics["series"]["labels"][0] == {"name": "scanning", "value": 1}
    assert metrics["activity"]["state"] == "dead"


def test_activity_state_turns_thinking_for_recent_online_telemetry(tmp_path, monkeypatch):
    now = time.time()
    log_dir = tmp_path / "logs"
    audit_log = tmp_path / "audit" / "analyze_events.jsonl"
    _write_jsonl(log_dir / "attacks.jsonl", [
        {
            "epoch": now,
            "prediction": {"label": "scanning", "confidence": 0.91},
            "triage": {"severity": "high"},
            "flow": {"src_ip": "10.0.0.4", "dst_ip": "10.0.0.8", "dst_port": 8081},
        }
    ])
    _write_jsonl(audit_log, [
        {
            "timestamp": now,
            "record_count": 1,
            "predictions": [{"predicted_label": "scanning", "route": "custom"}],
        }
    ])
    monkeypatch.setattr(dashboard_app, "_api_status", lambda _url: {
        "online": True,
        "status": "online",
        "model": "resnet-ensemble",
        "routing_enabled": True,
        "classes": [],
    })
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(log_dir),
        audit_log=str(audit_log),
        api_url="http://ids-api.local",
    )

    metrics = dashboard_app.build_metrics(settings)

    assert metrics["activity"]["state"] == "thinking"
    assert metrics["activity"]["recent_attack"] is True
    assert metrics["activity"]["signals"][1] == {"name": "Router", "value": "enabled"}


def test_build_metrics_hides_default_management_ports(tmp_path):
    now = time.time()
    log_dir = tmp_path / "logs"
    audit_log = tmp_path / "audit" / "analyze_events.jsonl"
    _write_jsonl(log_dir / "attacks.jsonl", [
        {
            "epoch": now,
            "prediction": {"label": "scanning", "confidence": 0.81},
            "triage": {"severity": "medium"},
            "flow": {"src_ip": "100.111.76.168", "dst_ip": "100.111.77.70", "dst_port": 5000},
        },
        {
            "epoch": now,
            "prediction": {"label": "xss", "confidence": 0.90},
            "triage": {"severity": "high"},
            "flow": {"src_ip": "10.0.0.2", "dst_ip": "10.0.0.3", "dst_port": 8081},
        },
    ])
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(log_dir),
        audit_log=str(audit_log),
        api_url="",
    )

    metrics = build_metrics(settings)

    assert metrics["kpis"]["threats_24h"] == 1
    assert metrics["series"]["labels"] == [{"name": "xss", "value": 1}]


def test_build_metrics_counts_incidents_not_flows(tmp_path):
    now = time.time()
    log_dir = tmp_path / "logs"
    audit_log = tmp_path / "audit" / "analyze_events.jsonl"
    rows = []
    for idx in range(5):
        rows.append({
            "epoch": now + idx * 0.001,
            "incident_id": "incident-scan-1",
            "prediction": {"label": "scanning", "confidence": 0.91},
            "triage": {
                "label": "scanning",
                "severity": "high",
                "incident_role": "primary",
                "incident_primary_label": "scanning",
            },
            "flow": {"src_ip": "10.0.0.2", "dst_ip": "10.0.0.3", "dst_port": 8000 + idx},
        })
    rows.extend([
        {
            "epoch": now,
            "incident_id": "incident-scan-1",
            "prediction": {"label": "unknown", "confidence": 0.70},
            "triage": {
                "label": "unknown",
                "severity": "review",
                "incident_role": "secondary",
                "incident_primary_label": "scanning",
            },
            "flow": {"src_ip": "10.0.0.2", "dst_ip": "10.0.0.3", "dst_port": 9000},
        },
        {
            "epoch": now + 0.002,
            "incident_id": "incident-scan-1",
            "prediction": {"label": "unknown", "confidence": 0.68},
            "triage": {
                "label": "unknown",
                "severity": "review",
                "incident_role": "secondary",
                "incident_primary_label": "scanning",
            },
            "flow": {"src_ip": "10.0.0.2", "dst_ip": "10.0.0.3", "dst_port": 9001},
        },
        {
            "epoch": now,
            "incident_id": "incident-scan-1",
            "prediction": {"label": "password", "confidence": 0.72},
            "triage": {
                "label": "password",
                "severity": "review",
                "incident_role": "secondary",
                "incident_primary_label": "scanning",
            },
            "flow": {"src_ip": "10.0.0.2", "dst_ip": "10.0.0.3", "dst_port": 22},
        },
    ])
    _write_jsonl(log_dir / "attacks.jsonl", rows)
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(log_dir),
        audit_log=str(audit_log),
        api_url="",
        ignored_ports=frozenset(),
    )

    metrics = build_metrics(settings)

    assert metrics["kpis"]["threats_24h"] == 1
    assert metrics["kpis"]["possible_threats_24h"] == 2
    assert sum(point["total"] for point in metrics["series"]["timeline"]) == 1
    assert len(metrics["recent"]["attacks"]) == 1
    assert metrics["recent"]["attacks"][0]["flow_count"] == 8
    assert metrics["recent"]["attacks"][0]["possible_count"] == 2
    assert metrics["recent"]["attacks"][0]["incident_summary"]["secondary_labels"] == [
        {"name": "unknown", "value": 2},
        {"name": "password", "value": 1},
    ]


def test_repeated_scan_incidents_keep_separate_dashboard_rows(tmp_path):
    now = time.time()
    log_dir = tmp_path / "logs"
    audit_log = tmp_path / "audit" / "analyze_events.jsonl"
    _write_jsonl(log_dir / "attacks.jsonl", [
        {
            "epoch": now - 60,
            "incident_id": "scan-old",
            "prediction": {"label": "scanning", "confidence": 0.91},
            "triage": {
                "label": "scanning",
                "severity": "high",
                "incident_role": "primary",
                "incident_primary_label": "scanning",
            },
            "flow": {"src_ip": "100.111.76.168", "dst_ip": "100.111.77.70", "dst_port": 4001},
            "actions": {"telegram_sent": True},
        },
        {
            "epoch": now,
            "incident_id": "scan-new",
            "prediction": {"label": "scanning", "confidence": 0.93},
            "triage": {
                "label": "scanning",
                "severity": "high",
                "incident_role": "primary",
                "incident_primary_label": "scanning",
            },
            "flow": {"src_ip": "100.111.76.168", "dst_ip": "100.111.77.70", "dst_port": 4002},
            "actions": {"telegram_sent": True},
        },
    ])
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(log_dir),
        audit_log=str(audit_log),
        api_url="",
        ignored_ports=frozenset(),
    )

    metrics = build_metrics(settings)

    assert metrics["kpis"]["threats_24h"] == 2
    assert [row["event_id"] for row in metrics["recent"]["attacks"]] == [
        "incident:scan-new",
        "incident:scan-old",
    ]


def test_recent_attacks_payload_can_feed_show_more(tmp_path):
    now = time.time()
    log_dir = tmp_path / "logs"
    audit_log = tmp_path / "audit" / "analyze_events.jsonl"
    _write_jsonl(log_dir / "attacks.jsonl", [
        {
            "epoch": now - idx,
            "prediction": {"label": "scanning", "confidence": 0.80},
            "triage": {"severity": "medium"},
            "flow": {"src_ip": f"10.0.0.{idx + 2}", "dst_ip": "10.0.0.3", "dst_port": 9000 + idx},
        }
        for idx in range(25)
    ])
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(log_dir),
        audit_log=str(audit_log),
        api_url="",
    )

    metrics = build_metrics(settings)

    assert len(metrics["recent"]["attacks"]) == 25


def test_recent_attack_contains_detail_payload(tmp_path):
    now = time.time()
    log_dir = tmp_path / "logs"
    audit_log = tmp_path / "audit" / "analyze_events.jsonl"
    _write_jsonl(log_dir / "attacks.jsonl", [
        {
            "epoch": now,
            "prediction": {
                "label": "injection",
                "confidence": 0.93,
                "route": "original",
                "router_confidence": 0.77,
                "probabilities": {"injection": 0.93, "xss": 0.04},
            },
            "triage": {
                "label": "injection",
                "severity": "high",
                "incident_role": "primary",
                "mitre_tactics": ["Initial Access"],
                "mitre_techniques": [
                    {"id": "T1190", "name": "Exploit Public-Facing Application", "reason": "web exploit"}
                ],
                "summary": "Possible injection attempt",
                "next_actions": ["Inspect web logs", "Patch vulnerable endpoint"],
            },
            "flow": {"src_ip": "10.0.0.2", "dst_ip": "10.0.0.3", "src_port": 44444, "dst_port": 8081},
            "actions": {
                "telegram_sent": True,
                "block_result": {"ip": "10.0.0.2", "applied": True, "ttl": 3600},
            },
            "reputation": {"badge": "Suspicious", "hit_count": 1},
            "audit_id": "abc",
        },
    ])
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(log_dir),
        audit_log=str(audit_log),
        api_url="",
    )

    attack = build_metrics(settings)["recent"]["attacks"][0]

    assert attack["event_id"]
    assert attack["epoch"] == round(now, 3)
    assert attack["telegram_sent"] is True
    assert attack["mitre_tactics"] == ["Initial Access"]
    assert attack["mitre_techniques"][0]["id"] == "T1190"
    assert attack["flow_count"] == 1
    assert attack["possible_count"] == 0
    assert attack["sample_flows"][0]["dst_port"] == 8081
    assert "Inspect application/API logs" in attack["next_actions"][0]
    assert attack["block_result"]["applied"] is True
    assert attack["reputation"]["badge"] == "Suspicious"
    assert attack["top_probabilities"][0] == {"label": "injection", "value": 0.93}


def test_flask_metrics_endpoint(tmp_path):
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(tmp_path / "logs"),
        audit_log=str(tmp_path / "audit.jsonl"),
        api_url="",
    )
    client = create_app(settings).test_client()

    response = client.get("/api/metrics")

    assert response.status_code == 200
    assert response.get_json()["api"]["status"] == "offline"
    assert response.get_json()["activity"]["state"] == "dead"


def test_flask_index_renders_dashboard(tmp_path):
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(tmp_path / "logs"),
        audit_log=str(tmp_path / "audit.jsonl"),
        api_url="",
    )
    client = create_app(settings).test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"Network Threat Operations" in response.data
    assert b"cdn.jsdelivr.net/npm/chart.js" in response.data
    assert b"fonts.googleapis.com" in response.data
    assert b'telegramToastStack' in response.data
    assert b"System Pulse" in response.data
    assert b"Loaded Model" in response.data
    assert b"Routing" in response.data
