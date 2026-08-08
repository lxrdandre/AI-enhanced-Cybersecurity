from __future__ import annotations

import json
import sqlite3
import time

import dashboard.app as dashboard_app
from clawdbot.control_plane import load_firewall_requests, load_protected_ips
from dashboard.app import DashboardSettings, build_metrics, create_app, load_health_panel, load_ip_intel, read_jsonl_tail


def _write_jsonl(path, rows):
    """Write jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_threat_db(path, now):
    """Write threat db."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE ip_intel (
            ip TEXT PRIMARY KEY,
            first_seen REAL NOT NULL,
            last_seen REAL NOT NULL,
            hit_count INTEGER NOT NULL DEFAULT 1,
            cumulative_severity INTEGER NOT NULL DEFAULT 0,
            labels TEXT NOT NULL DEFAULT '[]',
            abuseipdb_score INTEGER NOT NULL DEFAULT -1,
            vt_malicious INTEGER NOT NULL DEFAULT -1,
            otx_pulse_count INTEGER NOT NULL DEFAULT -1,
            api_checked_at REAL NOT NULL DEFAULT 0
        );
        CREATE TABLE mitre_attack (
            stix_id TEXT PRIMARY KEY,
            ext_id TEXT NOT NULL,
            name TEXT NOT NULL,
            tactics TEXT NOT NULL DEFAULT '[]',
            platforms TEXT NOT NULL DEFAULT '[]',
            updated_at REAL NOT NULL DEFAULT 0
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )
    conn.execute(
        """INSERT INTO ip_intel
           (ip, first_seen, last_seen, hit_count, cumulative_severity, labels,
            abuseipdb_score, vt_malicious, otx_pulse_count, api_checked_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("203.0.113.50", now - 3600, now, 3, 7, json.dumps(["scanning", "ddos_dos"]), 90, 4, 2, now),
    )
    conn.execute(
        """INSERT INTO ip_intel
           (ip, first_seen, last_seen, hit_count, cumulative_severity, labels,
            abuseipdb_score, vt_malicious, otx_pulse_count, api_checked_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("198.51.100.20", now - 7200, now - 120, 1, 1, json.dumps(["unknown"]), -1, -1, -1, 0),
    )
    conn.execute(
        """INSERT INTO mitre_attack
           (stix_id, ext_id, name, tactics, platforms, updated_at)
           VALUES ('attack-pattern--1', 'T1595', 'Active Scanning', '[]', '[]', ?)""",
        (now,),
    )
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('mitre_updated_at', ?)",
        (str(now),),
    )
    conn.commit()
    conn.close()


def test_read_jsonl_tail_ignores_bad_lines(tmp_path):
    """Verify that read jsonl tail ignores bad lines."""
    path = tmp_path / "events.jsonl"
    path.write_text('{"ok": 1}\nnot-json\n{"ok": 2}\n', encoding="utf-8")

    assert read_jsonl_tail(str(path), 10) == [{"ok": 1}, {"ok": 2}]


def test_build_metrics_counts_attack_and_audit_rows(tmp_path):
    """Verify that build metrics counts attack and audit rows."""
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
    """Verify that activity state turns thinking for recent online telemetry."""
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
    """Verify that build metrics hides default management ports."""
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


def test_dashboard_normalizes_originators_and_targets(tmp_path):
    """Verify that dashboard normalizes originators and targets."""
    now = time.time()
    log_dir = tmp_path / "logs"
    audit_log = tmp_path / "audit" / "analyze_events.jsonl"
    _write_jsonl(log_dir / "attacks.jsonl", [
        {
            "epoch": now,
            "incident_id": "scan-1",
            "prediction": {"label": "scanning", "confidence": 0.91},
            "triage": {
                "label": "scanning",
                "severity": "high",
                "incident_role": "primary",
                "incident_primary_label": "scanning",
            },
            "flow": {"src_ip": "192.0.2.10", "dst_ip": "10.0.0.5", "src_port": 55222, "dst_port": 80},
        },
        {
            "epoch": now + 0.001,
            "incident_id": "scan-1",
            "prediction": {"label": "scanning", "confidence": 0.90},
            "triage": {
                "label": "scanning",
                "severity": "high",
                "incident_role": "primary",
                "incident_primary_label": "scanning",
            },
            "flow": {"src_ip": "10.0.0.5", "dst_ip": "192.0.2.10", "src_port": 80, "dst_port": 55222},
        },
    ])
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(log_dir),
        audit_log=str(audit_log),
        api_url="",
        ignored_ports=frozenset(),
        protected_ips=frozenset({"10.0.0.5"}),
    )

    metrics = build_metrics(settings)

    assert metrics["series"]["sources"] == [{"name": "192.0.2.10", "value": 2}]
    assert metrics["series"]["destinations"] == [{"name": "10.0.0.5", "value": 2}]
    assert metrics["series"]["ports"] == [{"name": "80", "value": 2}]
    attack = metrics["recent"]["attacks"][0]
    assert attack["src"] == "192.0.2.10"
    assert attack["dst"] == "10.0.0.5"
    assert attack["sample_flows"][1]["src"] == "192.0.2.10"
    assert attack["sample_flows"][1]["dst"] == "10.0.0.5"


def test_build_metrics_counts_incidents_not_flows(tmp_path):
    """Verify that build metrics counts incidents not flows."""
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
    """Verify that repeated scan incidents keep separate dashboard rows."""
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
    """Verify that recent attacks payload can feed show more."""
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
    """Verify that recent attack contains detail payload."""
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


def test_load_ip_intel_reads_threat_cache(tmp_path):
    """Verify that load ip intel reads threat cache."""
    now = time.time()
    db_path = tmp_path / "data" / "threat_cache.db"
    _write_threat_db(db_path, now)

    intel = load_ip_intel(str(db_path))

    assert intel["db_exists"] is True
    assert intel["error"] == ""
    assert intel["total_ips"] == 2
    assert intel["total_hits"] == 4
    assert intel["known_bad"] == 1
    assert intel["external_checked"] == 1
    assert intel["mitre_count"] == 1
    assert intel["rows"][0]["ip"] == "203.0.113.50"
    assert intel["rows"][0]["badge"] == "Known-bad"
    assert intel["rows"][0]["labels"] == ["scanning", "ddos_dos"]


def test_load_ip_intel_excludes_protected_server_ip(tmp_path):
    """Verify that load ip intel excludes protected server ip."""
    now = time.time()
    db_path = tmp_path / "data" / "threat_cache.db"
    _write_threat_db(db_path, now)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO ip_intel
           (ip, first_seen, last_seen, hit_count, cumulative_severity, labels,
            abuseipdb_score, vt_malicious, otx_pulse_count, api_checked_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("100.111.77.70", now - 60, now, 99, 99, json.dumps(["server"]), 100, 9, 9, now),
    )
    conn.commit()
    conn.close()

    intel = load_ip_intel(str(db_path), frozenset({"100.111.77.70"}))

    assert "100.111.77.70" not in {row["ip"] for row in intel["rows"]}
    assert intel["total_ips"] == 2


def test_load_ip_intel_handles_missing_cache(tmp_path):
    """Verify that load ip intel handles missing cache."""
    intel = load_ip_intel(str(tmp_path / "missing.db"))

    assert intel["db_exists"] is False
    assert intel["rows"] == []
    assert intel["total_ips"] == 0


def test_flask_metrics_endpoint(tmp_path):
    """Verify that flask metrics endpoint."""
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


def test_flask_ip_intel_page_renders_records(tmp_path):
    """Verify that flask ip intel page renders records."""
    now = time.time()
    db_path = tmp_path / "data" / "threat_cache.db"
    _write_threat_db(db_path, now)
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(tmp_path / "logs"),
        audit_log=str(tmp_path / "audit.jsonl"),
        api_url="",
        threat_db=str(db_path),
    )
    client = create_app(settings).test_client()

    response = client.get("/ip-intel")

    assert response.status_code == 200
    assert b"IP Intelligence" in response.data
    assert b"203.0.113.50" in response.data
    assert b"Known-bad" in response.data


def test_flask_ip_intel_api_returns_records(tmp_path):
    """Verify that flask ip intel api returns records."""
    now = time.time()
    db_path = tmp_path / "data" / "threat_cache.db"
    _write_threat_db(db_path, now)
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(tmp_path / "logs"),
        audit_log=str(tmp_path / "audit.jsonl"),
        api_url="",
        threat_db=str(db_path),
    )
    client = create_app(settings).test_client()

    response = client.get("/api/ip-intel")

    assert response.status_code == 200
    assert response.get_json()["rows"][0]["ip"] == "203.0.113.50"


def test_flask_operations_pages_render(tmp_path):
    """Verify that flask operations pages render."""
    now = time.time()
    log_dir = tmp_path / "logs"
    audit_log = tmp_path / "audit" / "analyze_events.jsonl"
    db_path = tmp_path / "data" / "threat_cache.db"
    _write_threat_db(db_path, now)
    _write_jsonl(log_dir / "attacks.jsonl", [
        {
            "epoch": now,
            "incident_id": "incident-1",
            "prediction": {"label": "scanning", "confidence": 0.91, "route": "custom"},
            "triage": {
                "label": "scanning",
                "severity": "high",
                "incident_role": "primary",
                "incident_primary_label": "scanning",
                "mitre_tactics": ["Reconnaissance"],
                "mitre_techniques": [{"id": "T1595", "name": "Active Scanning"}],
            },
            "flow": {"src_ip": "203.0.113.50", "dst_ip": "100.111.77.70", "src_port": 55000, "dst_port": 80},
        }
    ])
    _write_jsonl(audit_log, [
        {
            "timestamp": now,
            "record_count": 1,
            "predictions": [{"predicted_label": "scanning", "confidence": 0.91, "route": "custom"}],
        }
    ])
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(log_dir),
        audit_log=str(audit_log),
        api_url="",
        threat_db=str(db_path),
        protected_ips=frozenset({"100.111.77.70"}),
        protected_ips_file=str(tmp_path / "data" / "protected_ips.json"),
        firewall_queue=str(tmp_path / "data" / "firewall_requests.json"),
    )
    client = create_app(settings).test_client()

    for path, marker in (
        ("/ip-intel/203.0.113.50", b"IP investigation"),
        ("/firewall", b"Firewall Control"),
        ("/protected-assets", b"Protected Assets"),
        ("/mitre", b"MITRE Matrix"),
        ("/model-quality", b"Model Quality"),
        ("/health-panel", b"Health Panel"),
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert marker in response.data


def test_health_panel_reports_clean_system_when_checks_pass(tmp_path, monkeypatch):
    """Verify that health panel reports clean system when checks pass."""
    now = time.time()
    log_dir = tmp_path / "logs"
    audit_log = tmp_path / "audit" / "analyze_events.jsonl"
    db_path = tmp_path / "data" / "threat_cache.db"
    _write_threat_db(db_path, now)
    _write_jsonl(log_dir / "actions.jsonl", [
        {"epoch": now, "action": "agent_start", "detail": {"interface": "wt0"}}
    ])
    _write_jsonl(audit_log, [
        {"timestamp": now, "record_count": 1, "predictions": [], "llm_error": None}
    ])
    monkeypatch.setattr(dashboard_app, "_api_status", lambda _url: {
        "online": True,
        "status": "online",
        "model": "sedwnet",
        "routing_enabled": True,
        "startup_error": "",
    })
    monkeypatch.setattr(dashboard_app, "_systemd_service_status", lambda service: {
        "name": service,
        "category": "Services",
        "status": "ok",
        "summary": "Online",
        "detail": "running",
        "value": "active",
    })
    monkeypatch.setattr(dashboard_app, "_telegram_health_item", lambda _settings: {
        "name": "Telegram bot",
        "category": "Integrations",
        "status": "ok",
        "summary": "Online",
        "detail": "configured chat reachable",
        "value": "online",
    })
    monkeypatch.setattr(dashboard_app, "_ollama_health_item", lambda _settings: {
        "name": "Ollama API",
        "category": "Integrations",
        "status": "ok",
        "summary": "Online",
        "detail": "models visible",
        "value": "online",
    })
    monkeypatch.setattr(dashboard_app, "_journal_error_item", lambda _services, _cutoff: {
        "name": "System journal errors",
        "category": "Errors",
        "status": "ok",
        "summary": "None",
        "detail": "clean",
        "value": "0",
        "lines": [],
    })
    monkeypatch.setattr(dashboard_app, "_local_log_error_item", lambda _log_dir, _cutoff: {
        "name": "Local log errors",
        "category": "Errors",
        "status": "ok",
        "summary": "None",
        "detail": "clean",
        "value": "0",
        "lines": [],
    })
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(log_dir),
        audit_log=str(audit_log),
        api_url="http://ids-api.local",
        threat_db=str(db_path),
        health_services=("ids-api.service", "clawdbot-agent.service"),
    )

    health = load_health_panel(settings)

    assert health["overall"]["status"] == "ok"
    assert health["overall"]["counts"]["critical"] == 0
    assert health["overall"]["counts"]["warning"] == 0


def test_health_panel_accepts_active_agent_service_without_lifecycle_log(tmp_path, monkeypatch):
    """Verify that health panel accepts active agent service without lifecycle log."""
    now = time.time()
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    audit_log = tmp_path / "audit" / "analyze_events.jsonl"
    db_path = tmp_path / "data" / "threat_cache.db"
    _write_threat_db(db_path, now)
    _write_jsonl(audit_log, [
        {"timestamp": now, "record_count": 1, "predictions": [], "llm_error": None}
    ])
    monkeypatch.setattr(dashboard_app, "_api_status", lambda _url: {
        "online": True,
        "status": "online",
        "model": "sedwnet",
        "routing_enabled": True,
        "startup_error": "",
    })
    monkeypatch.setattr(dashboard_app, "_systemd_service_status", lambda service: {
        "name": service,
        "category": "Services",
        "status": "ok",
        "summary": "Online",
        "detail": "running",
        "value": "active",
    })
    monkeypatch.setattr(dashboard_app, "_telegram_health_item", lambda _settings: {
        "name": "Telegram bot",
        "category": "Integrations",
        "status": "ok",
        "summary": "Online",
        "detail": "configured chat reachable",
        "value": "online",
    })
    monkeypatch.setattr(dashboard_app, "_ollama_health_item", lambda _settings: {
        "name": "Ollama API",
        "category": "Integrations",
        "status": "ok",
        "summary": "Online",
        "detail": "models visible",
        "value": "online",
    })
    monkeypatch.setattr(dashboard_app, "_journal_error_item", lambda _services, _cutoff: {
        "name": "System journal errors",
        "category": "Errors",
        "status": "ok",
        "summary": "None",
        "detail": "clean",
        "value": "0",
        "lines": [],
    })
    monkeypatch.setattr(dashboard_app, "_local_log_error_item", lambda _log_dir, _cutoff: {
        "name": "Local log errors",
        "category": "Errors",
        "status": "ok",
        "summary": "None",
        "detail": "clean",
        "value": "0",
        "lines": [],
    })
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(log_dir),
        audit_log=str(audit_log),
        api_url="http://ids-api.local",
        threat_db=str(db_path),
        health_services=("ids-api.service", "clawdbot-agent.service"),
    )

    health = load_health_panel(settings)
    runtime = next(group for group in health["groups"] if group["name"] == "Runtime")
    lifecycle = next(item for item in runtime["checks"] if item["name"] == "ClawdBot lifecycle")

    assert lifecycle["status"] == "ok"
    assert lifecycle["summary"] == "Online by service state"
    assert health["overall"]["status"] == "ok"


def test_health_panel_marks_critical_when_service_or_errors_fail(tmp_path, monkeypatch):
    """Verify that health panel marks critical when service or errors fail."""
    now = time.time()
    log_dir = tmp_path / "logs"
    audit_log = tmp_path / "audit" / "analyze_events.jsonl"
    db_path = tmp_path / "data" / "threat_cache.db"
    _write_threat_db(db_path, now)
    _write_jsonl(log_dir / "actions.jsonl", [
        {"epoch": now, "action": "agent_start", "detail": {"interface": "wt0"}}
    ])
    _write_jsonl(audit_log, [
        {"timestamp": now, "record_count": 1, "predictions": [], "llm_error": None}
    ])
    monkeypatch.setattr(dashboard_app, "_api_status", lambda _url: {
        "online": True,
        "status": "online",
        "model": "sedwnet",
        "routing_enabled": True,
        "startup_error": "",
    })
    monkeypatch.setattr(dashboard_app, "_systemd_service_status", lambda service: {
        "name": service,
        "category": "Services",
        "status": "critical",
        "summary": "Offline",
        "detail": "failed",
        "value": "failed",
    })
    monkeypatch.setattr(dashboard_app, "_telegram_health_item", lambda _settings: {
        "name": "Telegram bot",
        "category": "Integrations",
        "status": "ok",
        "summary": "Online",
        "detail": "configured chat reachable",
        "value": "online",
    })
    monkeypatch.setattr(dashboard_app, "_ollama_health_item", lambda _settings: {
        "name": "Ollama API",
        "category": "Integrations",
        "status": "ok",
        "summary": "Online",
        "detail": "models visible",
        "value": "online",
    })
    monkeypatch.setattr(dashboard_app, "_journal_error_item", lambda _services, _cutoff: {
        "name": "System journal errors",
        "category": "Errors",
        "status": "critical",
        "summary": "Errors detected",
        "detail": "1 recent error",
        "value": "1",
        "lines": ["ids-api error"],
    })
    monkeypatch.setattr(dashboard_app, "_local_log_error_item", lambda _log_dir, _cutoff: {
        "name": "Local log errors",
        "category": "Errors",
        "status": "ok",
        "summary": "None",
        "detail": "clean",
        "value": "0",
        "lines": [],
    })
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(log_dir),
        audit_log=str(audit_log),
        api_url="http://ids-api.local",
        threat_db=str(db_path),
        health_services=("ids-api.service",),
    )

    health = load_health_panel(settings)

    assert health["overall"]["status"] == "critical"
    assert health["overall"]["counts"]["critical"] == 2
    assert health["journal_lines"] == ["ids-api error"]


def test_journal_errors_before_latest_successful_start_are_resolved(monkeypatch):
    """Verify that journal errors before latest successful start are resolved."""
    now = time.time()

    def row(epoch, priority, message, ident="systemd"):
        """Build a synthetic journal row for health-panel tests."""
        return json.dumps({
            "__REALTIME_TIMESTAMP": str(int(epoch * 1_000_000)),
            "PRIORITY": str(priority),
            "SYSLOG_IDENTIFIER": ident,
            "_HOSTNAME": "diploma",
            "MESSAGE": message,
        })

    stdout = "\n".join([
        row(now - 120, 3, "Failed to start clawdbot-agent.service - ClawdBot IDS Traffic Capture Agent."),
        row(now - 60, 6, "Started clawdbot-agent.service - ClawdBot IDS Traffic Capture Agent."),
        row(now - 10, 6, "Capture thread started on wt0", "clawdbot"),
    ])
    monkeypatch.setattr(dashboard_app, "_run_readonly_command", lambda *_args, **_kwargs: (0, stdout, ""))

    item = dashboard_app._journal_error_item(("clawdbot-agent.service",), now - 3600)

    assert item["status"] == "ok"
    assert item["value"] == "0"
    assert item["lines"] == []


def test_journal_errors_after_latest_successful_start_are_reported(monkeypatch):
    """Verify that journal errors after latest successful start are reported."""
    now = time.time()

    def row(epoch, priority, message, ident="systemd"):
        """Build a synthetic journal row for health-panel tests."""
        return json.dumps({
            "__REALTIME_TIMESTAMP": str(int(epoch * 1_000_000)),
            "PRIORITY": str(priority),
            "SYSLOG_IDENTIFIER": ident,
            "_HOSTNAME": "diploma",
            "MESSAGE": message,
        })

    stdout = "\n".join([
        row(now - 120, 3, "Failed to start clawdbot-agent.service - ClawdBot IDS Traffic Capture Agent."),
        row(now - 60, 6, "Started clawdbot-agent.service - ClawdBot IDS Traffic Capture Agent."),
        row(now - 10, 3, "Runtime crash after service start", "clawdbot"),
    ])
    monkeypatch.setattr(dashboard_app, "_run_readonly_command", lambda *_args, **_kwargs: (0, stdout, ""))

    item = dashboard_app._journal_error_item(("clawdbot-agent.service",), now - 3600)

    assert item["status"] == "critical"
    assert item["value"] == "1"
    assert "Runtime crash after service start" in item["lines"][0]


def test_firewall_block_post_queues_request(tmp_path):
    """Verify that firewall block post queues request."""
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(tmp_path / "logs"),
        audit_log=str(tmp_path / "audit.jsonl"),
        api_url="",
        firewall_queue=str(tmp_path / "data" / "firewall_requests.json"),
    )
    client = create_app(settings).test_client()

    response = client.post("/firewall/block", data={"ip": "203.0.113.50", "ttl": "600", "reason": "test"})

    assert response.status_code == 302
    requests = load_firewall_requests(settings.firewall_queue)
    assert requests[0]["action"] == "block"
    assert requests[0]["ip"] == "203.0.113.50"


def test_protected_assets_add_and_remove(tmp_path):
    """Verify that protected assets add and remove."""
    protected_file = tmp_path / "data" / "protected_ips.json"
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(tmp_path / "logs"),
        audit_log=str(tmp_path / "audit.jsonl"),
        api_url="",
        protected_ips_file=str(protected_file),
        protected_ips=frozenset({"100.111.77.70"}),
    )
    client = create_app(settings).test_client()

    assert client.post("/protected-assets/add", data={"ip": "10.0.0.5"}).status_code == 302
    assert "10.0.0.5" in load_protected_ips(str(protected_file), settings.protected_ips)

    assert client.post("/protected-assets/remove", data={"ip": "10.0.0.5"}).status_code == 302
    assert "10.0.0.5" not in load_protected_ips(str(protected_file), settings.protected_ips)


def test_export_endpoints_return_csv(tmp_path):
    """Verify that export endpoints return csv."""
    now = time.time()
    db_path = tmp_path / "data" / "threat_cache.db"
    _write_threat_db(db_path, now)
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(tmp_path / "logs"),
        audit_log=str(tmp_path / "audit.jsonl"),
        api_url="",
        threat_db=str(db_path),
    )
    client = create_app(settings).test_client()

    ip_csv = client.get("/export/ip-intel.csv")
    incident_csv = client.get("/export/incidents.csv")

    assert ip_csv.status_code == 200
    assert b"203.0.113.50" in ip_csv.data
    assert incident_csv.status_code == 200
    assert b"label" in incident_csv.data


def test_flask_index_renders_dashboard(tmp_path):
    """Verify that flask index renders dashboard."""
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
    assert b"dashboard_theme.js" in response.data
    assert b"data-theme-toggle" in response.data
    assert b"Light theme" in response.data
    assert b"cdn.jsdelivr.net/npm/chart.js" in response.data
    assert b"fonts.googleapis.com" in response.data
    assert b'telegramToastStack' in response.data
    assert b"System Pulse" in response.data
    assert b"ai-core" in response.data
    assert b"core-eye" in response.data
    assert b"Loaded Model" in response.data
    assert b"Routing" in response.data
    assert b"Open IP Intel" in response.data
    assert b"Clear Logs" in response.data
    assert b"Are you sure you want to clear the incident, action, and audit logs?" in response.data
    assert b'class="brand-mark"' in response.data
    assert b"logo.png" not in response.data
    assert b"Threat Intelligence" in response.data


def test_flask_ip_intel_page_uses_shared_sidebar_navigation(tmp_path):
    """Verify that flask ip intel page uses shared sidebar navigation."""
    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(tmp_path / "logs"),
        audit_log=str(tmp_path / "audit.jsonl"),
        api_url="",
        threat_db=str(tmp_path / "missing.db"),
    )
    client = create_app(settings).test_client()

    response = client.get("/ip-intel")

    assert response.status_code == 200
    assert b'class="brand-mark"' in response.data
    assert b"logo.png" not in response.data
    assert b'href="/#overview"' in response.data
    assert b">Overview</a>" in response.data
    assert b">Events</a>" in response.data
    assert b"Clear Logs" not in response.data


def test_clear_logs_truncates_dashboard_files(tmp_path):
    """Verify that clear logs truncates dashboard files."""
    log_dir = tmp_path / "logs"
    audit_log = tmp_path / "audit" / "analyze_events.jsonl"
    _write_jsonl(log_dir / "attacks.jsonl", [{"label": "scanning"}])
    _write_jsonl(log_dir / "actions.jsonl", [{"event": "agent_start"}])
    _write_jsonl(audit_log, [{"record_count": 3}])

    settings = DashboardSettings(
        project_root=str(tmp_path),
        log_dir=str(log_dir),
        audit_log=str(audit_log),
        api_url="",
    )
    client = create_app(settings).test_client()

    response = client.post("/logs/clear", data={"next": "/ip-intel"})

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/ip-intel")
    assert (log_dir / "attacks.jsonl").read_text(encoding="utf-8") == ""
    assert (log_dir / "actions.jsonl").read_text(encoding="utf-8") == ""
    assert audit_log.read_text(encoding="utf-8") == ""
