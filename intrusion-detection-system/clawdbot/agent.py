"""ClawdBot orchestrator — capture → IDS API → Telegram alerting loop."""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

from app.triage import TriageService
from clawdbot.actuator import Actuator, _is_whitelisted
from clawdbot.capture import TrafficCapture
from clawdbot.telegram import TelegramNotifier
from clawdbot.threat_intel import ThreatIntel

log = logging.getLogger(__name__)

DEFAULT_LOG_DIR = "/data/ton-iot-project/fresh_start/logs"


class EventLogger:
    """Append-only JSONL logger for attack events and actions taken."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self._attacks_path = os.path.join(log_dir, "attacks.jsonl")
        self._actions_path = os.path.join(log_dir, "actions.jsonl")

    def log_attack(
        self,
        *,
        prediction: dict,
        triage: dict,
        flow_meta: dict | None,
        telegram_sent: bool,
        audit_id: str | None = None,
        incident_id: str | None = None,
        incident_summary: dict | None = None,
        block_result: dict | None = None,
        reputation: dict | None = None,
    ) -> None:
        """Log a detected attack with full context."""
        event = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "epoch": time.time(),
            "event": "attack_detected",
            "prediction": {
                "label": prediction.get("predicted_label"),
                "confidence": prediction.get("confidence"),
                "probabilities": prediction.get("probabilities"),
                "route": prediction.get("route"),
                "router_confidence": prediction.get("router_confidence"),
            },
            "triage": {
                "label": triage.get("label"),
                "severity": triage.get("severity"),
                "incident_role": triage.get("incident_role"),
                "incident_primary_label": triage.get("incident_primary_label"),
                "secondary_reason": triage.get("secondary_reason"),
                "llm_reclassified": triage.get("llm_reclassified", False),
                "mitre_tactics": triage.get("mitre_tactics", []),
                "mitre_techniques": triage.get("mitre_techniques", []),
                "summary": triage.get("summary", ""),
                "next_actions": triage.get("next_actions", []),
                "source": triage.get("source", "unknown"),
            },
            "flow": flow_meta,
            "actions": {
                "telegram_sent": telegram_sent,
                "block_result": block_result,
            },
            "reputation": reputation,
            "audit_id": audit_id,
            "incident_id": incident_id,
            "incident_summary": incident_summary or {},
        }
        self._append(self._attacks_path, event)

    def log_action(self, *, action: str, detail: dict | None = None) -> None:
        """Log a system-level action (agent start/stop, errors, etc.)."""
        event = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "epoch": time.time(),
            "event": action,
            "detail": detail or {},
        }
        self._append(self._actions_path, event)

    def _append(self, path: str, event: dict) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.error("Failed to write event log %s: %s", path, exc)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _post_analyze(api_url: str, records: list[dict], timeout: int = 30) -> dict | None:
    """POST records to the IDS /analyze endpoint and return the parsed response."""
    url = f"{api_url.rstrip('/')}/analyze"

    # Strip _meta from records (not model features)
    cleaned = [{k: v for k, v in r.items() if k != "_meta"} for r in records]

    payload = json.dumps({"records": cleaned}).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        method="POST",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        log.error("IDS API request failed: %s", exc)
        return None


MAX_BATCH_SIZE = 64  # cap records per API call to avoid timeouts
ALERT_COOLDOWN_SECONDS = 300  # suppress repeated alerts for same label set (5 min)

# Management ports ignored between whitelisted peers (SSH, etc.)
# Traffic on these ports between VPN/management IPs is considered routine.
DEFAULT_MGMT_PORTS = frozenset({22, 64295, 5000, 8000})
AUTH_PORTS = frozenset({21, 22, 23, 25, 110, 143, 389, 445, 3389, 5900, 3306, 5432})
SCAN_MIN_UNIQUE_PORTS = 10
SCAN_MIN_UNIQUE_HOSTS = 5
DDOS_MIN_FLOWS_PER_TARGET = 50
DDOS_MIN_UNIQUE_SOURCES = 5


def _parse_mgmt_ports(csv: str) -> frozenset[int]:
    """Parse comma-separated port list, always keeping built-in management ports."""
    ports: set[int] = set(DEFAULT_MGMT_PORTS)
    for entry in csv.split(","):
        entry = entry.strip()
        if entry.isdigit():
            ports.add(int(entry))
    return frozenset(ports)


def _is_management_record(record: dict, whitelist, mgmt_ports: frozenset[int]) -> bool:
    meta = record.get("_meta") or {}
    src = meta.get("src_ip", "")
    dst = meta.get("dst_ip", "")
    src_port = _port_int(meta.get("src_port"))
    dst_port = _port_int(meta.get("dst_port"))
    return (
        bool(src and dst)
        and _is_whitelisted(src, whitelist)
        and _is_whitelisted(dst, whitelist)
        and (src_port in mgmt_ports or dst_port in mgmt_ports)
    )


def _detection_label(det: dict) -> str:
    return str(det["triage"].get("label") or det["prediction"].get("predicted_label", "unknown"))


def _incident_summary(detections: list[dict]) -> dict:
    primary_labels = [
        _detection_label(det)
        for det in detections
        if det["triage"].get("incident_role") == "primary"
    ]
    primary_label = (
        detections[0]["triage"].get("incident_primary_label")
        if detections else "unknown"
    ) or (primary_labels[0] if primary_labels else "unknown")
    secondary_labels = sorted({
        _detection_label(det)
        for det in detections
        if det["triage"].get("incident_role") == "secondary"
        and _detection_label(det) != primary_label
    })
    return {
        "threat_count": 1 if detections else 0,
        "primary_label": primary_label,
        "possible_count": len(secondary_labels),
        "secondary_labels": secondary_labels,
    }


def _incident_summary_record(detections: list[dict]) -> dict:
    srcs = Counter(_flow_field(det, "src_ip", "") for det in detections)
    dsts = Counter(_flow_field(det, "dst_ip", "") for det in detections)
    ports = Counter(_port_int(_flow_field(det, "dst_port")) for det in detections)
    protos = Counter(str(_flow_field(det, "proto", "")).lower() for det in detections)
    labels = Counter(_detection_label(det) for det in detections)
    samples = []
    for det in detections[:8]:
        meta = det.get("flow_meta") or {}
        samples.append({
            "src_ip": meta.get("src_ip"),
            "dst_ip": meta.get("dst_ip"),
            "src_port": meta.get("src_port"),
            "dst_port": meta.get("dst_port"),
            "proto": meta.get("proto"),
            "model_label": _detection_label(det),
            "confidence": (det.get("prediction") or {}).get("confidence"),
        })

    return {
        "incident_flow_count": len(detections),
        "model_label_counts": dict(labels.most_common()),
        "top_sources": [ip for ip, _ in srcs.most_common(10) if ip],
        "top_destinations": [ip for ip, _ in dsts.most_common(10) if ip],
        "top_destination_ports": [port for port, _ in ports.most_common(25) if port is not None],
        "protocols": [proto for proto, _ in protos.most_common(5) if proto],
        "sample_flows": samples,
    }


def _flow_field(det: dict, field: str, default=None):
    meta = det.get("flow_meta") or {}
    return meta.get(field, default)


def _port_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _strip_mitre(triage: dict) -> None:
    triage["mitre_tactics"] = []
    triage["mitre_techniques"] = []


def _mark_secondary(det: dict, reason: str) -> None:
    triage = det["triage"]
    triage["incident_role"] = "secondary"
    triage["secondary_reason"] = reason
    triage["severity"] = "review"
    _strip_mitre(triage)


def _looks_like_scan(detections: list[dict]) -> bool:
    return bool(_scan_sources(detections))


def _scan_sources(detections: list[dict]) -> set[str]:
    by_src: dict[str, dict[str, set]] = {}
    for det in detections:
        src = _flow_field(det, "src_ip")
        if not src:
            continue
        stats = by_src.setdefault(src, {"ports": set(), "hosts": set()})
        dst_port = _port_int(_flow_field(det, "dst_port"))
        dst_ip = _flow_field(det, "dst_ip")
        if dst_port is not None:
            stats["ports"].add(dst_port)
        if dst_ip:
            stats["hosts"].add(dst_ip)
    return {
        src for src, stats in by_src.items()
        if len(stats["ports"]) >= SCAN_MIN_UNIQUE_PORTS
        or len(stats["hosts"]) >= SCAN_MIN_UNIQUE_HOSTS
    }


def _promote_scan(det: dict) -> None:
    det["triage"].update(
        {
            "label": "scanning",
            "severity": "high",
            "mitre_tactics": ["Reconnaissance"],
            "mitre_techniques": [
                {
                    "id": "T1595",
                    "name": "Active Scanning",
                    "confidence": "high",
                    "reason": "Source contacted many destination ports or hosts in one harvest window.",
                }
            ],
            "summary": "Port/host fan-out indicates active scanning.",
            "next_actions": [
                "Confirm whether this source is an approved scanner.",
                "If unauthorized, block or rate-limit the source IP.",
            ],
            "confidence_note": "Heuristic incident rule promoted this flow from model output to scanning.",
            "source": "heuristic:scan-fanout",
        }
    )


def _has_password_evidence(detections: list[dict]) -> bool:
    attempts: Counter[tuple[str, str, int]] = Counter()
    for det in detections:
        if _detection_label(det) != "password":
            continue
        dst_port = _port_int(_flow_field(det, "dst_port"))
        if dst_port not in AUTH_PORTS:
            continue
        key = (_flow_field(det, "src_ip", ""), _flow_field(det, "dst_ip", ""), dst_port)
        attempts[key] += 1
    return any(count >= 10 for count in attempts.values())


def _has_ddos_evidence(detections: list[dict]) -> bool:
    targets: dict[tuple[str, int], set[str]] = {}
    counts: Counter[tuple[str, int]] = Counter()
    for det in detections:
        if _detection_label(det) != "ddos_dos":
            continue
        dst_ip = _flow_field(det, "dst_ip")
        dst_port = _port_int(_flow_field(det, "dst_port"))
        if not dst_ip or dst_port is None:
            continue
        key = (dst_ip, dst_port)
        counts[key] += 1
        src = _flow_field(det, "src_ip")
        if src:
            targets.setdefault(key, set()).add(src)
    return any(
        count >= DDOS_MIN_FLOWS_PER_TARGET and len(targets.get(key, set())) >= DDOS_MIN_UNIQUE_SOURCES
        for key, count in counts.items()
    )


def _apply_incident_rules(detections: list[dict]) -> list[dict]:
    if not detections:
        return detections

    scan_sources = _scan_sources(detections)
    scan_campaign = bool(scan_sources)
    password_evidence = _has_password_evidence(detections)
    ddos_evidence = _has_ddos_evidence(detections)

    for det in detections:
        if scan_campaign and _detection_label(det) == "normal" and _flow_field(det, "src_ip") in scan_sources:
            _promote_scan(det)

    counts = Counter(_detection_label(det) for det in detections)
    known_labels = [label for label, _ in counts.most_common() if label not in {"normal", "unknown"}]
    primary = "scanning" if scan_campaign else (known_labels[0] if known_labels else "unknown")

    for det in detections:
        label = _detection_label(det)
        triage = det["triage"]
        triage["incident_primary_label"] = primary

        if label == primary:
            triage["incident_role"] = "primary"
        else:
            _mark_secondary(det, f"secondary signal in {primary} incident")

        if scan_campaign and label in {"unknown", "password", "ddos_dos"}:
            _mark_secondary(det, "suppressed because scan fan-out dominates this harvest window")
        elif label == "password" and not password_evidence:
            _mark_secondary(det, "password label lacks repeated auth-service evidence")
        elif label == "ddos_dos" and not ddos_evidence:
            _mark_secondary(det, "ddos_dos label lacks multi-source/high-volume target evidence")
        elif label == "unknown" and primary != "unknown":
            _mark_secondary(det, f"unknown signal is secondary to {primary}")

    return detections


def _classify_primary_unknown_incident(
    detections: list[dict],
    *,
    triage_service: TriageService,
) -> None:
    if not detections or not triage_service.enabled:
        return

    incident_summary = _incident_summary(detections)
    if incident_summary["primary_label"] != "unknown":
        return

    primary_unknowns = [
        det for det in detections
        if det["triage"].get("incident_role") == "primary"
        and _detection_label(det) == "unknown"
    ]
    if not primary_unknowns:
        return

    confidence = max(float((det.get("prediction") or {}).get("confidence", 0.0) or 0.0) for det in primary_unknowns)
    prediction = {
        "predicted_label": "unknown",
        "confidence": confidence,
        "probabilities": {},
    }
    context = {
        "source": "clawdbot",
        "unknown_priority": "primary",
        "classification_scope": "incident",
        "incident_primary_label": "unknown",
    }
    try:
        triage_items, llm_error = triage_service.triage_predictions(
            predictions=[prediction],
            records=[_incident_summary_record(primary_unknowns)],
            context=context,
        )
    except Exception as exc:
        log.warning("Primary unknown incident LLM classification failed: %s", exc)
        return

    if llm_error:
        log.warning("Primary unknown incident LLM classification warning: %s", llm_error)
    if not triage_items:
        return

    new_triage = triage_items[0]
    new_label = str(new_triage.get("label") or "unknown")
    new_triage["incident_role"] = "primary"
    new_triage["incident_primary_label"] = new_label
    new_triage["llm_reclassified"] = True

    for det in primary_unknowns:
        det["triage"] = dict(new_triage)
    for det in detections:
        det["triage"]["incident_primary_label"] = new_label


def run(
    *,
    interface: str,
    bpf_filter: str = "ip",
    api_url: str = "http://127.0.0.1:8000",
    api_timeout: int = 60,
    harvest_interval: float = 10.0,
    bot_token: str = "",
    chat_id: str = "",
    severity_threshold: str = "medium",
    log_dir: str = DEFAULT_LOG_DIR,
    actuator_enabled: bool = False,
    actuator_whitelist: str = "",
    actuator_default_ttl: int = 3600,
    actuator_dry_run: bool = False,
    ignore_ports: str = "",
    threat_intel_db: str = "",
    abuseipdb_key: str = "",
    virustotal_key: str = "",
    otx_key: str = "",
    threat_intel_api_ttl: int = 86400,
) -> None:
    """Main capture → analyze → alert loop."""

    notifier = TelegramNotifier(
        bot_token=bot_token,
        chat_id=chat_id,
        severity_threshold=severity_threshold,
    )

    actuator = Actuator(
        enabled=actuator_enabled,
        extra_whitelist=actuator_whitelist,
        default_ttl=actuator_default_ttl,
        dry_run=actuator_dry_run,
    )

    mgmt_ports = _parse_mgmt_ports(ignore_ports)

    intel = ThreatIntel(
        db_path=threat_intel_db or os.path.join(log_dir, "..", "data", "threat_cache.db"),
        abuseipdb_key=abuseipdb_key,
        virustotal_key=virustotal_key,
        otx_key=otx_key,
        api_cache_ttl=threat_intel_api_ttl,
    )
    incident_triage = TriageService(
        api_key=_env("GEMINI_API_KEY"),
        model=_env("TON_IOT_TRIAGE_MODEL", "gemini-2.0-flash"),
        timeout_seconds=int(_env("TON_IOT_TRIAGE_TIMEOUT_SECONDS", "30")),
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        ollama_model_tier1=_env("OLLAMA_MODEL_TIER1", "clawdbot-triage"),
        ollama_model_tier2=_env("OLLAMA_MODEL_TIER2", "llama3.1:70b-instruct-q8_0"),
        ollama_escalation_confidence=float(_env("OLLAMA_ESCALATION_CONFIDENCE", "0.75")),
        triage_backend=_env("TON_IOT_TRIAGE_BACKEND", "ollama"),
    )

    event_log = EventLogger(log_dir)
    capture = TrafficCapture(interface=interface, bpf_filter=bpf_filter)

    shutdown = False

    def _handle_signal(signum, _frame):
        nonlocal shutdown
        sig_name = signal.Signals(signum).name
        log.info("Received %s — shutting down", sig_name)
        shutdown = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info(
        "ClawdBot agent starting  iface=%s  api=%s  interval=%.1fs  telegram=%s  actuator=%s  logs=%s",
        interface,
        api_url,
        harvest_interval,
        "enabled" if notifier.enabled else "disabled",
        "enabled" + (" (dry-run)" if actuator.dry_run else "") if actuator.enabled else "disabled",
        log_dir,
    )

    event_log.log_action(action="agent_start", detail={
        "interface": interface, "api_url": api_url,
        "harvest_interval": harvest_interval, "telegram_enabled": notifier.enabled,
        "actuator_enabled": actuator.enabled, "actuator_dry_run": actuator.dry_run,
        "mgmt_ports_ignored": sorted(mgmt_ports),
    })

    actuator.setup()
    intel.setup()
    capture.start()

    # Cooldown tracker: frozenset of labels → last alert timestamp
    _last_alert_ts: dict[frozenset[str], float] = {}

    try:
        while not shutdown:
            time.sleep(harvest_interval)

            records = capture.harvest()
            if not records:
                log.debug("No flows harvested — sleeping")
                continue

            harvested_count = len(records)
            records = [
                record for record in records
                if not _is_management_record(record, actuator.whitelist, mgmt_ports)
            ]
            ignored_count = harvested_count - len(records)
            if ignored_count:
                log.info(
                    "Ignored %d management/dashboard flow(s) before IDS analysis",
                    ignored_count,
                )
            if not records:
                log.debug("No non-management flows harvested — sleeping")
                continue

            log.info("Harvested %d flow(s)", len(records))

            # Process in batches to avoid API timeouts on large harvests.
            # Per-incident filtering happens after all batches are collected.
            all_detections: list[dict] = []

            for batch_start in range(0, len(records), MAX_BATCH_SIZE):
                batch = records[batch_start : batch_start + MAX_BATCH_SIZE]
                log.info(
                    "Sending batch %d–%d of %d to IDS API",
                    batch_start + 1,
                    min(batch_start + MAX_BATCH_SIZE, len(records)),
                    len(records),
                )

                response = _post_analyze(api_url, batch, timeout=api_timeout)
                if response is None:
                    log.warning("API returned no response — skipping batch")
                    continue

                predictions = response.get("predictions", [])
                triage_items = response.get("triage", [])

                audit_id = response.get("audit_id")
                for i, (pred, tri) in enumerate(zip(predictions, triage_items)):
                    label = pred.get("predicted_label", "unknown")

                    # Recover flow metadata from the original record
                    flow_meta = batch[i].get("_meta") if i < len(batch) else None

                    # Skip management traffic between whitelisted peers
                    # (e.g. SSH sessions) — but NOT scans or attacks
                    if flow_meta and _is_management_record({"_meta": flow_meta}, actuator.whitelist, mgmt_ports):
                        log.debug("Skipping management traffic after IDS response (%s)", label)
                        continue

                    all_detections.append({
                        "prediction": pred,
                        "triage": tri,
                        "flow_meta": flow_meta,
                        "record": {k: v for k, v in batch[i].items() if k != "_meta"} if i < len(batch) else {},
                        "block_result": None,
                        "reputation": None,
                        "audit_id": audit_id,
                    })

            # Send ONE summary Telegram message for the entire harvest cycle
            if all_detections:
                all_detections = _apply_incident_rules(all_detections)
                all_detections = [
                    det for det in all_detections
                    if _detection_label(det) != "normal"
                ]
                if not all_detections:
                    log.info("Harvest summary: no threats after incident rules")
                    continue

                _classify_primary_unknown_incident(
                    all_detections,
                    triage_service=incident_triage,
                )
                all_detections = [
                    det for det in all_detections
                    if _detection_label(det) != "normal"
                ]
                if not all_detections:
                    log.info("Harvest summary: no threats after incident-level LLM classification")
                    continue

                for det in all_detections:
                    pred = det["prediction"]
                    tri = det["triage"]
                    flow_meta = det.get("flow_meta")
                    label = tri.get("label") or pred.get("predicted_label", "unknown")

                    src_ip = flow_meta.get("src_ip") if flow_meta else None
                    reputation = None
                    if src_ip:
                        mitre_ids = [
                            t.get("id", "") for t in tri.get("mitre_techniques", [])
                            if t.get("id", "").startswith("T")
                        ]
                        reputation = intel.enrich(
                            ip=src_ip,
                            severity=tri.get("severity", "low"),
                            label=label,
                            confidence=pred.get("confidence", 0.0),
                            mitre_technique_ids=mitre_ids,
                        )
                        det["reputation"] = reputation

                    effective_severity = tri.get("severity", "low")
                    if src_ip and reputation and intel.should_escalate_block(src_ip):
                        if effective_severity == "low":
                            effective_severity = "medium"
                        log.info(
                            "Escalating block severity for known-bad IP %s: %s → %s",
                            src_ip, tri.get("severity", "low"), effective_severity,
                        )

                    block_result = actuator.maybe_block_from_detection(
                        src_ip=src_ip,
                        severity=effective_severity,
                        label=label,
                        confidence=pred.get("confidence", 0.0),
                    )
                    det["block_result"] = block_result
                    if block_result:
                        event_log.log_action(action="firewall_block", detail=block_result)

                # Cooldown: suppress if same label set was alerted recently
                label_set = frozenset(
                    _detection_label(d)
                    for d in all_detections
                )
                now = time.time()
                incident_summary = _incident_summary(all_detections)
                incident_id = f"{int(now * 1000)}-{incident_summary['primary_label']}"
                last = _last_alert_ts.get(label_set, 0.0)

                if now - last >= ALERT_COOLDOWN_SECONDS:
                    sent = notifier.batch_alert(all_detections)
                    if sent:
                        _last_alert_ts[label_set] = now
                    log.info(
                        "Harvest summary: %d threat, %d possible, Telegram %s",
                        incident_summary["threat_count"],
                        incident_summary["possible_count"],
                        "sent" if sent else "skipped",
                    )
                else:
                    remaining = int(ALERT_COOLDOWN_SECONDS - (now - last))
                    sent = False
                    log.info(
                        "Harvest summary: %d threat, %d possible, Telegram suppressed (cooldown %ds remaining)",
                        incident_summary["threat_count"],
                        incident_summary["possible_count"],
                        remaining,
                    )

                for det in all_detections:
                    event_log.log_attack(
                        prediction=det["prediction"],
                        triage=det["triage"],
                        flow_meta=det.get("flow_meta"),
                        telegram_sent=sent,
                        audit_id=det.get("audit_id"),
                        incident_id=incident_id,
                        incident_summary=incident_summary,
                        block_result=det.get("block_result"),
                        reputation=det.get("reputation"),
                    )
    finally:
        capture.stop()
        actuator.teardown()
        event_log.log_action(action="agent_stop", detail={
            "blocks_active_at_shutdown": actuator.active_block_count,
            "ips_tracked": intel.cache.total_tracked(),
        })
        intel.close()
        log.info("ClawdBot agent stopped")


def main() -> None:
    """CLI entry point — reads config from environment variables."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    interface = _env("CLAWDBOT_INTERFACE")
    if not interface:
        log.error("CLAWDBOT_INTERFACE env var is required (e.g. eth0)")
        sys.exit(1)

    run(
        interface=interface,
        bpf_filter=_env("CLAWDBOT_BPF_FILTER", "ip"),
        api_url=_env("CLAWDBOT_API_URL", "http://127.0.0.1:8000"),
        api_timeout=int(_env("CLAWDBOT_API_TIMEOUT", "60")),
        harvest_interval=float(_env("CLAWDBOT_HARVEST_INTERVAL", "10")),
        bot_token=_env("TELEGRAM_BOT_TOKEN"),
        chat_id=_env("TELEGRAM_CHAT_ID"),
        severity_threshold=_env("CLAWDBOT_SEVERITY_THRESHOLD", "medium"),
        log_dir=_env("CLAWDBOT_LOG_DIR", DEFAULT_LOG_DIR),
        actuator_enabled=_env("CLAWDBOT_ACTUATOR_ENABLED", "").lower() in ("1", "true", "yes"),
        actuator_whitelist=_env("CLAWDBOT_ACTUATOR_WHITELIST", ""),
        actuator_default_ttl=int(_env("CLAWDBOT_ACTUATOR_TTL", "3600")),
        actuator_dry_run=_env("CLAWDBOT_ACTUATOR_DRY_RUN", "").lower() in ("1", "true", "yes"),
        ignore_ports=_env("CLAWDBOT_IGNORE_PORTS", ""),
        threat_intel_db=_env("CLAWDBOT_THREAT_INTEL_DB"),
        abuseipdb_key=_env("ABUSEIPDB_API_KEY"),
        virustotal_key=_env("VIRUSTOTAL_API_KEY"),
        otx_key=_env("OTX_API_KEY"),
        threat_intel_api_ttl=int(_env("CLAWDBOT_THREAT_INTEL_TTL", "86400")),
    )


if __name__ == "__main__":
    main()
