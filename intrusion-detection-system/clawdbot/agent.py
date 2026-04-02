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

from clawdbot.capture import TrafficCapture
from clawdbot.telegram import TelegramNotifier

log = logging.getLogger(__name__)

DEFAULT_LOG_DIR = "/data/ton-iot-project/fresh_start/logs"


class EventLogger:
    """Append-only JSONL logger for attack events and actions taken."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self._attacks_path = os.path.join(log_dir, "attacks.jsonl")
        self._actions_path = os.path.join(log_dir, "actions.jsonl")

    def log_attack(self, *, prediction: dict, triage: dict, flow_meta: dict | None,
                   telegram_sent: bool, audit_id: str | None = None) -> None:
        """Log a detected attack with full context."""
        event = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "epoch": time.time(),
            "event": "attack_detected",
            "prediction": {
                "label": prediction.get("predicted_label"),
                "confidence": prediction.get("confidence"),
                "probabilities": prediction.get("probabilities"),
            },
            "triage": {
                "severity": triage.get("severity"),
                "mitre_tactics": triage.get("mitre_tactics", []),
                "mitre_techniques": triage.get("mitre_techniques", []),
                "summary": triage.get("summary", ""),
                "next_actions": triage.get("next_actions", []),
                "source": triage.get("source", "unknown"),
            },
            "flow": flow_meta,
            "actions": {
                "telegram_sent": telegram_sent,
            },
            "audit_id": audit_id,
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
) -> None:
    """Main capture → analyze → alert loop."""

    notifier = TelegramNotifier(
        bot_token=bot_token,
        chat_id=chat_id,
        severity_threshold=severity_threshold,
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
        "ClawdBot agent starting  iface=%s  api=%s  interval=%.1fs  telegram=%s  logs=%s",
        interface,
        api_url,
        harvest_interval,
        "enabled" if notifier.enabled else "disabled",
        log_dir,
    )

    event_log.log_action(action="agent_start", detail={
        "interface": interface, "api_url": api_url,
        "harvest_interval": harvest_interval, "telegram_enabled": notifier.enabled,
    })

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

            log.info("Harvested %d flow(s)", len(records))

            # Process in batches to avoid API timeouts on large harvests
            all_detections: list[dict] = []  # collect for batch Telegram alert

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
                    if label == "normal":
                        continue

                    # Recover flow metadata from the original record
                    flow_meta = batch[i].get("_meta") if i < len(batch) else None

                    all_detections.append({
                        "prediction": pred,
                        "triage": tri,
                        "flow_meta": flow_meta,
                    })

                    # Persist attack event to disk (always — independent of Telegram)
                    event_log.log_attack(
                        prediction=pred,
                        triage=tri,
                        flow_meta=flow_meta,
                        telegram_sent=False,  # updated below
                        audit_id=audit_id,
                    )

            # Send ONE summary Telegram message for the entire harvest cycle
            if all_detections:
                # Cooldown: suppress if same label set was alerted recently
                label_set = frozenset(
                    d["prediction"].get("predicted_label", "unknown")
                    for d in all_detections
                )
                now = time.time()
                last = _last_alert_ts.get(label_set, 0.0)

                if now - last >= ALERT_COOLDOWN_SECONDS:
                    sent = notifier.batch_alert(all_detections)
                    if sent:
                        _last_alert_ts[label_set] = now
                    log.info(
                        "Harvest summary: %d threat(s) detected, Telegram %s",
                        len(all_detections),
                        "sent" if sent else "skipped",
                    )
                else:
                    remaining = int(ALERT_COOLDOWN_SECONDS - (now - last))
                    log.info(
                        "Harvest summary: %d threat(s) detected, Telegram suppressed (cooldown %ds remaining)",
                        len(all_detections),
                        remaining,
                    )
    finally:
        capture.stop()
        event_log.log_action(action="agent_stop")
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
    )


if __name__ == "__main__":
    main()
