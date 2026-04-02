"""Telegram SOC alert notifier for ClawdBot."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from html import escape

log = logging.getLogger(__name__)

SEVERITY_ICONS = {
    "low": "",
    "medium": "\u26a0\ufe0f",      # ⚠️
    "high": "\U0001f534",           # 🔴
    "critical": "\U0001f6a8",       # 🚨
    "review": "\U0001f7e1",         # 🟡
}

SEVERITY_ORDER = {"low": 0, "medium": 1, "review": 2, "high": 3, "critical": 4}


class TelegramNotifier:
    def __init__(
        self,
        *,
        bot_token: str,
        chat_id: str,
        severity_threshold: str = "medium",
        timeout_seconds: int = 10,
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.severity_threshold = severity_threshold
        self.timeout_seconds = timeout_seconds

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def should_alert(self, severity: str) -> bool:
        threshold = SEVERITY_ORDER.get(self.severity_threshold, 1)
        level = SEVERITY_ORDER.get(severity, 0)
        return level >= threshold

    def format_alert(self, prediction: dict, triage: dict, flow_meta: dict | None = None) -> str:
        severity = triage.get("severity", "review").upper()
        label = triage.get("label", "unknown")
        icon = SEVERITY_ICONS.get(triage.get("severity", "review"), "")
        confidence = prediction.get("confidence", 0.0)
        summary = triage.get("summary", "")
        source = triage.get("source", "unknown")

        lines = [f"{icon} <b>{severity} SEVERITY</b> — {escape(label)}", ""]

        # MITRE ATT&CK
        tactics = triage.get("mitre_tactics", [])
        techniques = triage.get("mitre_techniques", [])
        if tactics or techniques:
            lines.append("\U0001f50e <b>MITRE ATT&amp;CK</b>")
            for tactic in tactics:
                lines.append(f"  Tactic: {escape(str(tactic))}")
            for tech in techniques:
                tid = tech.get("id", "?")
                tname = tech.get("name", "?")
                lines.append(f"  Technique: {escape(tid)} — {escape(tname)}")
            lines.append("")

        # Confidence
        lines.append(f"\U0001f4ca Confidence: {confidence:.3f}")

        # Flow metadata
        if flow_meta:
            src = flow_meta.get("src_ip", "?")
            dst = flow_meta.get("dst_ip", "?")
            proto = flow_meta.get("proto", "?")
            lines.append(f"\U0001f310 {escape(proto.upper())} {escape(src)} \u2192 {escape(dst)}")

        # Summary
        if summary:
            lines.append(f"\U0001f4dd {escape(summary)}")
        lines.append("")

        # Next actions
        actions = triage.get("next_actions", [])
        if actions:
            lines.append("\u23ed <b>Next actions:</b>")
            for i, action in enumerate(actions[:5], 1):
                lines.append(f"  {i}. {escape(str(action))}")
            lines.append("")

        lines.append(f"<i>Source: {escape(source)}</i>")
        return "\n".join(lines)

    def send_message(self, text: str) -> bool:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            method="POST",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                if not body.get("ok"):
                    log.warning("Telegram API returned ok=false: %s", body)
                    return False
                return True
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            log.error("Failed to send Telegram alert: %s", exc)
            return False

    def alert(self, prediction: dict, triage: dict, flow_meta: dict | None = None) -> bool:
        severity = triage.get("severity", "low")
        if not self.should_alert(severity):
            log.debug("Skipping alert for severity=%s (threshold=%s)", severity, self.severity_threshold)
            return False

        if not self.enabled:
            log.warning("Telegram notifier not configured (missing bot_token or chat_id)")
            return False

        text = self.format_alert(prediction, triage, flow_meta)
        return self.send_message(text)

    # ------------------------------------------------------------------
    # Batch summary — single message per harvest cycle
    # ------------------------------------------------------------------

    def format_batch_summary(
        self,
        detections: list[dict],
    ) -> str:
        """Build one compact summary message for all non-normal detections.

        Each item in *detections* is ``{"prediction": …, "triage": …, "flow_meta": …}``.
        """
        from collections import Counter

        label_counts: Counter[str] = Counter()
        label_max_conf: dict[str, float] = {}
        all_techniques: dict[str, str] = {}  # id → name (deduped)
        top_severity = "low"
        sample_flows: dict[str, list[str]] = {}  # label → [flow_str, ...]

        for det in detections:
            pred = det["prediction"]
            tri = det["triage"]
            meta = det.get("flow_meta") or {}

            label = tri.get("label") or pred.get("predicted_label", "unknown")
            conf = pred.get("confidence", 0.0)
            sev = tri.get("severity", "low")

            label_counts[label] += 1
            label_max_conf[label] = max(label_max_conf.get(label, 0.0), conf)

            if SEVERITY_ORDER.get(sev, 0) > SEVERITY_ORDER.get(top_severity, 0):
                top_severity = sev

            for tech in tri.get("mitre_techniques", []):
                all_techniques[tech.get("id", "?")] = tech.get("name", "?")

            # Keep up to 3 sample flows per label
            if meta and len(sample_flows.get(label, [])) < 3:
                src = meta.get("src_ip", "?")
                dst = meta.get("dst_ip", "?")
                proto = meta.get("proto", "?")
                sample_flows.setdefault(label, []).append(
                    f"{proto.upper()} {src} → {dst}"
                )

        icon = SEVERITY_ICONS.get(top_severity, "")
        total = sum(label_counts.values())
        lines = [
            f"{icon} <b>IDS Batch Alert</b> — {total} threat(s) detected",
            "",
        ]

        # Per-label breakdown
        for label, count in label_counts.most_common():
            max_c = label_max_conf.get(label, 0.0)
            lines.append(f"  • <b>{escape(label)}</b>: {count} flow(s)  (max conf {max_c:.2f})")
            for flow_str in sample_flows.get(label, []):
                lines.append(f"      {escape(flow_str)}")

        # MITRE techniques (deduplicated across all detections)
        if all_techniques:
            lines.append("")
            lines.append("\U0001f50e <b>MITRE ATT&amp;CK techniques</b>")
            for tid, tname in list(all_techniques.items())[:8]:
                lines.append(f"  {escape(tid)} — {escape(tname)}")

        return "\n".join(lines)

    def batch_alert(self, detections: list[dict]) -> bool:
        """Send a single summary Telegram message for a list of detections.

        Returns True if the message was sent, False otherwise.
        Each item: ``{"prediction": …, "triage": …, "flow_meta": …}``.
        """
        if not detections:
            return False

        # Check if *any* detection meets the severity threshold
        dominated = all(
            not self.should_alert(d["triage"].get("severity", "low"))
            for d in detections
        )
        if dominated:
            log.debug("All %d detections below severity threshold — skipping batch alert", len(detections))
            return False

        if not self.enabled:
            log.warning("Telegram notifier not configured (missing bot_token or chat_id)")
            return False

        text = self.format_batch_summary(detections)
        return self.send_message(text)
