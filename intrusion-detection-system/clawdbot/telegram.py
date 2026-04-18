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


def _format_block_result(br: dict) -> str:
    """Format a single block_result dict into a human-readable line."""
    ip = br.get("ip", "?")
    if br.get("skipped_reason") == "whitelisted":
        return f"  {escape(ip)} — skipped (whitelisted)"
    if br.get("skipped_reason") == "actuator_disabled":
        return f"  {escape(ip)} — skipped (actuator disabled)"
    if br.get("skipped_reason"):
        return f"  {escape(ip)} — skipped ({escape(br['skipped_reason'])})"
    if br.get("applied"):
        ttl = br.get("ttl", 0)
        ttl_str = f"{ttl // 60}min" if ttl >= 60 else f"{ttl}s"
        prefix = "[DRY-RUN] " if br.get("dry_run") else ""
        return f"  {prefix}Blocked {escape(ip)} for {ttl_str}"
    return f"  {escape(ip)} — no action"


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

    def format_alert(self, prediction: dict, triage: dict, flow_meta: dict | None = None, block_result: dict | None = None, reputation: dict | None = None) -> str:
        severity = triage.get("severity", "review").upper()
        label = triage.get("label", "unknown")
        icon = SEVERITY_ICONS.get(triage.get("severity", "review"), "")
        confidence = prediction.get("confidence", 0.0)
        summary = triage.get("summary", "")
        source = triage.get("source", "unknown")
        llm_reclassified = triage.get("llm_reclassified", False)

        lines = [f"{icon} <b>{severity} SEVERITY</b> — {escape(label)}", ""]

        if llm_reclassified:
            lines.append("<b>Note:</b> Model confidence was below threshold; LLM reclassified this flow.")
            lines.append("")

        # MITRE ATT&CK
        tactics = triage.get("mitre_tactics", [])
        techniques = triage.get("mitre_techniques", [])
        if tactics or techniques:
            lines.append("<b>MITRE ATT&amp;CK</b>")
            for tactic in tactics:
                lines.append(f"  Tactic: {escape(str(tactic))}")
            for tech in techniques:
                tid = tech.get("id", "?")
                tname = tech.get("name", "?")
                lines.append(f"  Technique: {escape(tid)} — {escape(tname)}")
            lines.append("")

        # Confidence
        lines.append(f"Confidence: {confidence:.3f}")

        # Flow metadata
        if flow_meta:
            src = flow_meta.get("src_ip", "?")
            dst = flow_meta.get("dst_ip", "?")
            proto = flow_meta.get("proto", "?")
            lines.append(f"{escape(proto.upper())} {escape(src)} \u2192 {escape(dst)}")

        # Summary
        if summary:
            lines.append(f"{escape(summary)}")
        lines.append("")

        # Firewall action
        if block_result:
            lines.append("<b>Firewall action</b>")
            lines.append(_format_block_result(block_result))
            lines.append("")

        # IP reputation
        if reputation:
            badge = reputation.get("badge", "")
            hits = reputation.get("hit_count", 0)
            if badge:
                lines.append(f"Reputation: {escape(badge)} ({hits} hit(s))")
                lines.append("")

        # Next actions
        actions = triage.get("next_actions", [])
        if actions:
            lines.append("<b>Next actions:</b>")
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
        primary_hints: Counter[str] = Counter()
        label_max_conf: dict[str, float] = {}
        label_top_severity: dict[str, str] = {}
        top_severity = "low"
        sample_flows: dict[str, list[str]] = {}  # label → [flow_str, ...]
        ip_badges: dict[str, str] = {}  # ip → badge (deduped, worst wins)

        for det in detections:
            pred = det["prediction"]
            tri = det["triage"]
            meta = det.get("flow_meta") or {}

            label = tri.get("label") or pred.get("predicted_label", "unknown")
            conf = pred.get("confidence", 0.0)
            sev = tri.get("severity", "low")

            label_counts[label] += 1
            if tri.get("incident_role") == "primary":
                primary_hints[label] += 1
            label_max_conf[label] = max(label_max_conf.get(label, 0.0), conf)
            if SEVERITY_ORDER.get(sev, 0) > SEVERITY_ORDER.get(label_top_severity.get(label, "low"), 0):
                label_top_severity[label] = sev

            if SEVERITY_ORDER.get(sev, 0) > SEVERITY_ORDER.get(top_severity, 0):
                top_severity = sev

            # Keep up to 3 sample flows per label
            if meta and len(sample_flows.get(label, [])) < 3:
                src = meta.get("src_ip", "?")
                dst = meta.get("dst_ip", "?")
                proto = meta.get("proto", "?")
                sample_flows.setdefault(label, []).append(
                    f"{proto.upper()} {src} → {dst}"
                )

            # Collect reputation badges (worst badge per IP wins)
            rep = det.get("reputation")
            if rep and meta:
                src_ip = meta.get("src_ip", "")
                badge = rep.get("badge", "")
                if src_ip and badge:
                    # "Known-bad" > "Suspicious" > "Unknown"
                    existing = ip_badges.get(src_ip, "")
                    if "Known-bad" in badge or not existing:
                        ip_badges[src_ip] = badge
                    elif "Suspicious" in badge and "Known-bad" not in existing:
                        ip_badges[src_ip] = badge

        hinted_labels = [label for label, _ in primary_hints.most_common() if label != "unknown"]
        known_labels = [label for label, _ in label_counts.most_common() if label != "unknown"]
        if hinted_labels:
            primary_label = hinted_labels[0]
        elif primary_hints:
            primary_label = primary_hints.most_common(1)[0][0]
        elif known_labels:
            primary_label = known_labels[0]
        else:
            primary_label = label_counts.most_common(1)[0][0] if label_counts else "unknown"
        secondary_labels = [label for label, _ in label_counts.most_common() if label != primary_label]

        primary_techniques: dict[str, str] = {}
        for det in detections:
            tri = det["triage"]
            label = tri.get("label") or det["prediction"].get("predicted_label", "unknown")
            if label != primary_label:
                continue
            for tech in tri.get("mitre_techniques", []):
                primary_techniques[tech.get("id", "?")] = tech.get("name", "?")

        icon = SEVERITY_ICONS.get(top_severity, "")
        possible_count = len(secondary_labels)
        lines = [
            f"{icon} <b>IDS Batch Alert</b> — 1 threat, {possible_count} possible",
            "",
        ]

        primary_conf = label_max_conf.get(primary_label, 0.0)
        primary_sev = label_top_severity.get(primary_label, "low")
        lines.append(
            f"<b>Primary incident</b>: 1 threat — {escape(primary_label)}, "
            f"max conf {primary_conf:.2f}, severity {escape(primary_sev)}"
        )
        for flow_str in sample_flows.get(primary_label, []):
            lines.append(f"  {escape(flow_str)}")

        if secondary_labels:
            lines.append("")
            lines.append(f"<b>Secondary signals</b>: {possible_count} possible — context only, not MITRE-mapped")
            for label in secondary_labels:
                max_c = label_max_conf.get(label, 0.0)
                lines.append(f"  - <b>{escape(label)}</b> (max conf {max_c:.2f})")

        # LLM reclassification note
        reclassified_count = sum(
            1 for det in detections if det["triage"].get("llm_reclassified")
        )
        if reclassified_count:
            lines.append("")
            lines.append(f"<b>Note:</b> {reclassified_count} observation(s) reclassified by LLM (model confidence below threshold)")

        # MITRE techniques for primary incident only. Secondary labels are noisy
        # context in batch alerts and should not expand the incident mapping.
        if primary_techniques:
            lines.append("")
            lines.append(f"<b>MITRE ATT&amp;CK techniques</b> — primary {escape(primary_label)} only")
            for tid, tname in list(primary_techniques.items())[:8]:
                lines.append(f"  {escape(tid)} — {escape(tname)}")

        # Firewall actions summary
        blocked_ips: list[str] = []
        whitelisted_ips: list[str] = []
        dry_run_ips: list[str] = []
        for det in detections:
            br = det.get("block_result")
            if not br:
                continue
            ip = br.get("ip", "?")
            if br.get("skipped_reason") == "whitelisted":
                if ip not in whitelisted_ips:
                    whitelisted_ips.append(ip)
            elif br.get("applied"):
                ttl = br.get("ttl", 0)
                entry = f"{ip} ({ttl // 60}min)"
                if br.get("dry_run"):
                    if entry not in dry_run_ips:
                        dry_run_ips.append(entry)
                else:
                    if entry not in blocked_ips:
                        blocked_ips.append(entry)

        if blocked_ips or whitelisted_ips or dry_run_ips:
            lines.append("")
            lines.append("<b>Firewall actions</b>")
            for entry in blocked_ips[:6]:
                lines.append(f"  Blocked: {escape(entry)}")
            for entry in dry_run_ips[:4]:
                lines.append(f"  [DRY-RUN] Would block: {escape(entry)}")
            for ip_addr in whitelisted_ips[:4]:
                lines.append(f"  Skipped (whitelisted): {escape(ip_addr)}")

        # IP reputation badges
        if ip_badges:
            lines.append("")
            lines.append("<b>IP Reputation</b>")
            for ip_addr, badge_str in list(ip_badges.items())[:6]:
                lines.append(f"  {escape(ip_addr)}: {escape(badge_str)}")

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
