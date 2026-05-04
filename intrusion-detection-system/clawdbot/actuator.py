"""Active response actuator - block/unblock attacker IPs via nftables.

Maintains an nftables set of blocked IPs with automatic TTL-based expiry.
A configurable whitelist prevents accidental self-lockout (VPN, management,
localhost, etc.).

Design constraints:
  - Whitelist is **always** enforced - whitelisted IPs are silently skipped.
  - Every block/unblock action is logged to ``actions.jsonl`` via EventLogger.
  - Blocks use nftables set elements with a ``timeout`` so the kernel handles
    automatic expiry - no background reaper thread needed.
  - The nftables table/set is created idempotently on ``setup()``.
  - ``teardown()`` flushes the set but keeps the table (safe for restarts).
"""

from __future__ import annotations

import ipaddress
import logging
import os
import shlex
import subprocess
import time

log = logging.getLogger(__name__)

# -- Defaults -------------------------------------------------

TABLE_NAME = "clawdbot"
SET_NAME = "blocklist"
CHAIN_NAME = "input_filter"

# Default whitelist - NetBird mesh + loopback.  Extended via env var.
_DEFAULT_WHITELIST = [
    "127.0.0.0/8",         # loopback
    "100.111.0.0/16",      # NetBird VPN mesh (covers MacBook + SVM peers)
]

DEFAULT_BLOCK_SECONDS = 3600   # 1 hour
MAX_BLOCK_SECONDS = 86400      # 24 hours - hard cap
MIN_BLOCK_SECONDS = 60         # 1 minute - lower guard

# Map triage severity -> block duration (seconds)
SEVERITY_DURATION: dict[str, int] = {
    "medium":   1800,   # 30 min
    "high":     3600,   # 1 hour
    "critical": 7200,   # 2 hours
}

# Severities that trigger automatic blocking
ACTIONABLE_SEVERITIES = frozenset({"medium", "high", "critical"})


def _parse_whitelist(extra_csv: str) -> set[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Build the whitelist from defaults + a comma-separated env string."""
    nets: set[ipaddress.IPv4Network | ipaddress.IPv6Network] = set()
    for entry in _DEFAULT_WHITELIST:
        nets.add(ipaddress.ip_network(entry, strict=False))
    for entry in extra_csv.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            nets.add(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            log.warning("Ignoring invalid whitelist entry: %r", entry)
    return nets


def _is_whitelisted(
    addr: str,
    whitelist: set[ipaddress.IPv4Network | ipaddress.IPv6Network],
) -> bool:
    """Return True if *addr* falls inside any whitelisted network."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return True  # unparseable -> safe-side: do not block
    return any(ip in net for net in whitelist)


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run an nftables command, logging it first."""
    log.debug("nft: %s", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


class Actuator:
    """NFTables-based active response engine.

    Parameters
    ----------
    enabled : bool
        Master switch.  When *False*, all block/unblock calls are no-ops.
    extra_whitelist : str
        Comma-separated CIDRs or IPs to add to the default whitelist.
    default_ttl : int
        Fallback block duration in seconds (used when severity is not mapped).
    dry_run : bool
        When *True*, commands are logged but not executed (useful for testing).
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        extra_whitelist: str = "",
        default_ttl: int = DEFAULT_BLOCK_SECONDS,
        dry_run: bool = False,
    ):
        """Initialize the actuator instance."""
        self.enabled = enabled
        self.dry_run = dry_run
        self.default_ttl = max(MIN_BLOCK_SECONDS, min(default_ttl, MAX_BLOCK_SECONDS))
        self.whitelist = _parse_whitelist(extra_whitelist)
        # In-memory tracker: ip -> expiry epoch (mirrors kernel set for logging)
        self._active_blocks: dict[str, float] = {}
        self._setup_done = False

        if self.enabled:
            log.info(
                "Actuator ENABLED  dry_run=%s  default_ttl=%ds  whitelist=%s",
                self.dry_run,
                self.default_ttl,
                ", ".join(str(n) for n in sorted(self.whitelist, key=str)),
            )
        else:
            log.info("Actuator DISABLED - blocks will be skipped")

    # -- nftables lifecycle -----------------------------------

    def setup(self) -> None:
        """Create the nftables table, set, and drop chain idempotently."""
        if not self.enabled or self._setup_done:
            return
        if self.dry_run:
            log.info("[DRY-RUN] Would create nftables table/set/chain")
            self._setup_done = True
            return

        # Create table (idempotent with 'add')
        # nft add table inet clawdbot
        _run(["nft", "add", "table", "inet", TABLE_NAME], check=False)

        # Create set with timeout support
        # nft add set inet clawdbot blocklist { type ipv4_addr ; flags timeout ; }
        _run([
            "nft", "add", "set", "inet", TABLE_NAME, SET_NAME,
            "{ type ipv4_addr ; flags timeout ; }",
        ], check=False)

        # Create input chain - priority -10 so it evaluates BEFORE firewalld's
        # filter chain (priority 0) on Rocky Linux / RHEL systems.
        _run([
            "nft", "add", "chain", "inet", TABLE_NAME, CHAIN_NAME,
            "{ type filter hook input priority -10 ; policy accept ; }",
        ], check=False)

        # Add drop rule referencing the set (idempotent - nft ignores dupes)
        _run([
            "nft", "add", "rule", "inet", TABLE_NAME, CHAIN_NAME,
            "ip", "saddr", f"@{SET_NAME}", "counter", "drop",
        ], check=False)

        self._setup_done = True
        log.info("nftables table/set/chain ready (%s.%s.%s)", TABLE_NAME, SET_NAME, CHAIN_NAME)

    def teardown(self) -> None:
        """Flush the blocklist set (remove all blocks). Safe for restarts."""
        if not self.enabled or self.dry_run:
            return
        _run(["nft", "flush", "set", "inet", TABLE_NAME, SET_NAME], check=False)
        self._active_blocks.clear()
        log.info("Flushed nftables blocklist set")

    # -- Core block / unblock ---------------------------------

    def block(self, ip: str, *, ttl: int | None = None, reason: str = "") -> dict:
        """Add *ip* to the nftables blocklist with a TTL.

        Returns a dict describing the action taken (for logging).
        """
        result: dict = {
            "action": "block",
            "ip": ip,
            "ttl": ttl or self.default_ttl,
            "reason": reason,
            "applied": False,
            "skipped_reason": None,
        }

        if not self.enabled:
            result["skipped_reason"] = "actuator_disabled"
            return result

        # Validate IP before whitelist check (invalid addrs can't be checked)
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            result["skipped_reason"] = "invalid_ip"
            log.warning("Block SKIPPED (invalid IP): %r", ip)
            return result

        if _is_whitelisted(ip, self.whitelist):
            result["skipped_reason"] = "whitelisted"
            log.info("Block SKIPPED (whitelisted): %s", ip)
            return result

        effective_ttl = max(MIN_BLOCK_SECONDS, min(ttl or self.default_ttl, MAX_BLOCK_SECONDS))
        result["ttl"] = effective_ttl

        if self.dry_run:
            result["applied"] = True
            result["dry_run"] = True
            log.info("[DRY-RUN] Would block %s for %ds - %s", ip, effective_ttl, reason)
        else:
            proc = _run([
                "nft", "add", "element", "inet", TABLE_NAME, SET_NAME,
                "{ " + ip + f" timeout {effective_ttl}s" + " }",
            ], check=False)
            if proc.returncode == 0:
                result["applied"] = True
                log.info("BLOCKED %s for %ds - %s", ip, effective_ttl, reason)
            else:
                result["skipped_reason"] = f"nft_error: {proc.stderr.strip()}"
                log.error("nft block failed for %s: %s", ip, proc.stderr.strip())

        if result["applied"]:
            self._active_blocks[ip] = time.time() + effective_ttl

        return result

    def unblock(self, ip: str, *, reason: str = "manual") -> dict:
        """Remove *ip* from the blocklist immediately."""
        result: dict = {
            "action": "unblock",
            "ip": ip,
            "reason": reason,
            "applied": False,
            "skipped_reason": None,
        }

        if not self.enabled:
            result["skipped_reason"] = "actuator_disabled"
            return result

        if self.dry_run:
            result["applied"] = True
            result["dry_run"] = True
            log.info("[DRY-RUN] Would unblock %s - %s", ip, reason)
        else:
            proc = _run([
                "nft", "delete", "element", "inet", TABLE_NAME, SET_NAME,
                "{ " + ip + " }",
            ], check=False)
            if proc.returncode == 0:
                result["applied"] = True
                log.info("UNBLOCKED %s - %s", ip, reason)
            else:
                result["skipped_reason"] = f"nft_error: {proc.stderr.strip()}"

        self._active_blocks.pop(ip, None)
        return result

    # -- High-level helpers -----------------------------------

    def maybe_block_from_detection(
        self,
        *,
        src_ip: str | None,
        severity: str,
        label: str,
        confidence: float,
    ) -> dict | None:
        """Decide whether to block *src_ip* based on severity and confidence.

        Returns the action dict if a block was attempted, or *None* if no
        action was taken.
        """
        if not self.enabled:
            return None

        if not src_ip:
            return None

        if severity not in ACTIONABLE_SEVERITIES:
            return None

        # Already actively blocked - skip redundant nft call
        if src_ip in self._active_blocks and self._active_blocks[src_ip] > time.time():
            return None

        ttl = SEVERITY_DURATION.get(severity, self.default_ttl)
        reason = f"{label} (severity={severity}, confidence={confidence:.3f})"
        return self.block(src_ip, ttl=ttl, reason=reason)

    @property
    def active_block_count(self) -> int:
        """Number of IPs currently tracked as blocked (in-memory)."""
        now = time.time()
        # Lazy prune expired entries
        expired = [ip for ip, exp in self._active_blocks.items() if exp <= now]
        for ip in expired:
            del self._active_blocks[ip]
        return len(self._active_blocks)
