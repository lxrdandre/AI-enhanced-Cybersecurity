"""Threat intelligence enrichment - external API lookups + local SQLite cache.

Queries AbuseIPDB, VirusTotal, and OTX AlienVault for IP reputation data,
caches results in a local SQLite database, and tracks per-IP attack history
(first-seen, last-seen, cumulative severity score, labels observed).

The enrichment result is attached to each detection as a ``reputation`` dict
that feeds into both Telegram alerts (badge) and the actuator (escalation).

Design constraints:
  - All external API calls are optional - missing keys simply skip that source.
  - API failures are logged and silently skipped (never block the pipeline).
  - SQLite uses WAL mode for concurrent read safety.
  - Cached API results expire after a configurable TTL (default 24 h).
  - MITRE ATT&CK data is fetched from the official STIX 2.1 feed on startup,
    cached locally, and refreshed periodically.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

# -- Defaults -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = str(PROJECT_ROOT / "data" / "threat_cache.db")
API_CACHE_TTL = 86400  # 24 hours
MITRE_STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)
MITRE_CACHE_TTL = 604800  # 7 days

SEVERITY_SCORE = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# Reputation badge thresholds
BADGE_KNOWN_BAD = 5      # cumulative severity or high external score
BADGE_SUSPICIOUS = 2     # moderate history


def _badge(cum_severity: int, abuseipdb_score: int, vt_malicious: int, otx_pulses: int) -> str:
    """Pick a reputation badge string based on all available signals."""
    if abuseipdb_score >= 80 or vt_malicious >= 3 or cum_severity >= BADGE_KNOWN_BAD:
        return "Known-bad"
    if abuseipdb_score >= 40 or vt_malicious >= 1 or otx_pulses >= 3 or cum_severity >= BADGE_SUSPICIOUS:
        return "Suspicious"
    return "Unknown"


# -- SQLite threat cache --------------------------------------

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS ip_intel (
    ip                TEXT PRIMARY KEY,
    first_seen        REAL NOT NULL,
    last_seen         REAL NOT NULL,
    hit_count         INTEGER NOT NULL DEFAULT 1,
    cumulative_severity INTEGER NOT NULL DEFAULT 0,
    labels            TEXT NOT NULL DEFAULT '[]',
    abuseipdb_score   INTEGER NOT NULL DEFAULT -1,
    vt_malicious      INTEGER NOT NULL DEFAULT -1,
    otx_pulse_count   INTEGER NOT NULL DEFAULT -1,
    api_checked_at    REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS mitre_attack (
    stix_id    TEXT PRIMARY KEY,
    ext_id     TEXT NOT NULL,
    name       TEXT NOT NULL,
    tactics    TEXT NOT NULL DEFAULT '[]',
    platforms  TEXT NOT NULL DEFAULT '[]',
    updated_at REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class ThreatCache:
    """SQLite-backed per-IP threat intelligence cache."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """Initialize the threat cache instance."""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        log.info("ThreatCache opened: %s", db_path)

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    # -- IP record CRUD ---------------------------------------

    def get(self, ip: str) -> dict | None:
        """Return the cached record for *ip*, or None."""
        row = self._conn.execute(
            "SELECT * FROM ip_intel WHERE ip = ?", (ip,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def record_hit(self, ip: str, *, severity: str, label: str) -> dict:
        """Insert or update a hit for *ip*, returning the updated record."""
        now = time.time()
        sev_score = SEVERITY_SCORE.get(severity, 0)
        existing = self.get(ip)

        if existing is None:
            labels_json = json.dumps([label])
            self._conn.execute(
                """INSERT INTO ip_intel
                   (ip, first_seen, last_seen, hit_count, cumulative_severity, labels)
                   VALUES (?, ?, ?, 1, ?, ?)""",
                (ip, now, now, sev_score, labels_json),
            )
            self._conn.commit()
            return self.get(ip)  # type: ignore[return-value]

        labels = json.loads(existing["labels"])
        if label not in labels:
            labels.append(label)

        self._conn.execute(
            """UPDATE ip_intel SET
                 last_seen = ?,
                 hit_count = hit_count + 1,
                 cumulative_severity = cumulative_severity + ?,
                 labels = ?
               WHERE ip = ?""",
            (now, sev_score, json.dumps(labels), ip),
        )
        self._conn.commit()
        return self.get(ip)  # type: ignore[return-value]

    def update_api_scores(
        self,
        ip: str,
        *,
        abuseipdb_score: int | None = None,
        vt_malicious: int | None = None,
        otx_pulse_count: int | None = None,
    ) -> None:
        """Persist external API lookup results for *ip*."""
        now = time.time()
        updates: list[str] = ["api_checked_at = ?"]
        params: list = [now]
        if abuseipdb_score is not None:
            updates.append("abuseipdb_score = ?")
            params.append(abuseipdb_score)
        if vt_malicious is not None:
            updates.append("vt_malicious = ?")
            params.append(vt_malicious)
        if otx_pulse_count is not None:
            updates.append("otx_pulse_count = ?")
            params.append(otx_pulse_count)
        params.append(ip)
        self._conn.execute(
            f"UPDATE ip_intel SET {', '.join(updates)} WHERE ip = ?",
            params,
        )
        self._conn.commit()

    def needs_api_refresh(self, ip: str, ttl: int = API_CACHE_TTL) -> bool:
        """Return True if the API cache for *ip* is stale or missing."""
        row = self._conn.execute(
            "SELECT api_checked_at FROM ip_intel WHERE ip = ?", (ip,)
        ).fetchone()
        if row is None:
            return True
        return (time.time() - row["api_checked_at"]) > ttl

    def top_offenders(self, limit: int = 10) -> list[dict]:
        """Return the top N IPs by cumulative severity."""
        rows = self._conn.execute(
            "SELECT * FROM ip_intel ORDER BY cumulative_severity DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def total_tracked(self) -> int:
        """Return the number of IPs currently stored in the cache."""
        row = self._conn.execute("SELECT COUNT(*) AS c FROM ip_intel").fetchone()
        return row["c"]

    def delete_ips(self, ips) -> int:
        """Delete IPs from reputation history. Returns number of removed rows."""
        clean = [str(ip).strip() for ip in ips if str(ip).strip()]
        if not clean:
            return 0
        before = self._conn.total_changes
        self._conn.executemany("DELETE FROM ip_intel WHERE ip = ?", [(ip,) for ip in clean])
        self._conn.commit()
        return self._conn.total_changes - before

    # -- MITRE ATT&CK STIX cache -----------------------------

    def mitre_last_updated(self) -> float:
        """Return the last MITRE STIX refresh timestamp."""
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'mitre_updated_at'"
        ).fetchone()
        return float(row["value"]) if row else 0.0

    def mitre_needs_refresh(self, ttl: int = MITRE_CACHE_TTL) -> bool:
        """Return True when cached MITRE STIX data is stale."""
        return (time.time() - self.mitre_last_updated()) > ttl

    def store_mitre_techniques(self, techniques: list[dict]) -> int:
        """Upsert MITRE techniques from parsed STIX bundle. Returns count stored."""
        now = time.time()
        count = 0
        for t in techniques:
            self._conn.execute(
                """INSERT INTO mitre_attack (stix_id, ext_id, name, tactics, platforms, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(stix_id) DO UPDATE SET
                     ext_id=excluded.ext_id, name=excluded.name,
                     tactics=excluded.tactics, platforms=excluded.platforms,
                     updated_at=excluded.updated_at""",
                (
                    t["stix_id"],
                    t["ext_id"],
                    t["name"],
                    json.dumps(t.get("tactics", [])),
                    json.dumps(t.get("platforms", [])),
                    now,
                ),
            )
            count += 1
        self._conn.execute(
            """INSERT INTO meta (key, value) VALUES ('mitre_updated_at', ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (str(now),),
        )
        self._conn.commit()
        return count

    def lookup_mitre(self, ext_id: str) -> dict | None:
        """Look up a technique by external ID (e.g. 'T1595')."""
        row = self._conn.execute(
            "SELECT * FROM mitre_attack WHERE ext_id = ?", (ext_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["tactics"] = json.loads(result["tactics"])
        result["platforms"] = json.loads(result["platforms"])
        return result

    def mitre_technique_count(self) -> int:
        """Return the number of cached MITRE techniques."""
        row = self._conn.execute("SELECT COUNT(*) AS c FROM mitre_attack").fetchone()
        return row["c"]

    def techniques_for_tactics(self, tactics: list[str], limit: int = 15) -> list[dict]:
        """Return techniques whose tactics overlap with the given tactic names.

        *tactics* should use display names (e.g. "Persistence"); they are
        normalised to STIX kill-chain phase names before querying.
        """
        if not tactics:
            return []
        # Normalise "Command and Control" -> "command-and-control"
        phase_names = [t.lower().replace(" ", "-") for t in tactics]
        conditions = " OR ".join(["tactics LIKE ?"] * len(phase_names))
        params: list = [f"%{pn}%" for pn in phase_names]
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT ext_id, name, tactics, platforms FROM mitre_attack "
            f"WHERE {conditions} LIMIT ?",
            params,
        ).fetchall()
        return [
            {
                "id": row["ext_id"],
                "name": row["name"],
                "tactics": json.loads(row["tactics"]),
                "platforms": json.loads(row["platforms"]),
            }
            for row in rows
        ]


# -- External API clients -------------------------------------

def _api_get(url: str, headers: dict, timeout: int = 10) -> dict | None:
    """Generic GET with JSON parsing. Returns None on any failure."""
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        log.warning("API request failed (%s): %s", url[:80], exc)
        return None


def query_abuseipdb(ip: str, api_key: str) -> int | None:
    """Query AbuseIPDB v2 for abuse confidence score (0-100). Returns None on failure."""
    data = _api_get(
        f"https://api.abuseipdb.com/api/v2/check?ipAddress={ip}&maxAgeInDays=90",
        headers={"Key": api_key, "Accept": "application/json"},
    )
    if data is None:
        return None
    try:
        return int(data["data"]["abuseConfidenceScore"])
    except (KeyError, TypeError, ValueError):
        return None


def query_virustotal(ip: str, api_key: str) -> int | None:
    """Query VirusTotal v3 for number of malicious detections. Returns None on failure."""
    data = _api_get(
        f"https://www.virustotal.com/api/v3/ip_addresses/{ip}",
        headers={"x-apikey": api_key, "Accept": "application/json"},
    )
    if data is None:
        return None
    try:
        stats = data["data"]["attributes"]["last_analysis_stats"]
        return int(stats.get("malicious", 0))
    except (KeyError, TypeError, ValueError):
        return None


def query_otx(ip: str, api_key: str) -> int | None:
    """Query OTX AlienVault for pulse count referencing this IP. Returns None on failure."""
    data = _api_get(
        f"https://otx.alienvault.com/api/v1/indicators/IPv4/{ip}/general",
        headers={"X-OTX-API-KEY": api_key, "Accept": "application/json"},
    )
    if data is None:
        return None
    try:
        return int(data.get("pulse_info", {}).get("count", 0))
    except (TypeError, ValueError):
        return None


# -- MITRE STIX feed parser -----------------------------------

def _parse_stix_techniques(bundle: dict) -> list[dict]:
    """Extract attack-pattern objects from a STIX 2.1 bundle."""
    techniques: list[dict] = []
    # Build tactic lookup: stix_id -> tactic short-name
    tactic_map: dict[str, str] = {}
    for obj in bundle.get("objects", []):
        if obj.get("type") == "x-mitre-tactic":
            refs = obj.get("external_references", [])
            for ref in refs:
                if ref.get("source_name") == "mitre-attack":
                    tactic_map[obj["id"]] = ref.get("external_id", obj.get("name", ""))

    # Build relationship map: technique stix_id -> list of tactic stix_ids
    tech_to_tactics: dict[str, list[str]] = {}
    for obj in bundle.get("objects", []):
        if obj.get("type") == "relationship" and obj.get("relationship_type") == "uses":
            continue
        # kill-chain-phases is on the attack-pattern itself, handled below

    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        ext_id = ""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                ext_id = ref.get("external_id", "")
                break
        if not ext_id:
            continue

        # Extract tactics from kill_chain_phases
        tactics = []
        for phase in obj.get("kill_chain_phases", []):
            if phase.get("kill_chain_name") == "mitre-attack":
                tactics.append(phase["phase_name"])

        techniques.append({
            "stix_id": obj["id"],
            "ext_id": ext_id,
            "name": obj.get("name", ""),
            "tactics": tactics,
            "platforms": obj.get("x_mitre_platforms", []),
        })

    return techniques


def fetch_mitre_stix(url: str = MITRE_STIX_URL, timeout: int = 30) -> list[dict]:
    """Download and parse the MITRE ATT&CK STIX bundle. Returns parsed techniques."""
    log.info("Fetching MITRE ATT&CK STIX feed from %s", url[:80])
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            bundle = json.loads(raw.decode("utf-8"))
            techniques = _parse_stix_techniques(bundle)
            log.info("Parsed %d MITRE ATT&CK techniques from STIX feed", len(techniques))
            return techniques
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        log.error("Failed to fetch MITRE STIX feed: %s", exc)
        return []


# -- Main enrichment orchestrator -----------------------------

class ThreatIntel:
    """Orchestrates threat intelligence lookups and caching.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    abuseipdb_key : str
        AbuseIPDB API key (empty = skip).
    virustotal_key : str
        VirusTotal API key (empty = skip).
    otx_key : str
        OTX AlienVault API key (empty = skip).
    api_cache_ttl : int
        Seconds before re-querying external APIs for a known IP.
    mitre_stix_url : str
        URL to the MITRE ATT&CK STIX JSON bundle.
    """

    def __init__(
        self,
        *,
        db_path: str = DEFAULT_DB_PATH,
        abuseipdb_key: str = "",
        virustotal_key: str = "",
        otx_key: str = "",
        api_cache_ttl: int = API_CACHE_TTL,
        mitre_stix_url: str = MITRE_STIX_URL,
    ):
        """Initialize the threat intel instance."""
        self.cache = ThreatCache(db_path)
        self.abuseipdb_key = abuseipdb_key
        self.virustotal_key = virustotal_key
        self.otx_key = otx_key
        self.api_cache_ttl = api_cache_ttl
        self.mitre_stix_url = mitre_stix_url
        self._has_any_key = bool(abuseipdb_key or virustotal_key or otx_key)

        sources = []
        if abuseipdb_key:
            sources.append("AbuseIPDB")
        if virustotal_key:
            sources.append("VirusTotal")
        if otx_key:
            sources.append("OTX")
        log.info(
            "ThreatIntel initialised  sources=%s  cache=%s  api_ttl=%ds",
            ", ".join(sources) or "none (local-only)",
            db_path,
            api_cache_ttl,
        )

    @property
    def enabled(self) -> bool:
        """Return True because local cache enrichment is always available."""
        return True  # always enabled - local cache works without API keys

    def setup(self) -> None:
        """Refresh MITRE STIX data if stale (non-blocking on failure)."""
        if self.cache.mitre_needs_refresh():
            techniques = fetch_mitre_stix(self.mitre_stix_url)
            if techniques:
                stored = self.cache.store_mitre_techniques(techniques)
                log.info("Stored %d MITRE techniques in cache", stored)
            else:
                log.warning("MITRE STIX refresh failed - using stale data (%d techniques cached)",
                            self.cache.mitre_technique_count())

    def close(self) -> None:
        """Close the SQLite connection."""
        self.cache.close()

    def enrich(
        self,
        *,
        ip: str,
        severity: str,
        label: str,
        confidence: float,
        mitre_technique_ids: list[str] | None = None,
    ) -> dict:
        """Enrich a detection with threat intel. Returns a reputation dict.

        The returned dict contains:
          badge, hit_count, cumulative_severity, first_seen, last_seen,
          abuseipdb_score, vt_malicious, otx_pulse_count, labels,
          mitre_details (enriched from STIX cache if available)
        """
        # 1. Record the hit in local cache
        record = self.cache.record_hit(ip, severity=severity, label=label)

        # 2. External API lookups (if keys configured and cache stale)
        if self._has_any_key and self.cache.needs_api_refresh(ip, self.api_cache_ttl):
            self._query_apis(ip)
            record = self.cache.get(ip) or record

        # 3. Enrich MITRE technique IDs with STIX data
        mitre_details = []
        for tid in (mitre_technique_ids or []):
            info = self.cache.lookup_mitre(tid)
            if info:
                mitre_details.append({
                    "id": info["ext_id"],
                    "name": info["name"],
                    "tactics": info["tactics"],
                    "platforms": info["platforms"],
                })

        # 4. Compute badge
        badge = _badge(
            record["cumulative_severity"],
            max(record.get("abuseipdb_score", -1), 0),
            max(record.get("vt_malicious", -1), 0),
            max(record.get("otx_pulse_count", -1), 0),
        )

        return {
            "badge": badge,
            "hit_count": record["hit_count"],
            "cumulative_severity": record["cumulative_severity"],
            "first_seen": record["first_seen"],
            "last_seen": record["last_seen"],
            "abuseipdb_score": record.get("abuseipdb_score", -1),
            "vt_malicious": record.get("vt_malicious", -1),
            "otx_pulse_count": record.get("otx_pulse_count", -1),
            "labels": json.loads(record["labels"]) if isinstance(record["labels"], str) else record["labels"],
            "mitre_details": mitre_details,
        }

    def _query_apis(self, ip: str) -> None:
        """Query all configured external APIs and cache results."""
        abuse_score = None
        vt_malicious = None
        otx_pulses = None

        if self.abuseipdb_key:
            abuse_score = query_abuseipdb(ip, self.abuseipdb_key)
        if self.virustotal_key:
            vt_malicious = query_virustotal(ip, self.virustotal_key)
        if self.otx_key:
            otx_pulses = query_otx(ip, self.otx_key)

        self.cache.update_api_scores(
            ip,
            abuseipdb_score=abuse_score,
            vt_malicious=vt_malicious,
            otx_pulse_count=otx_pulses,
        )
        log.info(
            "API enrichment for %s: abuse=%s vt=%s otx=%s",
            ip, abuse_score, vt_malicious, otx_pulses,
        )

    def should_escalate_block(self, ip: str, *, threshold: int = BADGE_KNOWN_BAD) -> bool:
        """Return True if *ip* has enough history/intel to warrant faster blocking."""
        record = self.cache.get(ip)
        if record is None:
            return False
        if record["cumulative_severity"] >= threshold:
            return True
        if record.get("abuseipdb_score", -1) >= 80:
            return True
        if record.get("vt_malicious", -1) >= 3:
            return True
        return False
