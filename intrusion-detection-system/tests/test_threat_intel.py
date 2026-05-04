"""Tests for clawdbot.threat_intel - external API enrichment + SQLite cache."""

from __future__ import annotations

import json
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from clawdbot.threat_intel import (
    API_CACHE_TTL,
    BADGE_KNOWN_BAD,
    BADGE_SUSPICIOUS,
    MITRE_STIX_URL,
    ThreatCache,
    ThreatIntel,
    _badge,
    _parse_stix_techniques,
    fetch_mitre_stix,
    query_abuseipdb,
    query_otx,
    query_virustotal,
)


# -- Helpers --------------------------------------------------

@pytest.fixture()
def db_path(tmp_path):
    """Provide a temporary threat-intel database path."""
    return str(tmp_path / "test_threat.db")


@pytest.fixture()
def cache(db_path):
    """Provide a temporary ThreatCache fixture."""
    c = ThreatCache(db_path)
    yield c
    c.close()


@pytest.fixture()
def intel(db_path):
    """Provide a ThreatIntel fixture backed by temporary storage."""
    ti = ThreatIntel(db_path=db_path)
    yield ti
    ti.close()


SAMPLE_STIX_BUNDLE = {
    "type": "bundle",
    "id": "bundle--1",
    "objects": [
        {
            "type": "attack-pattern",
            "id": "attack-pattern--1234",
            "name": "Active Scanning",
            "external_references": [
                {"source_name": "mitre-attack", "external_id": "T1595"}
            ],
            "kill_chain_phases": [
                {"kill_chain_name": "mitre-attack", "phase_name": "reconnaissance"}
            ],
            "x_mitre_platforms": ["PRE"],
        },
        {
            "type": "attack-pattern",
            "id": "attack-pattern--5678",
            "name": "Brute Force",
            "external_references": [
                {"source_name": "mitre-attack", "external_id": "T1110"}
            ],
            "kill_chain_phases": [
                {"kill_chain_name": "mitre-attack", "phase_name": "credential-access"}
            ],
            "x_mitre_platforms": ["Linux", "Windows"],
        },
        {
            "type": "attack-pattern",
            "id": "attack-pattern--revoked",
            "name": "Revoked Technique",
            "revoked": True,
            "external_references": [
                {"source_name": "mitre-attack", "external_id": "T9999"}
            ],
        },
    ],
}


# -- Badge logic ----------------------------------------------

class TestBadge:
    """Group tests covering badge behavior."""
    def test_known_bad_high_abuse_score(self):
        """Verify that known bad high abuse score."""
        result = _badge(0, 85, 0, 0)
        assert "Known-bad" in result

    def test_known_bad_high_vt(self):
        """Verify that known bad high vt."""
        result = _badge(0, 0, 5, 0)
        assert "Known-bad" in result

    def test_known_bad_high_cumulative(self):
        """Verify that known bad high cumulative."""
        result = _badge(BADGE_KNOWN_BAD, 0, 0, 0)
        assert "Known-bad" in result

    def test_suspicious_moderate_abuse(self):
        """Verify that suspicious moderate abuse."""
        result = _badge(0, 50, 0, 0)
        assert "Suspicious" in result

    def test_suspicious_one_vt_hit(self):
        """Verify that suspicious one vt hit."""
        result = _badge(0, 0, 1, 0)
        assert "Suspicious" in result

    def test_suspicious_otx_pulses(self):
        """Verify that suspicious otx pulses."""
        result = _badge(0, 0, 0, 3)
        assert "Suspicious" in result

    def test_suspicious_moderate_cumulative(self):
        """Verify that suspicious moderate cumulative."""
        result = _badge(BADGE_SUSPICIOUS, 0, 0, 0)
        assert "Suspicious" in result

    def test_unknown_clean(self):
        """Verify that unknown clean."""
        result = _badge(0, 0, 0, 0)
        assert "Unknown" in result


# -- ThreatCache - SQLite operations -------------------------

class TestThreatCacheBasic:
    """Group tests covering threat cache basic behavior."""
    def test_get_nonexistent(self, cache):
        """Verify that get nonexistent."""
        assert cache.get("1.2.3.4") is None

    def test_record_hit_creates_entry(self, cache):
        """Verify that record hit creates entry."""
        record = cache.record_hit("1.2.3.4", severity="high", label="scanning")
        assert record["ip"] == "1.2.3.4"
        assert record["hit_count"] == 1
        assert record["cumulative_severity"] == 2  # high=2
        assert "scanning" in json.loads(record["labels"])

    def test_record_hit_increments(self, cache):
        """Verify that record hit increments."""
        cache.record_hit("1.2.3.4", severity="medium", label="scanning")
        record = cache.record_hit("1.2.3.4", severity="high", label="ddos_dos")
        assert record["hit_count"] == 2
        assert record["cumulative_severity"] == 3  # medium(1) + high(2)
        labels = json.loads(record["labels"])
        assert "scanning" in labels
        assert "ddos_dos" in labels

    def test_record_hit_deduplicates_labels(self, cache):
        """Verify that record hit deduplicates labels."""
        cache.record_hit("1.2.3.4", severity="medium", label="scanning")
        record = cache.record_hit("1.2.3.4", severity="medium", label="scanning")
        labels = json.loads(record["labels"])
        assert labels.count("scanning") == 1

    def test_first_seen_preserved(self, cache):
        """Verify that first seen preserved."""
        r1 = cache.record_hit("1.2.3.4", severity="low", label="scanning")
        time.sleep(0.01)
        r2 = cache.record_hit("1.2.3.4", severity="low", label="scanning")
        assert r2["first_seen"] == r1["first_seen"]
        assert r2["last_seen"] >= r1["last_seen"]

    def test_total_tracked(self, cache):
        """Verify that total tracked."""
        assert cache.total_tracked() == 0
        cache.record_hit("1.1.1.1", severity="low", label="a")
        cache.record_hit("2.2.2.2", severity="low", label="b")
        assert cache.total_tracked() == 2

    def test_delete_ips_removes_protected_reputation_entries(self, cache):
        """Verify that delete ips removes protected reputation entries."""
        cache.record_hit("100.111.77.70", severity="critical", label="server")
        cache.record_hit("203.0.113.50", severity="high", label="scanning")

        removed = cache.delete_ips({"100.111.77.70"})

        assert removed == 1
        assert cache.get("100.111.77.70") is None
        assert cache.get("203.0.113.50") is not None


class TestThreatCacheAPIScores:
    """Group tests covering threat cache apiscores behavior."""
    def test_update_api_scores(self, cache):
        """Verify that update api scores."""
        cache.record_hit("1.2.3.4", severity="low", label="test")
        cache.update_api_scores("1.2.3.4", abuseipdb_score=75, vt_malicious=2, otx_pulse_count=5)
        record = cache.get("1.2.3.4")
        assert record["abuseipdb_score"] == 75
        assert record["vt_malicious"] == 2
        assert record["otx_pulse_count"] == 5
        assert record["api_checked_at"] > 0

    def test_partial_update(self, cache):
        """Verify that partial update."""
        cache.record_hit("1.2.3.4", severity="low", label="test")
        cache.update_api_scores("1.2.3.4", abuseipdb_score=50)
        record = cache.get("1.2.3.4")
        assert record["abuseipdb_score"] == 50
        assert record["vt_malicious"] == -1  # unchanged default

    def test_needs_api_refresh_new_ip(self, cache):
        """Verify that needs api refresh new ip."""
        cache.record_hit("1.2.3.4", severity="low", label="test")
        assert cache.needs_api_refresh("1.2.3.4") is True

    def test_needs_api_refresh_fresh(self, cache):
        """Verify that needs api refresh fresh."""
        cache.record_hit("1.2.3.4", severity="low", label="test")
        cache.update_api_scores("1.2.3.4", abuseipdb_score=10)
        assert cache.needs_api_refresh("1.2.3.4") is False

    def test_needs_api_refresh_stale(self, cache):
        """Verify that needs api refresh stale."""
        cache.record_hit("1.2.3.4", severity="low", label="test")
        cache.update_api_scores("1.2.3.4", abuseipdb_score=10)
        # Force staleness by setting api_checked_at far back
        cache._conn.execute(
            "UPDATE ip_intel SET api_checked_at = ? WHERE ip = ?",
            (time.time() - API_CACHE_TTL - 1, "1.2.3.4"),
        )
        cache._conn.commit()
        assert cache.needs_api_refresh("1.2.3.4") is True

    def test_needs_api_refresh_unknown_ip(self, cache):
        """Verify that needs api refresh unknown ip."""
        assert cache.needs_api_refresh("9.9.9.9") is True


class TestThreatCacheTopOffenders:
    """Group tests covering threat cache top offenders behavior."""
    def test_top_offenders_ordering(self, cache):
        """Verify that top offenders ordering."""
        cache.record_hit("1.1.1.1", severity="low", label="a")
        cache.record_hit("2.2.2.2", severity="critical", label="b")
        cache.record_hit("3.3.3.3", severity="medium", label="c")
        top = cache.top_offenders(limit=2)
        assert len(top) == 2
        assert top[0]["ip"] == "2.2.2.2"  # critical=3, highest


# -- MITRE ATT&CK STIX ---------------------------------------

class TestMitreStix:
    """Group tests covering mitre stix behavior."""
    def test_parse_stix_techniques(self):
        """Verify that parse stix techniques."""
        techniques = _parse_stix_techniques(SAMPLE_STIX_BUNDLE)
        assert len(techniques) == 2  # revoked excluded
        ids = {t["ext_id"] for t in techniques}
        assert "T1595" in ids
        assert "T1110" in ids
        assert "T9999" not in ids  # revoked

    def test_parse_stix_tactics(self):
        """Verify that parse stix tactics."""
        techniques = _parse_stix_techniques(SAMPLE_STIX_BUNDLE)
        t1595 = next(t for t in techniques if t["ext_id"] == "T1595")
        assert "reconnaissance" in t1595["tactics"]

    def test_parse_stix_platforms(self):
        """Verify that parse stix platforms."""
        techniques = _parse_stix_techniques(SAMPLE_STIX_BUNDLE)
        t1110 = next(t for t in techniques if t["ext_id"] == "T1110")
        assert "Linux" in t1110["platforms"]

    def test_store_and_lookup(self, cache):
        """Verify that store and lookup."""
        techniques = _parse_stix_techniques(SAMPLE_STIX_BUNDLE)
        stored = cache.store_mitre_techniques(techniques)
        assert stored == 2

        t = cache.lookup_mitre("T1595")
        assert t is not None
        assert t["name"] == "Active Scanning"
        assert "reconnaissance" in t["tactics"]

    def test_lookup_missing(self, cache):
        """Verify that lookup missing."""
        assert cache.lookup_mitre("T0000") is None

    def test_mitre_needs_refresh_empty(self, cache):
        """Verify that mitre needs refresh empty."""
        assert cache.mitre_needs_refresh() is True

    def test_mitre_needs_refresh_after_store(self, cache):
        """Verify that mitre needs refresh after store."""
        techniques = _parse_stix_techniques(SAMPLE_STIX_BUNDLE)
        cache.store_mitre_techniques(techniques)
        assert cache.mitre_needs_refresh() is False

    def test_mitre_technique_count(self, cache):
        """Verify that mitre technique count."""
        assert cache.mitre_technique_count() == 0
        techniques = _parse_stix_techniques(SAMPLE_STIX_BUNDLE)
        cache.store_mitre_techniques(techniques)
        assert cache.mitre_technique_count() == 2

    @patch("clawdbot.threat_intel.urllib.request.urlopen")
    def test_fetch_mitre_stix_success(self, mock_urlopen):
        """Verify that fetch mitre stix success."""
        resp_mock = MagicMock()
        resp_mock.read.return_value = json.dumps(SAMPLE_STIX_BUNDLE).encode()
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp_mock

        techniques = fetch_mitre_stix()
        assert len(techniques) == 2

    @patch("clawdbot.threat_intel.urllib.request.urlopen")
    def test_fetch_mitre_stix_failure(self, mock_urlopen):
        """Verify that fetch mitre stix failure."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("connection refused")
        techniques = fetch_mitre_stix()
        assert techniques == []


# -- External API client tests ------------------------------

class TestAPIClients:
    """Group tests covering apiclients behavior."""
    @patch("clawdbot.threat_intel._api_get")
    def test_query_abuseipdb_success(self, mock_get):
        """Verify that query abuseipdb success."""
        mock_get.return_value = {"data": {"abuseConfidenceScore": 85}}
        result = query_abuseipdb("1.2.3.4", "fake-key")
        assert result == 85

    @patch("clawdbot.threat_intel._api_get")
    def test_query_abuseipdb_failure(self, mock_get):
        """Verify that query abuseipdb failure."""
        mock_get.return_value = None
        result = query_abuseipdb("1.2.3.4", "fake-key")
        assert result is None

    @patch("clawdbot.threat_intel._api_get")
    def test_query_virustotal_success(self, mock_get):
        """Verify that query virustotal success."""
        mock_get.return_value = {
            "data": {"attributes": {"last_analysis_stats": {"malicious": 3, "harmless": 70}}}
        }
        result = query_virustotal("1.2.3.4", "fake-key")
        assert result == 3

    @patch("clawdbot.threat_intel._api_get")
    def test_query_virustotal_failure(self, mock_get):
        """Verify that query virustotal failure."""
        mock_get.return_value = None
        result = query_virustotal("1.2.3.4", "fake-key")
        assert result is None

    @patch("clawdbot.threat_intel._api_get")
    def test_query_otx_success(self, mock_get):
        """Verify that query otx success."""
        mock_get.return_value = {"pulse_info": {"count": 7}}
        result = query_otx("1.2.3.4", "fake-key")
        assert result == 7

    @patch("clawdbot.threat_intel._api_get")
    def test_query_otx_failure(self, mock_get):
        """Verify that query otx failure."""
        mock_get.return_value = None
        result = query_otx("1.2.3.4", "fake-key")
        assert result is None

    @patch("clawdbot.threat_intel._api_get")
    def test_query_otx_malformed(self, mock_get):
        """Verify that query otx malformed."""
        mock_get.return_value = {"unexpected": "format"}
        result = query_otx("1.2.3.4", "fake-key")
        assert result == 0  # pulse_info missing -> defaults to 0


# -- ThreatIntel orchestrator ---------------------------------

class TestThreatIntelEnrich:
    """Group tests covering threat intel enrich behavior."""
    def test_enrich_basic_no_keys(self, intel):
        """Verify that enrich basic no keys."""
        rep = intel.enrich(ip="203.0.113.50", severity="medium", label="scanning", confidence=0.8)
        assert rep["badge"] is not None
        assert rep["hit_count"] == 1
        assert rep["cumulative_severity"] == 1
        assert "scanning" in rep["labels"]

    def test_enrich_accumulates_severity(self, intel):
        """Verify that enrich accumulates severity."""
        intel.enrich(ip="203.0.113.50", severity="high", label="scanning", confidence=0.8)
        rep = intel.enrich(ip="203.0.113.50", severity="critical", label="ddos_dos", confidence=0.9)
        assert rep["hit_count"] == 2
        assert rep["cumulative_severity"] == 5  # high(2) + critical(3)

    def test_enrich_with_mitre_details(self, intel):
        # Pre-populate MITRE cache
        """Verify that enrich with mitre details."""
        intel.cache.store_mitre_techniques(_parse_stix_techniques(SAMPLE_STIX_BUNDLE))
        rep = intel.enrich(
            ip="203.0.113.50", severity="high", label="scanning",
            confidence=0.9, mitre_technique_ids=["T1595"],
        )
        assert len(rep["mitre_details"]) == 1
        assert rep["mitre_details"][0]["id"] == "T1595"
        assert "reconnaissance" in rep["mitre_details"][0]["tactics"]

    def test_enrich_unknown_mitre_id_skipped(self, intel):
        """Verify that enrich unknown mitre id skipped."""
        rep = intel.enrich(
            ip="203.0.113.50", severity="high", label="scanning",
            confidence=0.9, mitre_technique_ids=["T0000"],
        )
        assert rep["mitre_details"] == []

    @patch("clawdbot.threat_intel.query_abuseipdb")
    @patch("clawdbot.threat_intel.query_virustotal")
    @patch("clawdbot.threat_intel.query_otx")
    def test_enrich_queries_apis_when_keys_set(self, mock_otx, mock_vt, mock_abuse, db_path):
        """Verify that enrich queries apis when keys set."""
        mock_abuse.return_value = 90
        mock_vt.return_value = 5
        mock_otx.return_value = 10

        ti = ThreatIntel(
            db_path=db_path,
            abuseipdb_key="key1",
            virustotal_key="key2",
            otx_key="key3",
        )
        rep = ti.enrich(ip="203.0.113.50", severity="high", label="scanning", confidence=0.9)
        ti.close()

        assert rep["abuseipdb_score"] == 90
        assert rep["vt_malicious"] == 5
        assert rep["otx_pulse_count"] == 10
        assert "Known-bad" in rep["badge"]
        mock_abuse.assert_called_once()
        mock_vt.assert_called_once()
        mock_otx.assert_called_once()

    @patch("clawdbot.threat_intel.query_abuseipdb")
    def test_enrich_skips_api_when_cache_fresh(self, mock_abuse, db_path):
        """Verify that enrich skips api when cache fresh."""
        mock_abuse.return_value = 50

        ti = ThreatIntel(db_path=db_path, abuseipdb_key="key1")
        ti.enrich(ip="203.0.113.50", severity="medium", label="scanning", confidence=0.8)
        mock_abuse.reset_mock()
        # Second call should use cache
        ti.enrich(ip="203.0.113.50", severity="medium", label="scanning", confidence=0.8)
        ti.close()

        mock_abuse.assert_not_called()


class TestThreatIntelEscalation:
    """Group tests covering threat intel escalation behavior."""
    def test_should_escalate_unknown_ip(self, intel):
        """Verify that should escalate unknown ip."""
        assert intel.should_escalate_block("9.9.9.9") is False

    def test_should_escalate_high_cumulative(self, intel):
        """Verify that should escalate high cumulative."""
        for _ in range(3):
            intel.enrich(ip="203.0.113.50", severity="critical", label="ddos_dos", confidence=0.9)
        # cumulative = 3*3 = 9 > BADGE_KNOWN_BAD(5)
        assert intel.should_escalate_block("203.0.113.50") is True

    def test_should_not_escalate_low_cumulative(self, intel):
        """Verify that should not escalate low cumulative."""
        intel.enrich(ip="203.0.113.50", severity="low", label="scanning", confidence=0.5)
        assert intel.should_escalate_block("203.0.113.50") is False

    @patch("clawdbot.threat_intel.query_abuseipdb")
    def test_should_escalate_high_abuse_score(self, mock_abuse, db_path):
        """Verify that should escalate high abuse score."""
        mock_abuse.return_value = 90
        ti = ThreatIntel(db_path=db_path, abuseipdb_key="key1")
        ti.enrich(ip="203.0.113.50", severity="low", label="test", confidence=0.5)
        assert ti.should_escalate_block("203.0.113.50") is True
        ti.close()

    @patch("clawdbot.threat_intel.query_virustotal")
    def test_should_escalate_high_vt(self, mock_vt, db_path):
        """Verify that should escalate high vt."""
        mock_vt.return_value = 5
        ti = ThreatIntel(db_path=db_path, virustotal_key="key1")
        ti.enrich(ip="203.0.113.50", severity="low", label="test", confidence=0.5)
        assert ti.should_escalate_block("203.0.113.50") is True
        ti.close()


class TestThreatIntelSetup:
    """Group tests covering threat intel setup behavior."""
    @patch("clawdbot.threat_intel.fetch_mitre_stix")
    def test_setup_fetches_mitre_when_stale(self, mock_fetch, intel):
        """Verify that setup fetches mitre when stale."""
        mock_fetch.return_value = _parse_stix_techniques(SAMPLE_STIX_BUNDLE)
        intel.setup()
        mock_fetch.assert_called_once()
        assert intel.cache.mitre_technique_count() == 2

    @patch("clawdbot.threat_intel.fetch_mitre_stix")
    def test_setup_skips_mitre_when_fresh(self, mock_fetch, intel):
        # Pre-populate
        """Verify that setup skips mitre when fresh."""
        intel.cache.store_mitre_techniques(_parse_stix_techniques(SAMPLE_STIX_BUNDLE))
        intel.setup()
        mock_fetch.assert_not_called()

    @patch("clawdbot.threat_intel.fetch_mitre_stix")
    def test_setup_survives_mitre_failure(self, mock_fetch, intel):
        """Verify that setup survives mitre failure."""
        mock_fetch.return_value = []
        intel.setup()  # should not raise
        assert intel.cache.mitre_technique_count() == 0


class TestThreatIntelEnabled:
    """Group tests covering threat intel enabled behavior."""
    def test_always_enabled(self, intel):
        """Verify that always enabled."""
        assert intel.enabled is True

    def test_enabled_without_keys(self, db_path):
        """Verify that enabled without keys."""
        ti = ThreatIntel(db_path=db_path)
        assert ti.enabled is True
        ti.close()
