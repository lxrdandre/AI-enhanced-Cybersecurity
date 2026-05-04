"""Unit tests for triage module (MITRE mapping, severity, fallback logic, Ollama)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.triage import DEFAULT_MITRE_MAP, TriageService, _default_triage_item, _severity_for_label, _extract_json


# -- Severity helper -----------------------------------------


class TestSeverityForLabel:
    """Group tests covering severity for label behavior."""
    def test_normal_always_low(self):
        """Verify that normal always low."""
        assert _severity_for_label("normal", 0.99) == "low"

    def test_high_confidence_attack(self):
        """Verify that high confidence attack."""
        assert _severity_for_label("ddos_dos", 0.95) == "high"

    def test_medium_confidence_attack(self):
        """Verify that medium confidence attack."""
        assert _severity_for_label("backdoor", 0.75) == "medium"

    def test_low_confidence_attack(self):
        """Verify that low confidence attack."""
        assert _severity_for_label("scanning", 0.5) == "review"

    def test_boundary_high(self):
        """Verify that boundary high."""
        assert _severity_for_label("xss", 0.9) == "high"

    def test_boundary_medium(self):
        """Verify that boundary medium."""
        assert _severity_for_label("xss", 0.7) == "medium"


# -- Default triage item -------------------------------------


class TestDefaultTriageItem:
    """Group tests covering default triage item behavior."""
    def test_normal_prediction(self):
        """Verify that normal prediction."""
        pred = {"predicted_label": "normal", "confidence": 0.98}
        result = _default_triage_item(pred)
        assert result["label"] == "normal"
        assert result["severity"] == "low"
        assert result["mitre_tactics"] == []
        assert result["source"] == "fallback"

    def test_ddos_prediction(self):
        """Verify that ddos prediction."""
        pred = {"predicted_label": "ddos_dos", "confidence": 0.92}
        result = _default_triage_item(pred)
        assert result["label"] == "ddos_dos"
        assert result["severity"] == "high"
        assert "Impact" in result["mitre_tactics"]
        assert any(t["id"] == "T1498" for t in result["mitre_techniques"])

    def test_unknown_label_fallback(self):
        """Verify that unknown label fallback."""
        pred = {"predicted_label": "unknown_class", "confidence": 0.5}
        result = _default_triage_item(pred)
        assert result["label"] == "unknown_class"
        assert result["mitre_tactics"] == []

    @pytest.mark.parametrize("label", list(DEFAULT_MITRE_MAP.keys()))
    def test_all_mapped_labels_produce_valid_triage(self, label):
        """Verify that all mapped labels produce valid triage."""
        pred = {"predicted_label": label, "confidence": 0.85}
        result = _default_triage_item(pred)
        assert "label" in result
        assert "severity" in result
        assert "mitre_tactics" in result
        assert "source" in result


# -- JSON extraction ------------------------------------------


class TestExtractJson:
    """Group tests covering extract json behavior."""
    def test_plain_json(self):
        """Verify that plain json."""
        data = '{"label": "xss", "severity": "high"}'
        parsed = _extract_json(data)
        assert parsed["label"] == "xss"

    def test_fenced_json(self):
        """Verify that fenced json."""
        data = '```json\n{"label": "backdoor"}\n```'
        parsed = _extract_json(data)
        assert parsed["label"] == "backdoor"

    def test_fenced_without_lang(self):
        """Verify that fenced without lang."""
        data = '```\n{"label": "scanning"}\n```'
        parsed = _extract_json(data)
        assert parsed["label"] == "scanning"

    def test_json_with_surrounding_text(self):
        """Verify that json with surrounding text."""
        data = 'Here is the result: {"label": "injection"} -- done'
        parsed = _extract_json(data)
        assert parsed["label"] == "injection"

    def test_invalid_json_raises(self):
        """Verify that invalid json raises."""
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _extract_json("not json at all")


# -- TriageService (without LLM) -----------------------------


class TestTriageServiceFallback:
    """Group tests covering triage service fallback behavior."""
    def test_disabled_when_fallback_backend(self):
        """Verify that disabled when fallback backend."""
        svc = TriageService(api_key=None, model="gemini-2.0-flash", timeout_seconds=10, triage_backend="fallback")
        assert svc.enabled is False

    def test_enabled_when_ollama_backend(self):
        """Verify that enabled when ollama backend."""
        svc = TriageService(api_key=None, model="gemini-2.0-flash", timeout_seconds=10, triage_backend="ollama")
        assert svc.enabled is True

    def test_enabled_when_gemini_backend_with_key(self):
        """Verify that enabled when gemini backend with key."""
        svc = TriageService(api_key="test-key", model="gemini-2.0-flash", timeout_seconds=10, triage_backend="gemini")
        assert svc.enabled is True

    def test_disabled_when_gemini_backend_without_key(self):
        """Verify that disabled when gemini backend without key."""
        svc = TriageService(api_key=None, model="gemini-2.0-flash", timeout_seconds=10, triage_backend="gemini")
        assert svc.enabled is False

    def test_fallback_triage_for_single_prediction(self):
        """Verify that fallback triage for single prediction."""
        svc = TriageService(api_key=None, model="gemini-2.0-flash", timeout_seconds=10, triage_backend="fallback")
        predictions = [{"predicted_label": "password", "confidence": 0.88}]
        records = [{"duration": 1, "src_bytes": 0, "dst_bytes": 0, "proto": "tcp"}]
        triage, llm_error = svc.triage_predictions(predictions=predictions, records=records, context=None)
        assert len(triage) == 1
        assert triage[0]["source"] == "fallback"
        assert triage[0]["label"] == "password"
        assert llm_error is None

    def test_fallback_triage_batch(self):
        """Verify that fallback triage batch."""
        svc = TriageService(api_key=None, model="gemini-2.0-flash", timeout_seconds=10, triage_backend="fallback")
        predictions = [
            {"predicted_label": "normal", "confidence": 0.99},
            {"predicted_label": "ddos_dos", "confidence": 0.91},
            {"predicted_label": "xss", "confidence": 0.65},
        ]
        records = [{"proto": "tcp"}] * 3
        triage, _ = svc.triage_predictions(predictions=predictions, records=records, context=None)
        assert len(triage) == 3
        assert triage[0]["severity"] == "low"
        assert triage[1]["severity"] == "high"
        assert triage[2]["severity"] == "review"


# -- Ollama backend (mocked) ---------------------------------

MOCK_OLLAMA_RESPONSE = json.dumps({
    "label": "ddos_dos",
    "severity": "high",
    "mitre_tactics": ["Impact"],
    "mitre_techniques": [
        {"id": "T1498", "name": "Network Denial of Service", "confidence": "high", "reason": "Volumetric flood"}
    ],
    "summary": "DDoS attack detected",
    "next_actions": ["Block source IP"],
    "confidence_note": "High confidence",
})


def _make_ollama_svc(**overrides):
    """Build a TriageService configured for mocked Ollama calls."""
    defaults = dict(
        api_key=None,
        model="gemini-2.0-flash",
        timeout_seconds=30,
        triage_backend="ollama",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model_tier1="mistral-small:24b",
        ollama_model_tier2="llama3.1:70b-instruct-q8_0",
        ollama_escalation_confidence=0.75,
    )
    defaults.update(overrides)
    return TriageService(**defaults)


class TestOllamaTriage:
    """Group tests covering ollama triage behavior."""
    def test_tier1_only_high_confidence(self):
        """High-confidence prediction -> tier-1 only, no escalation."""
        svc = _make_ollama_svc()
        pred = {"predicted_label": "ddos_dos", "confidence": 0.95}
        record = {"proto": "tcp", "duration": 1}

        with patch.object(svc, "_ollama_chat", return_value=MOCK_OLLAMA_RESPONSE) as mock_chat:
            triage, llm_error = svc.triage_predictions(
                predictions=[pred], records=[record], context=None,
            )
            assert mock_chat.call_count == 1
            assert mock_chat.call_args.kwargs["model"] == "mistral-small:24b"

        assert llm_error is None
        assert triage[0]["label"] == "ddos_dos"
        assert triage[0]["severity"] == "high"
        assert "ollama:mistral-small:24b" in triage[0]["source"]

    def test_escalation_on_low_confidence(self):
        """Low confidence -> escalates to tier-2."""
        svc = _make_ollama_svc(ollama_escalation_confidence=0.75)
        pred = {"predicted_label": "injection", "confidence": 0.55}
        record = {"proto": "tcp"}

        tier2_response = json.dumps({
            "label": "injection",
            "severity": "medium",
            "mitre_tactics": ["Initial Access"],
            "mitre_techniques": [{"id": "T1190", "name": "Exploit Public-Facing Application", "confidence": "medium", "reason": "Low confidence injection"}],
            "summary": "Possible injection, needs review",
            "next_actions": ["Inspect payload"],
            "confidence_note": "Escalated to tier-2 due to low confidence",
        })

        call_count = {"n": 0}

        def mock_chat(*, model, prompt_text):
            """Return a mocked LLM chat response."""
            call_count["n"] += 1
            if model == "mistral-small:24b":
                return MOCK_OLLAMA_RESPONSE
            return tier2_response

        with patch.object(svc, "_ollama_chat", side_effect=mock_chat):
            triage, llm_error = svc.triage_predictions(
                predictions=[pred], records=[record], context=None,
            )

        assert call_count["n"] == 2
        assert llm_error is None
        assert "70b" in triage[0]["source"]

    def test_review_severity_does_not_escalate_by_itself(self):
        """Severity=review alone stays on tier-1."""
        svc = _make_ollama_svc()
        pred = {"predicted_label": "scanning", "confidence": 0.95}
        record = {"proto": "udp"}

        call_count = {"n": 0}

        def mock_chat(*, model, prompt_text):
            """Return a mocked LLM chat response."""
            call_count["n"] += 1
            return MOCK_OLLAMA_RESPONSE

        with (
            patch("app.triage._severity_for_label", return_value="review"),
            patch.object(svc, "_ollama_chat", side_effect=mock_chat),
        ):
            svc.triage_predictions(predictions=[pred], records=[record], context=None)

        assert call_count["n"] == 1

    def test_escalation_on_critical_severity(self):
        """Severity=critical escalates even when confidence is otherwise high."""
        svc = _make_ollama_svc()
        pred = {"predicted_label": "scanning", "confidence": 0.95}
        record = {"proto": "udp"}

        call_count = {"n": 0}

        def mock_chat(*, model, prompt_text):
            """Return a mocked LLM chat response."""
            call_count["n"] += 1
            return MOCK_OLLAMA_RESPONSE

        with (
            patch("app.triage._severity_for_label", return_value="critical"),
            patch.object(svc, "_ollama_chat", side_effect=mock_chat),
        ):
            svc.triage_predictions(predictions=[pred], records=[record], context=None)

        assert call_count["n"] == 2

    def test_primary_unknown_uses_tier2_directly(self):
        """Verify that primary unknown uses tier2 directly."""
        svc = _make_ollama_svc()
        pred = {"predicted_label": "unknown", "confidence": 0.40}
        record = {"proto": "tcp"}

        with patch.object(svc, "_ollama_chat", return_value=MOCK_OLLAMA_RESPONSE) as mock_chat:
            triage, llm_error = svc.triage_predictions(
                predictions=[pred],
                records=[record],
                context={"unknown_priority": "primary"},
            )

        assert llm_error is None
        assert mock_chat.call_count == 1
        assert mock_chat.call_args.kwargs["model"] == "llama3.1:70b-instruct-q8_0"
        assert triage[0]["llm_reclassified"] is True

    def test_secondary_unknown_uses_tier1_only(self):
        """Verify that secondary unknown uses tier1 only."""
        svc = _make_ollama_svc()
        pred = {"predicted_label": "unknown", "confidence": 0.40}
        record = {"proto": "tcp"}

        with patch.object(svc, "_ollama_chat", return_value=MOCK_OLLAMA_RESPONSE) as mock_chat:
            triage, llm_error = svc.triage_predictions(
                predictions=[pred],
                records=[record],
                context={"unknown_priority": "secondary"},
            )

        assert llm_error is None
        assert mock_chat.call_count == 1
        assert mock_chat.call_args.kwargs["model"] == "mistral-small:24b"
        assert triage[0]["llm_reclassified"] is True

    def test_tier2_failure_falls_back_to_tier1(self):
        """If tier-2 fails, tier-1 result is kept."""
        svc = _make_ollama_svc()
        pred = {"predicted_label": "xss", "confidence": 0.50}
        record = {"proto": "tcp"}

        call_count = {"n": 0}

        def mock_chat(*, model, prompt_text):
            """Return a mocked LLM chat response."""
            call_count["n"] += 1
            if "70b" in model:
                raise TimeoutError("tier-2 timed out")
            return MOCK_OLLAMA_RESPONSE

        with patch.object(svc, "_ollama_chat", side_effect=mock_chat):
            triage, llm_error = svc.triage_predictions(
                predictions=[pred], records=[record], context=None,
            )

        assert call_count["n"] == 2
        assert llm_error is None
        assert "mistral" in triage[0]["source"]

    def test_ollama_total_failure_falls_back_to_heuristic(self):
        """If tier-1 itself fails, returns heuristic fallback."""
        svc = _make_ollama_svc()
        pred = {"predicted_label": "password", "confidence": 0.90}
        record = {"proto": "tcp"}

        with patch.object(svc, "_ollama_chat", side_effect=TimeoutError("connection refused")):
            triage, llm_error = svc.triage_predictions(
                predictions=[pred], records=[record], context=None,
            )

        assert triage[0]["source"] == "fallback"
        assert llm_error is not None

    def test_no_escalation_when_tier2_empty(self):
        """If tier-2 model name is empty, no escalation even for low confidence."""
        svc = _make_ollama_svc(ollama_model_tier2="")
        pred = {"predicted_label": "backdoor", "confidence": 0.40}
        record = {"proto": "tcp"}

        with patch.object(svc, "_ollama_chat", return_value=MOCK_OLLAMA_RESPONSE) as mock_chat:
            triage, _ = svc.triage_predictions(
                predictions=[pred], records=[record], context=None,
            )
            assert mock_chat.call_count == 1

    def test_prompt_builder_produces_valid_json(self):
        """Verify that prompt builder produces valid json."""
        pred = {"predicted_label": "scanning", "confidence": 0.85}
        record = {"proto": "tcp", "duration": 100}
        context = {"source": "clawdbot"}
        prompt = TriageService._build_prompt(pred, record, context)
        parsed = json.loads(prompt)
        assert parsed["task"] == "SOC triage and MITRE enrichment"
        assert parsed["classifier_prediction"] == pred
        assert parsed["context"] == context

    def test_stix_context_injected_when_threat_cache_present(self):
        """When a threat_cache is provided, STIX techniques are injected into the prompt context."""
        fake_techniques = [
            {"id": "T1595", "name": "Active Scanning", "tactics": ["reconnaissance"], "platforms": ["Linux"]},
            {"id": "T1046", "name": "Network Service Discovery", "tactics": ["discovery"], "platforms": ["Linux"]},
        ]
        mock_cache = MagicMock()
        mock_cache.techniques_for_tactics.return_value = fake_techniques

        svc = _make_ollama_svc(threat_cache=mock_cache)
        pred = {"predicted_label": "scanning", "confidence": 0.95}
        record = {"proto": "tcp"}

        captured_prompts = []

        def mock_chat(*, model, prompt_text):
            """Return a mocked LLM chat response."""
            captured_prompts.append(prompt_text)
            return MOCK_OLLAMA_RESPONSE

        with patch.object(svc, "_ollama_chat", side_effect=mock_chat):
            svc.triage_predictions(predictions=[pred], records=[record], context=None)

        assert len(captured_prompts) == 1
        prompt_data = json.loads(captured_prompts[0])
        assert "mitre_stix_techniques" in prompt_data["context"]
        assert len(prompt_data["context"]["mitre_stix_techniques"]) == 2
        assert prompt_data["context"]["mitre_stix_techniques"][0]["id"] == "T1595"
        mock_cache.techniques_for_tactics.assert_called_once_with(["Reconnaissance"], limit=15)

    def test_no_stix_context_when_threat_cache_none(self):
        """Without a threat_cache, context is not enriched with STIX data."""
        svc = _make_ollama_svc(threat_cache=None)
        pred = {"predicted_label": "scanning", "confidence": 0.95}
        record = {"proto": "tcp"}

        captured_prompts = []

        def mock_chat(*, model, prompt_text):
            """Return a mocked LLM chat response."""
            captured_prompts.append(prompt_text)
            return MOCK_OLLAMA_RESPONSE

        with patch.object(svc, "_ollama_chat", side_effect=mock_chat):
            svc.triage_predictions(predictions=[pred], records=[record], context=None)

        prompt_data = json.loads(captured_prompts[0])
        assert "mitre_stix_techniques" not in prompt_data.get("context", {})
