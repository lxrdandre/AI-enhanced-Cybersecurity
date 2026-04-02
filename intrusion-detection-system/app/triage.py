from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request

log = logging.getLogger(__name__)


DEFAULT_MITRE_MAP = {
    "backdoor": {
        "tactics": ["Persistence", "Command and Control"],
        "techniques": [
            {"id": "T1053", "name": "Scheduled Task/Job"},
            {"id": "T1071", "name": "Application Layer Protocol"},
        ],
    },
    "ddos_dos": {
        "tactics": ["Impact"],
        "techniques": [
            {"id": "T1498", "name": "Network Denial of Service"},
        ],
    },
    "injection": {
        "tactics": ["Initial Access", "Execution"],
        "techniques": [
            {"id": "T1190", "name": "Exploit Public-Facing Application"},
        ],
    },
    "password": {
        "tactics": ["Credential Access"],
        "techniques": [
            {"id": "T1110", "name": "Brute Force"},
        ],
    },
    "scanning": {
        "tactics": ["Reconnaissance"],
        "techniques": [
            {"id": "T1595", "name": "Active Scanning"},
        ],
    },
    "xss": {
        "tactics": ["Initial Access", "Execution"],
        "techniques": [
            {"id": "T1189", "name": "Drive-by Compromise"},
        ],
    },
    "normal": {
        "tactics": [],
        "techniques": [],
    },
    "unknown": {
        "tactics": ["Unknown"],
        "techniques": [
            {"id": "T0000", "name": "Unclassified — model confidence below threshold"},
        ],
    },
}


def _severity_for_label(label: str, confidence: float) -> str:
    if label == "normal":
        return "low"
    if label == "unknown":
        return "review"
    if confidence >= 0.9:
        return "high"
    if confidence >= 0.7:
        return "medium"
    return "review"


def _default_triage_item(prediction: dict) -> dict:
    label = str(prediction.get("predicted_label", "unknown"))
    confidence = float(prediction.get("confidence", 0.0))
    mitre = DEFAULT_MITRE_MAP.get(label, {"tactics": [], "techniques": []})

    return {
        "label": label,
        "severity": _severity_for_label(label, confidence),
        "mitre_tactics": list(mitre["tactics"]),
        "mitre_techniques": [
            {
                "id": str(t["id"]),
                "name": str(t["name"]),
                "confidence": "medium",
                "reason": f"Heuristic mapping for class '{label}'.",
            }
            for t in mitre["techniques"]
        ],
        "summary": f"Classifier flagged '{label}' with confidence {confidence:.3f}.",
        "next_actions": [
            "Review correlated logs for same source/destination context.",
            "Validate whether this pattern matches expected baseline behavior.",
        ],
        "confidence_note": (
            "Model confidence is high." if confidence >= 0.8 else "Model confidence is moderate/low; review recommended."
        ),
        "source": "fallback",
    }


def _extract_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])

    return json.loads(text)


class TriageService:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        timeout_seconds: int,
        ollama_base_url: str = "http://127.0.0.1:11434",
        ollama_model_tier1: str = "mistral-small:24b",
        ollama_model_tier2: str = "llama3.1:70b-instruct-q8_0",
        ollama_escalation_confidence: float = 0.75,
        triage_backend: str = "ollama",
    ):
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model_tier1 = ollama_model_tier1
        self.ollama_model_tier2 = ollama_model_tier2
        self.ollama_escalation_confidence = ollama_escalation_confidence
        self.triage_backend = triage_backend  # "ollama" | "gemini" | "fallback"

    @property
    def enabled(self) -> bool:
        if self.triage_backend == "ollama":
            return True
        if self.triage_backend == "gemini":
            return bool(self.api_key)
        return False

    # ── Shared prompt builder ────────────────────────────────

    @staticmethod
    def _build_prompt(prediction: dict, record: dict, context: dict | None) -> str:
        prompt = {
            "task": "SOC triage and MITRE enrichment",
            "constraints": [
                "Do not relabel classifier output.",
                "Return JSON only.",
                "If uncertain, set severity to review.",
            ],
            "output_schema": {
                "label": "string",
                "severity": "low|medium|high|critical|review",
                "mitre_tactics": ["string"],
                "mitre_techniques": [
                    {
                        "id": "Txxxx",
                        "name": "string",
                        "confidence": "high|medium|low",
                        "reason": "string",
                    }
                ],
                "summary": "string",
                "next_actions": ["string"],
                "confidence_note": "string",
            },
            "classifier_prediction": prediction,
            "record": record,
            "context": context or {},
        }
        return json.dumps(prompt, ensure_ascii=False)

    # ── Ollama backend ───────────────────────────────────────

    def _ollama_chat(self, *, model: str, prompt_text: str) -> str:
        url = f"{self.ollama_base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a SOC analyst assistant. Respond ONLY with valid JSON "
                        "matching the requested output_schema. No markdown fences, no commentary."
                    ),
                },
                {"role": "user", "content": prompt_text},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        req = urllib.request.Request(
            url=url,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        return parsed.get("message", {}).get("content", "")

    def _ollama_triage(
        self,
        *,
        prediction: dict,
        record: dict,
        context: dict | None,
    ) -> dict:
        prompt_text = self._build_prompt(prediction, record, context)
        confidence = float(prediction.get("confidence", 0.0))
        label = str(prediction.get("predicted_label", "unknown"))
        severity = _severity_for_label(label, confidence)

        # Tier-1: fast model
        text = self._ollama_chat(model=self.ollama_model_tier1, prompt_text=prompt_text)
        triage = _extract_json(text)
        source = f"ollama:{self.ollama_model_tier1}"

        # Escalate to Tier-2 when conditions are met
        needs_escalation = (
            confidence < self.ollama_escalation_confidence
            or severity in ("review", "critical")
        )
        if needs_escalation and self.ollama_model_tier2:
            log.info(
                "Escalating to tier-2 (%s): confidence=%.3f severity=%s label=%s",
                self.ollama_model_tier2, confidence, severity, label,
            )
            try:
                text_t2 = self._ollama_chat(model=self.ollama_model_tier2, prompt_text=prompt_text)
                triage = _extract_json(text_t2)
                source = f"ollama:{self.ollama_model_tier2}"
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                log.warning("Tier-2 escalation failed, keeping tier-1 result: %s", exc)

        triage.setdefault("label", prediction.get("predicted_label", "unknown"))
        triage.setdefault("source", source)
        triage.setdefault("mitre_tactics", [])
        triage.setdefault("mitre_techniques", [])
        triage.setdefault("next_actions", [])
        triage.setdefault("confidence_note", "")
        triage.setdefault("summary", "")
        triage.setdefault("severity", "review")

        return triage

    # ── Gemini backend (unchanged) ───────────────────────────

    def _gemini_triage(self, *, prediction: dict, record: dict, context: dict | None) -> dict:
        prompt_text = self._build_prompt(prediction, record, context)

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt_text,
                        }
                    ]
                }
            ]
        }

        req = urllib.request.Request(
            url=url,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)

        text = (
            parsed.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        triage = _extract_json(text)

        triage.setdefault("label", prediction.get("predicted_label", "unknown"))
        triage.setdefault("source", "gemini")
        triage.setdefault("mitre_tactics", [])
        triage.setdefault("mitre_techniques", [])
        triage.setdefault("next_actions", [])
        triage.setdefault("confidence_note", "")
        triage.setdefault("summary", "")
        triage.setdefault("severity", "review")

        return triage

    def triage_predictions(self, *, predictions: list[dict], records: list[dict], context: dict | None) -> tuple[list[dict], str | None]:
        triage_results: list[dict] = []
        llm_error: str | None = None

        for idx, prediction in enumerate(predictions):
            record = records[idx] if idx < len(records) else {}
            label = str(prediction.get("predicted_label", "unknown"))

            # Skip LLM for benign and unknown traffic
            if label in ("normal", "unknown"):
                triage_results.append(_default_triage_item(prediction))
                continue

            if self.triage_backend == "ollama":
                try:
                    triage_item = self._ollama_triage(prediction=prediction, record=record, context=context)
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                    llm_error = str(exc)
                    log.warning("Ollama triage failed, using fallback: %s", exc)
                    triage_item = _default_triage_item(prediction)
            elif self.triage_backend == "gemini" and self.api_key:
                try:
                    triage_item = self._gemini_triage(prediction=prediction, record=record, context=context)
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                    llm_error = str(exc)
                    triage_item = _default_triage_item(prediction)
            else:
                triage_item = _default_triage_item(prediction)

            triage_results.append(triage_item)

        return triage_results, llm_error
