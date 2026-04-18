"""Evaluate base vs fine-tuned triage model side-by-side.

Sends the same triage prompts to both models via Ollama and compares
JSON parse rate, severity accuracy, MITRE overlap, and field completeness.

Usage::

    python -m lora.evaluate \\
        --test-data data/triage_test.jsonl \\
        --base-model mistral-small:24b \\
        --tuned-model clawdbot-triage
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REQUIRED_FIELDS = {
    "label", "severity", "mitre_tactics", "mitre_techniques",
    "summary", "next_actions", "confidence_note",
}
VALID_SEVERITIES = {"low", "medium", "high", "critical", "review"}


def _ollama_chat(
    model: str, system: str, user: str,
    base_url: str = "http://127.0.0.1:11434", timeout: int = 120,
) -> str:
    """Send a chat request to Ollama and return the assistant text."""
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("message", {}).get("content", "")


def _parse_json(text: str) -> dict | None:
    """Best-effort JSON extraction from LLM output."""
    text = text.strip()
    # Raw JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Substring search
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _score(parsed: dict | None, gold: dict) -> dict:
    """Score one response against the gold standard."""
    if parsed is None:
        return {
            "json_valid": False,
            "fields": 0,
            "severity_match": False,
            "severity_valid": False,
            "technique_overlap": 0.0,
        }

    fields = sum(1 for f in REQUIRED_FIELDS if f in parsed)
    severity_match = parsed.get("severity") == gold.get("severity")
    severity_valid = parsed.get("severity") in VALID_SEVERITIES

    gold_ids = {t["id"] for t in gold.get("mitre_techniques", [])}
    pred_ids = set()
    for t in parsed.get("mitre_techniques", []):
        if isinstance(t, dict) and "id" in t:
            pred_ids.add(t["id"])
    overlap = len(gold_ids & pred_ids) / max(len(gold_ids), 1)

    return {
        "json_valid": True,
        "fields": fields,
        "severity_match": severity_match,
        "severity_valid": severity_valid,
        "technique_overlap": overlap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned triage model")
    parser.add_argument("--test-data", required=True, help="Test JSONL file")
    parser.add_argument("--base-model", default="mistral-small:24b")
    parser.add_argument("--tuned-model", default="clawdbot-triage")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout (s)")
    parser.add_argument("--limit", type=int, default=0, help="Max examples (0 = all)")
    args = parser.parse_args()

    test_path = Path(args.test_data)
    if not test_path.exists():
        print(f"Test data not found: {test_path}", file=sys.stderr)
        sys.exit(1)

    examples = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if args.limit > 0:
        examples = examples[: args.limit]

    print(f"Evaluating {len(examples)} examples")
    print(f"  Base:  {args.base_model}")
    print(f"  Tuned: {args.tuned_model}")
    print()

    results: dict[str, list[dict]] = {"base": [], "tuned": []}

    for i, ex in enumerate(examples):
        system_msg = ex["messages"][0]["content"]
        user_msg = ex["messages"][1]["content"]
        gold = json.loads(ex["messages"][2]["content"])
        label = gold.get("label", "?")

        print(f"  [{i + 1}/{len(examples)}] {label:<12}", end="", flush=True)

        for key, model_name in [("base", args.base_model), ("tuned", args.tuned_model)]:
            try:
                raw = _ollama_chat(model_name, system_msg, user_msg, args.ollama_url, args.timeout)
                parsed = _parse_json(raw)
                score = _score(parsed, gold)
            except Exception as exc:
                score = {
                    "json_valid": False, "fields": 0,
                    "severity_match": False, "severity_valid": False,
                    "technique_overlap": 0.0, "error": str(exc),
                }
            results[key].append(score)

        b_ok = "\u2713" if results["base"][-1]["json_valid"] else "\u2717"
        t_ok = "\u2713" if results["tuned"][-1]["json_valid"] else "\u2717"
        print(f"  base={b_ok}  tuned={t_ok}")

    # ── Aggregate & print comparison table ───────────────────────────────
    print()
    print("=" * 62)
    print(f"{'Metric':<32} {'Base':>12} {'Fine-tuned':>14}")
    print("-" * 62)

    agg = {}
    for key in ("base", "tuned"):
        r = results[key]
        n = len(r)
        agg[key] = {
            "json_rate": sum(1 for s in r if s["json_valid"]) / n,
            "avg_fields": sum(s["fields"] for s in r) / n,
            "sev_match": sum(1 for s in r if s["severity_match"]) / n,
            "sev_valid": sum(1 for s in r if s["severity_valid"]) / n,
            "tech_overlap": sum(s["technique_overlap"] for s in r) / n,
        }

    b, t = agg["base"], agg["tuned"]
    print(f"{'JSON parse rate':<32} {b['json_rate']:>11.1%} {t['json_rate']:>13.1%}")
    print(f"{'Avg fields present (of 7)':<32} {b['avg_fields']:>12.1f} {t['avg_fields']:>14.1f}")
    print(f"{'Severity accuracy':<32} {b['sev_match']:>11.1%} {t['sev_match']:>13.1%}")
    print(f"{'Severity valid':<32} {b['sev_valid']:>11.1%} {t['sev_valid']:>13.1%}")
    print(f"{'MITRE technique overlap':<32} {b['tech_overlap']:>11.1%} {t['tech_overlap']:>13.1%}")
    print("=" * 62)


if __name__ == "__main__":
    main()
