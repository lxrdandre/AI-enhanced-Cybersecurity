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
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
import urllib.error
import urllib.request

REQUIRED_FIELDS = {
    "label", "severity", "mitre_tactics", "mitre_techniques",
    "summary", "next_actions", "confidence_note",
}
VALID_SEVERITIES = {"low", "medium", "high", "critical", "review"}
SEVERITY_ORDER = ["review", "low", "medium", "high", "critical"]
SEVERITY_INDEX = {severity: index for index, severity in enumerate(SEVERITY_ORDER)}
SEVERITY_CONFUSION_LABELS = SEVERITY_ORDER + ["invalid"]
DEFAULT_RESULTS_DIR = Path("results") / "lora_evaluation"


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
            "severity_adjacent": False,
            "technique_overlap": 0.0,
        }

    fields = sum(1 for f in REQUIRED_FIELDS if f in parsed)
    predicted_severity = parsed.get("severity")
    gold_severity = gold.get("severity")
    severity_match = predicted_severity == gold_severity
    severity_valid = predicted_severity in VALID_SEVERITIES
    severity_adjacent = False
    if severity_valid and gold_severity in VALID_SEVERITIES:
        severity_adjacent = abs(SEVERITY_INDEX[predicted_severity] - SEVERITY_INDEX[gold_severity]) <= 1

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
        "severity_adjacent": severity_adjacent,
        "technique_overlap": overlap,
    }


def _safe_name(value: str) -> str:
    """Return a filesystem-safe name."""
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip().lower())
    return cleaned.strip("_") or "model"


def _aggregate(scores: list[dict]) -> dict[str, float]:
    """Aggregate evaluation rows by model and test set."""
    n = len(scores)
    if n == 0:
        return {
            "json_rate": 0.0,
            "avg_fields": 0.0,
            "sev_match": 0.0,
            "sev_valid": 0.0,
            "sev_adjacent": 0.0,
            "tech_overlap": 0.0,
        }
    return {
        "json_rate": sum(1 for s in scores if s["json_valid"]) / n,
        "avg_fields": sum(s["fields"] for s in scores) / n,
        "sev_match": sum(1 for s in scores if s["severity_match"]) / n,
        "sev_valid": sum(1 for s in scores if s["severity_valid"]) / n,
        "sev_adjacent": sum(1 for s in scores if s["severity_adjacent"]) / n,
        "tech_overlap": sum(s["technique_overlap"] for s in scores) / n,
    }


def _severity_bucket(value: object) -> str:
    """Map severity text into a stable report bucket."""
    severity = str(value or "").strip().lower()
    return severity if severity in VALID_SEVERITIES else "invalid"


def _build_severity_confusion(per_example_rows: list[dict], model_key: str) -> dict[str, object]:
    """Build severity confusion."""
    matrix = {
        gold: {pred: 0 for pred in SEVERITY_CONFUSION_LABELS}
        for gold in SEVERITY_CONFUSION_LABELS
    }
    for row in per_example_rows:
        gold = _severity_bucket((row.get("gold") or {}).get("severity"))
        parsed = ((row.get(model_key) or {}).get("parsed") or {})
        pred = _severity_bucket(parsed.get("severity"))
        matrix[gold][pred] += 1
    return {"labels": list(SEVERITY_CONFUSION_LABELS), "matrix": matrix}


def _format_confusion_text(title: str, confusion: dict[str, object]) -> list[str]:
    """Format confusion text for display or logging."""
    labels = confusion["labels"]
    matrix = confusion["matrix"]
    col_width = 8
    lines = [title]
    header = "gold\\pred".ljust(12) + "".join(label[:col_width].rjust(col_width) for label in labels)
    lines.append(header)
    for gold in labels:
        row = gold[:12].ljust(12) + "".join(str(matrix[gold][pred]).rjust(col_width) for pred in labels)
        lines.append(row)
    return lines


def _report_text(
    *,
    base_model: str,
    tuned_model: str,
    examples_count: int,
    agg: dict[str, dict[str, float]],
    base_confusion: dict[str, object],
    tuned_confusion: dict[str, object],
) -> str:
    """Build a human-readable evaluation report."""
    b, t = agg["base"], agg["tuned"]
    lines = [
        f"Examples: {examples_count}",
        f"Base model: {base_model}",
        f"Tuned model: {tuned_model}",
        "",
        "=" * 62,
        f"{'Metric':<32} {'Base':>12} {'Fine-tuned':>14}",
        "-" * 62,
        f"{'JSON parse rate':<32} {b['json_rate']:>11.1%} {t['json_rate']:>13.1%}",
        f"{'Avg fields present (of 7)':<32} {b['avg_fields']:>12.1f} {t['avg_fields']:>14.1f}",
        f"{'Severity accuracy':<32} {b['sev_match']:>11.1%} {t['sev_match']:>13.1%}",
        f"{'Adjacent severity accuracy':<32} {b['sev_adjacent']:>11.1%} {t['sev_adjacent']:>13.1%}",
        f"{'Severity valid':<32} {b['sev_valid']:>11.1%} {t['sev_valid']:>13.1%}",
        f"{'MITRE technique overlap':<32} {b['tech_overlap']:>11.1%} {t['tech_overlap']:>13.1%}",
        "=" * 62,
        "",
    ]
    lines.extend(_format_confusion_text("Base severity confusion matrix", base_confusion))
    lines.append("")
    lines.extend(_format_confusion_text("Fine-tuned severity confusion matrix", tuned_confusion))
    return "\n".join(lines)


def _write_results_bundle(
    *,
    output_dir: Path,
    base_model: str,
    tuned_model: str,
    test_data: str,
    ollama_url: str,
    timeout: int,
    limit: int,
    examples: list[dict],
    per_example_rows: list[dict],
    agg: dict[str, dict[str, float]],
    base_confusion: dict[str, object],
    tuned_confusion: dict[str, object],
) -> dict[str, str]:
    """Write results bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "test_data": test_data,
        "base_model": base_model,
        "tuned_model": tuned_model,
        "ollama_url": ollama_url,
        "timeout": timeout,
        "limit": limit,
        "examples_evaluated": len(examples),
        "aggregate": agg,
        "severity_confusion": {
            "base": base_confusion,
            "tuned": tuned_confusion,
        },
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    report_text = _report_text(
        base_model=base_model,
        tuned_model=tuned_model,
        examples_count=len(examples),
        agg=agg,
        base_confusion=base_confusion,
        tuned_confusion=tuned_confusion,
    )
    report_path = output_dir / "report.txt"
    report_path.write_text(report_text + "\n", encoding="utf-8")

    jsonl_path = output_dir / "per_example.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for row in per_example_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    csv_path = output_dir / "per_example.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "label",
                "gold_severity",
                "base_predicted_severity",
                "base_json_valid",
                "base_fields",
                "base_severity_match",
                "base_severity_valid",
                "base_severity_adjacent",
                "base_technique_overlap",
                "tuned_predicted_severity",
                "tuned_json_valid",
                "tuned_fields",
                "tuned_severity_match",
                "tuned_severity_valid",
                "tuned_severity_adjacent",
                "tuned_technique_overlap",
            ],
        )
        writer.writeheader()
        for row in per_example_rows:
            writer.writerow(
                {
                    "index": row["index"],
                    "label": row["label"],
                    "gold_severity": row["gold"]["severity"],
                    "base_predicted_severity": ((row["base"].get("parsed") or {}).get("severity")),
                    "base_json_valid": row["base"]["score"]["json_valid"],
                    "base_fields": row["base"]["score"]["fields"],
                    "base_severity_match": row["base"]["score"]["severity_match"],
                    "base_severity_valid": row["base"]["score"]["severity_valid"],
                    "base_severity_adjacent": row["base"]["score"]["severity_adjacent"],
                    "base_technique_overlap": row["base"]["score"]["technique_overlap"],
                    "tuned_predicted_severity": ((row["tuned"].get("parsed") or {}).get("severity")),
                    "tuned_json_valid": row["tuned"]["score"]["json_valid"],
                    "tuned_fields": row["tuned"]["score"]["fields"],
                    "tuned_severity_match": row["tuned"]["score"]["severity_match"],
                    "tuned_severity_valid": row["tuned"]["score"]["severity_valid"],
                    "tuned_severity_adjacent": row["tuned"]["score"]["severity_adjacent"],
                    "tuned_technique_overlap": row["tuned"]["score"]["technique_overlap"],
                }
            )

    confusion_paths = {}
    for key, confusion in (("base", base_confusion), ("tuned", tuned_confusion)):
        matrix_csv_path = output_dir / f"severity_confusion_{key}.csv"
        with open(matrix_csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            labels = confusion["labels"]
            writer.writerow(["gold\\pred", *labels])
            for gold in labels:
                writer.writerow([gold, *[confusion["matrix"][gold][pred] for pred in labels]])
        confusion_paths[f"severity_confusion_{key}_csv"] = str(matrix_csv_path)

    return {
        "output_dir": str(output_dir),
        "summary_json": str(summary_path),
        "report_txt": str(report_path),
        "per_example_jsonl": str(jsonl_path),
        "per_example_csv": str(csv_path),
        **confusion_paths,
    }


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned triage model")
    parser.add_argument("--test-data", required=True, help="Test JSONL file")
    parser.add_argument("--base-model", default="mistral-small:24b")
    parser.add_argument("--tuned-model", default="clawdbot-triage")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout (s)")
    parser.add_argument("--limit", type=int, default=0, help="Max examples (0 = all)")
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory where evaluation result folders are created.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name. Defaults to a timestamped name.",
    )
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

    if not examples:
        print("No evaluation examples found.", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(examples)} examples")
    print(f"  Base:  {args.base_model}")
    print(f"  Tuned: {args.tuned_model}")
    print()

    results: dict[str, list[dict]] = {"base": [], "tuned": []}
    per_example_rows: list[dict] = []

    for i, ex in enumerate(examples):
        system_msg = ex["messages"][0]["content"]
        user_msg = ex["messages"][1]["content"]
        gold = json.loads(ex["messages"][2]["content"])
        label = gold.get("label", "?")
        row = {
            "index": i,
            "label": label,
            "gold": gold,
            "base": {},
            "tuned": {},
        }

        print(f"  [{i + 1}/{len(examples)}] {label:<12}", end="", flush=True)

        for key, model_name in [("base", args.base_model), ("tuned", args.tuned_model)]:
            try:
                raw = _ollama_chat(model_name, system_msg, user_msg, args.ollama_url, args.timeout)
                parsed = _parse_json(raw)
                score = _score(parsed, gold)
            except Exception as exc:
                raw = ""
                parsed = None
                score = {
                    "json_valid": False, "fields": 0,
                    "severity_match": False, "severity_valid": False,
                    "severity_adjacent": False,
                    "technique_overlap": 0.0, "error": str(exc),
                }
            results[key].append(score)
            row[key] = {
                "model": model_name,
                "raw": raw,
                "parsed": parsed,
                "score": score,
            }

        b_ok = "\u2713" if results["base"][-1]["json_valid"] else "\u2717"
        t_ok = "\u2713" if results["tuned"][-1]["json_valid"] else "\u2717"
        print(f"  base={b_ok}  tuned={t_ok}")
        per_example_rows.append(row)

    # -- Aggregate & print comparison table -------------------------------
    print()
    agg = {key: _aggregate(results[key]) for key in ("base", "tuned")}
    base_confusion = _build_severity_confusion(per_example_rows, "base")
    tuned_confusion = _build_severity_confusion(per_example_rows, "tuned")
    report_text = _report_text(
        base_model=args.base_model,
        tuned_model=args.tuned_model,
        examples_count=len(examples),
        agg=agg,
        base_confusion=base_confusion,
        tuned_confusion=tuned_confusion,
    )
    print(report_text)

    run_name = args.run_name
    if not run_name:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
        run_name = f"{timestamp}_{_safe_name(args.base_model)}_vs_{_safe_name(args.tuned_model)}"
    output_dir = Path(args.results_dir) / run_name
    saved = _write_results_bundle(
        output_dir=output_dir,
        base_model=args.base_model,
        tuned_model=args.tuned_model,
        test_data=str(test_path),
        ollama_url=args.ollama_url,
        timeout=args.timeout,
        limit=args.limit,
        examples=examples,
        per_example_rows=per_example_rows,
        agg=agg,
        base_confusion=base_confusion,
        tuned_confusion=tuned_confusion,
    )
    print()
    print(f"Saved results to: {saved['output_dir']}")
    print(f"  Summary JSON:   {saved['summary_json']}")
    print(f"  Text report:    {saved['report_txt']}")
    print(f"  Per-example:    {saved['per_example_jsonl']}")
    print(f"  Per-example CSV:{saved['per_example_csv']}")
    print(f"  Base confusion: {saved['severity_confusion_base_csv']}")
    print(f"  Tuned confusion:{saved['severity_confusion_tuned_csv']}")


if __name__ == "__main__":
    main()
