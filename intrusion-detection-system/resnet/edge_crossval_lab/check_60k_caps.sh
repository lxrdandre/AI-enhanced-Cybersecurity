#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$LAB_DIR"

# shellcheck source=lib.sh
source "${LAB_DIR}/lib.sh"

EXPECTED_CAP=60000
CHECK_EDGE="${CHECK_EDGE:-0}"
status=0

ok() {
  echo "OK: $*"
}

warn() {
  echo "WARN: $*"
}

fail() {
  echo "FAIL: $*"
  status=1
}

require_line() {
  local file="$1"
  local text="$2"
  local label="$3"
  if grep -Fq -- "$text" "$file"; then
    ok "$label"
  else
    fail "$label is missing from $file"
  fi
}

effective_target_per_class() {
  local raw="${TARGET_PER_CLASS:-$EXPECTED_CAP}"
  if ! [[ "$raw" =~ ^[0-9]+$ ]]; then
    fail "TARGET_PER_CLASS must be numeric; current value is '$raw'" >&2
    echo "0"
    return
  fi

  local value=$((10#$raw))
  if (( value > EXPECTED_CAP )); then
    warn "TARGET_PER_CLASS=$value is above $EXPECTED_CAP; build_dataset_from_pcaps.sh will force it to $EXPECTED_CAP" >&2
    echo "$EXPECTED_CAP"
  elif (( value < EXPECTED_CAP )); then
    warn "TARGET_PER_CLASS=$value is below $EXPECTED_CAP; build_dataset_from_pcaps.sh will force it to $EXPECTED_CAP" >&2
    echo "$EXPECTED_CAP"
  else
    echo "$value"
  fi
}

echo "=== 60k Cap Diagnostic ==="
echo "LAB_DIR=$LAB_DIR"
echo "Config file=$([[ -f "${LAB_DIR}/config.env" ]] && echo "${LAB_DIR}/config.env" || echo "${LAB_DIR}/config.env.example")"
echo "PCAP_PARSER=${PCAP_PARSER:-unset}"
echo "BUILD_DISTRIBUTION=${BUILD_DISTRIBUTION:-unset}"
echo "LIMIT_PER_PCAP=${LIMIT_PER_PCAP:-unset}"
echo "TARGET_PER_CLASS(raw)=${TARGET_PER_CLASS:-unset}"
echo "CHECK_EDGE=$CHECK_EDGE"

effective_target="$(effective_target_per_class)"
echo "TARGET_PER_CLASS(effective)=$effective_target"
if (( effective_target != EXPECTED_CAP )); then
  fail "effective TARGET_PER_CLASS is $effective_target, expected $EXPECTED_CAP. Fix config.env or run: TARGET_PER_CLASS=$EXPECTED_CAP bash build_dataset_from_pcaps.sh"
else
  ok "wrapper effective target is $EXPECTED_CAP"
fi

if [[ "${PCAP_PARSER:-zeek}" != "zeek" && "${BUILD_DISTRIBUTION:-balanced}" != "balanced" ]]; then
  fail "non-Zeek edge_like distribution will not make backdoor 60k; set BUILD_DISTRIBUTION=balanced or use PCAP_PARSER=zeek"
fi

if [[ "${LIMIT_PER_PCAP:-}" =~ ^[0-9]+$ ]] && (( 10#${LIMIT_PER_PCAP} < EXPECTED_CAP )); then
  warn "LIMIT_PER_PCAP=${LIMIT_PER_PCAP} is per source PCAP, not per class. One PCAP alone cannot produce 60k extracted rows with this limit."
fi

echo
echo "=== Script Checks ==="
require_line build_dataset_from_pcaps.sh "MAX_TARGET_PER_CLASS=60000" "wrapper hard cap is present"
require_line build_dataset_from_pcaps.sh "--cap-per-class \"\$TARGET_PER_CLASS\"" "Zeek builder receives TARGET_PER_CLASS"
require_line build_zeek_crossval_dataset.py "MAX_CAP_PER_CLASS = 60000" "Zeek Python builder max is 60k"
if [[ "$CHECK_EDGE" == "1" ]]; then
  require_line build_dataset_from_pcaps.sh "--cap-per-major-class \"\$TARGET_PER_CLASS\"" "Edge-style builder receives TARGET_PER_CLASS"
  require_line build_edge_crossval_dataset.py "MAX_CAP_PER_CLASS = 60000" "Edge-style Python builder max is 60k"
fi

PYTHONPATH="$LAB_DIR" python3 - <<'PY' || status=1
import contextlib
import io

import build_zeek_crossval_dataset as zeek

assert zeek.MAX_CAP_PER_CLASS == 60000, zeek.MAX_CAP_PER_CLASS
with contextlib.redirect_stdout(io.StringIO()):
    assert zeek.normalize_cap(100000, "--test") == 60000
    assert zeek.normalize_cap(10000, "--test") == 60000
print("OK: Zeek Python builder clamp function forces 60k")
PY

if [[ "$CHECK_EDGE" == "1" ]]; then
  PYTHONPATH="$LAB_DIR" python3 - <<'PY' || status=1
import contextlib
import io

import build_edge_crossval_dataset as edge

assert edge.MAX_CAP_PER_CLASS == 60000, edge.MAX_CAP_PER_CLASS
with contextlib.redirect_stdout(io.StringIO()):
    assert edge.normalize_cap(100000, "--test") == 60000
print("OK: Edge-style Python builder clamp function caps oversized values at 60k")
PY
fi

check_report() {
  local report="$1"
  local label="$2"
  [[ -f "$report" ]] || {
    warn "$label report missing: $report"
    return
  }

  python3 - "$report" "$EXPECTED_CAP" "$label" <<'PY' || status=1
import json
import sys
from pathlib import Path

report = Path(sys.argv[1])
expected = int(sys.argv[2])
label = sys.argv[3]
data = json.loads(report.read_text(encoding="utf-8"))

raw = data.get("raw_class_counts", {})
sampled = data.get("sampled_class_counts", {})
shortfalls = data.get("shortfalls", {})

if "cap_per_class" in data:
    caps = {cls: int(data["cap_per_class"]) for cls in sampled}
else:
    caps = {cls: int(value) for cls, value in data.get("target_caps", {}).items()}

print(f"{label} report={report}")
print(f"{label} caps={caps}")
print(f"{label} raw_class_counts={raw}")
print(f"{label} sampled_class_counts={sampled}")
if shortfalls:
    print(f"{label} shortfalls={shortfalls}")

bad_caps = {cls: cap for cls, cap in caps.items() if cap != expected}
if bad_caps:
    print(f"FAIL: {label} report target caps are not {expected}: {bad_caps}")
    raise SystemExit(1)

short = {cls: expected - int(count) for cls, count in sampled.items() if int(count) < expected}
if short:
    print(f"WARN: {label} output is below {expected} for: {short}")
    enough_raw = {cls: int(raw.get(cls, 0)) for cls in short if int(raw.get(cls, 0)) >= expected}
    if enough_raw:
        print(f"FAIL: {label} had enough raw rows but sampled below {expected}: {enough_raw}")
        raise SystemExit(1)
    print(f"WARN: {label} did not have enough extracted rows to fill 60k for those classes")
else:
    print(f"OK: {label} sampled counts are at least {expected} for every reported class")
PY
}

check_csv() {
  local csv="$1"
  local label="$2"
  [[ -f "$csv" ]] || {
    warn "$label CSV missing: $csv"
    return
  }

  python3 - "$csv" "$EXPECTED_CAP" "$label" <<'PY' || status=1
import csv
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
label = sys.argv[3]
counts = Counter()
with path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    if not reader.fieldnames or "type" not in reader.fieldnames:
        print(f"FAIL: {label} CSV has no type column: {path}")
        raise SystemExit(1)
    for row in reader:
        counts[str(row.get("type", "")).strip()] += 1

target_classes = ["backdoor", "dos_ddos", "injection", "normal", "password", "scanning"]
class_counts = {cls: int(counts[cls]) for cls in target_classes}
print(f"{label} CSV={path}")
print(f"{label} CSV class_counts={class_counts}")
short = {cls: expected - count for cls, count in class_counts.items() if count and count < expected}
if short:
    print(f"WARN: {label} CSV has classes below {expected}: {short}")
PY
}

echo
echo "=== Existing Output Checks ==="
check_report "${EDGE_CROSSVAL_ROOT}/zeek_crossval_report.json" "Zeek"
check_csv "${EDGE_CROSSVAL_ROOT}/zeek_crossval.csv" "Zeek"
if [[ "$CHECK_EDGE" == "1" ]]; then
  check_report "${EDGE_CROSSVAL_ROOT}/edge_like_crossval_report.json" "Edge-style"
  check_csv "${EDGE_CROSSVAL_ROOT}/edge_like_crossval.csv" "Edge-style"
fi

echo
if (( status == 0 )); then
  echo "RESULT: 60k cap configuration looks correct."
else
  echo "RESULT: 60k cap configuration has failures. Fix the FAIL lines above."
fi
exit "$status"
