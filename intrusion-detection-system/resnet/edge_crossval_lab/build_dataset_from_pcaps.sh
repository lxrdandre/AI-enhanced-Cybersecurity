#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${LAB_DIR}/lib.sh"

MAX_TARGET_PER_CLASS=60000
TARGET_PER_CLASS="${TARGET_PER_CLASS:-$MAX_TARGET_PER_CLASS}"
LIMIT_PER_PCAP="${LIMIT_PER_PCAP:-30000}"
BUILD_DISTRIBUTION="${BUILD_DISTRIBUTION:-balanced}"
OVERWRITE_CSVS="${OVERWRITE_CSVS:-1}"
PCAP_PARSER="${PCAP_PARSER:-zeek}"
BUILD_WINDOWS="${BUILD_WINDOWS:-0}"
WINDOW_SECONDS="${WINDOW_SECONDS:-5}"
FLOW_CONTEXT_WINDOWS="${FLOW_CONTEXT_WINDOWS:-5,15,60}"
ZEEK_BIN="${ZEEK_BIN:-zeek}"
SSH_PORT="${SSH_PORT:-22}"
CANONICAL_SSH_PORT="${CANONICAL_SSH_PORT:-22}"
if [[ "$PCAP_PARSER" == "zeek" ]]; then
  DEFAULT_OUTPUT_CSV="${EDGE_CROSSVAL_ROOT}/zeek_crossval.csv"
  DEFAULT_REPORT_JSON="${EDGE_CROSSVAL_ROOT}/zeek_crossval_report.json"
  DEFAULT_COUNTS_JSON="${EDGE_CROSSVAL_ROOT}/zeek_crossval_counts.json"
  DEFAULT_CSV_DIR="${EDGE_CROSSVAL_ROOT}/zeek_csv"
  DEFAULT_WINDOW_OUTPUT_CSV="${EDGE_CROSSVAL_ROOT}/zeek_crossval_windows_5s.csv"
  DEFAULT_WINDOW_REPORT_JSON="${EDGE_CROSSVAL_ROOT}/zeek_crossval_windows_5s_report.json"
else
  DEFAULT_OUTPUT_CSV="${EDGE_CROSSVAL_ROOT}/edge_like_crossval.csv"
  DEFAULT_REPORT_JSON="${EDGE_CROSSVAL_ROOT}/edge_like_crossval_report.json"
  DEFAULT_COUNTS_JSON="${EDGE_CROSSVAL_ROOT}/edge_like_crossval_counts.json"
  DEFAULT_CSV_DIR="${EDGE_CSV_DIR}"
  DEFAULT_WINDOW_OUTPUT_CSV=""
  DEFAULT_WINDOW_REPORT_JSON=""
fi
OUTPUT_CSV="${OUTPUT_CSV:-${DEFAULT_OUTPUT_CSV}}"
REPORT_JSON="${REPORT_JSON:-${DEFAULT_REPORT_JSON}}"
COUNTS_JSON="${COUNTS_JSON:-${DEFAULT_COUNTS_JSON}}"
BUILD_CSV_DIR="${BUILD_CSV_DIR:-${DEFAULT_CSV_DIR}}"
WINDOW_OUTPUT_CSV="${WINDOW_OUTPUT_CSV:-${DEFAULT_WINDOW_OUTPUT_CSV}}"
WINDOW_REPORT_JSON="${WINDOW_REPORT_JSON:-${DEFAULT_WINDOW_REPORT_JSON}}"

require_cmd python3
if [[ "$PCAP_PARSER" == "zeek" ]]; then
  require_cmd "$ZEEK_BIN"
else
  require_cmd tshark
fi
ensure_dirs
mkdir -p "$BUILD_CSV_DIR"

if ! python3 -c "import pandas" >/dev/null 2>&1; then
  echo "Missing Python package: pandas" >&2
  echo "Install it first, for example: python3 -m pip install --user pandas" >&2
  exit 1
fi

if ! ls "${EDGE_RAW_DIR}"/*.meta.json >/dev/null 2>&1; then
  echo "No capture meta files found in ${EDGE_RAW_DIR}" >&2
  echo "Run the capture schedule first." >&2
  exit 2
fi

if ! [[ "$TARGET_PER_CLASS" =~ ^[0-9]+$ ]]; then
  echo "TARGET_PER_CLASS must be a positive integer, got: ${TARGET_PER_CLASS}" >&2
  exit 2
fi
TARGET_PER_CLASS_VALUE=$((10#$TARGET_PER_CLASS))
if (( TARGET_PER_CLASS_VALUE < 1 )); then
  echo "TARGET_PER_CLASS must be at least 1, got: ${TARGET_PER_CLASS}" >&2
  exit 2
fi
if (( TARGET_PER_CLASS_VALUE != MAX_TARGET_PER_CLASS )); then
  echo "TARGET_PER_CLASS=${TARGET_PER_CLASS} differs from the required ${MAX_TARGET_PER_CLASS}; using ${MAX_TARGET_PER_CLASS}." >&2
fi
TARGET_PER_CLASS="$MAX_TARGET_PER_CLASS"

echo "=== Edge Crossval Dataset Build From PCAPs ==="
echo "RAW_DIR=${EDGE_RAW_DIR}"
echo "PCAP_PARSER=${PCAP_PARSER}"
echo "CSV_DIR=${BUILD_CSV_DIR}"
echo "TARGET_PER_CLASS=${TARGET_PER_CLASS}"
echo "MAX_TARGET_PER_CLASS=${MAX_TARGET_PER_CLASS}"
echo "LIMIT_PER_PCAP=${LIMIT_PER_PCAP}"
echo "BUILD_DISTRIBUTION=${BUILD_DISTRIBUTION}"
echo "BUILD_WINDOWS=${BUILD_WINDOWS}"
echo "WINDOW_SECONDS=${WINDOW_SECONDS}"
echo "FLOW_CONTEXT_WINDOWS=${FLOW_CONTEXT_WINDOWS}"
echo "SSH_PORT=${SSH_PORT}"
echo "CANONICAL_SSH_PORT=${CANONICAL_SSH_PORT}"
echo "OUTPUT_CSV=${OUTPUT_CSV}"
echo "REPORT_JSON=${REPORT_JSON}"
if [[ "$PCAP_PARSER" == "zeek" && "$BUILD_WINDOWS" == "1" ]]; then
  echo "WINDOW_OUTPUT_CSV=${WINDOW_OUTPUT_CSV}"
  echo "WINDOW_REPORT_JSON=${WINDOW_REPORT_JSON}"
fi
echo

if [[ "$PCAP_PARSER" == "zeek" ]]; then
  EXTRACT_ARGS=(
    python3 "${LAB_DIR}/extract_all_pcaps_zeek.py"
    --raw-dir "$EDGE_RAW_DIR"
    --csv-dir "$BUILD_CSV_DIR"
    --zeek "$ZEEK_BIN"
    --target-ip "$TARGET_IP"
    --kali-ip "$KALI_IP"
    --ssh-port "$SSH_PORT"
    --canonical-ssh-port "$CANONICAL_SSH_PORT"
    --limit-per-pcap "$LIMIT_PER_PCAP"
  )
else
  EXTRACT_ARGS=(
    python3 "${LAB_DIR}/extract_all_pcaps.py"
    --raw-dir "$EDGE_RAW_DIR"
    --csv-dir "$BUILD_CSV_DIR"
    --limit-per-pcap "$LIMIT_PER_PCAP"
  )
fi
if [[ "$OVERWRITE_CSVS" == "1" ]]; then
  EXTRACT_ARGS+=(--overwrite)
fi

echo "Extracting PCAPs to labelled ${PCAP_PARSER} CSVs..."
"${EXTRACT_ARGS[@]}"

echo
echo "Counting extracted rows..."
python3 "${LAB_DIR}/count_edge_csv_rows.py" --input-dir "$BUILD_CSV_DIR" --json | tee "$COUNTS_JSON"

if [[ "$PCAP_PARSER" == "zeek" ]]; then
  BUILD_ARGS=(
    python3 "${LAB_DIR}/build_zeek_crossval_dataset.py"
    --input-dir "$BUILD_CSV_DIR"
    --output-csv "$OUTPUT_CSV"
    --report-json "$REPORT_JSON"
    --cap-per-class "$TARGET_PER_CLASS"
    --quota-mode even
    --context-windows "$FLOW_CONTEXT_WINDOWS"
  )
else
  BUILD_ARGS=(
    python3 "${LAB_DIR}/build_edge_crossval_dataset.py"
    --input-dir "$BUILD_CSV_DIR"
    --output-csv "$OUTPUT_CSV"
    --report-json "$REPORT_JSON"
    --distribution "$BUILD_DISTRIBUTION"
    --cap-per-major-class "$TARGET_PER_CLASS"
    --quota-mode even
  )
  if [[ "$BUILD_DISTRIBUTION" == "balanced" ]]; then
    BUILD_ARGS+=(--backdoor-cap "$TARGET_PER_CLASS")
  fi
fi

echo
echo "Building final capped dataset..."
"${BUILD_ARGS[@]}"

if [[ "$PCAP_PARSER" == "zeek" && "$BUILD_WINDOWS" == "1" ]]; then
  echo
  echo "Building 5-second Zeek window dataset..."
  python3 "${LAB_DIR}/build_zeek_window_dataset.py" \
    --input-csv "$OUTPUT_CSV" \
    --output-csv "$WINDOW_OUTPUT_CSV" \
    --report-json "$WINDOW_REPORT_JSON" \
    --window-seconds "$WINDOW_SECONDS"
fi

echo
echo "DONE"
echo "CSV=${OUTPUT_CSV}"
echo "REPORT=${REPORT_JSON}"
echo "COUNTS=${COUNTS_JSON}"
if [[ "$PCAP_PARSER" == "zeek" && "$BUILD_WINDOWS" == "1" ]]; then
  echo "WINDOW_CSV=${WINDOW_OUTPUT_CSV}"
  echo "WINDOW_REPORT=${WINDOW_REPORT_JSON}"
fi
