#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALLER_MAX_ROUNDS="${MAX_ROUNDS-}"
CALLER_START_ROUND="${START_ROUND-}"
CALLER_CAPTURE_CLASSES="${CAPTURE_CLASSES-}"
CALLER_WINDOW_PAD_SECONDS="${WINDOW_PAD_SECONDS-}"
CALLER_START_DELAY_SECONDS="${START_DELAY_SECONDS-}"
CALLER_SCHEDULE_START_EPOCH="${SCHEDULE_START_EPOCH-}"
CALLER_SKIP_EXISTING_PCAPS="${SKIP_EXISTING_PCAPS-}"

# shellcheck source=lib.sh
source "${LAB_DIR}/lib.sh"
# shellcheck source=schedule_lib.sh
source "${LAB_DIR}/schedule_lib.sh"

MAX_ROUNDS="${CALLER_MAX_ROUNDS:-14}"
START_ROUND="${CALLER_START_ROUND:-${START_ROUND:-0}}"
CAPTURE_CLASSES="${CALLER_CAPTURE_CLASSES:-${CAPTURE_CLASSES:-normal dos_ddos injection password scanning backdoor}}"
WINDOW_PAD_SECONDS="${CALLER_WINDOW_PAD_SECONDS:-${WINDOW_PAD_SECONDS:-20}}"
START_DELAY_SECONDS="${CALLER_START_DELAY_SECONDS:-${START_DELAY_SECONDS:-120}}"
SCHEDULE_START_EPOCH="${CALLER_SCHEDULE_START_EPOCH:-${SCHEDULE_START_EPOCH:-$(( $(date +%s) + START_DELAY_SECONDS ))}}"
SKIP_EXISTING_PCAPS="${CALLER_SKIP_EXISTING_PCAPS:-${SKIP_EXISTING_PCAPS:-1}}"

require_lab_target "$TARGET_IP"
require_cmd tcpdump
require_cmd timeout
ensure_dirs

LOG_DIR="${EDGE_CROSSVAL_ROOT}/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="${LOG_DIR}/debian_capture_schedule_$(date +%Y%m%d_%H%M%S).log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$RUN_LOG"
}

require_uint() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer, got: ${value}" >&2
    exit 2
  fi
}

write_meta() {
  local source_label="$1"
  local type_label="$2"
  local duration="$3"
  local pcap="$4"
  local meta="${EDGE_RAW_DIR}/${source_label}.meta.json"
  cat > "$meta" <<EOF
{
  "source_label": "${source_label}",
  "type": "${type_label}",
  "duration_seconds": ${duration},
  "target_ip": "${TARGET_IP}",
  "kali_ip": "${KALI_IP}",
  "interface": "${IFACE}",
  "pcap": "${pcap}"
}
EOF
}

capture_window() {
  local source_label="$1"
  local type_label="$2"
  local attack_duration="$3"
  local window_duration=$((attack_duration + WINDOW_PAD_SECONDS))
  local pcap="${EDGE_RAW_DIR}/${source_label}.pcap"

  if [[ "$SKIP_EXISTING_PCAPS" == "1" && -s "$pcap" ]]; then
    log "SKIP ${source_label}/${type_label}: existing ${pcap}"
    return 0
  fi

  log "CAPTURE ${source_label}/${type_label}: ${window_duration}s on ${IFACE} -> ${pcap}"
  sudo timeout "$window_duration" tcpdump -i "$IFACE" -w "$pcap" \
    "host ${TARGET_IP} or host ${KALI_IP}" 2>&1 | tee -a "$RUN_LOG" || true
  write_meta "$source_label" "$type_label" "$window_duration" "$pcap"
  log "SAVED ${pcap}"
}

cleanup() {
  stop_sudo_keepalive
}
trap cleanup EXIT

require_uint "MAX_ROUNDS" "$MAX_ROUNDS"
require_uint "START_ROUND" "$START_ROUND"
require_uint "WINDOW_PAD_SECONDS" "$WINDOW_PAD_SECONDS"
require_uint "SCHEDULE_START_EPOCH" "$SCHEDULE_START_EPOCH"

log "=== Debian PCAP capture schedule ==="
print_lab_config | tee -a "$RUN_LOG"
log "MAX_ROUNDS=${MAX_ROUNDS}"
log "START_ROUND=${START_ROUND}"
log "CAPTURE_CLASSES=${CAPTURE_CLASSES}"
log "WINDOW_PAD_SECONDS=${WINDOW_PAD_SECONDS}"
log "SCHEDULE_START_EPOCH=${SCHEDULE_START_EPOCH}"
log "Run the Kali schedule with the same SCHEDULE_START_EPOCH."

start_sudo_keepalive

END_ROUND=$((START_ROUND + MAX_ROUNDS))
SLOT_START_EPOCH="$SCHEDULE_START_EPOCH"
for ((round = START_ROUND; round < END_ROUND; round++)); do
  log "=== Round ${round} (${START_ROUND}..$((END_ROUND - 1))) ==="
  for class_name in $CAPTURE_CLASSES; do
    IFS=":" read -r source_label type_label duration < <(schedule_source "$class_name" "$round")
    window_duration=$((duration + WINDOW_PAD_SECONDS))
    log "WINDOW ${source_label}: planned_start=${SLOT_START_EPOCH} window=${window_duration}s"
    wait_until_epoch "$SLOT_START_EPOCH"
    capture_window "$source_label" "$type_label" "$duration"
    SLOT_START_EPOCH=$((SLOT_START_EPOCH + window_duration))
  done
done

log "DONE"
log "RAW_PCAP_DIR=${EDGE_RAW_DIR}"
log "LOG=${RUN_LOG}"
