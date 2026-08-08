#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALLER_MAX_ROUNDS="${MAX_ROUNDS-}"
CALLER_START_ROUND="${START_ROUND-}"
CALLER_CAPTURE_CLASSES="${CAPTURE_CLASSES-}"
CALLER_WINDOW_PAD_SECONDS="${WINDOW_PAD_SECONDS-}"
CALLER_START_DELAY_SECONDS="${START_DELAY_SECONDS-}"
CALLER_SCHEDULE_START_EPOCH="${SCHEDULE_START_EPOCH-}"
CALLER_C2_PORT="${C2_PORT-}"

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
C2_PORT="${CALLER_C2_PORT:-${C2_PORT:-8090}}"

require_lab_target "$TARGET_IP"
require_cmd timeout

LOG_DIR="${EDGE_CROSSVAL_ROOT}/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="${LOG_DIR}/kali_attack_schedule_$(date +%Y%m%d_%H%M%S).log"

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

maybe_start_local_c2() {
  if [[ "${START_LOCAL_C2:-0}" != "1" ]]; then
    return 0
  fi
  if [[ -n "${LOCAL_C2_PID:-}" ]] && kill -0 "$LOCAL_C2_PID" >/dev/null 2>&1; then
    return 0
  fi
  require_cmd python3
  log "Starting local benign C2 server on 0.0.0.0:${C2_PORT}"
  python3 "${LAB_DIR}/attacks/backdoor_c2_server.py" --host 0.0.0.0 --port "$C2_PORT" >>"$RUN_LOG" 2>&1 &
  LOCAL_C2_PID=$!
  sleep 2
}

run_attack_window() {
  local source_label="$1"
  local duration="$2"
  local window_duration=$((duration + WINDOW_PAD_SECONDS))
  local started_at
  local target_end
  local ended_at
  local remaining

  if [[ "$source_label" == backdoor_http_c2* ]]; then
    maybe_start_local_c2
  fi

  started_at="$(date +%s)"
  target_end=$((started_at + window_duration))
  log "ATTACK ${source_label}: duration=${duration}s window=${window_duration}s"
  set +e
  timeout "$window_duration" bash "${LAB_DIR}/kali_attack_worker.sh" "$source_label" "$duration" 2>&1 | tee -a "$RUN_LOG"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -eq 124 ]]; then
    log "WARN ${source_label}: worker reached ${window_duration}s window timeout"
  elif [[ "$status" -ne 0 ]]; then
    log "WARN ${source_label}: worker exited ${status}"
  fi

  ended_at="$(date +%s)"
  remaining=$((target_end - ended_at))
  if (( remaining > 0 )); then
    log "WAIT ${source_label}: attack finished early; sleeping ${remaining}s until capture window ends"
    sleep "$remaining"
  fi
}

cleanup() {
  stop_sudo_keepalive
  if [[ -n "${LOCAL_C2_PID:-}" ]]; then
    kill "$LOCAL_C2_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

require_uint "MAX_ROUNDS" "$MAX_ROUNDS"
require_uint "START_ROUND" "$START_ROUND"
require_uint "WINDOW_PAD_SECONDS" "$WINDOW_PAD_SECONDS"
require_uint "SCHEDULE_START_EPOCH" "$SCHEDULE_START_EPOCH"

log "=== Kali attack generation schedule ==="
print_lab_config | tee -a "$RUN_LOG"
log "MAX_ROUNDS=${MAX_ROUNDS}"
log "START_ROUND=${START_ROUND}"
log "CAPTURE_CLASSES=${CAPTURE_CLASSES}"
log "WINDOW_PAD_SECONDS=${WINDOW_PAD_SECONDS}"
log "SCHEDULE_START_EPOCH=${SCHEDULE_START_EPOCH}"
log "The Debian capture schedule must use the same SCHEDULE_START_EPOCH."

start_sudo_keepalive

END_ROUND=$((START_ROUND + MAX_ROUNDS))
SLOT_START_EPOCH="$SCHEDULE_START_EPOCH"
for ((round = START_ROUND; round < END_ROUND; round++)); do
  log "=== Round ${round} (${START_ROUND}..$((END_ROUND - 1))) ==="
  for class_name in $CAPTURE_CLASSES; do
    IFS=":" read -r source_label _type_label duration < <(schedule_source "$class_name" "$round")
    window_duration=$((duration + WINDOW_PAD_SECONDS))
    log "WINDOW ${source_label}: planned_start=${SLOT_START_EPOCH} window=${window_duration}s"
    wait_until_epoch "$SLOT_START_EPOCH"
    run_attack_window "$source_label" "$duration"
    SLOT_START_EPOCH=$((SLOT_START_EPOCH + window_duration))
  done
done

log "DONE"
log "LOG=${RUN_LOG}"
