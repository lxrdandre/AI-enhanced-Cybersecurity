#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${LAB_DIR}/lib.sh"

MAX_ROUNDS="${MAX_ROUNDS:-14}"
KALI_SSH_USER="${KALI_SSH_USER:-kali}"
KALI_SSH_PORT="${KALI_SSH_PORT:-22}"
KALI_LAB_DIR="${KALI_LAB_DIR:-~/edge_crossval_lab}"
SSH_BATCH_MODE="${SSH_BATCH_MODE:-yes}"
CAPTURE_CLASSES="${CAPTURE_CLASSES:-normal dos_ddos injection password scanning backdoor}"
SKIP_EXISTING_PCAPS="${SKIP_EXISTING_PCAPS:-1}"
C2_PORT="${C2_PORT:-8090}"

require_lab_target "$TARGET_IP"
require_cmd tcpdump
require_cmd ssh
ensure_dirs

LOG_DIR="${EDGE_CROSSVAL_ROOT}/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="${LOG_DIR}/debian_capture_$(date +%Y%m%d_%H%M%S).log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$RUN_LOG"
}

ssh_kali() {
  ssh \
    -p "$KALI_SSH_PORT" \
    -o "BatchMode=${SSH_BATCH_MODE}" \
    -o StrictHostKeyChecking=accept-new \
    "${KALI_SSH_USER}@${KALI_IP}" \
    "$@"
}

run_kali_source() {
  local source_label="$1"
  local duration="$2"
  local remote_cmd
  printf -v remote_cmd "cd %s && source config.env && bash kali_attack_worker.sh %q %q" \
    "$KALI_LAB_DIR" "$source_label" "$duration"

  if command -v timeout >/dev/null 2>&1; then
    timeout "$((duration + 90))" ssh \
      -p "$KALI_SSH_PORT" \
      -o "BatchMode=${SSH_BATCH_MODE}" \
      -o StrictHostKeyChecking=accept-new \
      "${KALI_SSH_USER}@${KALI_IP}" \
      "$remote_cmd"
  else
    ssh_kali "$remote_cmd"
  fi
}

run_pair() {
  local source_label="$1"
  local type_label="$2"
  local duration="$3"
  local pcap="${EDGE_RAW_DIR}/${source_label}.pcap"

  if [[ "$SKIP_EXISTING_PCAPS" == "1" && -s "$pcap" ]]; then
    log "SKIP ${source_label}/${type_label}: existing PCAP ${pcap}"
    return 0
  fi

  log "START ${source_label}/${type_label}: local capture=${duration}s remote Kali generator"
  bash "${LAB_DIR}/capture_source.sh" "$source_label" "$type_label" "$duration" >>"$RUN_LOG" 2>&1 &
  local capture_pid=$!
  sleep 3

  set +e
  run_kali_source "$source_label" "$duration" >>"$RUN_LOG" 2>&1
  local generator_status=$?
  set -e
  if [[ "$generator_status" -ne 0 ]]; then
    log "Kali generator exited non-zero for ${source_label}: ${generator_status}"
  fi

  wait "$capture_pid" || true
  log "DONE ${source_label}/${type_label}: saved ${pcap}"
}

maybe_start_local_c2() {
  if [[ "${START_LOCAL_C2:-1}" != "1" ]]; then
    return 0
  fi
  if [[ -n "${LOCAL_C2_PID:-}" ]] && kill -0 "$LOCAL_C2_PID" >/dev/null 2>&1; then
    return 0
  fi
  log "Starting benign C2 server on Debian victim: 0.0.0.0:${C2_PORT}"
  require_cmd python3
  python3 "${LAB_DIR}/attacks/backdoor_c2_server.py" --host 0.0.0.0 --port "$C2_PORT" >>"$RUN_LOG" 2>&1 &
  LOCAL_C2_PID=$!
  sleep 2
}

cleanup() {
  if [[ -n "${LOCAL_C2_PID:-}" ]]; then
    kill "$LOCAL_C2_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

select_source() {
  local class_name="$1"
  local round="$2"
  local slot=$((round % 4))

  case "$class_name" in
    normal)
      if [[ "${ENABLE_MQTT:-0}" == "1" && "$slot" -eq 1 ]]; then
        echo "normal_mqtt_r${round}:120"
      else
        echo "normal_http_dns_icmp_r${round}:180"
      fi
      ;;
    dos_ddos)
      case "$slot" in
        0) echo "ddos_tcp_syn_r${round}:35" ;;
        1) echo "ddos_udp_r${round}:35" ;;
        2) echo "ddos_icmp_r${round}:35" ;;
        3) echo "ddos_http_r${round}:90" ;;
      esac
      ;;
    scanning)
      case "$slot" in
        0) echo "port_scanning_r${round}:120" ;;
        1) echo "os_fingerprinting_r${round}:120" ;;
        2|3) echo "vulnerability_scanner_r${round}:240" ;;
      esac
      ;;
    password)
      if [[ "${ENABLE_HTTP_PASSWORD:-0}" == "1" && "$slot" -eq 1 ]]; then
        echo "password_http_r${round}:150"
      else
        echo "password_ssh_r${round}:180"
      fi
      ;;
    injection)
      case "$slot" in
        0) echo "sql_injection_r${round}:120" ;;
        1) echo "xss_r${round}:120" ;;
        2|3) echo "uploading_r${round}:120" ;;
      esac
      ;;
    backdoor)
      maybe_start_local_c2
      echo "backdoor_http_c2_r${round}:180"
      ;;
    *)
      return 2
      ;;
  esac
}

log "=== Debian capture-only + Kali generator Edge cross-validation run ==="
print_lab_config | tee -a "$RUN_LOG"
log "KALI_SSH_USER=${KALI_SSH_USER}"
log "KALI_SSH_PORT=${KALI_SSH_PORT}"
log "KALI_LAB_DIR=${KALI_LAB_DIR}"
log "MAX_ROUNDS=${MAX_ROUNDS}"
log "CAPTURE_CLASSES=${CAPTURE_CLASSES}"
log "SKIP_EXISTING_PCAPS=${SKIP_EXISTING_PCAPS}"
log "This run captures PCAPs only. It does not extract CSVs or build the dataset."

log "Checking Kali worker over SSH"
ssh_kali "cd ${KALI_LAB_DIR} && source config.env && bash kali_attack_worker.sh --check" | tee -a "$RUN_LOG"

for ((round = 0; round < MAX_ROUNDS; round++)); do
  log "=== Round ${round}/${MAX_ROUNDS} ==="
  for class_name in $CAPTURE_CLASSES; do
    source_spec="$(select_source "$class_name" "$round")"
    source_label="${source_spec%%:*}"
    duration="${source_spec##*:}"
    run_pair "$source_label" "$class_name" "$duration"
  done
done

log "DONE"
log "RAW_PCAP_DIR=${EDGE_RAW_DIR}"
log "LOG=${RUN_LOG}"
log "Later, extract with: python3 ${LAB_DIR}/extract_all_pcaps.py --raw-dir ${EDGE_RAW_DIR} --csv-dir ${EDGE_CSV_DIR} --overwrite"
