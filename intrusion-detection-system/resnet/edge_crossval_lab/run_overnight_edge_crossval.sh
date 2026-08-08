#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${LAB_DIR}/lib.sh"

MAX_ROUNDS="${MAX_ROUNDS:-14}"
CAPTURE_CLASSES="${CAPTURE_CLASSES:-normal dos_ddos injection password scanning backdoor}"
SKIP_EXISTING_PCAPS="${SKIP_EXISTING_PCAPS:-1}"
HTTP_PORT="${HTTP_PORT:-8080}"
C2_PORT="${C2_PORT:-8090}"

require_lab_target "$TARGET_IP"
require_cmd tcpdump
ensure_dirs

LOG_DIR="${EDGE_CROSSVAL_ROOT}/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="${LOG_DIR}/overnight_$(date +%Y%m%d_%H%M%S).log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$RUN_LOG"
}

run_pair() {
  local source_label="$1"
  local type_label="$2"
  local duration="$3"
  local generator="$4"
  local pcap="${EDGE_RAW_DIR}/${source_label}.pcap"

  if [[ "$SKIP_EXISTING_PCAPS" == "1" && -s "$pcap" ]]; then
    log "SKIP ${source_label}/${type_label}: existing PCAP ${pcap}"
    return 0
  fi

  log "START ${source_label}/${type_label}: capture=${duration}s generator=${generator}"
  bash "${LAB_DIR}/capture_source.sh" "$source_label" "$type_label" "$duration" >>"$RUN_LOG" 2>&1 &
  local capture_pid=$!
  sleep 3

  set +e
  if command -v timeout >/dev/null 2>&1; then
    timeout "$((duration + 30))" bash -lc "$generator" >>"$RUN_LOG" 2>&1
  else
    bash -lc "$generator" >>"$RUN_LOG" 2>&1
  fi
  local generator_status=$?
  set -e
  if [[ "$generator_status" -ne 0 ]]; then
    log "Generator exited non-zero for ${source_label}: ${generator_status}"
  fi

  wait "$capture_pid" || true
  log "DONE ${source_label}/${type_label}: saved ${pcap}"
}

maybe_start_local_c2() {
  if [[ "${START_LOCAL_C2:-0}" != "1" ]]; then
    return 0
  fi
  if [[ -n "${LOCAL_C2_PID:-}" ]] && kill -0 "$LOCAL_C2_PID" >/dev/null 2>&1; then
    return 0
  fi
  log "Starting local benign C2 server on 0.0.0.0:${C2_PORT}"
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

run_class_round() {
  local class_name="$1"
  local round="$2"
  local slot=$((round % 4))
  local source=""
  local duration=""
  local generator=""

  case "$class_name" in
    normal)
      if [[ "${ENABLE_MQTT:-0}" == "1" && "$slot" -eq 1 ]]; then
        source="normal_mqtt_r${round}"
        duration="120"
        generator="source '${LAB_DIR}/config.env'; bash '${LAB_DIR}/attacks/mqtt_normal.sh' 120"
      else
        source="normal_http_dns_icmp_r${round}"
        duration="180"
        generator="source '${LAB_DIR}/config.env'; bash '${LAB_DIR}/attacks/normal_mix.sh' mixed 180"
      fi
      ;;
    dos_ddos)
      case "$slot" in
        0) source="ddos_tcp_syn_r${round}"; duration="35"; generator="source '${LAB_DIR}/config.env'; bash '${LAB_DIR}/attacks/dos_ddos.sh' tcp_syn 35 '${HTTP_PORT}'" ;;
        1) source="ddos_udp_r${round}"; duration="35"; generator="source '${LAB_DIR}/config.env'; bash '${LAB_DIR}/attacks/dos_ddos.sh' udp 35 '${HTTP_PORT}'" ;;
        2) source="ddos_icmp_r${round}"; duration="35"; generator="source '${LAB_DIR}/config.env'; bash '${LAB_DIR}/attacks/dos_ddos.sh' icmp 35" ;;
        3) source="ddos_http_r${round}"; duration="90"; generator="source '${LAB_DIR}/config.env'; bash '${LAB_DIR}/attacks/dos_ddos.sh' http 90 '${HTTP_PORT}'" ;;
      esac
      ;;
    scanning)
      case "$slot" in
        0) source="port_scanning_r${round}"; duration="120"; generator="source '${LAB_DIR}/config.env'; bash '${LAB_DIR}/attacks/scanning.sh' port" ;;
        1) source="os_fingerprinting_r${round}"; duration="120"; generator="source '${LAB_DIR}/config.env'; bash '${LAB_DIR}/attacks/scanning.sh' os" ;;
        2|3) source="vulnerability_scanner_r${round}"; duration="240"; generator="source '${LAB_DIR}/config.env'; bash '${LAB_DIR}/attacks/scanning.sh' vuln" ;;
      esac
      ;;
    password)
      if [[ "${ENABLE_HTTP_PASSWORD:-0}" == "1" && "$slot" -eq 1 ]]; then
        source="password_http_r${round}"
        duration="150"
        generator="source '${LAB_DIR}/config.env'; bash '${LAB_DIR}/attacks/password.sh' http 150"
      else
        source="password_ssh_r${round}"
        duration="180"
        generator="source '${LAB_DIR}/config.env'; bash '${LAB_DIR}/attacks/password.sh' ssh 180"
      fi
      ;;
    injection)
      case "$slot" in
        0) source="sql_injection_r${round}"; duration="120"; generator="source '${LAB_DIR}/config.env'; bash '${LAB_DIR}/attacks/injection.sh' sql 2500" ;;
        1) source="xss_r${round}"; duration="120"; generator="source '${LAB_DIR}/config.env'; bash '${LAB_DIR}/attacks/injection.sh' xss 2500" ;;
        2|3) source="uploading_r${round}"; duration="120"; generator="source '${LAB_DIR}/config.env'; bash '${LAB_DIR}/attacks/injection.sh' uploading 2500" ;;
      esac
      ;;
    backdoor)
      maybe_start_local_c2
      source="backdoor_http_c2_r${round}"
      duration="180"
      generator="source '${LAB_DIR}/config.env'; python3 '${LAB_DIR}/attacks/backdoor_beacon_client.py' --url '${C2_URL}' --duration 180"
      ;;
    *)
      log "Unknown class: ${class_name}"
      return 2
      ;;
  esac

  run_pair "$source" "$class_name" "$duration" "$generator"
}

log "=== Edge-style overnight capture-only run ==="
print_lab_config | tee -a "$RUN_LOG"
log "MAX_ROUNDS=${MAX_ROUNDS}"
log "CAPTURE_CLASSES=${CAPTURE_CLASSES}"
log "SKIP_EXISTING_PCAPS=${SKIP_EXISTING_PCAPS}"
log "This run captures PCAPs only. It does not extract CSVs or build the dataset."

for ((round = 0; round < MAX_ROUNDS; round++)); do
  log "=== Round ${round}/${MAX_ROUNDS} ==="
  for class_name in $CAPTURE_CLASSES; do
    run_class_round "$class_name" "$round"
  done
done

log "DONE"
log "RAW_PCAP_DIR=${EDGE_RAW_DIR}"
log "LOG=${RUN_LOG}"
log "Later, extract with: python3 ${LAB_DIR}/extract_all_pcaps.py --raw-dir ${EDGE_RAW_DIR} --csv-dir ${EDGE_CSV_DIR} --overwrite"
