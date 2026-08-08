#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${LAB_DIR}/lib.sh"

SOURCE_LABEL="${1:-}"
DURATION="${2:-}"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash kali_attack_worker.sh <source_label> [duration_seconds]
  bash kali_attack_worker.sh --check

Examples:
  bash kali_attack_worker.sh ddos_tcp_syn_r0 45
  bash kali_attack_worker.sh port_scanning_r0 120
EOF
}

check_cmd_optional() {
  local cmd="$1"
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "ok: ${cmd}"
  else
    echo "missing: ${cmd}"
  fi
}

if [[ "$SOURCE_LABEL" == "--check" ]]; then
  print_lab_config
  for cmd in bash python3 curl dig ping nmap hping3 hydra ab nping masscan nikto gobuster ffuf wfuzz sqlmap medusa slowhttptest; do
    check_cmd_optional "$cmd"
  done
  exit 0
fi

if [[ -z "$SOURCE_LABEL" ]]; then
  usage
  exit 2
fi

require_lab_target "$TARGET_IP"
HTTP_PORT="${HTTP_PORT:-8080}"

base_label="$SOURCE_LABEL"
base_label="${base_label%_r[0-9]*}"

case "$base_label" in
  normal_mixed|normal_http_dns_icmp)
    bash "${LAB_DIR}/attacks/normal_mix.sh" mixed "${DURATION:-180}"
    ;;
  normal_mixed_burst)
    bash "${LAB_DIR}/attacks/normal_mix.sh" mixed_burst "${DURATION:-180}"
    ;;
  normal_web)
    bash "${LAB_DIR}/attacks/normal_mix.sh" web "${DURATION:-180}"
    ;;
  normal_web_burst)
    bash "${LAB_DIR}/attacks/normal_mix.sh" web_burst "${DURATION:-180}"
    ;;
  normal_dns)
    bash "${LAB_DIR}/attacks/normal_mix.sh" dns "${DURATION:-120}"
    ;;
  normal_dns_burst)
    bash "${LAB_DIR}/attacks/normal_mix.sh" dns_burst "${DURATION:-150}"
    ;;
  normal_icmp)
    bash "${LAB_DIR}/attacks/normal_mix.sh" icmp "${DURATION:-90}"
    ;;
  normal_api)
    bash "${LAB_DIR}/attacks/normal_mix.sh" api "${DURATION:-180}"
    ;;
  normal_api_burst)
    bash "${LAB_DIR}/attacks/normal_mix.sh" api_burst "${DURATION:-180}"
    ;;
  normal_ssh)
    bash "${LAB_DIR}/attacks/normal_mix.sh" ssh "${DURATION:-120}"
    ;;
  normal_ssh_burst)
    bash "${LAB_DIR}/attacks/normal_mix.sh" ssh_burst "${DURATION:-150}"
    ;;
  normal_mqtt)
    bash "${LAB_DIR}/attacks/mqtt_normal.sh" "${DURATION:-120}"
    ;;
  ddos_tcp_syn)
    bash "${LAB_DIR}/attacks/dos_ddos.sh" tcp_syn "${DURATION:-45}" "$HTTP_PORT"
    ;;
  ddos_tcp_syn_rand)
    bash "${LAB_DIR}/attacks/dos_ddos.sh" tcp_syn_rand "${DURATION:-45}" "$HTTP_PORT"
    ;;
  ddos_udp)
    bash "${LAB_DIR}/attacks/dos_ddos.sh" udp "${DURATION:-45}" "$HTTP_PORT"
    ;;
  ddos_udp_rand)
    bash "${LAB_DIR}/attacks/dos_ddos.sh" udp_rand "${DURATION:-45}" "$HTTP_PORT"
    ;;
  ddos_icmp)
    bash "${LAB_DIR}/attacks/dos_ddos.sh" icmp "${DURATION:-45}"
    ;;
  ddos_http_ab|ddos_http)
    bash "${LAB_DIR}/attacks/dos_ddos.sh" http_ab "${DURATION:-120}" "$HTTP_PORT"
    ;;
  ddos_http_curl)
    bash "${LAB_DIR}/attacks/dos_ddos.sh" http_curl "${DURATION:-120}" "$HTTP_PORT"
    ;;
  ddos_nping_tcp)
    bash "${LAB_DIR}/attacks/dos_ddos.sh" nping_tcp "${DURATION:-60}" "$HTTP_PORT"
    ;;
  ddos_tcp_connect_burst)
    bash "${LAB_DIR}/attacks/dos_ddos.sh" tcp_connect_burst "${DURATION:-90}" "$HTTP_PORT"
    ;;
  ddos_udp_burst)
    bash "${LAB_DIR}/attacks/dos_ddos.sh" udp_burst "${DURATION:-90}" "$HTTP_PORT"
    ;;
  ddos_http_burst)
    bash "${LAB_DIR}/attacks/dos_ddos.sh" http_burst "${DURATION:-120}" "$HTTP_PORT"
    ;;
  ddos_slow_http)
    bash "${LAB_DIR}/attacks/dos_ddos.sh" slow_http "${DURATION:-120}" "$HTTP_PORT"
    ;;
  ddos_mixed)
    bash "${LAB_DIR}/attacks/dos_ddos.sh" mixed "${DURATION:-90}" "$HTTP_PORT"
    ;;
  scan_syn_top|port_scanning)
    bash "${LAB_DIR}/attacks/scanning.sh" syn_top "${DURATION:-180}"
    ;;
  scan_syn_full)
    bash "${LAB_DIR}/attacks/scanning.sh" syn_full "${DURATION:-240}"
    ;;
  scan_tcp_connect)
    bash "${LAB_DIR}/attacks/scanning.sh" tcp_connect "${DURATION:-180}"
    ;;
  scan_version)
    bash "${LAB_DIR}/attacks/scanning.sh" version "${DURATION:-180}"
    ;;
  scan_os|os_fingerprinting)
    bash "${LAB_DIR}/attacks/scanning.sh" os "${DURATION:-180}"
    ;;
  scan_udp_top)
    bash "${LAB_DIR}/attacks/scanning.sh" udp_top "${DURATION:-240}"
    ;;
  scan_vuln|vulnerability_scanner)
    bash "${LAB_DIR}/attacks/scanning.sh" vuln "${DURATION:-240}"
    ;;
  scan_web_enum)
    bash "${LAB_DIR}/attacks/scanning.sh" web_enum "${DURATION:-180}"
    ;;
  scan_masscan)
    bash "${LAB_DIR}/attacks/scanning.sh" masscan "${DURATION:-120}"
    ;;
  password_ssh_hydra|password_ssh)
    bash "${LAB_DIR}/attacks/password.sh" ssh_hydra "${DURATION:-180}"
    ;;
  password_ssh_slow)
    bash "${LAB_DIR}/attacks/password.sh" ssh_slow "${DURATION:-180}"
    ;;
  password_http_post|password_http)
    bash "${LAB_DIR}/attacks/password.sh" http_post "${DURATION:-180}"
    ;;
  password_http_get)
    bash "${LAB_DIR}/attacks/password.sh" http_get "${DURATION:-120}"
    ;;
  password_ftp)
    bash "${LAB_DIR}/attacks/password.sh" ftp "${DURATION:-120}"
    ;;
  password_telnet)
    bash "${LAB_DIR}/attacks/password.sh" telnet "${DURATION:-120}"
    ;;
  password_medusa_ssh)
    bash "${LAB_DIR}/attacks/password.sh" medusa_ssh "${DURATION:-180}"
    ;;
  password_mixed)
    bash "${LAB_DIR}/attacks/password.sh" mixed "${DURATION:-180}"
    ;;
  sql_injection)
    bash "${LAB_DIR}/attacks/injection.sh" sql 3500
    ;;
  xss)
    bash "${LAB_DIR}/attacks/injection.sh" xss 3500
    ;;
  uploading)
    bash "${LAB_DIR}/attacks/injection.sh" uploading 3500
    ;;
  path_traversal)
    bash "${LAB_DIR}/attacks/injection.sh" traversal 3500
    ;;
  command_injection)
    bash "${LAB_DIR}/attacks/injection.sh" command 3500
    ;;
  injection_fuzz)
    bash "${LAB_DIR}/attacks/injection.sh" fuzz 3500
    ;;
  sqlmap_probe)
    bash "${LAB_DIR}/attacks/injection.sh" sqlmap 2000
    ;;
  injection_mixed)
    bash "${LAB_DIR}/attacks/injection.sh" mixed 4000
    ;;
  backdoor_http_c2)
    python3 "${LAB_DIR}/attacks/backdoor_beacon_client.py" --url "${C2_URL}" --duration "${DURATION:-180}" --sleep-min 0.05 --sleep-max 0.6 --burst 3
    ;;
  backdoor_http_c2_fast)
    python3 "${LAB_DIR}/attacks/backdoor_beacon_client.py" --url "${C2_URL}" --duration "${DURATION:-180}" --sleep-min 0.01 --sleep-max 0.12 --burst 6
    ;;
  backdoor_http_c2_slow)
    python3 "${LAB_DIR}/attacks/backdoor_beacon_client.py" --url "${C2_URL}" --duration "${DURATION:-180}" --sleep-min 0.6 --sleep-max 2.5 --burst 1
    ;;
  backdoor_http_c2_post|backdoor_http_c2_jitter)
    python3 "${LAB_DIR}/attacks/backdoor_beacon_client.py" --url "${C2_URL}" --duration "${DURATION:-180}" --sleep-min 0.03 --sleep-max 1.5 --burst 4
    ;;
  *)
    echo "Unknown source_label: ${SOURCE_LABEL}" >&2
    usage
    exit 2
    ;;
esac
