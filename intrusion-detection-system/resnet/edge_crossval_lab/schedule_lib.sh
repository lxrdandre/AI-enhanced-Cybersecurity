#!/usr/bin/env bash
set -euo pipefail

schedule_source() {
  local class_name="$1"
  local round="$2"
  local slot=$((round % 8))

  case "$class_name" in
    normal)
      case "$slot" in
        0) echo "normal_mixed_burst_r${round}:normal:180" ;;
        1) echo "normal_web_burst_r${round}:normal:180" ;;
        2) echo "normal_dns_burst_r${round}:normal:150" ;;
        3) echo "normal_api_burst_r${round}:normal:180" ;;
        4) echo "normal_web_burst_r${round}:normal:180" ;;
        5) echo "normal_ssh_burst_r${round}:normal:150" ;;
        6) echo "normal_mixed_burst_r${round}:normal:180" ;;
        7)
          if [[ "${ENABLE_MQTT:-0}" == "1" ]]; then
            echo "normal_mqtt_r${round}:normal:120"
          else
            echo "normal_web_burst_r${round}:normal:180"
          fi
          ;;
      esac
      ;;
    dos_ddos)
      case "$slot" in
        0) echo "ddos_tcp_syn_r${round}:dos_ddos:45" ;;
        1) echo "ddos_tcp_connect_burst_r${round}:dos_ddos:90" ;;
        2) echo "ddos_udp_burst_r${round}:dos_ddos:90" ;;
        3) echo "ddos_http_burst_r${round}:dos_ddos:120" ;;
        4) echo "ddos_icmp_r${round}:dos_ddos:45" ;;
        5) echo "ddos_tcp_connect_burst_r${round}:dos_ddos:90" ;;
        6) echo "ddos_udp_burst_r${round}:dos_ddos:90" ;;
        7) echo "ddos_nping_tcp_r${round}:dos_ddos:60" ;;
      esac
      ;;
    injection)
      case "$slot" in
        0) echo "sql_injection_r${round}:injection:150" ;;
        1) echo "xss_r${round}:injection:150" ;;
        2) echo "uploading_r${round}:injection:150" ;;
        3) echo "path_traversal_r${round}:injection:150" ;;
        4) echo "command_injection_r${round}:injection:150" ;;
        5) echo "injection_fuzz_r${round}:injection:180" ;;
        6) echo "sqlmap_probe_r${round}:injection:180" ;;
        7) echo "injection_mixed_r${round}:injection:180" ;;
      esac
      ;;
    password)
      case "$slot" in
        0) echo "password_http_post_r${round}:password:180" ;;
        1) echo "password_ftp_r${round}:password:180" ;;
        2) echo "password_telnet_r${round}:password:180" ;;
        3) echo "password_mixed_r${round}:password:180" ;;
        4) echo "password_http_get_r${round}:password:180" ;;
        5) echo "password_telnet_r${round}:password:180" ;;
        6) echo "password_ftp_r${round}:password:180" ;;
        7) echo "password_http_post_r${round}:password:180" ;;
      esac
      ;;
    scanning)
      case "$slot" in
        0) echo "scan_syn_top_r${round}:scanning:240" ;;
        1) echo "scan_syn_full_r${round}:scanning:240" ;;
        2) echo "scan_tcp_connect_r${round}:scanning:240" ;;
        3) echo "scan_version_r${round}:scanning:240" ;;
        4) echo "scan_os_r${round}:scanning:240" ;;
        5) echo "scan_web_enum_r${round}:scanning:240" ;;
        6) echo "scan_vuln_r${round}:scanning:240" ;;
        7) echo "scan_syn_top_r${round}:scanning:240" ;;
      esac
      ;;
    backdoor)
      case "$slot" in
        0) echo "backdoor_http_c2_r${round}:backdoor:180" ;;
        1) echo "backdoor_http_c2_fast_r${round}:backdoor:180" ;;
        2) echo "backdoor_http_c2_slow_r${round}:backdoor:180" ;;
        3) echo "backdoor_http_c2_post_r${round}:backdoor:180" ;;
        4) echo "backdoor_http_c2_jitter_r${round}:backdoor:180" ;;
        5) echo "backdoor_http_c2_fast_r${round}:backdoor:180" ;;
        6) echo "backdoor_http_c2_r${round}:backdoor:180" ;;
        7) echo "backdoor_http_c2_jitter_r${round}:backdoor:180" ;;
      esac
      ;;
    *)
      echo "Unknown class in CAPTURE_CLASSES: ${class_name}" >&2
      return 2
      ;;
  esac
}

wait_until_epoch() {
  local start_epoch="$1"
  local now
  now="$(date +%s)"
  if (( now < start_epoch )); then
    echo "Waiting $((start_epoch - now))s until schedule start epoch ${start_epoch}"
    sleep "$((start_epoch - now))"
  fi
}

start_sudo_keepalive() {
  if [[ "${KEEP_SUDO_ALIVE:-1}" != "1" ]]; then
    return 0
  fi
  sudo -v
  (
    while true; do
      sudo -n true >/dev/null 2>&1 || exit
      sleep 60
    done
  ) &
  SUDO_KEEPALIVE_PID=$!
}

stop_sudo_keepalive() {
  if [[ -n "${SUDO_KEEPALIVE_PID:-}" ]]; then
    kill "$SUDO_KEEPALIVE_PID" >/dev/null 2>&1 || true
  fi
}
