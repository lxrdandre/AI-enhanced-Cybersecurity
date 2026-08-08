#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../lib.sh
source "${LAB_DIR}/lib.sh"

MODE="${1:-mixed}"
DURATION="${2:-180}"
require_lab_target "$TARGET_IP"
require_cmd nmap

IOT_TCP_PORTS="${IOT_TCP_PORTS:-21,22,23,80,81,102,443,502,554,631,1883,2000,2323,2404,4840,5357,8080,8081,9000,44818}"
SCAN_TOP_PORTS="${SCAN_TOP_PORTS:-3000}"
SCAN_CONNECT_PORTS="${SCAN_CONNECT_PORTS:-1-20000}"
SCAN_REPEAT_DELAY="${SCAN_REPEAT_DELAY:-1}"
DEADLINE=$((SECONDS + DURATION))

WEB_ENUM_PATHS=(
  /
  /login
  /admin
  /status
  /api
  /api/v1/telemetry
  /device
  /device/update
  /config
  /setup.xml
  /onvif/device_service
  /cgi-bin/status
  /camera
  /snapshot
  /firmware
  /upload
  /download
  /search
  /metrics
  /debug
)

remaining_seconds() {
  local remaining=$((DEADLINE - SECONDS))
  if (( remaining < 1 )); then
    echo 0
  else
    echo "$remaining"
  fi
}

run_left() {
  local remaining
  remaining="$(remaining_seconds)"
  (( remaining > 0 )) || return 0
  timeout "$remaining" "$@" || true
}

run_left_sudo() {
  local remaining
  remaining="$(remaining_seconds)"
  (( remaining > 0 )) || return 0
  timeout "$remaining" sudo "$@" || true
}

pause_if_time_left() {
  if (( $(remaining_seconds) > 0 && SCAN_REPEAT_DELAY > 0 )); then
    sleep "$SCAN_REPEAT_DELAY"
  fi
}

repeat_until_deadline() {
  local label="$1"
  shift
  local pass=1
  while (( $(remaining_seconds) > 0 )); do
    echo "scan/${MODE} ${label} pass ${pass}, remaining $(remaining_seconds)s"
    "$@"
    pass=$((pass + 1))
    pause_if_time_left
  done
}

scan_syn_top_once() {
  run_left_sudo nmap -sS -T4 --max-retries 1 --top-ports "$SCAN_TOP_PORTS" "$TARGET_IP"
  run_left nmap -sT -T4 --max-retries 1 --top-ports "$SCAN_TOP_PORTS" "$TARGET_IP"
}

scan_version_once() {
  run_left nmap -sV -T4 --version-light --max-retries 1 -p "$IOT_TCP_PORTS" "$TARGET_IP"
  run_left nmap -sV -T4 --version-light --top-ports 1000 "$TARGET_IP"
}

scan_os_once() {
  run_left_sudo nmap -O --osscan-guess -sS -T4 --max-retries 1 --top-ports 1000 "$TARGET_IP"
  run_left_sudo nmap -O --osscan-guess -sS -T4 --max-retries 1 -p "$IOT_TCP_PORTS" "$TARGET_IP"
}

web_path_probe() {
  local pass=1
  require_cmd curl
  while (( $(remaining_seconds) > 0 )); do
    echo "scan/${MODE} web path probe pass ${pass}, remaining $(remaining_seconds)s"
    for path in "${WEB_ENUM_PATHS[@]}"; do
      (( $(remaining_seconds) > 0 )) || break
      timeout "$(remaining_seconds)" curl -sk -o /dev/null --connect-timeout 2 -m 4 \
        -A "EdgeScanner/${pass}" "${TARGET_URL}${path}?probe=${pass}${RANDOM}" || true
    done
    pass=$((pass + 1))
  done
}

echo "Generating scanning/${MODE} for about ${DURATION}s against ${TARGET_IP}"

case "$MODE" in
  syn_top)
    repeat_until_deadline syn-top scan_syn_top_once
    ;;
  syn_full)
    repeat_until_deadline syn-full run_left_sudo nmap -sS -T4 --max-retries 1 -p 1-65535 "$TARGET_IP"
    ;;
  tcp_connect)
    repeat_until_deadline tcp-connect run_left nmap -sT -T4 --max-retries 1 -p "$SCAN_CONNECT_PORTS" "$TARGET_IP"
    ;;
  version)
    repeat_until_deadline version scan_version_once
    ;;
  os)
    repeat_until_deadline os scan_os_once
    ;;
  udp_top)
    repeat_until_deadline udp-top run_left_sudo nmap -sU -T3 --max-retries 1 --top-ports 200 "$TARGET_IP"
    ;;
  vuln)
    run_left nmap -sV --script vuln -T3 --max-retries 1 -p "$IOT_TCP_PORTS" "$TARGET_IP"
    if command -v nikto >/dev/null 2>&1; then
      run_left nikto -h "$TARGET_URL"
    fi
    web_path_probe
    ;;
  web_enum)
    web_path_probe
    if command -v gobuster >/dev/null 2>&1 && [[ -f /usr/share/wordlists/dirb/common.txt ]]; then
      run_left gobuster dir -u "$TARGET_URL" -w /usr/share/wordlists/dirb/common.txt -q
    elif command -v ffuf >/dev/null 2>&1 && [[ -f /usr/share/wordlists/dirb/common.txt ]]; then
      run_left ffuf -u "${TARGET_URL}/FUZZ" -w /usr/share/wordlists/dirb/common.txt -s
    else
      run_left nmap --script http-enum -T3 -p 80,81,443,631,5357,8080,8081 "$TARGET_IP"
    fi
    ;;
  masscan)
    if command -v masscan >/dev/null 2>&1; then
      run_left_sudo masscan "$TARGET_IP" -p1-65535 --rate 3000
    else
      run_left_sudo nmap -sS -T5 -p 1-65535 "$TARGET_IP"
    fi
    ;;
  mixed)
    "$0" syn_top "$((DURATION / 3 + 1))"
    "$0" version "$((DURATION / 3 + 1))"
    "$0" web_enum "$((DURATION / 3 + 1))"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Modes: syn_top, syn_full, tcp_connect, version, os, udp_top, vuln, web_enum, masscan, mixed" >&2
    exit 2
    ;;
esac
