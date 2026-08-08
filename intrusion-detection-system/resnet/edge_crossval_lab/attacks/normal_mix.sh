#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../lib.sh
source "${LAB_DIR}/lib.sh"

MODE="${1:-mixed}"
DURATION="${2:-180}"
require_lab_target "$TARGET_IP"

DNS_SERVER="${DNS_SERVER:-${TARGET_IP}}"
HTTP_CONCURRENCY="${HTTP_CONCURRENCY:-4}"
SSH_PORT="${SSH_PORT:-22}"
HTTP_PORT="${HTTP_PORT:-8080}"

USER_AGENTS=(
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Firefox/124"
  "curl/8.5.0"
  "python-requests/2.31.0"
)
PATHS=(
  "/" "/index.html" "/login" "/dashboard" "/status" "/api/v1/status"
  "/api/v1/sensors" "/search?q=temperature" "/assets/app.js" "/css/style.css"
  "/images/logo.png" "/favicon.ico"
)
DOMAINS=("google.com" "github.com" "ubuntu.com" "debian.org" "pool.ntp.org" "time.cloudflare.com")

http_once() {
  local ua path
  ua="${USER_AGENTS[$RANDOM % ${#USER_AGENTS[@]}]}"
  path="${PATHS[$RANDOM % ${#PATHS[@]}]}"
  case $((RANDOM % 5)) in
    0)
      curl -s -o /dev/null -A "$ua" --connect-timeout 2 -m 4 \
        -X POST -d "sensor=temp&value=$((18 + RANDOM % 14))" "${TARGET_URL}${path}" || true
      ;;
    1)
      curl -s -o /dev/null -A "$ua" --connect-timeout 2 -m 4 \
        -H "Accept: application/json" "${TARGET_URL}${path}?r=$RANDOM" || true
      ;;
    *)
      curl -s -o /dev/null -A "$ua" --connect-timeout 2 -m 4 "${TARGET_URL}${path}" || true
      ;;
  esac
}

dns_once() {
  dig +time=1 +tries=1 @"$DNS_SERVER" "${DOMAINS[$RANDOM % ${#DOMAINS[@]}]}" >/dev/null 2>&1 || true
}

icmp_once() {
  ping -c 1 -W 1 "$TARGET_IP" >/dev/null 2>&1 || true
}

ssh_once() {
  timeout 3 bash -lc ":</dev/tcp/${TARGET_IP}/${SSH_PORT}" >/dev/null 2>&1 || true
}

echo "Generating normal/${MODE} for ${DURATION}s toward ${TARGET_URL}"
case "$MODE" in
  web_burst)
    python3 "${LAB_DIR}/attacks/flow_burst.py" --mode http --target-ip "$TARGET_IP" --port "$HTTP_PORT" --duration "$DURATION" --rate "${NORMAL_HTTP_RATE:-120}" --workers 96
    exit 0
    ;;
  api_burst)
    python3 "${LAB_DIR}/attacks/flow_burst.py" --mode http --target-ip "$TARGET_IP" --port "$HTTP_PORT" --duration "$DURATION" --rate "${NORMAL_API_RATE:-100}" --workers 96
    exit 0
    ;;
  dns_burst)
    python3 "${LAB_DIR}/attacks/flow_burst.py" --mode dns --target-ip "$DNS_SERVER" --port 53 --duration "$DURATION" --rate "${NORMAL_DNS_RATE:-120}" --workers 64
    exit 0
    ;;
  ssh_burst)
    python3 "${LAB_DIR}/attacks/flow_burst.py" --mode tcp --target-ip "$TARGET_IP" --port "$SSH_PORT" --duration "$DURATION" --rate "${NORMAL_SSH_RATE:-80}" --workers 64
    exit 0
    ;;
  mixed_burst)
    python3 "${LAB_DIR}/attacks/flow_burst.py" --mode http --target-ip "$TARGET_IP" --port "$HTTP_PORT" --duration "$((DURATION / 2 + 1))" --rate "${NORMAL_HTTP_RATE:-120}" --workers 96
    python3 "${LAB_DIR}/attacks/flow_burst.py" --mode dns --target-ip "$DNS_SERVER" --port 53 --duration "$((DURATION / 3 + 1))" --rate "${NORMAL_DNS_RATE:-100}" --workers 64
    python3 "${LAB_DIR}/attacks/flow_burst.py" --mode tcp --target-ip "$TARGET_IP" --port "$SSH_PORT" --duration "$((DURATION / 4 + 1))" --rate "${NORMAL_SSH_RATE:-60}" --workers 64
    exit 0
    ;;
esac

end=$((SECONDS + DURATION))
while (( SECONDS < end )); do
  case "$MODE" in
    web|http)
      for _ in $(seq 1 "$HTTP_CONCURRENCY"); do http_once & done
      wait || true
      ;;
    dns)
      for _ in {1..12}; do dns_once & done
      wait || true
      ;;
    icmp)
      for _ in {1..4}; do icmp_once & done
      wait || true
      ;;
    api)
      for _ in $(seq 1 "$HTTP_CONCURRENCY"); do http_once & done
      dns_once &
      wait || true
      ;;
    ssh)
      ssh_once
      http_once
      ;;
    mixed)
      for _ in $(seq 1 "$HTTP_CONCURRENCY"); do http_once & done
      (( RANDOM % 2 == 0 )) && dns_once &
      (( RANDOM % 4 == 0 )) && icmp_once &
      (( RANDOM % 12 == 0 )) && ssh_once &
      wait || true
      ;;
    *)
      echo "Unknown normal mode: ${MODE}" >&2
      echo "Modes: mixed, web, dns, icmp, api, ssh" >&2
      exit 2
      ;;
  esac
  sleep 0.15
done
