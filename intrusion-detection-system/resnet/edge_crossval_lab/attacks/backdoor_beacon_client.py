"""Benign HTTP C2-like beacon client for lab traffic generation.

This does not execute commands. It only creates request/response patterns that
look like periodic backdoor beaconing for the IDS validation dataset.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import random
import socket
import time
import urllib.error
import urllib.parse
import urllib.request


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "curl/7.81.0",
    "python-requests/2.31.0",
]


def require_lab_url(url: str, allow_non_private: bool) -> None:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    if allow_non_private or not host:
        return
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(host))
    except OSError as exc:
        raise SystemExit(f"Could not resolve C2 host {host!r}: {exc}") from exc
    if not (ip.is_private or ip.is_loopback):
        raise SystemExit(f"Refusing non-private C2 target {ip}. Set ALLOW_NON_PRIVATE=1 only for an authorized lab.")


def request(url: str, method: str = "GET", payload: bytes | None = None) -> tuple[int, bytes]:
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "*/*",
        "Cache-Control": "no-cache",
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return int(resp.status), resp.read(4096)
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(1024)
    except OSError:
        return 0, b""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_c2_port = os.environ.get("C2_PORT", "8090")
    default_base = os.environ.get("C2_URL") or f"http://{os.environ.get('TARGET_IP', '192.168.56.20')}:{default_c2_port}"
    parser.add_argument("--url", default=default_base.rstrip("/"))
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--sleep-min", type=float, default=0.03)
    parser.add_argument("--sleep-max", type=float, default=0.45)
    parser.add_argument("--burst", type=int, default=3)
    parser.add_argument("--host-id", default=f"host-{random.randint(1000, 9999)}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_lab_url(args.url, os.environ.get("ALLOW_NON_PRIVATE", "0") == "1")
    end = time.monotonic() + max(1, args.duration)
    count = 0
    while time.monotonic() < end:
        for _ in range(max(1, args.burst)):
            nonce = random.randint(100000, 999999)
            endpoint = random.choice(["/api/v1/update", "/static/config.json", "/api/v1/task"])
            request(f"{args.url}{endpoint}?id={args.host_id}&n={nonce}")
            telemetry = {
                "id": args.host_id,
                "ts": time.time(),
                "pid": random.randint(1000, 9000),
                "user": random.choice(["www-data", "iot", "svc", "root"]),
                "status": random.choice(["idle", "collect", "sleep", "exec"]),
                "bytes": random.randint(64, 8192),
            }
            post_path = random.choice(["/api/v1/telemetry", "/submit.php", "/api/v1/result"])
            request(f"{args.url}{post_path}", method="POST", payload=json.dumps(telemetry).encode("utf-8"))
            count += 1
        time.sleep(random.uniform(args.sleep_min, args.sleep_max))
    print(f"Sent {count} beacon cycles to {args.url}")


if __name__ == "__main__":
    main()
