"""Generate many short-lived flows for Zeek-based lab datasets.

This is used for retraining data because Zeek counts flows, not packets. Packet
floods can be huge but still produce very few Zeek conn.log rows if the 5-tuple
does not change. These modes create many distinct client-side source ports.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import random
import socket
import struct
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor


PATHS = [
    "/",
    "/index.html",
    "/login",
    "/status",
    "/api/v1/status",
    "/api/v1/sensors",
    "/search?q=sensor",
    "/assets/app.js",
    "/css/style.css",
    "/favicon.ico",
]


def require_private_target(host: str) -> None:
    if os.environ.get("ALLOW_NON_PRIVATE", "0") == "1":
        return
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(host))
    except OSError as exc:
        raise SystemExit(f"Could not resolve target {host!r}: {exc}") from exc
    if not (ip.is_private or ip.is_loopback):
        raise SystemExit(f"Refusing non-private target {ip}. Set ALLOW_NON_PRIVATE=1 only for an authorized lab.")


def make_dns_query(name: str) -> bytes:
    txid = random.randint(0, 65535)
    header = struct.pack("!HHHHHH", txid, 0x0100, 1, 0, 0, 0)
    qname = b"".join(bytes([len(part)]) + part.encode("ascii", errors="ignore") for part in name.split(".")) + b"\x00"
    return header + qname + struct.pack("!HH", 1, 1)


def tcp_once(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def http_once(host: str, port: int, timeout: float) -> bool:
    path = random.choice(PATHS)
    request = (
        f"GET {path}?r={random.randint(1, 999999)} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        "User-Agent: edge-lab-flow-burst/1.0\r\n"
        "Accept: */*\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii", errors="ignore")
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(request)
            try:
                sock.recv(256)
            except OSError:
                pass
            return True
    except OSError:
        return False


def udp_once(host: str, port: int, timeout: float) -> bool:
    payload = os.urandom(random.randint(16, 96))
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)
            sock.sendto(payload, (host, port))
            return True
    except OSError:
        return False


def dns_once(host: str, port: int, timeout: float) -> bool:
    domains = ["google.com", "github.com", "ubuntu.com", "debian.org", "pool.ntp.org", "time.cloudflare.com"]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)
            sock.sendto(make_dns_query(random.choice(domains)), (host, port))
            try:
                sock.recvfrom(512)
            except OSError:
                pass
            return True
    except OSError:
        return False


def choose_port(ports: list[int]) -> int:
    return random.choice(ports)


def parse_ports(value: str) -> list[int]:
    ports: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ports.extend(range(int(start), int(end) + 1))
        else:
            ports.append(int(part))
    return ports or [80]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["tcp", "http", "udp", "dns"], required=True)
    parser.add_argument("--target-ip", default=os.environ.get("TARGET_IP", "192.168.56.20"))
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--ports", default="")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--rate", type=float, default=100.0, help="Approximate attempts per second.")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_private_target(args.target_ip)
    ports = parse_ports(args.ports) if args.ports else [args.port]
    end = time.monotonic() + max(1.0, args.duration)
    interval = 1.0 / max(1.0, args.rate)
    lock = threading.Lock()
    counts = {"attempted": 0, "ok": 0}

    def submit_one() -> bool:
        port = choose_port(ports)
        if args.mode == "tcp":
            return tcp_once(args.target_ip, port, args.timeout)
        if args.mode == "http":
            return http_once(args.target_ip, port, args.timeout)
        if args.mode == "udp":
            return udp_once(args.target_ip, port, args.timeout)
        if args.mode == "dns":
            return dns_once(args.target_ip, port, args.timeout)
        return False

    def done_callback(future) -> None:
        ok = False
        try:
            ok = bool(future.result())
        except Exception:
            ok = False
        with lock:
            counts["attempted"] += 1
            counts["ok"] += int(ok)
            if counts["attempted"] % 1000 == 0:
                print(f"flow_burst {args.mode}: attempted={counts['attempted']} ok={counts['ok']}", flush=True)

    print(
        f"flow_burst mode={args.mode} target={args.target_ip} ports={args.ports or args.port} "
        f"duration={args.duration}s rate={args.rate}/s workers={args.workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        while time.monotonic() < end:
            future = pool.submit(submit_one)
            future.add_done_callback(done_callback)
            time.sleep(interval)
    print(f"flow_burst done: attempted={counts['attempted']} ok={counts['ok']}", flush=True)


if __name__ == "__main__":
    main()
