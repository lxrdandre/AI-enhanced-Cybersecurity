#!/usr/bin/env python3
"""Benign HTTP C2-like server for lab traffic generation.

This server returns harmless command strings and accepts telemetry posts. It is
only meant to create Edge-IIoTset-style backdoor/C2 traffic patterns in a
private lab.
"""

from __future__ import annotations

import argparse
import base64
import random
from http.server import BaseHTTPRequestHandler, HTTPServer


COMMANDS = [
    "whoami",
    "id",
    "uname -a",
    "hostname",
    "uptime",
    "netstat -tuln",
    "ps aux | head -n 10",
    "sleep",
]


class C2Handler(BaseHTTPRequestHandler):
    counter = 0

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path.split("?")[0] not in {"/api/v1/update", "/static/config.json"}:
            self.send_response(404)
            self.end_headers()
            return
        command = COMMANDS[C2Handler.counter % len(COMMANDS)]
        if random.random() < 0.15:
            command = "sleep"
        C2Handler.counter += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(command.encode("utf-8"))
        print(f"[>] command={command}")

    def do_POST(self):
        if self.path.split("?")[0] not in {"/submit.php", "/api/v1/telemetry"}:
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        data = self.rfile.read(length)
        try:
            decoded = base64.b64decode(data, validate=False).decode("utf-8", errors="replace")
        except Exception:
            decoded = data.decode("utf-8", errors="replace")
        print(f"[<] telemetry bytes={len(data)} decoded_preview={decoded[:80]!r}")
        self.send_response(200)
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), C2Handler)
    print(f"Benign C2-like server listening on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
