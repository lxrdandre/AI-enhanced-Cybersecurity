#!/usr/bin/env python3
"""Run benign IoT/OT-looking services for lab scan and traffic generation.

The services are intentionally fake: they return small banners or protocol
stubs so scanners see open ports without deploying real vulnerable daemons.
Run only inside an authorized lab.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TcpService:
    port: int
    name: str
    kind: str


@dataclass(frozen=True)
class UdpService:
    port: int
    name: str
    kind: str


TCP_SERVICES = [
    TcpService(21, "iot-ftpd", "ftp"),
    TcpService(23, "iot-telnet", "telnet"),
    TcpService(80, "iot-web", "http"),
    TcpService(81, "camera-web", "http_camera"),
    TcpService(102, "s7comm", "raw"),
    TcpService(443, "iot-https-placeholder", "http"),
    TcpService(502, "modbus", "modbus"),
    TcpService(554, "rtsp-camera", "rtsp"),
    TcpService(631, "printer-ipp", "http_printer"),
    TcpService(1883, "mqtt", "mqtt"),
    TcpService(2000, "cisco-sccp-like", "banner"),
    TcpService(2323, "alt-telnet", "telnet"),
    TcpService(2404, "iec104", "raw"),
    TcpService(4840, "opcua", "opcua"),
    TcpService(5357, "wsdapi", "http_wsd"),
    TcpService(8080, "iot-web-alt", "http"),
    TcpService(8081, "camera-web-alt", "http_camera"),
    TcpService(9000, "dvr-debug", "banner"),
    TcpService(44818, "ethernet-ip", "raw"),
]

UDP_SERVICES = [
    UdpService(69, "tftp", "tftp"),
    UdpService(161, "snmp", "snmp"),
    UdpService(1900, "ssdp", "ssdp"),
    UdpService(5353, "mdns", "mdns"),
    UdpService(5683, "coap", "coap"),
    UdpService(47808, "bacnet", "bacnet"),
]


def http_response(kind: str, data: bytes, port: int) -> bytes:
    try:
        first_line = data.decode("iso-8859-1", errors="ignore").splitlines()[0]
        path = first_line.split()[1] if len(first_line.split()) >= 2 else "/"
    except Exception:
        path = "/"

    headers = [
        "Server: Boa/0.94.14rc21",
        "Connection: close",
        "X-Device-Type: edge-lab-iot",
    ]

    if path.startswith("/api") or path.startswith("/status"):
        body = (
            b'{"device":"edge-lab-sensor","model":"EL-2048",'
            b'"uptime":86400,"temperature":23.7,"humidity":48.2,"relay":false}\n'
        )
        content_type = "application/json"
    elif "onvif" in path.lower() or kind == "http_camera":
        body = (
            b"<?xml version=\"1.0\"?>\n"
            b"<Device><Manufacturer>EdgeLab</Manufacturer>"
            b"<Model>Cam-IR-720</Model><Firmware>1.0.7</Firmware></Device>\n"
        )
        content_type = "application/xml"
    elif kind == "http_printer":
        body = b"<html><title>IPP Printer</title><body>EdgeLab Printer Ready</body></html>\n"
        content_type = "text/html"
    elif kind == "http_wsd":
        body = b"<?xml version=\"1.0\"?><wsd:Device>EdgeLab WSD Printer</wsd:Device>\n"
        content_type = "application/soap+xml"
    elif path.startswith("/login"):
        body = (
            b"<html><title>Device Login</title><body>"
            b"<form method='post'><input name='username'><input name='password' type='password'></form>"
            b"</body></html>\n"
        )
        content_type = "text/html"
    else:
        body = (
            f"<html><title>EdgeLab IoT Device</title><body>"
            f"<h1>EdgeLab IoT Gateway</h1><p>port={port}</p>"
            f"<a href='/status'>status</a> <a href='/login'>login</a>"
            f"</body></html>\n"
        ).encode("ascii")
        content_type = "text/html"

    status = "HTTP/1.1 200 OK"
    response_headers = headers + [f"Content-Type: {content_type}", f"Content-Length: {len(body)}"]
    return (status + "\r\n" + "\r\n".join(response_headers) + "\r\n\r\n").encode("ascii") + body


def rtsp_response(data: bytes) -> bytes:
    text = data.decode("iso-8859-1", errors="ignore")
    cseq = "1"
    for line in text.splitlines():
        if line.lower().startswith("cseq:"):
            cseq = line.split(":", 1)[1].strip() or "1"
            break
    return (
        "RTSP/1.0 200 OK\r\n"
        f"CSeq: {cseq}\r\n"
        "Server: EdgeLab-Cam/1.2\r\n"
        "Public: OPTIONS, DESCRIBE, SETUP, TEARDOWN, PLAY\r\n\r\n"
    ).encode("ascii")


def modbus_response(data: bytes) -> bytes:
    if len(data) < 8:
        return b""
    transaction = data[0:2]
    protocol = b"\x00\x00"
    unit = data[6:7]
    function = data[7] | 0x80
    payload = unit + bytes([function, 0x01])
    return transaction + protocol + len(payload).to_bytes(2, "big") + payload


def mqtt_response(data: bytes) -> bytes:
    if not data:
        return b""
    packet_type = data[0] >> 4
    if packet_type == 1:
        return b"\x20\x02\x00\x00"
    if packet_type == 12:
        return b"\xd0\x00"
    return b""


def opcua_response() -> bytes:
    return b"ACKF\x1c\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\xff\xff\x00\x00\xff\xff\x00\x00\x00\x00\x00\x00"


def tcp_payload(service: TcpService, data: bytes) -> bytes:
    if service.kind == "ftp":
        return b"220 EdgeLab IoT FTP update service ready\r\n"
    if service.kind == "telnet":
        return b"EdgeLab embedded Linux\r\nlogin: "
    if service.kind.startswith("http"):
        return http_response(service.kind, data, service.port)
    if service.kind == "rtsp":
        return rtsp_response(data)
    if service.kind == "modbus":
        return modbus_response(data)
    if service.kind == "mqtt":
        return mqtt_response(data)
    if service.kind == "opcua":
        return opcua_response()
    if service.kind == "banner":
        return f"{service.name} EdgeLab device service ready\r\n".encode("ascii")
    return b""


def run_tcp_service(host: str, service: TcpService, quiet: bool, stop: threading.Event) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, service.port))
        sock.listen(200)
        sock.settimeout(1.0)
    except OSError as exc:
        print(f"WARN tcp/{service.port} {service.name}: {exc}", flush=True)
        sock.close()
        return

    print(f"OPEN tcp/{service.port:<5} {service.name}", flush=True)
    count = 0
    while not stop.is_set():
        try:
            conn, addr = sock.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        count += 1
        threading.Thread(
            target=handle_tcp_client,
            args=(conn, addr, service, quiet, count),
            daemon=True,
        ).start()
    sock.close()


def handle_tcp_client(conn: socket.socket, addr: tuple[str, int], service: TcpService, quiet: bool, count: int) -> None:
    try:
        conn.settimeout(2.0)
        if service.kind == "ftp":
            handle_ftp_client(conn)
        elif service.kind == "telnet":
            handle_telnet_client(conn)
        else:
            try:
                data = conn.recv(4096)
            except socket.timeout:
                data = b""
            payload = tcp_payload(service, data)
            if payload:
                conn.sendall(payload)
        if not quiet and (count <= 5 or count % 100 == 0):
            print(f"tcp/{service.port} {service.name}: hit {count} from {addr[0]}:{addr[1]}", flush=True)
    except OSError:
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


def read_line(conn: socket.socket) -> bytes:
    data = b""
    while len(data) < 512:
        chunk = conn.recv(1)
        if not chunk:
            break
        data += chunk
        if chunk in {b"\n", b"\r"}:
            break
    return data


def handle_ftp_client(conn: socket.socket) -> None:
    conn.sendall(b"220 EdgeLab IoT FTP update service ready\r\n")
    read_line(conn)
    conn.sendall(b"331 Password required\r\n")
    read_line(conn)
    conn.sendall(b"530 Login incorrect\r\n")


def handle_telnet_client(conn: socket.socket) -> None:
    conn.sendall(b"EdgeLab embedded Linux\r\nlogin: ")
    read_line(conn)
    conn.sendall(b"Password: ")
    read_line(conn)
    conn.sendall(b"\r\nLogin incorrect\r\n")


def udp_payload(service: UdpService, data: bytes, host: str) -> bytes:
    if service.kind == "ssdp":
        return (
            "HTTP/1.1 200 OK\r\n"
            "CACHE-CONTROL: max-age=120\r\n"
            f"LOCATION: http://{host}:8080/setup.xml\r\n"
            "SERVER: Linux/5.10 UPnP/1.0 EdgeLab-IoT/1.0\r\n"
            "ST: upnp:rootdevice\r\n"
            "USN: uuid:edge-lab-gateway::upnp:rootdevice\r\n\r\n"
        ).encode("ascii")
    if service.kind == "coap":
        message_id = data[2:4] if len(data) >= 4 else b"\x00\x01"
        return b"\x60\x45" + message_id + b"\xffok"
    if service.kind == "mdns":
        return b"\x00\x00\x84\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    if service.kind == "snmp":
        return b"0\x1d\x02\x01\x01\x04\x06public\xa2\x10\x02\x04\x00\x00\x00\x01\x02\x01\x00\x02\x01\x00\x30\x02\x05\x00"
    if service.kind == "tftp":
        return b"\x00\x05\x00\x04edge-lab tftp disabled\x00"
    if service.kind == "bacnet":
        return b"\x81\x0a\x00\x11\x01\x00\x30\x01\x0c\x0c\x02\x3f\xff\xff\x19\x4b"
    return b"ok"


def run_udp_service(host: str, service: UdpService, quiet: bool, stop: threading.Event) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, service.port))
        sock.settimeout(1.0)
    except OSError as exc:
        print(f"WARN udp/{service.port} {service.name}: {exc}", flush=True)
        sock.close()
        return

    print(f"OPEN udp/{service.port:<5} {service.name}", flush=True)
    count = 0
    while not stop.is_set():
        try:
            data, addr = sock.recvfrom(4096)
        except socket.timeout:
            continue
        except OSError:
            break
        count += 1
        payload = udp_payload(service, data, host if host != "0.0.0.0" else os.environ.get("TARGET_IP", "192.168.56.20"))
        if payload:
            try:
                sock.sendto(payload, addr)
            except OSError:
                pass
        if not quiet and (count <= 5 or count % 100 == 0):
            print(f"udp/{service.port} {service.name}: hit {count} from {addr[0]}:{addr[1]}", flush=True)
    sock.close()


def parse_port_filter(value: str) -> set[int] | None:
    if not value:
        return None
    ports: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ports.update(range(int(start), int(end) + 1))
        else:
            ports.add(int(part))
    return ports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--tcp-ports", default="", help="Optional comma/range filter, e.g. 80,502,1883,8080.")
    parser.add_argument("--udp-ports", default="", help="Optional comma/range filter, e.g. 161,1900,5683.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tcp_filter = parse_port_filter(args.tcp_ports)
    udp_filter = parse_port_filter(args.udp_ports)
    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    threads: list[threading.Thread] = []
    for service in TCP_SERVICES:
        if tcp_filter is not None and service.port not in tcp_filter:
            continue
        thread = threading.Thread(target=run_tcp_service, args=(args.host, service, args.quiet, stop), daemon=True)
        thread.start()
        threads.append(thread)
    for service in UDP_SERVICES:
        if udp_filter is not None and service.port not in udp_filter:
            continue
        thread = threading.Thread(target=run_udp_service, args=(args.host, service, args.quiet, stop), daemon=True)
        thread.start()
        threads.append(thread)

    print("IoT lab services running. Press Ctrl-C to stop.", flush=True)
    while not stop.is_set():
        time.sleep(0.5)
    print("Stopping IoT lab services.", flush=True)


if __name__ == "__main__":
    main()
