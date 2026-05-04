"""Flow role helpers for attacker/target reporting.

Captured packet direction is not always the same thing as incident role:
server responses have the protected host as ``src_ip`` and the attacker as
``dst_ip``.  These helpers normalize a raw flow into originator/target roles
for alerting, dashboards, and response decisions.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

CLIENT_PORT_MIN = 32768


def parse_ip_csv(value: str | Iterable[str] | None) -> frozenset[str]:
    """Parse a comma-separated IP list into normalized strings."""
    if value is None:
        return frozenset()
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = value
    return frozenset(str(item).strip() for item in items if str(item).strip())


def is_protected_ip(addr: str | None, protected_ips: Iterable[str] | None) -> bool:
    """Return True when *addr* is one of the configured protected/server IPs."""
    if not addr:
        return False
    return str(addr).strip() in parse_ip_csv(protected_ips)


def _port_int(value: Any) -> int | None:
    """Parse a port value into an integer, or None when unavailable."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_client_port(port: int | None) -> bool:
    """Return True when a port looks like an ephemeral client port."""
    return port is not None and port >= CLIENT_PORT_MIN


def _looks_like_response(src_port: int | None, dst_port: int | None) -> bool:
    """Return True when raw src/dst likely represent server -> client."""
    return (
        src_port not in {None, 0}
        and dst_port not in {None, 0}
        and not _is_client_port(src_port)
        and _is_client_port(dst_port)
    )


def normalize_flow_roles(
    flow: dict[str, Any] | None,
    protected_ips: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return normalized originator/target fields for a raw flow dict.

    ``protected_ips`` is authoritative when supplied.  Otherwise, the fallback
    heuristic swaps flows that look like server responses: source port is a
    service/non-ephemeral port and destination port is an ephemeral client port.
    """
    flow = flow or {}
    protected = parse_ip_csv(protected_ips)
    src_ip = str(flow.get("src_ip") or "")
    dst_ip = str(flow.get("dst_ip") or "")
    src_port = _port_int(flow.get("src_port"))
    dst_port = _port_int(flow.get("dst_port"))

    reverse = False
    if protected:
        src_protected = src_ip in protected
        dst_protected = dst_ip in protected
        if src_protected and not dst_protected:
            reverse = True
        elif dst_protected and not src_protected:
            reverse = False
        else:
            reverse = _looks_like_response(src_port, dst_port)
    else:
        reverse = _looks_like_response(src_port, dst_port)

    if reverse:
        return {
            "originator_ip": dst_ip,
            "target_ip": src_ip,
            "originator_port": dst_port,
            "target_port": src_port,
            "direction": "response",
        }

    return {
        "originator_ip": src_ip,
        "target_ip": dst_ip,
        "originator_port": src_port,
        "target_port": dst_port,
        "direction": "request",
    }
