from __future__ import annotations

from clawdbot.flow_roles import is_protected_ip, normalize_flow_roles, parse_ip_csv


def test_parse_ip_csv_trims_and_skips_empty_values():
    """Verify that parse ip csv trims and skips empty values."""
    assert parse_ip_csv(" 10.0.0.5, ,192.0.2.10 ") == frozenset({"10.0.0.5", "192.0.2.10"})


def test_is_protected_ip_matches_configured_server():
    """Verify that is protected ip matches configured server."""
    assert is_protected_ip("100.111.77.70", {"100.111.77.70"})
    assert not is_protected_ip("203.0.113.50", {"100.111.77.70"})


def test_normalize_keeps_client_to_server_flow():
    """Verify that normalize keeps client to server flow."""
    roles = normalize_flow_roles({
        "src_ip": "192.0.2.10",
        "dst_ip": "10.0.0.5",
        "src_port": 55222,
        "dst_port": 80,
    })

    assert roles["originator_ip"] == "192.0.2.10"
    assert roles["target_ip"] == "10.0.0.5"
    assert roles["target_port"] == 80
    assert roles["direction"] == "request"


def test_normalize_swaps_server_response_by_ports():
    """Verify that normalize swaps server response by ports."""
    roles = normalize_flow_roles({
        "src_ip": "10.0.0.5",
        "dst_ip": "192.0.2.10",
        "src_port": 80,
        "dst_port": 55222,
    })

    assert roles["originator_ip"] == "192.0.2.10"
    assert roles["target_ip"] == "10.0.0.5"
    assert roles["target_port"] == 80
    assert roles["direction"] == "response"


def test_protected_ip_overrides_ambiguous_ports():
    """Verify that protected ip overrides ambiguous ports."""
    roles = normalize_flow_roles(
        {
            "src_ip": "10.0.0.5",
            "dst_ip": "192.0.2.10",
            "src_port": 0,
            "dst_port": 0,
        },
        protected_ips={"10.0.0.5"},
    )

    assert roles["originator_ip"] == "192.0.2.10"
    assert roles["target_ip"] == "10.0.0.5"
