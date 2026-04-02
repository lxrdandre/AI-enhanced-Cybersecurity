"""Shared fixtures for TON IoT IDS test suite."""

from __future__ import annotations

import numpy as np
import pytest


# ────────────────────────────────────────────────────────────
# Reusable sample records (valid network-flow feature dicts)
# ────────────────────────────────────────────────────────────

SAMPLE_NORMAL_RECORD = {
    "duration": 12,
    "src_bytes": 442,
    "dst_bytes": 1290,
    "proto": "tcp",
}

SAMPLE_ATTACK_RECORDS = [
    {
        "duration": 0,
        "src_bytes": 0,
        "dst_bytes": 0,
        "proto": "icmp",
    },
    {
        "duration": 999,
        "src_bytes": 100000,
        "dst_bytes": 50,
        "proto": "udp",
    },
]


@pytest.fixture()
def normal_record():
    return dict(SAMPLE_NORMAL_RECORD)


@pytest.fixture()
def attack_records():
    return [dict(r) for r in SAMPLE_ATTACK_RECORDS]


@pytest.fixture()
def batch_records():
    """10 random-ish records for batch / throughput testing."""
    rng = np.random.default_rng(42)
    protos = ["tcp", "udp", "icmp"]
    records = []
    for _ in range(10):
        records.append(
            {
                "duration": float(rng.integers(0, 5000)),
                "src_bytes": float(rng.integers(0, 100_000)),
                "dst_bytes": float(rng.integers(0, 100_000)),
                "proto": protos[int(rng.integers(0, len(protos)))],
            }
        )
    return records
