"""Unit tests for the preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from app.preprocessing import SafeLabelEncoder


class TestSafeLabelEncoder:
    def test_fit_transform_known_labels(self):
        enc = SafeLabelEncoder()
        enc.fit(["tcp", "udp", "icmp"])
        result = enc.transform(["tcp", "udp", "icmp"])
        assert result.dtype == np.int32
        assert set(result.tolist()) == {1, 2, 3}

    def test_unknown_label_maps_to_zero(self):
        enc = SafeLabelEncoder()
        enc.fit(["tcp", "udp"])
        result = enc.transform(["tcp", "unknown_proto"])
        assert result[1] == 0

    def test_deterministic_ordering(self):
        enc = SafeLabelEncoder()
        enc.fit(["b", "a", "c"])
        r1 = enc.transform(["a", "b", "c"])
        r2 = enc.transform(["a", "b", "c"])
        np.testing.assert_array_equal(r1, r2)

    def test_empty_fit(self):
        enc = SafeLabelEncoder()
        enc.fit([])
        result = enc.transform(["anything"])
        assert result[0] == 0
