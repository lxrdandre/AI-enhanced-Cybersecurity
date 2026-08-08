"""Compatibility wrapper for the live SE-DWNet Edge-IIoTset classifier.

This file used to run a CIC-style live classifier. The project now uses the
random-holdout Edge-IIoTset SE-DWNet model as the active classifier, so this
wrapper forwards old commands to live_se_dwnet_edge_classifier.py.
"""

from __future__ import annotations

try:
    from .live_se_dwnet_edge_classifier import main
except ImportError:
    from live_se_dwnet_edge_classifier import main


if __name__ == "__main__":
    main()
