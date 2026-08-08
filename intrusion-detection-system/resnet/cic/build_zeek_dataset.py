"""Public PCAP-to-Zeek dataset builder wrapper for CIC/Edge-style public PCAPs."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / "build_cic_zeek_dataset.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
