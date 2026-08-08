"""Latest custom Zeek dataset builder wrapper."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / "build_zeek_crossval_dataset.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
