"""CIC public SE-DWNet trainer wrapper."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / "train_onefile.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
