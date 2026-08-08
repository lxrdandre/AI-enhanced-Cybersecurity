"""TON-IoT SE-DWNet trainer wrapper.

This delegates to the local one-file trainer in this folder while providing a
canonical dataset-specific path:

    python -m resnet.ton_iot ...
    python resnet/ton_iot/train.py ...
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / "train_onefile.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
