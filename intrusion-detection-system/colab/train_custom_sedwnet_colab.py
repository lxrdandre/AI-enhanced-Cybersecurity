"""Colab launcher for the custom Zeek SE-DWNet trainer.

This script is intentionally thin: it reuses the maintained one-file trainer in
``resnet/resnet_zeek_crossval.py`` and only injects Colab-friendly defaults.
Override any default by passing the normal trainer flag.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESNET_DIR = REPO_ROOT / "resnet"
DEFAULT_CSV = "/content/drive/MyDrive/thesis_ids/data/zeek_crossval.csv"
DEFAULT_OUTPUT_DIR = "/content/drive/MyDrive/thesis_ids/artifacts/se_dwnet_zeek_crossval_colab"


def _add_repo_paths() -> None:
    for path in (str(REPO_ROOT), str(RESNET_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _has_arg(argv: list[str], flag: str) -> bool:
    return flag in argv or any(arg.startswith(flag + "=") for arg in argv)


def _inject_default(argv: list[str], flag: str, value: str) -> None:
    if not _has_arg(argv, flag):
        argv.extend([flag, value])


def main() -> None:
    _add_repo_paths()

    argv = sys.argv[1:]
    _inject_default(argv, "--csv", DEFAULT_CSV)
    _inject_default(argv, "--output-dir", DEFAULT_OUTPUT_DIR)
    _inject_default(argv, "--split", "random")
    _inject_default(argv, "--final-holdout-size", "0.05")
    _inject_default(argv, "--final-holdout-mode", "random")
    _inject_default(argv, "--target-k", "192")
    _inject_default(argv, "--smote", "auto")
    _inject_default(argv, "--loss", "ce")
    _inject_default(argv, "--batch-size", "1024")
    _inject_default(argv, "--max-epochs", "100")

    sys.argv = [sys.argv[0], *argv]
    from resnet_zeek_crossval import main as trainer_main

    trainer_main()


if __name__ == "__main__":
    main()
