"""Modular training entry point for Edge-IIoTset SE-DWNet.

The original single-file implementation remains at resnet/resnet_edge_iiotset.py.
This module delegates to that battle-tested trainer, then normalizes artifact
names so deployment uses SE-DWNet terminology.
"""

from __future__ import annotations

import argparse
import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
RESNET_DIR = os.path.normpath(os.path.join(PACKAGE_DIR, ".."))
if RESNET_DIR not in sys.path:
    sys.path.insert(0, RESNET_DIR)

try:
    from .artifacts import normalize_artifact_names  # type: ignore  # noqa: E402
    from .config import default_output_dir  # type: ignore  # noqa: E402
except ImportError:
    if PACKAGE_DIR not in sys.path:
        sys.path.insert(0, PACKAGE_DIR)
    from artifacts import normalize_artifact_names  # type: ignore  # noqa: E402
    from config import default_output_dir  # type: ignore  # noqa: E402
from resnet_edge_iiotset import main as legacy_main  # type: ignore  # noqa: E402


def _extract_output_dir(argv: list[str]) -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", default=None)
    args, _ = parser.parse_known_args(argv)
    return args.output_dir


def main() -> None:
    argv = sys.argv[1:]
    output_dir = _extract_output_dir(argv)
    if output_dir is None:
        output_dir = default_output_dir()
        sys.argv.extend(["--output-dir", output_dir])

    legacy_main()
    artifacts = normalize_artifact_names(output_dir)
    print("\nSE-DWNet artifact aliases:")
    for key, value in artifacts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
