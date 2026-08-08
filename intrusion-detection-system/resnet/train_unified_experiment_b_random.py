from __future__ import annotations

from unified_sedwnet_experiment import build_parser, config_from_args, run_experiment


def main() -> None:
    parser = build_parser("artifacts/resnet_unified_7class_random")
    args = parser.parse_args()
    run_experiment(config_from_args(args, split_mode="random"))


if __name__ == "__main__":
    main()
