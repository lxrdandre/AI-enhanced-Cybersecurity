#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import time

import flwr as fl
import torch

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from federated_common import (
    WEIGHTS_DIR,
    build_model_from_params,
    evaluate_global,
    get_device,
    get_initial_parameters,
)

# Server config
SERVER_ADDRESS = "127.0.0.1:8080"
NUM_ROUNDS = 180
ROUND_SLEEP_SEC = 60


def main() -> None:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    def _evaluate_and_save(server_round: int, params, config):
        model = build_model_from_params(params)
        weights_path = os.path.join(WEIGHTS_DIR, f"round_{server_round:03d}.pt")
        torch.save(model.state_dict(), weights_path)
        time.sleep(ROUND_SLEEP_SEC)
        return 0.0, evaluate_global(model, get_device())[1]

    initial_parameters = fl.common.ndarrays_to_parameters(get_initial_parameters())

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_fn=_evaluate_and_save,
        initial_parameters=initial_parameters,
    )

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
