#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
import time
import math

import flwr as fl
import torch

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from federated_common import (
    DATA_DIR,
    DEVICE_FILES,
    WEIGHTS_DIR,
    build_model_from_params,
    evaluate_global,
    get_device,
    get_initial_parameters,
)

# Server config
SERVER_ADDRESS = "127.0.0.1:8080"
NUM_ROUNDS = 180
ROUND_SLEEP_SEC = 10


def main() -> None:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    profile_path = os.path.join(DATA_DIR, "client_profile.json")
    if not os.path.exists(profile_path):
        profile = {
            "plan": "Fiber Secure",
            "next_bill": "2026-02-15",
            "due_date": "2026-02-20",
            "billing_cycle": "Monthly",
            "router_model": "ISP-Edge-R1",
            "uptime": 99.2,
            "speed_consistency": 97.8,
        }
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f)

    class ISPFedAvg(fl.server.strategy.FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            aggregated = super().aggregate_fit(server_round, results, failures)
            if results:
                anomalies_by_router = {}
                for _, fit_res in results:
                    metrics = fit_res.metrics or {}
                    router_id = metrics.get("router_id")
                    count = metrics.get("anomalies_count")
                    if router_id is not None and count is not None:
                        anomalies_by_router[router_id] = int(count)

                if anomalies_by_router:
                    payload = {
                        "round": server_round,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                        "anomalies_by_router": anomalies_by_router,
                        "total_anomalies": int(sum(anomalies_by_router.values())),
                    }
                    metrics_path = os.path.join(
                        WEIGHTS_DIR, f"round_{server_round:03d}_metrics.json"
                    )
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f)
            return aggregated

    def _evaluate_and_save(server_round: int, params, config):
        model = build_model_from_params(params)
        weights_path = os.path.join(WEIGHTS_DIR, f"round_{server_round:03d}.pt")
        torch.save(model.state_dict(), weights_path)
        time.sleep(ROUND_SLEEP_SEC)
        return 0.0, evaluate_global(model, get_device())[1]

    initial_parameters = fl.common.ndarrays_to_parameters(get_initial_parameters())

    total_clients = len(DEVICE_FILES)
    min_clients = max(1, math.ceil(total_clients / 3))

    strategy = ISPFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
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
