#!/usr/bin/env python3


from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from federated_common import (
    DEVICE_FILES,
    LOCAL_EPOCHS,
    NORMAL_LABEL,
    Autoencoder,
    get_dataloader,
    get_device,
    get_model_parameters,
    load_features,
    set_model_parameters,
    train_local,
)

SERVER_ADDRESS = "127.0.0.1:8080"
CLIENT_ID = int(os.getenv("CLIENT_ID", "2"))


class AutoencoderClient(fl.client.NumPyClient):
    def __init__(self, train_file: str, input_dim: int, device: torch.device) -> None:
        self.train_file = train_file
        self.device = device
        self.model = Autoencoder(input_dim=input_dim).to(device)

        x_train, y_train, _ = load_features(train_file)
        normal_mask = y_train == str(NORMAL_LABEL)
        self.x_train = x_train[normal_mask]

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        set_model_parameters(self.model, parameters)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        self.set_parameters(parameters)
        loader = get_dataloader(self.x_train, shuffle=True)
        epoch_losses = []
        for _ in range(LOCAL_EPOCHS):
            epoch_losses.append(train_local(self.model, loader, self.device))

        print("Client update: sending model weights only (no raw data).")
        return (
            get_model_parameters(self.model),
            len(self.x_train),
            {"loss": float(np.mean(epoch_losses))},
        )


def main() -> None:
    device = get_device()
    train_file = DEVICE_FILES[CLIENT_ID]
    x_train, _, _ = load_features(train_file)
    client = AutoencoderClient(
        train_file=train_file, input_dim=x_train.shape[1], device=device
    )

    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
