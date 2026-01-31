#!/usr/bin/env python3

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Paths
DATA_DIR = os.getenv("FEDLEARNING_DATA_DIR", "/ton-iot-project/fedlearning/data/processed")
DEVICE_FILES = [
    os.path.join(DATA_DIR, "device_1.csv"),
    os.path.join(DATA_DIR, "device_2.csv"),
    os.path.join(DATA_DIR, "device_3.csv"),
]
TRAIN_FILE = os.path.join(DATA_DIR, "train_supervised.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_supervised.csv")

# Data/label config
LABEL_COL = "label"

# Training config (router-optimized)
BATCH_SIZE = 64
LOCAL_EPOCHS = 1
LR = 5e-4
THRESHOLD = 0.5
RANDOM_STATE = 42

WEIGHTS_DIR = os.path.join(DATA_DIR, "global_weights")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def load_features(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column '{LABEL_COL}' in {path}")
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    x = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].astype(int).to_numpy()
    return x, y, feature_cols


def get_dataloader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_local(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optim.zero_grad()
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x_batch.size(0)
    return total_loss / len(loader.dataset)


def predict_proba(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            prob = torch.sigmoid(logits)
            probs.append(prob.detach().cpu().numpy())
    return np.concatenate(probs, axis=0)


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, params: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    new_state = {
        k: torch.tensor(v, dtype=state_dict[k].dtype) for k, v in zip(keys, params)
    }
    model.load_state_dict(new_state, strict=True)


def evaluate_global(model: nn.Module, device: torch.device) -> Tuple[float, Dict[str, float]]:
    x_train, y_train, _ = load_features(TRAIN_FILE)
    x_test, y_test, _ = load_features(TEST_FILE)

    train_loader = get_dataloader(x_train, y_train, shuffle=False)
    test_loader = get_dataloader(x_test, y_test, shuffle=False)

    test_probs = predict_proba(model, test_loader, device)
    preds = (test_probs > THRESHOLD).astype(int)

    y_true = y_test
    acc = float((preds == y_true).mean())
    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    fnr = fn / (fn + tp + 1e-8)

    return 0.0, {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "threshold": float(THRESHOLD),
    }


def build_model_from_params(params: List[np.ndarray]) -> nn.Module:
    x_train, _, _ = load_features(DEVICE_FILES[0])
    model = MLPClassifier(input_dim=x_train.shape[1]).to(get_device())
    set_model_parameters(model, params)
    return model


def get_initial_parameters() -> List[np.ndarray]:
    x_train, _, _ = load_features(DEVICE_FILES[0])
    model = MLPClassifier(input_dim=x_train.shape[1]).to(get_device())
    return get_model_parameters(model)
