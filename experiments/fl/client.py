from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fl.model import CNN3


class TorchClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        local_epochs: int,
        lr: float,
        log_path: Path | None,
    ) -> None:
        self.client_id = client_id
        self.train_loader = train_loader
        self.local_epochs = local_epochs
        self.lr = lr
        self.model = CNN3()
        self.criterion = nn.CrossEntropyLoss()
        self.log_path = log_path
        self._previous_fit_end_ts: float | None = None

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        state = OrderedDict(
            (k, torch.tensor(v))
            for k, v in zip(self.model.state_dict().keys(), parameters)
        )
        self.model.load_state_dict(state, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        round_num = int(config.get("round", -1))
        fit_start_ts = time.time()
        deser_start_perf = time.perf_counter()

        self.set_parameters(parameters)
        training_start_ts = time.time()
        training_start_perf = time.perf_counter()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.local_epochs):
            for inputs, targets in self.train_loader:
                optimizer.zero_grad()
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                loss.backward()
                optimizer.step()

        training_end_perf = time.perf_counter()
        training_end_ts = time.time()
        new_params = self.get_parameters({})
        ser_end_perf = time.perf_counter()
        fit_end_ts = time.time()

        record: Dict[str, float | int | None] = {
            "round": round_num,
            "client_id": self.client_id,
            "fit_start_ts": fit_start_ts,
            "training_start_ts": training_start_ts,
            "training_end_ts": training_end_ts,
            "fit_end_ts": fit_end_ts,
            "training_time_s": training_end_perf - training_start_perf,
            "local_serde_time_s": (
                (training_start_perf - deser_start_perf)
                + (ser_end_perf - training_end_perf)
            ),
            "update_exchange_time_s": (
                fit_start_ts - self._previous_fit_end_ts
                if self._previous_fit_end_ts is not None
                else None
            ),
            "samples": len(self.train_loader.dataset),
        }
        self._previous_fit_end_ts = fit_end_ts

        if self.log_path is not None:
            with self.log_path.open("a") as fp:
                fp.write(json.dumps(record) + "\n")

        return new_params, int(record["samples"]), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        return 0.0, 0, {}
