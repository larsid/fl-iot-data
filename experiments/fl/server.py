from __future__ import annotations

import csv
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import NDArrays, Scalar
from torch.utils.data import DataLoader

from fl.model import CNN3


class AccuracyLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        with self.path.open("w", newline="") as fp:
            csv.writer(fp).writerow(
                ["round", "test_loss", "test_accuracy", "eval_time_s"]
            )

    def write(self, rnd: int, loss: float, acc: float, eval_time_s: float) -> None:
        with self.path.open("a", newline="") as fp:
            csv.writer(fp).writerow(
                [rnd, f"{loss:.6f}", f"{acc:.6f}", f"{eval_time_s:.6f}"]
            )


def make_evaluate_fn(test_loader: DataLoader, logger: AccuracyLogger):
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model = CNN3()

    def evaluate(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        t0 = time.perf_counter()
        state = OrderedDict(
            (k, torch.tensor(v)) for k, v in zip(model.state_dict().keys(), parameters)
        )
        model.load_state_dict(state, strict=True)
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                logits = model(inputs)
                total_loss += criterion(logits, targets).item()
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        loss = total_loss / total
        acc = correct / total
        eval_time_s = time.perf_counter() - t0
        logger.write(server_round, loss, acc, eval_time_s)
        return loss, {"accuracy": acc}

    return evaluate


def fit_config(server_round: int) -> Dict[str, Scalar]:
    return {"round": server_round}


def _noop_fit_metrics(results) -> Dict[str, Scalar]:
    return {}


def make_strategy(
    initial_parameters: List[np.ndarray],
    evaluate_fn,
    num_clients: int,
) -> fl.server.strategy.FedAvg:
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=_noop_fit_metrics,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
    )
