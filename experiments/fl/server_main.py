from __future__ import annotations

import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

from pathlib import Path

import flwr as fl

from fl.data import build_test_loader, load_cifar10_test
from fl.model import CNN3
from fl.seed import set_seed
from fl.server import AccuracyLogger, make_evaluate_fn, make_strategy


def main() -> None:
    seed = int(os.environ.get("SEED", "42"))
    num_clients = int(os.environ.get("NUM_CLIENTS", "8"))
    rounds = int(os.environ.get("ROUNDS", "500"))
    data_root = os.environ.get("DATA_ROOT", "/data")
    out_path = Path(os.environ.get("OUT_PATH", "/app/logs/accuracy.csv"))
    server_address = os.environ.get("SERVER_ADDRESS", "0.0.0.0:9191")

    set_seed(seed)

    test_set = load_cifar10_test(data_root)
    test_loader = build_test_loader(test_set, batch_size=256)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    accuracy_logger = AccuracyLogger(out_path)

    initial_model = CNN3()
    initial_params = [v.cpu().numpy() for v in initial_model.state_dict().values()]

    strategy = make_strategy(
        initial_params,
        make_evaluate_fn(test_loader, accuracy_logger),
        num_clients,
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
