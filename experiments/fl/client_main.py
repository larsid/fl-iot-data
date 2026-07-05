from __future__ import annotations

import os

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import socket
import time
from pathlib import Path

import flwr as fl

from fl.client import TorchClient
from fl.data import (
    PartitionSpec,
    build_client_loader,
    build_client_partitions,
    load_cifar10_train,
)
from fl.seed import set_seed


def wait_for_server(host: str, port: int, timeout_s: int = 120) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                print(f"[client] server {host}:{port} reachable", flush=True)
                return
        except OSError as err:
            last_err = err
            time.sleep(1.0)
    raise RuntimeError(
        f"timed out waiting for {host}:{port} after {timeout_s}s ({last_err})"
    )


def main() -> None:
    seed = int(os.environ.get("SEED", "42"))
    client_id = int(os.environ.get("CLIENT_ID", "1"))
    num_clients = int(os.environ.get("NUM_CLIENTS", "8"))
    local_epochs = int(os.environ.get("LOCAL_EPOCHS", "2"))
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    lr = float(os.environ.get("LR", "0.01"))
    data_root = os.environ.get("DATA_ROOT", "/data")
    log_path = Path(os.environ.get("LOG_PATH", "/app/logs/clients.jsonl"))
    server_ip = os.environ.get("FL_SERVER_IP")
    server_port = os.environ.get("FL_SERVER_PORT", "9191")

    if server_ip is None:
        raise RuntimeError("FL_SERVER_IP env var required")
    server_address = f"{server_ip}:{server_port}"

    wait_for_server(server_ip, int(server_port), timeout_s=120)

    set_seed(seed)

    train_set = load_cifar10_train(data_root)
    spec = PartitionSpec(num_partitions=64, num_clients=num_clients)
    partitions = build_client_partitions(train_set, seed=seed, spec=spec)

    idx = client_id - 1
    loader = build_client_loader(train_set, partitions[idx], batch_size)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    client = TorchClient(
        client_id=client_id,
        train_loader=loader,
        local_epochs=local_epochs,
        lr=lr,
        log_path=log_path,
    ).to_client()

    fl.client.start_client(server_address=server_address, client=client)


if __name__ == "__main__":
    main()
