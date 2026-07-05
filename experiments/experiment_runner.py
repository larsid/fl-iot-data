from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from fogbed import Container, FogbedDistributedExperiment

from config import (
    CLIENT_IMAGE,
    FL_PARAMS,
    FL_SERVER_PORT,
    NUM_CLIENTS,
    SERVER_BANDWIDTH,
    SERVER_IMAGE,
    SERVER_RESOURCES,
    WORKER_IP,
    WORKER_PORT,
)

REPO_DIR = Path(__file__).resolve().parent
COLLECTOR_IMAGE = os.environ.get("COLLECTOR_IMAGE", "fl-base:1.13.1")
COLLECTOR_NAME = "fl_collector"
COLLECTOR_CPUSET = os.environ.get("COLLECTOR_CPUSET", "0-3")


def _start_collector(log_dir: Path) -> None:
    subprocess.run(
        ["docker", "rm", "-f", COLLECTOR_NAME],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cmd = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        COLLECTOR_NAME,
        "--entrypoint",
        "",
        "--cpuset-cpus",
        COLLECTOR_CPUSET,
        "-v",
        "/var/run/docker.sock:/var/run/docker.sock:ro",
        "-v",
        "/sys/fs/cgroup:/sys/fs/cgroup:ro",
        "-v",
        f"{REPO_DIR}:/app",
        "-v",
        f"{log_dir}:/app/logs",
        "-w",
        "/app",
        COLLECTOR_IMAGE,
        "python",
        "-u",
        "collector/collect.py",
        "--out",
        "/app/logs/stats.csv",
        "--interval",
        "1",
        "--max-stalls",
        "300",
    ]
    subprocess.run(cmd, check=True)
    print(f"--- collector started ({COLLECTOR_NAME} on cpus {COLLECTOR_CPUSET}) ---")


def _stop_collector() -> None:
    subprocess.run(
        ["docker", "rm", "-f", COLLECTOR_NAME],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _list_fl_containers() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", r"name=^mn\."],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.split()


def _list_running_fl_containers() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "-q", "--filter", r"name=^mn\."],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.split()


def _server_container_id() -> str | None:
    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", r"name=^mn\.server$"],
        capture_output=True,
        text=True,
        check=False,
    )
    ids = result.stdout.split()
    return ids[0] if ids else None


def _force_remove(container_ids: list[str]) -> None:
    if not container_ids:
        return
    subprocess.run(
        ["docker", "rm", "-f", *container_ids],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_completion(expected_containers: int) -> None:
    if os.environ.get("INTERACTIVE", "0") == "1":
        input("Experimento em execucao. Pressione Enter para encerrar... ")
        return
    timeout_s = int(os.environ.get("WAIT_TIMEOUT_S", "86400"))
    grace_s = int(os.environ.get("CLIENT_EXIT_GRACE_S", "60"))

    spawn_deadline = time.monotonic() + 120
    while time.monotonic() < spawn_deadline:
        container_ids = _list_fl_containers()
        if len(container_ids) >= expected_containers:
            break
        time.sleep(1)
    else:
        container_ids = _list_fl_containers()
        print(
            f"!!! only {len(container_ids)}/{expected_containers} mn.* containers"
            f" visible after 120s; proceeding anyway",
            flush=True,
        )

    if not container_ids:
        raise RuntimeError(
            "no mn.* containers found after exp.start(); "
            "Fogbed/Containernet failed to spawn the topology"
        )

    server_id = _server_container_id()
    if server_id is None:
        print("!!! mn.server not found; waiting on all mn.* (bounded)", flush=True)
        try:
            subprocess.run(
                ["docker", "wait", *container_ids],
                check=False,
                stdout=subprocess.DEVNULL,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            print(f"!!! timeout after {timeout_s}s; proceeding to cleanup", flush=True)
        return

    print(
        f"--- waiting for FL server (mn.server) to exit "
        f"(timeout={timeout_s}s; set INTERACTIVE=1 to wait for Enter) ---"
    )
    try:
        subprocess.run(
            ["docker", "wait", server_id],
            check=False,
            stdout=subprocess.DEVNULL,
            timeout=timeout_s,
        )
        print("--- FL server exited; FL run complete ---")
    except subprocess.TimeoutExpired:
        print(
            f"!!! timeout after {timeout_s}s waiting for the server to exit;"
            f" proceeding to cleanup",
            flush=True,
        )
        return

    if grace_s > 0:
        print(f"--- grace period {grace_s}s for clients to flush and exit ---")
        time.sleep(grace_s)

    stuck = _list_running_fl_containers()
    if stuck:
        print(f"--- force-removing {len(stuck)} stuck mn.* container(s) ---")
        _force_remove(stuck)
    else:
        print("--- all FL containers exited cleanly ---")


def run_experiment(
    *,
    level_label: str,
    client_resources: Mapping[str, Any],
    client_bandwidth: Mapping[str, Any],
    controlled: bool,
) -> None:
    rounds = int(os.environ.get("ROUNDS", "500"))
    seed = int(os.environ.get("SEED", "42"))

    bw_override = os.environ.get("BW")
    if bw_override is not None and not controlled:
        raise SystemExit(
            f"BW={bw_override} is not valid at Level 0 (controlled=False); "
            f"the client link is unshaped by design"
        )
    if bw_override is not None:
        client_bandwidth = {"bw": int(bw_override)}

    suffix = f"_bw{bw_override}" if bw_override else ""
    log_subdir = f"level{level_label}{suffix}"
    log_dir = REPO_DIR.parent / "results" / log_subdir / time.strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    shared_env = {
        "PYTHONPATH": "/app",
        "SEED": str(seed),
        "NUM_CLIENTS": str(NUM_CLIENTS),
        "PYTHONHASHSEED": str(seed),
    }
    server_thread_cap = str(SERVER_RESOURCES["cpus"])
    server_env = {
        **shared_env,
        "ROUNDS": str(rounds),
        "OUT_PATH": "/app/logs/accuracy.csv",
        "SERVER_ADDRESS": f"0.0.0.0:{FL_SERVER_PORT}",
        "OMP_NUM_THREADS": server_thread_cap,
        "MKL_NUM_THREADS": server_thread_cap,
    }
    common_volumes = [
        f"{REPO_DIR}:/app",
        f"{log_dir}:/app/logs",
    ]

    manifest = {
        "level": level_label,
        "controlled": controlled,
        "rounds": rounds,
        "seed": seed,
        "num_clients": NUM_CLIENTS,
        "fl_params": dict(FL_PARAMS),
        "client_resources": dict(client_resources),
        "client_bandwidth": dict(client_bandwidth),
        "server_resources": dict(SERVER_RESOURCES),
        "server_bandwidth": dict(SERVER_BANDWIDTH),
        "images": {
            "server": SERVER_IMAGE,
            "client": CLIENT_IMAGE,
            "collector": COLLECTOR_IMAGE,
        },
        "env_overrides": {
            "pythonhashseed": str(seed),
            "server_omp_num_threads": server_thread_cap,
            "server_mkl_num_threads": server_thread_cap,
            "client_omp_num_threads": "1",
            "client_mkl_num_threads": "1",
        },
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "collector": {
            "image": COLLECTOR_IMAGE,
            "cpuset_cpus": COLLECTOR_CPUSET,
            "interval_s": 1,
            "name_pattern": r"^mn\.(client_\d+|server)$",
        },
    }
    (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    exp = FogbedDistributedExperiment()
    cloud = exp.add_virtual_instance("cloud")
    edge = exp.add_virtual_instance("edge")

    server_kwargs: dict[str, Any] = {
        "name": "server",
        "dimage": SERVER_IMAGE,
        "dcmd": "/entrypoint.sh python -m fl.server_main",
        "environment": server_env,
        "volumes": common_volumes,
        "cap_add": ["NET_ADMIN"],
        "link_params": dict(SERVER_BANDWIDTH),
        **SERVER_RESOURCES,
    }
    server = Container(**server_kwargs)
    exp.add_docker(server, cloud)

    for i in range(1, NUM_CLIENTS + 1):
        client_env = {
            **shared_env,
            "CLIENT_ID": str(i),
            "FL_SERVER_IP": server.ip,
            "FL_SERVER_PORT": str(FL_SERVER_PORT),
            "LOG_PATH": f"/app/logs/client_{i}.jsonl",
            "LOCAL_EPOCHS": str(FL_PARAMS["local_epochs"]),
            "BATCH_SIZE": str(FL_PARAMS["batch_size"]),
            "LR": str(FL_PARAMS["learning_rate"]),
        }
        client_env["OMP_NUM_THREADS"] = "1"
        client_env["MKL_NUM_THREADS"] = "1"
        client_kwargs: dict[str, Any] = {
            "name": f"client_{i}",
            "dimage": CLIENT_IMAGE,
            "dcmd": "/entrypoint.sh python -m fl.client_main",
            "environment": client_env,
            "volumes": common_volumes,
            "cap_add": ["NET_ADMIN"],
        }
        if controlled:
            client_kwargs["cpuset_cpus"] = str(3 + i)
        if client_bandwidth:
            client_kwargs["link_params"] = dict(client_bandwidth)
        client_kwargs.update(client_resources)
        client = Container(**client_kwargs)
        exp.add_docker(client, edge)

    worker = exp.add_worker(ip=WORKER_IP, port=WORKER_PORT)
    worker.add(cloud, reachable=True)
    worker.add(edge)
    worker.add_link(cloud, edge)

    print(
        f"--- Nivel {level_label}: rounds={rounds}, seed={seed}, controlled={controlled}, logs={log_dir} ---"
    )
    expected_containers = 1 + NUM_CLIENTS
    try:
        _start_collector(log_dir)
        exp.start()
        _wait_for_completion(expected_containers)
    finally:
        _stop_collector()
        try:
            exp.stop()
        except Exception as cleanup_err:
            print(f"!!! exp.stop() failed during cleanup: {cleanup_err}", flush=True)
