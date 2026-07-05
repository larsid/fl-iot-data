from __future__ import annotations

import argparse
import csv
import http.client
import json
import re
import socket
import sys
import time
from pathlib import Path
from typing import List, Tuple

CGROUP_BASE = Path("/sys/fs/cgroup")
DOCKER_SOCK = "/var/run/docker.sock"


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, sock_path: str) -> None:
        super().__init__("localhost")
        self.sock_path = sock_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.sock_path)


def _docker_api(path: str):
    conn = _UnixHTTPConnection(DOCKER_SOCK)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read().decode()
    conn.close()
    return json.loads(body)


def list_target_containers(pattern: re.Pattern) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for container in _docker_api("/containers/json?all=false"):
        for name in container.get("Names", []):
            n = name.lstrip("/")
            if pattern.match(n):
                out.append((container["Id"], n))
                break
    return out


_CGROUP_PATH_CACHE: dict[str, Path] = {}


def cgroup_path(container_id: str) -> Path:
    cached = _CGROUP_PATH_CACHE.get(container_id)
    if cached is not None:
        return cached
    info = _docker_api(f"/containers/{container_id}/json")
    parent = info.get("HostConfig", {}).get("CgroupParent") or "system.slice"
    path = CGROUP_BASE / parent.lstrip("/") / f"docker-{container_id}.scope"
    _CGROUP_PATH_CACHE[container_id] = path
    return path


def read_cpu_usec(cgroup: Path) -> int:
    with (cgroup / "cpu.stat").open() as fp:
        for line in fp:
            if line.startswith("usage_usec"):
                return int(line.split()[1])
    return 0


def read_memory_bytes(cgroup: Path) -> int:
    return int((cgroup / "memory.current").read_text().strip())


def read_memory_limit(cgroup: Path) -> int:
    val = (cgroup / "memory.max").read_text().strip()
    return -1 if val == "max" else int(val)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--name-pattern", default=r"^mn\.(client_\d+|server)$")
    parser.add_argument(
        "--max-stalls",
        type=int,
        default=15,
        help="exit after N consecutive cycles with no target containers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pattern = re.compile(args.name_pattern)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "timestamp",
                "container",
                "cpu_usec",
                "mem_bytes",
                "mem_limit_bytes",
                "container_id",
            ]
        )
        fp.flush()

        print(
            f"[collector] out={out_path} interval={args.interval}s "
            f"pattern={args.name_pattern}",
            flush=True,
        )

        stalls = 0
        next_tick = time.monotonic()
        while True:
            try:
                targets = list_target_containers(pattern)
            except Exception as exc:
                print(f"[collector] list error: {exc}", file=sys.stderr, flush=True)
                targets = []

            ts = time.time()
            if not targets:
                stalls += 1
                if stalls >= args.max_stalls:
                    print(
                        f"[collector] no containers for {stalls} cycles; exiting",
                        flush=True,
                    )
                    return
            else:
                stalls = 0

            for cid, name in targets:
                try:
                    cg = cgroup_path(cid)
                    writer.writerow(
                        [
                            f"{ts:.3f}",
                            name,
                            read_cpu_usec(cg),
                            read_memory_bytes(cg),
                            read_memory_limit(cg),
                            cid[:12],
                        ]
                    )
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    print(
                        f"[collector] sample error {name}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
            fp.flush()
            next_tick += args.interval
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()


if __name__ == "__main__":
    main()
