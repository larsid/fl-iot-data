from __future__ import annotations

import argparse
import bisect
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


Sample = Tuple[float, int, int, int]


def load_manifest(log_dir: Path) -> dict:
    return json.loads((log_dir / "manifest.json").read_text())


def load_accuracy(log_dir: Path) -> List[Dict[str, float]]:
    rows = []
    with (log_dir / "accuracy.csv").open() as fp:
        for row in csv.DictReader(fp):
            entry = {
                "round": int(row["round"]),
                "test_loss": float(row["test_loss"]),
                "test_accuracy": float(row["test_accuracy"]),
            }

            if row.get("eval_time_s") not in (None, ""):
                entry["eval_time_s"] = float(row["eval_time_s"])
            rows.append(entry)
    return rows


def load_client_jsonl(log_dir: Path, client_id: int) -> List[Dict]:
    path = log_dir / f"client_{client_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


def load_stats(log_dir: Path) -> Dict[str, List[List[Sample]]]:
    by_key: Dict[Tuple[str, str], List[Sample]] = {}
    path = log_dir / "stats.csv"
    if not path.exists():
        return {}
    with path.open() as fp:
        for row in csv.DictReader(fp):
            key = (row["container"], row.get("container_id", ""))
            by_key.setdefault(key, []).append(
                (
                    float(row["timestamp"]),
                    int(row["cpu_usec"]),
                    int(row["mem_bytes"]),
                    int(row["mem_limit_bytes"]),
                )
            )
    by_name: Dict[str, List[List[Sample]]] = {}
    for (name, _cid), series in by_key.items():
        series.sort(key=lambda s: s[0])
        by_name.setdefault(name, []).append(series)
    return by_name


def _cpu_is_monotonic(series: List[Sample]) -> bool:
    return all(series[i][1] >= series[i - 1][1] for i in range(1, len(series)))


def resolve_series(
    series_list: List[List[Sample]],
    t_ref: float,
    container: str,
) -> List[Sample]:
    if not series_list:
        return []
    if len(series_list) == 1:
        chosen = series_list[0]
    else:
        covering = [s for s in series_list if s and s[0][0] <= t_ref <= s[-1][0]]
        chosen = max(covering or series_list, key=len)
        print(
            f"!!! {container}: {len(series_list)} container series share this "
            f"name (orphan?); selected the one covering the FL window "
            f"(n={len(chosen)} samples).",
            file=sys.stderr,
            flush=True,
        )
    if not _cpu_is_monotonic(chosen):
        print(
            f"!!! {container}: cpu_usec is non-monotonic within the selected "
            f"series (counter reset / merged containers); CPU% may be unreliable.",
            file=sys.stderr,
            flush=True,
        )
    return chosen


def parse_cpuset(cpuset: str) -> int:
    if not cpuset:
        return 0
    count = 0
    for part in cpuset.split(","):
        if "-" in part:
            a, b = part.split("-")
            count += int(b) - int(a) + 1
        elif part:
            count += 1
    return count


def cpu_denominator(manifest: dict, kind: str) -> Tuple[float, str]:
    resources = manifest.get(f"{kind}_resources") or {}
    if resources.get("cpus"):
        return float(resources["cpus"]), "cpus_limit"
    cpuset = resources.get("cpuset_cpus", "")
    if cpuset:
        n = parse_cpuset(cpuset)
        if n > 0:
            return float(n), "cpuset_size"
    return 1.0, "default_1"


def _interp_cumulative(
    samples: List[Sample],
    timestamps: List[float],
    t: float,
) -> Optional[float]:
    if not samples:
        return None
    i = bisect.bisect_left(timestamps, t)
    if i < len(samples) and timestamps[i] == t:
        return float(samples[i][1])
    if i == 0 or i >= len(samples):
        return None
    t_prev, u_prev = samples[i - 1][0], samples[i - 1][1]
    t_next, u_next = samples[i][0], samples[i][1]
    if t_next == t_prev:
        return float(u_prev)
    return u_prev + (u_next - u_prev) * (t - t_prev) / (t_next - t_prev)


def window_metrics(
    samples: List[Sample],
    t_start: float,
    t_end: float,
    cpus: float,
) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    if not samples or t_end < t_start or cpus <= 0:
        return None, None, None
    timestamps = [s[0] for s in samples]

    u_start = _interp_cumulative(samples, timestamps, t_start)
    u_end = _interp_cumulative(samples, timestamps, t_end)
    cpu_pct: Optional[float]
    if u_start is None or u_end is None or t_end <= t_start:
        cpu_pct = None
    else:
        cpu_pct = (u_end - u_start) / 1_000_000 / (t_end - t_start) / cpus * 100

    i0 = bisect.bisect_left(timestamps, t_start)
    i1 = bisect.bisect_right(timestamps, t_end) - 1
    in_window = samples[i0 : i1 + 1] if i0 <= i1 and i0 < len(samples) else []
    if in_window:
        mems = [s[2] for s in in_window]
        mem_mean = sum(mems) / len(mems)
        mem_lim = in_window[0][3]
    else:
        t_mid = (t_start + t_end) / 2
        j = bisect.bisect_left(timestamps, t_mid)
        candidates = []
        if j > 0:
            candidates.append(samples[j - 1])
        if j < len(samples):
            candidates.append(samples[j])
        if not candidates:
            return cpu_pct, None, None
        nearest = min(candidates, key=lambda s: abs(s[0] - t_mid))
        mem_mean = float(nearest[2])
        mem_lim = nearest[3]

    return cpu_pct, mem_mean, mem_lim


def derive_convergence_speed(accuracy_rows: List[Dict]) -> int:
    rounds_only = [r for r in accuracy_rows if r["round"] > 0]
    if not rounds_only:
        return 0
    final = rounds_only[-1]["test_accuracy"]
    for r in rounds_only:
        if r["test_accuracy"] >= final:
            return r["round"]
    return rounds_only[-1]["round"]


def _stat(xs: List[float]) -> Optional[Dict[str, float]]:
    if not xs:
        return None
    out: Dict[str, float] = {"mean": statistics.mean(xs), "n": len(xs)}
    if len(xs) > 1:
        out["std"] = statistics.stdev(xs)
    return out


def reduce_run(log_dir: Path) -> dict:
    manifest = load_manifest(log_dir)
    accuracy_rows = load_accuracy(log_dir)
    stats_by_container = load_stats(log_dir)
    num_clients = int(manifest.get("num_clients", 8))

    client_cpus, client_cpu_basis = cpu_denominator(manifest, "client")
    server_cpus, server_cpu_basis = cpu_denominator(manifest, "server")

    client_data = {
        cid: load_client_jsonl(log_dir, cid) for cid in range(1, num_clients + 1)
    }

    client_series: Dict[int, List[Sample]] = {}
    for cid in range(1, num_clients + 1):
        rows = client_data[cid]
        if rows:
            first = rows[0]
            t_ref = first.get("training_start_ts", first.get("fit_start_ts", 0.0))
        else:
            t_ref = 0.0
        client_series[cid] = resolve_series(
            stats_by_container.get(f"mn.client_{cid}", []), t_ref, f"mn.client_{cid}"
        )

    eval_time_by_round = {
        r["round"]: r["eval_time_s"] for r in accuracy_rows if "eval_time_s" in r
    }

    per_round: List[dict] = []
    for acc_row in accuracy_rows:
        rnd = acc_row["round"]
        if rnd == 0:
            continue
        round_entry = {
            "round": rnd,
            "test_loss": acc_row["test_loss"],
            "test_accuracy": acc_row["test_accuracy"],
            "clients": [],
        }
        for cid in range(1, num_clients + 1):
            rows = client_data[cid]
            r_data = next((r for r in rows if r["round"] == rnd), None)
            if r_data is None:
                continue
            next_r = next((r for r in rows if r["round"] == rnd + 1), None)
            samples = client_series[cid]

            t_train_start = r_data.get("training_start_ts", r_data["fit_start_ts"])
            t_train_end = r_data.get("training_end_ts", r_data["fit_end_ts"])
            cpu_t, mem_t, mem_lim_t = window_metrics(
                samples, t_train_start, t_train_end, cpus=client_cpus
            )
            mem_pct_t = (
                mem_t / mem_lim_t * 100
                if (mem_t is not None and mem_lim_t is not None and mem_lim_t > 0)
                else None
            )

            t_full_end = (
                next_r.get("training_start_ts", next_r["fit_start_ts"])
                if next_r
                else t_train_end
            )
            cpu_f, mem_f, mem_lim_f = window_metrics(
                samples, t_train_start, t_full_end, cpus=client_cpus
            )
            mem_pct_f = (
                mem_f / mem_lim_f * 100
                if (mem_f is not None and mem_lim_f is not None and mem_lim_f > 0)
                else None
            )

            round_entry["clients"].append(
                {
                    "client_id": cid,
                    "samples": r_data.get("samples"),
                    "training_time_s": r_data["training_time_s"],
                    "update_exchange_time_s": r_data.get("update_exchange_time_s"),
                    "update_exchange_net_s": (
                        r_data["update_exchange_time_s"] - eval_time_by_round[rnd - 1]
                        if (
                            r_data.get("update_exchange_time_s") is not None
                            and (rnd - 1) in eval_time_by_round
                        )
                        else None
                    ),
                    "local_serde_time_s": r_data.get("local_serde_time_s"),
                    "training_window": {
                        "cpu_pct": cpu_t,
                        "mem_bytes_mean": mem_t,
                        "mem_limit_bytes": mem_lim_t,
                        "mem_pct": mem_pct_t,
                    },
                    "full_window": {
                        "cpu_pct": cpu_f,
                        "mem_bytes_mean": mem_f,
                        "mem_limit_bytes": mem_lim_f,
                        "mem_pct": mem_pct_f,
                    },
                }
            )
        per_round.append(round_entry)

    train_vals, upd_vals, upd_net_vals = [], [], []
    cpu_t_vals, mem_pct_t_vals, mem_bytes_t_vals = [], [], []
    cpu_f_vals, mem_pct_f_vals = [], []
    for r in per_round:
        for c in r["clients"]:
            train_vals.append(c["training_time_s"])
            if c["update_exchange_time_s"] is not None:
                upd_vals.append(c["update_exchange_time_s"])
            if c.get("update_exchange_net_s") is not None:
                upd_net_vals.append(c["update_exchange_net_s"])
            tw = c["training_window"]
            if tw["cpu_pct"] is not None:
                cpu_t_vals.append(tw["cpu_pct"])
            if tw["mem_pct"] is not None:
                mem_pct_t_vals.append(tw["mem_pct"])
            if tw["mem_bytes_mean"] is not None:
                mem_bytes_t_vals.append(tw["mem_bytes_mean"])
            fw = c["full_window"]
            if fw["cpu_pct"] is not None:
                cpu_f_vals.append(fw["cpu_pct"])
            if fw["mem_pct"] is not None:
                mem_pct_f_vals.append(fw["mem_pct"])

    eval_vals = [v for rnd_k, v in eval_time_by_round.items() if rnd_k >= 1]

    final_acc = next(
        (r["test_accuracy"] for r in reversed(accuracy_rows) if r["round"] > 0), None
    )
    final_loss = next(
        (r["test_loss"] for r in reversed(accuracy_rows) if r["round"] > 0), None
    )

    rounds_with_eval = sorted({r["round"] for r in accuracy_rows if r["round"] > 0})
    rounds_completed = len(rounds_with_eval)
    rounds_expected = int(manifest.get("rounds", 0))
    last_round = rounds_with_eval[-1] if rounds_with_eval else 0
    is_complete = (
        rounds_expected > 0
        and rounds_completed == rounds_expected
        and last_round == rounds_expected
    )
    if not is_complete and rounds_expected > 0:
        print(
            f"!!! incomplete run at {log_dir}: "
            f"{rounds_completed}/{rounds_expected} rounds, last={last_round}",
            file=sys.stderr,
            flush=True,
        )

    return {
        "log_dir": str(log_dir),
        "manifest": manifest,
        "completeness": {
            "rounds_expected": rounds_expected,
            "rounds_completed": rounds_completed,
            "last_round": last_round,
            "is_complete": is_complete,
        },
        "denominators": {
            "client_cpu_basis": client_cpu_basis,
            "client_cpu_denominator": client_cpus,
            "server_cpu_basis": server_cpu_basis,
            "server_cpu_denominator": server_cpus,
        },
        "aggregate": {
            "test_accuracy_final": final_acc,
            "test_loss_final": final_loss,
            "convergence_speed_rounds": derive_convergence_speed(accuracy_rows),
            "avg_training_time_s": _stat(train_vals),
            "avg_update_exchange_time_s": _stat(upd_vals),
            "avg_update_exchange_net_s": _stat(upd_net_vals),
            "avg_eval_time_s": _stat(eval_vals),
            "avg_cpu_pct": _stat(cpu_t_vals),
            "avg_mem_pct": _stat(mem_pct_t_vals),
            "avg_mem_bytes": _stat(mem_bytes_t_vals),
            "diagnostic_full_window": {
                "avg_cpu_pct": _stat(cpu_f_vals),
                "avg_mem_pct": _stat(mem_pct_f_vals),
            },
        },
        "per_round": per_round,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    args = parser.parse_args()
    log_dir = args.log_dir.resolve()
    summary = reduce_run(log_dir)
    out_path = log_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    a = summary["aggregate"]
    c = summary["completeness"]
    print(f"Wrote {out_path}")
    print(f"  level                          : {summary['manifest'].get('level')}")
    print(
        f"  completeness                   : "
        f"{c['rounds_completed']}/{c['rounds_expected']} rounds "
        f"({'complete' if c['is_complete'] else 'INCOMPLETE'})"
    )
    print(f"  test_accuracy_final            : {a['test_accuracy_final']}")
    print(f"  convergence_speed_rounds       : {a['convergence_speed_rounds']}")
    print(f"  avg_training_time_s            : {a['avg_training_time_s']}")
    print(f"  avg_update_exchange_time_s     : {a['avg_update_exchange_time_s']}")
    print(f"  avg_update_exchange_net_s      : {a['avg_update_exchange_net_s']}")
    print(f"  avg_eval_time_s                : {a['avg_eval_time_s']}")
    print(f"  avg_cpu_pct                    : {a['avg_cpu_pct']}")
    print(f"  avg_mem_pct                    : {a['avg_mem_pct']}")


if __name__ == "__main__":
    main()
