from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_DIR = Path(__file__).resolve().parent
STATE_PATH = REPO_DIR.parent / "results" / "orchestrator_state.json"

MAIN_CONFIGS = [
    ("level0", "experiment_level0.py", {}),
    ("level1", "experiment_level1.py", {}),
    ("level2", "experiment_level2.py", {}),
    ("level3", "experiment_level3.py", {}),
]

SECONDARY_CONFIGS = [
    ("level3_bw100", "experiment_level3.py", {"BW": "100"}),
    ("level3_bw50", "experiment_level3.py", {"BW": "50"}),
    ("level3_bw25", "experiment_level3.py", {"BW": "25"}),
]

DEFAULT_DURATION_S = {
    "level0": 90 * 60,
    "level1": 90 * 60,
    "level2": 90 * 60,
    "level3": 90 * 60,
    "level3_bw100": 90 * 60,
    "level3_bw50": 95 * 60,
    "level3_bw25": 110 * 60,
}


def _config_key(name: str) -> str:
    return name.rsplit("_rep", 1)[0]


def _build_plan(meta_seed: int, reps: int) -> List[Dict]:
    rng = random.Random(meta_seed)
    plan = []
    for label, script, env in MAIN_CONFIGS + SECONDARY_CONFIGS:
        for rep in range(reps):
            plan.append(
                {
                    "name": f"{label}_rep{rep + 1}",
                    "script": script,
                    "env_extra": env,
                    "seed": rng.randint(1, 2**31 - 1),
                }
            )
    return plan


def load_state() -> Dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: Dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def ensure_state() -> Dict:
    state = load_state()
    if state.get("plan"):
        return state

    meta_seed = int(os.environ.get("META_SEED", "42"))
    rounds = int(os.environ.get("ROUNDS", "500"))
    reps = int(os.environ.get("REPS", "5"))
    state = {
        "meta_seed": meta_seed,
        "rounds": rounds,
        "reps": reps,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "plan": _build_plan(meta_seed, reps),
        "runs": [],
    }
    save_state(state)
    print(f"\n[orquestrador] estado inicializado em {STATE_PATH}")
    print(
        f"  meta_seed={meta_seed}  rounds={rounds}  reps={reps}  "
        f"total={len(state['plan'])} experimentos\n"
    )
    return state


def _last_ok_run(state: Dict, name: str) -> Optional[Dict]:
    for run in reversed(state.get("runs", [])):
        if run.get("name") == name and run.get("status") == "ok":
            return run
    return None


def _last_run(state: Dict, name: str) -> Optional[Dict]:
    for run in reversed(state.get("runs", [])):
        if run.get("name") == name:
            return run
    return None


def _estimate_duration_s(state: Dict, name: str) -> float:
    key = _config_key(name)
    completed_durations = [
        r["duration_s"]
        for r in state.get("runs", [])
        if r.get("status") == "ok"
        and r.get("duration_s") is not None
        and _config_key(r["name"]) == key
    ]
    if completed_durations:
        return sum(completed_durations) / len(completed_durations)
    return DEFAULT_DURATION_S.get(key, 90 * 60)


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _describe(name: str) -> tuple[str, str, str]:
    config_part, _, rep_part = name.rpartition("_rep")
    rep_str = str(int(rep_part)) if rep_part.isdigit() else "0"
    if "_bw" in config_part:
        level, _, bw = config_part.partition("_bw")
        level_num = level.replace("level", "")
        return "secundario", f"N{level_num} (bw={bw}Mbps)", rep_str
    level_num = config_part.replace("level", "")
    return "principal", f"N{level_num}", rep_str


def _render_row(entry: Dict, state: Dict, reps_total: int) -> str:
    name = entry["name"]
    seed = entry["seed"]
    ok = _last_ok_run(state, name)
    last = _last_run(state, name)
    if ok:
        status = "[ OK ]"
        real = _format_duration(ok.get("duration_s"))
    elif last and last.get("status") != "ok":
        status = "[FAIL]"
        real = "-"
    else:
        status = "[    ]"
        real = "-"
    estimated = _format_duration(_estimate_duration_s(state, name))
    _, level_label, rep_str = _describe(name)
    rep_label = f"{rep_str}/{reps_total}"
    return (
        f"  {status:<7} {level_label:<16} {rep_label:<5} "
        f"{name:<22} {seed:>11}  {estimated:>8}  {real:>9}"
    )


def cmd_status(state: Dict) -> None:
    plan = state["plan"]
    reps_total = state["reps"]
    print(
        f"\nPlano: {len(plan)} experimentos  "
        f"(META_SEED={state['meta_seed']}, "
        f"ROUNDS={state['rounds']}, REPS={reps_total})\n"
    )

    header = (
        f"  {'Status':<7} {'Nível/Variante':<16} {'Rep':<5} "
        f"{'ID':<22} {'Seed':>11}  {'Estim.':>8}  {'Real':>9}"
    )
    div = "  " + "-" * (len(header) - 2)

    def render_group(title: str, group_key: str) -> None:
        print()
        print(f"  {title}")
        print(div)
        print(header)
        print(div)
        for entry in plan:
            g, _, _ = _describe(entry["name"])
            if g != group_key:
                continue
            print(_render_row(entry, state, reps_total))

    render_group(
        "EXPERIMENTO PRINCIPAL (20 runs: 4 níveis × 5 reps)",
        "principal",
    )
    render_group(
        "EXPERIMENTO SECUNDÁRIO (15 runs: 3 bandwidths × 5 reps)",
        "secundario",
    )

    done = sum(1 for e in plan if _last_ok_run(state, e["name"]))
    failed = sum(
        1
        for e in plan
        if _last_run(state, e["name"]) is not None
        and _last_run(state, e["name"]).get("status") != "ok"
        and _last_ok_run(state, e["name"]) is None
    )
    pending = len(plan) - done
    total_done_s = sum(
        (_last_ok_run(state, e["name"]) or {}).get("duration_s") or 0.0 for e in plan
    )
    total_remaining_s = sum(
        _estimate_duration_s(state, e["name"])
        for e in plan
        if _last_ok_run(state, e["name"]) is None
    )

    print()
    print(
        f"  Resumo: {done} ok / {failed} falharam / {pending} pendentes "
        f"de {len(plan)} totais"
    )
    print(f"  Tempo já gasto:        {_format_duration(total_done_s)}")
    print(f"  Tempo restante estim.: {_format_duration(total_remaining_s)}")
    print()


def cmd_plan(state: Dict) -> None:
    plan = state["plan"]
    reps_total = state["reps"]
    print(
        f"\nPlano completo: {len(plan)} experimentos "
        f"(META_SEED={state['meta_seed']}, ROUNDS={state['rounds']})\n"
    )
    print(
        f"  {'Nível/Variante':<16} {'Rep':<5} {'ID':<22} "
        f"{'Script':<25} {'Seed':>11}"
    )
    print("  " + "-" * 86)
    for entry in plan:
        _, level_label, rep_str = _describe(entry["name"])
        rep_label = f"{rep_str}/{reps_total}"
        print(
            f"  {level_label:<16} {rep_label:<5} {entry['name']:<22} "
            f"{entry['script']:<25} {entry['seed']:>11}"
        )
    print()


def _next_pending(state: Dict) -> Optional[Dict]:
    for entry in state["plan"]:
        if _last_ok_run(state, entry["name"]) is None:
            return entry
    return None


def _resolve_entry(state: Dict, name: str) -> Optional[Dict]:
    for entry in state["plan"]:
        if entry["name"] == name:
            return entry
    return None


def _execute(state: Dict, entry: Dict) -> None:
    name = entry["name"]
    script = entry["script"]
    env_extra = entry["env_extra"]
    seed = entry["seed"]
    rounds = state["rounds"]

    env = os.environ.copy()
    env.pop("BW", None)
    env.update(env_extra)
    env["ROUNDS"] = str(rounds)
    env["SEED"] = str(seed)

    print(
        f"\n>>> Executando [{name}]  seed={seed}  rounds={rounds}  "
        f"env_extra={env_extra}"
    )
    print(f"    Estimativa: {_format_duration(_estimate_duration_s(state, name))}\n")

    started_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    t0 = time.monotonic()

    subprocess.run(["bash", str(REPO_DIR / "clean.sh")], check=False)
    result = subprocess.run([sys.executable, str(REPO_DIR / script)], env=env)
    duration_s = time.monotonic() - t0
    finished_at = time.strftime("%Y-%m-%d %H:%M:%S %z")

    record: Dict = {
        "name": name,
        "seed": seed,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(duration_s, 1),
        "env_extra": env_extra,
        "rounds": rounds,
    }

    if result.returncode != 0:
        record["status"] = "failed"
        record["returncode"] = result.returncode
        state.setdefault("runs", []).append(record)
        save_state(state)
        print(
            f"\n!!! [{name}] FALHOU (rc={result.returncode}) "
            f"apos {_format_duration(duration_s)}"
        )
        return

    bw = env_extra.get("BW")
    level_label = name.split("_")[0].removeprefix("level")
    log_subdir = f"level{level_label}" + (f"_bw{bw}" if bw else "")
    candidates = sorted(
        p for p in (REPO_DIR.parent / "results" / log_subdir).glob("*") if p.is_dir()
    )
    if candidates:
        record["log_dir"] = str(candidates[-1])
        reduce_rc = subprocess.run(
            [
                sys.executable,
                str(REPO_DIR / "reduce_run.py"),
                record["log_dir"],
            ],
        ).returncode
        record["status"] = "ok" if reduce_rc == 0 else "ok_no_reduce"
    else:
        record["status"] = "no_logs"

    state.setdefault("runs", []).append(record)
    save_state(state)
    print(f"\n=== [{name}] {record['status']} em {_format_duration(duration_s)} ===\n")


def cmd_next(state: Dict) -> None:
    entry = _next_pending(state)
    if entry is None:
        print("\nTodos os experimentos ja foram concluidos com sucesso.\n")
        return
    print(
        f"\nProximo pendente: {entry['name']} "
        f"(estimado {_format_duration(_estimate_duration_s(state, entry['name']))})"
    )
    if not _confirm("Executar agora?"):
        return
    _execute(state, entry)


def cmd_run(state: Dict, name: str) -> None:
    entry = _resolve_entry(state, name)
    if entry is None:
        print(f"\nExperimento '{name}' nao consta no plano. Use 'plan' para listar.")
        return
    if _last_ok_run(state, name) is not None:
        print(f"\nExperimento '{name}' ja foi concluido com sucesso.")
        if not _confirm("Re-executar mesmo assim?"):
            return
    _execute(state, entry)


def cmd_reset(state: Dict, name: str) -> None:
    entry = _resolve_entry(state, name)
    if entry is None:
        print(f"\nExperimento '{name}' nao consta no plano.")
        return
    before = len(state.get("runs", []))
    state["runs"] = [r for r in state.get("runs", []) if r.get("name") != name]
    removed = before - len(state["runs"])
    save_state(state)
    print(
        f"\n{removed} registro(s) removido(s) para '{name}'; "
        f"agora marcado como pendente.\n"
    )


def _confirm(prompt: str) -> bool:
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes", "s", "sim")


def interactive_menu(state: Dict) -> None:
    while True:
        cmd_status(state)
        print("Opcoes:")
        print("  [n]        rodar proximo pendente")
        print("  [r <id>]   rodar um experimento especifico (ex.: r level1_rep2)")
        print("  [p]        ver plano completo")
        print("  [s]        atualizar status")
        print("  [reset <id>] limpar registros de um experimento")
        print("  [q]        sair")
        try:
            cmd = input("> ").strip()
        except EOFError:
            print()
            return
        if not cmd:
            continue
        if cmd == "q":
            return
        if cmd == "s":
            state = load_state()
            continue
        if cmd == "n":
            cmd_next(state)
            state = load_state()
            continue
        if cmd == "p":
            cmd_plan(state)
            continue
        if cmd.startswith("r "):
            cmd_run(state, cmd[2:].strip())
            state = load_state()
            continue
        if cmd.startswith("reset "):
            cmd_reset(state, cmd[6:].strip())
            state = load_state()
            continue
        print(f"Comando desconhecido: {cmd!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("status", help="tabela de status (read-only)")
    sub.add_parser("plan", help="lista plano completo + seeds")
    sub.add_parser("next", help="executa o proximo experimento pendente")
    p_run = sub.add_parser("run", help="executa um experimento por nome")
    p_run.add_argument("name", help="nome do experimento (ex.: level1_rep2)")
    p_reset = sub.add_parser("reset", help="reseta o status de um experimento")
    p_reset.add_argument("name", help="nome do experimento")

    args = parser.parse_args()
    state = ensure_state()

    if args.cmd == "status":
        cmd_status(state)
    elif args.cmd == "plan":
        cmd_plan(state)
    elif args.cmd == "next":
        cmd_next(state)
    elif args.cmd == "run":
        cmd_run(state, args.name)
    elif args.cmd == "reset":
        cmd_reset(state, args.name)
    else:
        interactive_menu(state)


if __name__ == "__main__":
    main()
