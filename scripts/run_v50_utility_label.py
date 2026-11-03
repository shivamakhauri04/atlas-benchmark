"""The label hypothesis within AgentDojo (prereg-v50 A16, registered before running).

Same runs and features as E1, relabeled: 1 = the user's task failed (`utility=False`). The primary
set is the attacked important_instructions runs; the secondary set is the clean `none` runs. Ten
task splits, class-balanced, five length-free features, and the ten-split permutation null.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import run_v50_cross_agent as e1
import run_v50_cross_agent_tensplit_perm as tp

from data import domain_loader as dl
from data.schema import Trace
from data.trace_loader import _agentdojo_actions, load_agentdojo
from traceshift.v50 import is_reversed

OUT = Path("output/tables/v50_utility_label.json")
SECURITY_EXCESS_TAU01 = 0.2248 - 0.0582


def _utility(run_dir: Path, attack: str) -> dict[str, bool]:
    out = {}
    for path in run_dir.rglob(f"{attack}/*.json"):
        raw = json.loads(path.read_text())
        suite = path.relative_to(run_dir).parts[0]
        key = (f"agentdojo_{suite}_{raw.get('user_task_id', 'unk')}_"
               f"{raw.get('injection_task_id', 'unk')}_{attack}")
        out[key] = bool(raw.get("utility"))
    return out


def _clean_traces(run_dir: Path) -> tuple[list[Trace], np.ndarray]:
    """No-attack runs, which the injection loader skips because their attack_type is null."""
    traces, y = [], []
    for path in sorted(run_dir.rglob("none/none.json")):
        raw = json.loads(path.read_text())
        suite = path.relative_to(run_dir).parts[0]
        actions = _agentdojo_actions(raw.get("messages", []), suite, 0)
        if len(actions) >= 1:
            user_task = raw.get("user_task_id", "unk")
            traces.append(Trace(trace_id=f"{suite}_{user_task}", actions=actions,
                                source="agentdojo", task=f"{suite}|{user_task}"))
            y.append(0 if raw.get("utility") else 1)
    return traces, np.array(y)


def _corpus(run: str, attack: str):
    run_dir = e1.RUNS / run
    if attack == "none":
        traces, y = _clean_traces(run_dir)
    else:
        traces = load_agentdojo(run_dir, min_tool_calls=1, attack=attack)
        util = _utility(run_dir, attack)
        traces = [t for t in traces if t.trace_id in util]
        y = np.array([0 if util[t.trace_id] else 1 for t in traces])
    if len(y) == 0 or min(int(y.sum()), int(len(y) - y.sum())) < e1.MIN_CLASS:
        return None
    feats = dl._extract_all_features(traces)
    x = np.column_stack([np.asarray(feats[f], float) for f in e1.FEATURES])
    return x, y, [t.task for t in traces]


def _run(attack: str) -> dict:
    runs = sorted(p.name for p in e1.RUNS.iterdir() if p.is_dir())
    corpora = {r: c for r in runs if (c := _corpus(r, attack))}
    pairs = list(itertools.combinations(corpora, 2))
    splits = [tp._halves(corpora, s) for s in e1.SEEDS]
    cross = tp._observed(corpora, pairs, splits)
    within = {}
    for t in tp.TAUS:
        flips = []
        for halves in splits:
            for r in corpora:
                da = e1._d(corpora[r][0], corpora[r][1], halves[r][0])
                db = e1._d(corpora[r][0], corpora[r][1], halves[r][1])
                flips += [is_reversed(a, b, tau=t) for a, b in zip(da, db)]
        within[t] = float(np.mean(flips))
    rng = np.random.default_rng(0)
    null = [tp._null_draw(corpora, pairs, splits, rng) for _ in range(tp.N_PERM)]
    out = {"attack": attack, "n_configurations": len(corpora), "configurations": list(corpora),
           "failure_rate": {r: float(c[1].mean()) for r, c in corpora.items()}}
    for t in tp.TAUS:
        nt = [n[t] for n in null]
        out[f"tau={t}"] = {"cross": cross[t], "within": within[t],
                           "perm_null_p95": float(np.percentile(nt, 95)),
                           "perm_p": float((1 + sum(n >= cross[t] for n in nt)) / (1 + tp.N_PERM))}
    return out


def main() -> None:
    result = {"security_excess_tau0.1": SECURITY_EXCESS_TAU01,
              "primary_attacked": _run("important_instructions"),
              "secondary_clean": _run("none")}
    p = result["primary_attacked"]["tau=0.1"]
    result["h_label_refuted"] = bool(p["cross"] - p["within"] >= SECURITY_EXCESS_TAU01)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                      if kk not in ("configurations", "failure_rate")}) for k, v in result.items()},
                     indent=1))


if __name__ == "__main__":
    main()
