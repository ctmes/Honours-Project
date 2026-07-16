"""
One-command production evaluation: sweep checkpoints -> thesis numbers.

Runs the full proposal Section-4 analysis over the three trained arms plus the
Avellaneda-Stoikov fixed-policy baseline, on the held-out 2024_test split:

  - per-seed rollouts under attack-on / attack-off / mixed (detection AUROC)
  - COMMON adversary for every under-attack rollout (config-3's, by seed index)
  - contrasts 1-vs-2 (adversarial co-training), 2-vs-3 (detection+regime),
    with Holm correction over the pre-registered confirmatory family
  - H2 no-degradation TOST, H3 AUROC-vs-chance, H4 regime-split estimates
  - the Phase-1 progression gate (baseline IPPO vs A-S on clean data)

Analysis decisions come EXCLUSIVELY from preregistration.json. Confirmatory
p-values are only produced once that file is signed off; before sign-off the
script still runs but marks all output ESTIMATION-ONLY.

Usage (on Kaya, from the repo root — typically inside slurm_eval.sh):
  python run_production_eval.py --out results/eval_$(date +%Y%m%d)
  python run_production_eval.py --arms baseline,as --n-seeds 2   # partial dry-run

Runtime: each seed evaluates under 3 attack modes, each rebuilding the env
(~2-4 min cache load) — budget ~10 min/seed/arm, i.e. a full 4-arm x 20-seed
pass is a workday on one GPU. Use --n-seeds for a quick partial pass first.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np


def _load_prereg(root: Path) -> dict:
    with open(root / "preregistration.json") as f:
        return json.load(f)


def _serialise(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialise(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="repo root (must contain config/, checkpoints/)")
    ap.add_argument("--out", default="results/eval", help="output prefix (writes .json and .txt)")
    ap.add_argument("--arms", default="baseline,adversarial,full,as",
                    help="comma-separated subset of arms to evaluate")
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="evaluate only the first N seeds (partial dry-run)")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step (default: preregistration checkpoint_step; "
                         "pass -1 for latest available)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))

    from gymnax_exchange.jaxrl.MARL.adversarial_eval.run_evaluation import (
        run_full_evaluation, format_report,
    )

    prereg = _load_prereg(root)
    seeds = prereg["seeds"][: args.n_seeds] if args.n_seeds else prereg["seeds"]
    run_names = [f"seed_{s}" for s in seeds]
    step = prereg["checkpoint_step"] if args.step is None else (
        None if args.step == -1 else args.step)
    ppy = float(prereg["periods_per_year"])
    signed_off = bool(prereg.get("signed_off"))

    partial = bool(args.n_seeds) or set(args.arms.split(",")) != {
        "baseline", "adversarial", "full", "as"}
    if not signed_off:
        print("=" * 70)
        print("PREREGISTRATION NOT SIGNED OFF — output is ESTIMATION-ONLY.")
        print("Do not quote p-values from this run as confirmatory results.")
        print("=" * 70)

    adv_kw = {
        "adv_project": prereg["common_adversary"]["project"],
        "adv_run_names": run_names,
        "adv_step": step,
    }
    all_arms = {
        "baseline": dict(project="v1_config1_baseline", run_names=run_names,
                         yaml_path="config/rl_configs/eval_2024_test_config1.yaml",
                         n_envs=args.n_envs, periods_per_year=ppy, step=step,
                         seeds=list(range(len(run_names))), **adv_kw),
        "adversarial": dict(project="v1_config2_adversarial", run_names=run_names,
                            yaml_path="config/rl_configs/eval_2024_test_config2.yaml",
                            n_envs=args.n_envs, periods_per_year=ppy, step=step,
                            seeds=list(range(len(run_names))), **adv_kw),
        "full": dict(project="v1_config3_full", run_names=run_names,
                     yaml_path="config/rl_configs/eval_2024_test_config3.yaml",
                     n_envs=args.n_envs, periods_per_year=ppy, step=step,
                     seeds=list(range(len(run_names))), **adv_kw),
        "as": dict(fixed_policy=True, n_seeds=len(run_names),
                   yaml_path="config/rl_configs/eval_2024_test_as.yaml",
                   n_envs=args.n_envs, periods_per_year=ppy,
                   seeds=list(range(len(run_names)))),
    }
    configs = {name: kw for name, kw in all_arms.items()
               if name in args.arms.split(",")}
    print(f"evaluating arms: {list(configs)}  seeds: {len(run_names)}  "
          f"step: {step}  common adversary: {adv_kw['adv_project']}")

    report = run_full_evaluation(
        configs,
        primary_metrics=tuple(prereg["primary_metrics"]),
        equivalence_margins={k: v for k, v in prereg["equivalence_margins"].items()
                             if not k.startswith("_")},
        gate_as="as" if "as" in configs else None,
        gate_ippo="baseline" if "baseline" in configs else None,
        gate_kwargs={"sharpe_margin": prereg["progression_gate"]["sharpe_margin"],
                     "sortino_margin": prereg["progression_gate"]["sortino_margin"],
                     "inv_sd_factor": prereg["progression_gate"]["inv_sd_factor"]},
    )
    report["_meta"] = {
        "signed_off": signed_off,
        "partial_run": partial,
        "confirmatory": signed_off and not partial,
        "arms": list(configs), "seeds": seeds, "checkpoint_step": step,
        "periods_per_year": ppy,
        "common_adversary": adv_kw["adv_project"],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.json", "w") as f:
        json.dump(_serialise(report), f, indent=2, default=str)
    text = format_report(report)
    banner = ("" if report["_meta"]["confirmatory"] else
              "*** ESTIMATION-ONLY (preregistration not signed off, or partial run) ***\n\n")
    with open(f"{out}.txt", "w") as f:
        f.write(banner + text)
    print(banner + text)
    print(f"\nwrote {out}.json and {out}.txt")


if __name__ == "__main__":
    main()
