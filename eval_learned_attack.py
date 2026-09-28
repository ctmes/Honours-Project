"""WS10a: does a LEARNED optimal attack hurt the frozen market maker?

The strongest test of H1 available in this project (Zhang et al. 2021, ATLA): for
a FIXED victim the optimal observation attack is itself an MDP, so it can be
learned. The v4opt_* runs (config/rl_configs/kaya_v4opt_*.yaml) load a trained v4
market maker (MM_INIT_FROM), never update it (FREEZE_MM, deterministic actions),
and train a cost-free adversary that observes the victim's own observation
(observe_victim) with its gate always on and a 1e7 budget. The adversary is free to
learn when, where and on which side to inject.

This script evaluates each of those runs twice on the held-out 2024-Q4 period, with
the same rollout keys: 'off' (gate off) and 'learned' (gate on, the trained
adversary's own actions). Because the victim's params are bit-identical to the v4
checkpoint, 'off' must reproduce eval_17313.json's *_off values EXACTLY for that
arm -- --reference-eval checks it, and no 'learned' number should be trusted if it
does not.

Keys are suffixed _learned/_off so a file can never be mistaken for a forced or a
co-trained result. Analyse with:
    python -m analysis.h1_attack_effect_check results/learned_attack_v4.json --on-suffix _learned
    python -m analysis.permutation_check results/learned_attack_v4.json

EXPLORATORY (preregistration.json -> amendments -> tier4_extensions_amendment_2026-09-28).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from eval_forced_attack import (
    CONTRASTS, _parse_seeds, evaluate_arm, reference_check,
)

# arm -> (attack-run project, eval yaml, source v4 project)
ARMS = {
    "baseline":    ("v4opt_baseline",    "config/rl_configs/eval_2024_test_v4opt_baseline.yaml",
                    "v4_config1_baseline"),
    "adversarial": ("v4opt_adversarial", "config/rl_configs/eval_2024_test_v4opt_adversarial.yaml",
                    "v4_config2_adversarial"),
    "full":        ("v4opt_full",        "config/rl_configs/eval_2024_test_v4opt_full.yaml",
                    "v4_config3_full"),
}
ON = "learned"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default="baseline,adversarial,full")
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--budget", type=float, default=1e7,
                    help="must equal the attack runs' training budget_per_episode")
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--step", type=int, default=None,
                    help="attack-run checkpoint step (default: latest available)")
    ap.add_argument("--reference-eval", default=None,
                    help="e.g. results/eval_17313.json: 'off' must reproduce it exactly")
    ap.add_argument("--yaml-override", default=None,
                    help="arm=path pairs (comma list) to substitute eval yamls, for local smoke tests")
    ap.add_argument("--project-override", default=None,
                    help="arm=project pairs (comma list), for local smoke tests")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    with open(root / "preregistration.json") as f:
        ppy = float(json.load(f)["periods_per_year"])
    arms = [a.strip() for a in args.arms.split(",")]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {list(ARMS)}")
    seeds = _parse_seeds(args.seeds)
    yaml_over = dict(kv.split("=", 1) for kv in args.yaml_override.split(",")) if args.yaml_override else {}
    proj_over = dict(kv.split("=", 1) for kv in args.project_override.split(",")) if args.project_override else {}

    banner = ("*** EXPLORATORY — LEARNED OPTIMAL ATTACK on a FROZEN market maker (WS10a). "
              "Not a co-trained adversary, not the confirmatory result. ***")
    print(banner)
    per_seed = {}
    for arm in arms:
        project, yaml_path, _src = ARMS[arm]
        project = proj_over.get(arm, project)
        yaml_path = yaml_over.get(arm, yaml_path)
        step = args.step
        if step is None:
            from gymnax_exchange.jaxrl.MARL.adversarial_eval.rollout import load_merged_config
            import orbax.checkpoint as oxcp
            cfg = load_merged_config(yaml_path)
            d = f'{cfg["world_config"]["alphatradePath"]}/checkpoints/MARLCheckpoints/{project}/seed_{seeds[0]}'
            step = oxcp.CheckpointManager(d, oxcp.PyTreeCheckpointer()).latest_step()
        per_seed[arm] = evaluate_arm(arm, project, yaml_path, seeds, "bid", args.budget,
                                     args.n_envs, args.n_steps, step, ppy,
                                     override=False, on_label=ON)

    from gymnax_exchange.jaxrl.MARL.adversarial_eval.aggregate import compare_configs, summarize_seeds
    from gymnax_exchange.jaxrl.MARL.adversarial_eval.stats import holm_adjust, paired_comparison

    report = {"summaries": {a: summarize_seeds(m) for a, m in per_seed.items()},
              "per_seed": {a: {k: v.tolist() for k, v in m.items()} for a, m in per_seed.items()},
              "holm": {}}
    metrics = [f"sharpe_{ON}", f"sortino_{ON}", f"cvar_{ON}"]
    if len(seeds) >= 3:
        for label, a, b in CONTRASTS:
            if a in per_seed and b in per_seed:
                res = compare_configs(per_seed, a, b, metrics)
                report[label] = res
                report["holm"][label] = holm_adjust({m: r.p_value for m, r in res.items()})
    report["_meta"] = {
        "attack_construction": "learned_optimal_on_frozen_mm",
        "budget": args.budget, "arms": arms, "seeds": seeds, "n_envs": args.n_envs,
        "n_steps": args.n_steps, "periods_per_year": ppy,
        "signed_off": False, "confirmatory": False, "exploratory": True, "partial_run": True,
        "amendment": "preregistration.json -> amendments -> tier4_extensions_amendment_2026-09-28",
    }

    lines = [banner, "", f"=== per-arm means ({ON} vs off) ==="]
    for arm, m in per_seed.items():
        lines.append(f"[{arm}]")
        for k in ("sharpe", "sortino", "cvar", "inventory_sd", "mean_attack_rate",
                  "mean_bid_volume_injected", "mean_ask_volume_injected", "qi_mean"):
            lines.append(f"  {k:<26} {ON}={np.nanmean(m[f'{k}_{ON}']):+10.4g}  "
                         f"off={np.nanmean(m[f'{k}_off']):+10.4g}")
        bid, ask = m[f"mean_bid_volume_injected_{ON}"], m[f"mean_ask_volume_injected_{ON}"]
        side = np.abs(bid - ask) / np.maximum(bid + ask, 1e-12)
        lines.append(f"  {'sidedness |b-a|/(b+a)':<26} {ON}={np.nanmean(side):10.4g}  (1 = one-sided)")
    if len(seeds) >= 3:
        lines += ["", f"=== within-arm: {ON} vs off, paired by seed ==="]
        for arm, m in per_seed.items():
            for k in ("sharpe", "sortino", "cvar", "inventory_sd"):
                x, y = m[f"{k}_{ON}"], m[f"{k}_off"]
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 3:
                    continue
                r = paired_comparison(x[ok], y[ok])
                lines.append(f"  {arm:<12} {k:<13} diff={r.mean_diff:+9.4g} d={r.cohens_d:+.3g} "
                             f"{r.test:<9} p={r.p_value:.3g}")
    if args.reference_eval:
        lines += [""] + reference_check(per_seed, seeds, args.reference_eval)
    text = "\n".join(lines)
    print("\n" + text)
    if args.out:
        from run_production_eval import _serialise
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(f"{out}.json", "w") as f:
            json.dump(_serialise(report), f, indent=2, default=str)
        with open(f"{out}.txt", "w") as f:
            f.write(text + "\n")
        print(f"\nwrote {out}.json and {out}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
