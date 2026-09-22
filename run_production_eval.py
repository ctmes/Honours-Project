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
  python run_production_eval.py --project-prefix v4 --out results/eval_v4

Runtime: each seed evaluates under 3 attack modes, each rebuilding the env
(~2-4 min cache load) — budget ~10 min/seed/arm, i.e. a full 4-arm x 20-seed
pass is a workday on one GPU. Use --n-seeds for a quick partial pass first.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
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
    ap.add_argument("--arms",
                    default="baseline,adversarial,detection,regime,full,unconstrained,as",
                    help="comma-separated subset of arms to evaluate")
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="evaluate only the first N seeds (partial dry-run)")
    # Each seed's metrics are averaged over n_envs parallel eval episodes, so this
    # sets the WITHIN-seed noise floor. The A-S arm is a deterministic fixed policy
    # yet showed sharpe_off = 0.119 +/- 41.81 across seeds in eval_1136928 — that
    # entire spread is rollout noise, and at n_envs=16 it swamped every between-arm
    # effect (observed diffs 11-24 against a standard error of ~9). 64 episodes cuts
    # the within-seed standard error in half; the eval nodes have ample RAM for it.
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step (default: preregistration checkpoint_step; "
                         "pass -1 for latest available)")
    # v3 is the CONFIRMATORY RESULT OF RECORD and stays the default, so an
    # unqualified invocation reproduces eval 1179095 exactly. v4 selects the
    # exploratory spread_skew arms; see preregistration.json -> amendments ->
    # v4_spread_skew_amendment, whose inference_status forbids reporting any v4
    # p-value as a significance result.
    ap.add_argument("--project-prefix", default="v3", choices=["v3", "v4"],
                    help="checkpoint project version to evaluate (default: v3). "
                         "Selects the arm projects, the matching eval configs and "
                         "the common adversary together - they cannot be mixed.")
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
    PFX = args.project_prefix

    # signed_off refers to the v3 CONFIRMATORY design and must NOT license a
    # later version as confirmatory. v4_spread_skew_amendment's inference_status:
    # "No confirmatory claim about H1-H4 may be made from v4 under any
    # multiplicity scheme, and no v4 p-value may be reported as a significance
    # result." Without this gate a full 7-arm 20-seed v4 pass emits
    # confirmatory: true with no banner, indistinguishable from eval 1179095.
    exploratory = PFX != "v3"

    partial = bool(args.n_seeds) or set(args.arms.split(",")) != {
        "baseline", "adversarial", "detection", "regime", "full", "unconstrained", "as"}
    if not signed_off or exploratory:
        print("=" * 70)
        print("%s — output is ESTIMATION-ONLY."
              % ("%s IS AN EXPLORATORY EXTENSION" % PFX.upper() if exploratory
                 else "PREREGISTRATION NOT SIGNED OFF"))
        print("Do not quote p-values from this run as confirmatory results.")
        if exploratory:
            print("The confirmatory result of record remains v3 (eval 1179095).")
        print("=" * 70)

    # v3's eval configs keep their historical filenames so the confirmatory run
    # is byte-identical; later versions use a version-tagged set. Both stay
    # evaluable from this one script, which is why this is a flag and not a
    # rename of the v3 files.
    def _eval_yaml(n: int) -> str:
        stem = ("eval_2024_test_config%d" % n if PFX == "v3"
                else "eval_2024_test_%s_config%d" % (PFX, n))
        return "config/rl_configs/%s.yaml" % stem

    # The pre-registered common-adversary design is "config-3's adversary, paired
    # by seed index (internal validity for H1)", and preregistration.json freezes
    # that as a v3 project name. Evaluating a later version applies the DESIGN
    # within that version: a v4 market maker faces the v4 config-3 adversary,
    # which is the only one co-trained against a price-setting (spread_skew) MM.
    # A v3 adversary never had a price lever to learn against, so pairing it with
    # a v4 MM would under-attack it and confound the one thing v4 exists to test.
    # Chosen 2026-09-15, BEFORE any v4 eval was run.
    adv_project = re.sub(r"^v\d+_", PFX + "_", prereg["common_adversary"]["project"])
    adv_kw = {
        "adv_project": adv_project,
        "adv_run_names": run_names,
        "adv_step": step,
    }
    all_arms = {
        "baseline": dict(project=f"{PFX}_config1_baseline", run_names=run_names,
                         yaml_path=_eval_yaml(1),
                         n_envs=args.n_envs, periods_per_year=ppy, step=step,
                         seeds=list(range(len(run_names))), **adv_kw),
        "adversarial": dict(project=f"{PFX}_config2_adversarial", run_names=run_names,
                            yaml_path=_eval_yaml(2),
                            n_envs=args.n_envs, periods_per_year=ppy, step=step,
                            seeds=list(range(len(run_names))), **adv_kw),
        "detection": dict(project=f"{PFX}_config4_detection", run_names=run_names,
                          yaml_path=_eval_yaml(4),
                          n_envs=args.n_envs, periods_per_year=ppy, step=step,
                          seeds=list(range(len(run_names))), **adv_kw),
        "regime": dict(project=f"{PFX}_config5_regime", run_names=run_names,
                       yaml_path=_eval_yaml(5),
                       n_envs=args.n_envs, periods_per_year=ppy, step=step,
                       seeds=list(range(len(run_names))), **adv_kw),
        "full": dict(project=f"{PFX}_config3_full", run_names=run_names,
                     yaml_path=_eval_yaml(3),
                     n_envs=args.n_envs, periods_per_year=ppy, step=step,
                     seeds=list(range(len(run_names))), **adv_kw),
        "unconstrained": dict(project=f"{PFX}_config6_unconstrained", run_names=run_names,
                              yaml_path=_eval_yaml(6),
                              n_envs=args.n_envs, periods_per_year=ppy, step=step,
                              seeds=list(range(len(run_names))), **adv_kw),
        # ARM G - EXPLORATORY, not in the pre-registered A-F design. Opt in with
        # --arms ...,detection_noobs; including it makes the run `partial`, which
        # is correct: it is an extension, not the registered analysis.
        # G vs adversarial isolates the auxiliary BCE loss (observation identical);
        # detection vs G isolates the fed-back detection channel (loss identical).
        "detection_noobs": dict(project=f"{PFX}_config7_detection_noobs",
                                run_names=run_names, yaml_path=_eval_yaml(7),
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
          f"step: {step}  version: {PFX}  "
          f"common adversary: {adv_kw['adv_project']}")

    # Pre-flight: verify every checkpoint exists BEFORE evaluating anything. A
    # missing seed otherwise aborts the run wherever it happens to fall — job
    # 1121705 died on the first arm's first seed, but a gap in the middle of the
    # list would have burned hours of completed rollouts first. Skipped when
    # --step -1 (latest available) is requested.
    if step is not None:
        ckpt_root = root / "checkpoints" / "MARLCheckpoints"
        wanted = [(kw["project"], rn) for kw in configs.values()
                  if not kw.get("fixed_policy") for rn in kw["run_names"]]
        if any(not kw.get("fixed_policy") for kw in configs.values()):
            wanted += [(adv_kw["adv_project"], rn) for rn in adv_kw["adv_run_names"]]
        missing = sorted({f"{proj}/{rn}/{step}" for proj, rn in wanted
                          if not (ckpt_root / proj / rn / str(step)).is_dir()})
        if missing:
            print()
            print(f"MISSING {len(missing)} of {len(set(wanted))} checkpoint(s) at "
                  f"step {step} - nothing was evaluated:")
            for m in missing:
                print(f"  {m}")
            print()
            print("Training has probably not finished. Check ./status.sh")
            sys.exit(1)
        print(f"pre-flight OK: {len(set(wanted))} checkpoints present at step {step}")

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
        "project_prefix": PFX,
        "exploratory": exploratory,
        "confirmatory": signed_off and not partial and not exploratory,
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
              "*** ESTIMATION-ONLY (exploratory version, prereg not signed off, or partial run) ***\n\n")
    with open(f"{out}.txt", "w") as f:
        f.write(banner + text)
    print(banner + text)
    print(f"\nwrote {out}.json and {out}.txt")


if __name__ == "__main__":
    main()
