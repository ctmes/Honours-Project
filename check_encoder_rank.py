"""RC4: does the detection head's auxiliary loss preserve the encoder's feature rank?

The v4 "fifth finding": arms with a trained detection head (detection, full,
detection_noobs) avoid the no-trade collapse that the no-head arms fall into, even
when the policy never sees the head's output (detection_noobs). Moalla et al.
(2024, "No Representation, No Trust") show that PPO actors collapse as the rank of
their representation degrades under non-stationarity, and that an auxiliary loss
on the representation prevents it. If that is the mechanism here, head arms should
have a higher-rank SharedEncoder output than their no-head minimal pairs.

Design (pre-registered: preregistration.json -> amendments ->
robustness_checks_amendment_2026-09-28, RC4)
---------------------------------------------------------------------------
* ONE fixed observation set: a clean (attack off) rollout of the reference
  checkpoint (default v4 baseline, seed 0), n_envs x n_steps MM observations.
  Every checkpoint's encoder is applied to the SAME inputs, and every arm has the
  identical SharedEncoder (no-head arms only skip the BCE gradient,
  ippo_adversarial.py), so a difference comes from training, not from data or
  architecture.
* Per checkpoint: effective rank (Roy & Vetterli 2007), srank_0.01 (Kumar et al.
  2021) and the fraction of dead ReLU units, on the 128-d SharedEncoder output.
* Primary: the three minimal-pair isolations of the head factor, paired by seed
  index -- detection vs adversarial, full vs regime, detection_noobs vs
  adversarial -- on effective rank, exact sign-flip p, Holm over 3. Supported if
  all three differences are positive and >= 2 of 3 survive Holm.
* Secondary (descriptive): --own-steps also scores each checkpoint on its OWN clean
  rollout; --eval-ref correlates rank with the collapse indicator inventory_sd_off
  across every arm x seed.

    python check_encoder_rank.py --arms all --seeds 0-19 \\
        --eval-ref results/eval_17313.json --out results/encoder_rank_v4

Nothing in the environment, agents, training loop or confirmatory eval is changed;
the rollout is eval_forced_attack.run_rollout_with_override with no override, the
loop whose 'off' output reproduces eval_17313.json exactly.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

from eval_forced_attack import (
    ARMS, EXTRA_ARMS, EXTRA_HEAD_ARMS, HEAD_ARMS, _parse_seeds, resolve_arm,
    run_rollout_with_override,
)
from gymnax_exchange.jaxrl.MARL.adversarial_eval.metrics import (
    dead_unit_fraction, effective_rank, srank,
)

ENCODER_KEY = "SharedEncoder_0"
MINIMAL_PAIRS = [
    ("detection_vs_adversarial", "detection", "adversarial"),
    ("full_vs_regime", "full", "regime"),
    ("detection_noobs_vs_adversarial", "detection_noobs", "adversarial"),
]
_OBS_DIM = 45


def rank_metrics(features) -> dict:
    return {"erank": effective_rank(features),
            "srank": float(srank(features, delta=0.01)),
            "dead_frac": dead_unit_fraction(features)}


def encoder_features(mm_params, obs) -> np.ndarray:
    """SharedEncoder output for `obs` under one MM checkpoint's parameters."""
    import jax.numpy as jnp
    from gymnax_exchange.jaxrl.MARL.attack_aware_policy import SharedEncoder
    inner = mm_params["params"] if "params" in mm_params else mm_params
    if ENCODER_KEY not in inner:
        raise KeyError(f"{ENCODER_KEY} not in MM params (keys: {sorted(inner)}); "
                       "is this an AttackAwarePolicyNet checkpoint?")
    out = SharedEncoder().apply({"params": inner[ENCODER_KEY]}, jnp.asarray(obs))
    return np.asarray(out, dtype=np.float64)


def _build(pfx: str, arm: str, n_envs: int):
    from gymnax_exchange.jaxrl.MARL.adversarial_eval.rollout import (
        build_eval, load_merged_config, set_attack_mode,
    )
    cfg = set_attack_mode(load_merged_config(resolve_arm(arm, pfx)[1]), "off")
    return build_eval(cfg, n_envs)


def _clean_obs(env, nets, ts, cfg, seed: int, n_envs: int, n_steps: int) -> np.ndarray:
    import jax
    arrays = run_rollout_with_override(env, nets, ts, cfg, 0.0, None, 0.0,
                                       jax.random.PRNGKey(seed), n_envs, n_steps,
                                       collect_mm_obs=True)
    obs = arrays["obs_mm"].reshape(-1, arrays["obs_mm"].shape[-1])
    if obs.shape[-1] != _OBS_DIM:
        raise SystemExit(f"MM obs is {obs.shape[-1]}-dim, expected {_OBS_DIM}")
    return obs


def collect_reference_obs(pfx, arm, seed, step, n_envs, n_steps, cache: Path) -> np.ndarray:
    if cache.exists():
        z = np.load(cache)
        meta = json.loads(str(z["meta"]))
        want = {"prefix": pfx, "arm": arm, "seed": seed, "step": step,
                "n_envs": n_envs, "n_steps": n_steps}
        if meta == want:
            print(f"[rank] reusing fixed observation set {cache} {z['obs'].shape}")
            return z["obs"]
        print(f"[rank] cache {cache} is for {meta}, not {want}; rebuilding")
    from gymnax_exchange.jaxrl.MARL.adversarial_eval.rollout import restore_checkpoint
    env, nets, ts_template, cfg = _build(pfx, arm, n_envs)
    steps = int(n_steps) if n_steps is not None else int(cfg["NUM_STEPS"])
    ts, used = restore_checkpoint(cfg, ts_template, resolve_arm(arm, pfx)[0], f"seed_{seed}", step)
    obs = _clean_obs(env, nets, ts, cfg, seed, n_envs, steps)
    cache.parent.mkdir(parents=True, exist_ok=True)
    meta = {"prefix": pfx, "arm": arm, "seed": seed, "step": step,
            "n_envs": n_envs, "n_steps": n_steps}
    np.savez_compressed(cache, obs=obs, meta=json.dumps(meta))
    print(f"[rank] fixed observation set: {arm} seed {seed} step {used}, {obs.shape} -> {cache}")
    del env, nets, ts_template, ts
    gc.collect()
    return obs


def evaluate_arm(pfx, arm, seeds, step, obs_fixed, n_envs, own_steps) -> dict:
    import jax
    from gymnax_exchange.jaxrl.MARL.adversarial_eval.rollout import restore_checkpoint
    env, nets, ts_template, cfg = _build(pfx, arm, n_envs)
    mm_idx = env._mm_idx
    rows = {}
    for s in seeds:
        ts, used = restore_checkpoint(cfg, ts_template, resolve_arm(arm, pfx)[0], f"seed_{s}", step)
        row = {f"{k}_fixed": v for k, v in
               rank_metrics(encoder_features(ts[mm_idx].params, obs_fixed)).items()}
        if own_steps:
            own = _clean_obs(env, nets, ts, cfg, s, n_envs, own_steps)
            row.update({f"{k}_own": v for k, v in
                        rank_metrics(encoder_features(ts[mm_idx].params, own)).items()})
        rows[s] = row
        print(f"[rank] arm={arm:<16} seed={s:<3} step={used} "
              + " ".join(f"{k}={v:.4g}" for k, v in row.items()), flush=True)
        del ts
        gc.collect()
    del env, nets, ts_template, cfg
    jax.clear_caches()
    gc.collect()
    keys = list(rows[seeds[0]])
    return {k: np.array([rows[s][k] for s in seeds], dtype=np.float64) for k in keys}


def rc4_tests(per_seed: dict) -> dict:
    """Primary RC4 statistic: minimal pairs on erank_fixed, sign-flip p, Holm over 3."""
    from gymnax_exchange.jaxrl.MARL.adversarial_eval.stats import (
        bca_ci, holm_adjust, paired_comparison, sign_flip_permutation,
    )
    out, perm_p = {}, {}
    for label, a, b in MINIMAL_PAIRS:
        if a not in per_seed or b not in per_seed:
            continue
        x, y = per_seed[a]["erank_fixed"], per_seed[b]["erank_fixed"]
        ok = np.isfinite(x) & np.isfinite(y)
        d = x[ok] - y[ok]
        r = paired_comparison(x[ok], y[ok])
        p = sign_flip_permutation(d)
        lo, hi = bca_ci(d)
        perm_p[label] = p
        out[label] = {"arm_a": a, "arm_b": b, "n": int(ok.sum()),
                      "mean_diff": float(d.mean()), "cohens_d": r.cohens_d,
                      "p_sign_flip": p, "bca_low": lo, "bca_high": hi,
                      "preregistered_style_test": r.test, "p_preregistered_style": r.p_value}
    holm = holm_adjust(perm_p) if perm_p else {}
    for label in out:
        out[label]["p_holm"] = holm[label]
    positive = [r["mean_diff"] > 0 for r in out.values()]
    sig_positive = [r["mean_diff"] > 0 and r["p_holm"] < 0.05 for r in out.values()]
    if len(out) < 3:
        verdict = "INCOMPLETE: not all three minimal pairs were evaluated"
    elif all(positive) and sum(sig_positive) >= 2:
        verdict = "SUPPORTED: head arms have higher encoder effective rank"
    elif not any(sig_positive):
        verdict = ("NOT SUPPORTED: no minimal pair shows a significantly higher-rank head "
                   "arm; the collapse mechanism is not shown to be feature-rank loss")
    else:
        verdict = "INCONCLUSIVE under the pre-registered rule"
    return {"minimal_pairs": out, "verdict": verdict,
            "rule": "all three diffs > 0 and >= 2 of 3 Holm p < 0.05"}


def rank_vs_collapse(per_seed: dict, seeds: list[int], eval_ref: str) -> dict:
    """Descriptive: Spearman rho of erank_fixed vs inventory_sd_off over arm x seed."""
    from scipy.stats import spearmanr
    with open(eval_ref) as f:
        ref = json.load(f)
    ref_seeds = ref.get("_meta", {}).get("seeds", list(range(20)))
    xs, ys = [], []
    for arm, m in per_seed.items():
        vals = ref.get("per_seed", {}).get(arm, {}).get("inventory_sd_off")
        if vals is None:
            continue
        for i, s in enumerate(seeds):
            if s in ref_seeds:
                xs.append(m["erank_fixed"][i])
                ys.append(vals[ref_seeds.index(s)])
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {"n": int(ok.sum()), "spearman_rho": float("nan"), "p": float("nan")}
    rho, p = spearmanr(x[ok], y[ok])
    return {"n": int(ok.sum()), "spearman_rho": float(rho), "p": float(p),
            "source": eval_ref, "note": "descriptive only; arm x seed pooled"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default="all", help=f"comma list or 'all' ({','.join(ARMS)})")
    ap.add_argument("--project-prefix", default="v4", choices=["v3", "v4"])
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step (default: preregistration.json checkpoint_step)")
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=None,
                    help="reference rollout length (default: config NUM_STEPS)")
    ap.add_argument("--reference-arm", default="baseline")
    ap.add_argument("--reference-seed", type=int, default=0)
    ap.add_argument("--own-steps", type=int, default=0,
                    help="secondary: also score each checkpoint on its own clean rollout "
                         "of this many steps (0 = skip)")
    ap.add_argument("--obs-cache", default=None,
                    help="npz for the fixed observation set (default outputs/encoder_rank_obs_<prefix>.npz)")
    ap.add_argument("--eval-ref", default=None,
                    help="eval_*.json for the descriptive rank-vs-inventory_sd correlation")
    ap.add_argument("--out", default=None, help="output prefix (.json and .txt)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    with open(root / "preregistration.json") as f:
        prereg = json.load(f)
    step = prereg["checkpoint_step"] if args.step is None else args.step
    pfx = args.project_prefix
    arms = list(ARMS) if args.arms == "all" else [a.strip() for a in args.arms.split(",")]
    unknown = [a for a in arms if a not in ARMS and a not in EXTRA_ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {list(ARMS) + list(EXTRA_ARMS)}")
    seeds = _parse_seeds(args.seeds)
    cache = Path(args.obs_cache or f"outputs/encoder_rank_obs_{pfx}.npz")

    print(f"*** EXPLORATORY — RC4 encoder rank. arms {arms} seeds {seeds} step {step} ***")
    obs_fixed = collect_reference_obs(pfx, args.reference_arm, args.reference_seed, step,
                                      args.n_envs, args.n_steps, cache)

    per_seed = {arm: evaluate_arm(pfx, arm, seeds, step, obs_fixed, args.n_envs, args.own_steps)
                for arm in arms}

    from gymnax_exchange.jaxrl.MARL.adversarial_eval.aggregate import summarize_seeds
    report = {"summaries": {a: summarize_seeds(m) for a, m in per_seed.items()},
              "per_seed": {a: {k: v.tolist() for k, v in m.items()} for a, m in per_seed.items()},
              "rc4": rc4_tests(per_seed) if len(seeds) >= 3 else None}
    if args.eval_ref:
        report["rank_vs_inventory_sd_off"] = rank_vs_collapse(per_seed, seeds, args.eval_ref)
    report["_meta"] = {
        "analysis": "RC4 encoder effective rank",
        "amendment": "preregistration.json -> amendments -> robustness_checks_amendment_2026-09-28",
        "project_prefix": pfx, "arms": arms, "seeds": seeds, "checkpoint_step": step,
        "fixed_obs": {"arm": args.reference_arm, "seed": args.reference_seed,
                      "n_obs": int(obs_fixed.shape[0]), "cache": str(cache)},
        "own_steps": args.own_steps, "head_arms": list(HEAD_ARMS) + list(EXTRA_HEAD_ARMS),
        "confirmatory": False, "exploratory": True,
    }

    lines = ["=== RC4 encoder rank (fixed observation set) ==="]
    for arm, m in per_seed.items():
        tag = "head" if (arm in HEAD_ARMS or arm in EXTRA_HEAD_ARMS) else "no-head"
        lines.append(f"  {arm:<16} [{tag:<7}] erank={np.nanmean(m['erank_fixed']):7.3f} "
                     f"(sd {np.nanstd(m['erank_fixed'], ddof=1):.3f})  "
                     f"srank={np.nanmean(m['srank_fixed']):6.1f}  "
                     f"dead={np.nanmean(m['dead_frac_fixed']):.3f}")
    if report["rc4"]:
        lines += ["", "=== RC4 primary: minimal pairs on erank_fixed ==="]
        for label, r in report["rc4"]["minimal_pairs"].items():
            lines.append(f"  {label:<32} diff={r['mean_diff']:+8.3f} d={r['cohens_d']:+.3g} "
                         f"BCa=[{r['bca_low']:+.3f}, {r['bca_high']:+.3f}] "
                         f"p_signflip={r['p_sign_flip']:.3g} p_holm={r['p_holm']:.3g}")
        lines.append(f"  verdict: {report['rc4']['verdict']}")
    if args.eval_ref:
        c = report["rank_vs_inventory_sd_off"]
        lines.append(f"\n  descriptive: Spearman(erank, inventory_sd_off) = {c['spearman_rho']:+.3f} "
                     f"(n={c['n']}, p={c['p']:.3g})")
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
