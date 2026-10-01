"""Apply the pre-registered WS10 decision rules (tier4_extensions_amendment_2026-09-28).

Inputs (any subset; each section runs only if its file is given):
  --learned   eval_learned_attack.py output         (WS10a learned optimal attack)
  --eval      run_production_eval.py --project-prefix v4 output containing
              adversarial, detection_noobs, shuffled_noobs, baseline, full,
              recurrent_baseline, recurrent_full    (WS10b, WS10c clean performance)
  --forced-bid / --forced-ask  eval_forced_attack.py outputs on baseline, full,
              recurrent_baseline, recurrent_full    (WS10c robustness)

Every test is the exact sign-flip permutation test on paired per-seed differences
(RC1), with the Holm families fixed in the amendment. Sharpe/Sortino differences
are reported per step.

    python -m analysis.tier4_check --learned results/learned_attack_v4.json \\
        --eval results/eval_tier4.json --forced-bid results/forced_attack_tier4_bid.json \\
        --forced-ask results/forced_attack_tier4_ask.json --out results/tier4_check
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from gymnax_exchange.jaxrl.MARL.adversarial_eval.stats import (
    bca_ci, holm_adjust, sign_flip_permutation,
)

ALPHA = 0.05


def _load(path):
    with open(path) as f:
        return json.load(f)


def _ps(d, arm, key):
    return np.asarray(d["per_seed"][arm][key], dtype=np.float64)


def _scale(d, metric):
    ppy = float(d.get("_meta", {}).get("periods_per_year") or 0.0)
    return 1.0 / math.sqrt(ppy) if ppy > 0 and metric.startswith(("sharpe", "sortino")) else 1.0


def _test(diff, scale=1.0):
    diff = diff[np.isfinite(diff)]
    lo, hi = bca_ci(diff)
    return {"n": int(diff.size), "mean_diff": float(diff.mean()) * scale,
            "bca_low": lo * scale, "bca_high": hi * scale,
            "p": sign_flip_permutation(diff)}


def ws10a(d: dict) -> dict:
    arms = [a for a in ("baseline", "adversarial", "full") if a in d["per_seed"]]
    out = {"within_arm": {}, "did": {}, "adversary": {}}
    for metric in ("sharpe", "sortino"):
        fam = {}
        for arm in arms:
            diff = _ps(d, arm, f"{metric}_learned") - _ps(d, arm, f"{metric}_off")
            fam[arm] = _test(diff, _scale(d, metric))
        adj = holm_adjust({a: r["p"] for a, r in fam.items()})
        for a, r in fam.items():
            r["p_holm"] = adj[a]
        out["within_arm"][metric] = fam
    for arm in arms:
        bid = _ps(d, arm, "mean_bid_volume_injected_learned")
        ask = _ps(d, arm, "mean_ask_volume_injected_learned")
        side = np.abs(bid - ask) / np.maximum(bid + ask, 1e-12)
        qi = _ps(d, arm, "qi_mean_learned") - _ps(d, arm, "qi_mean_off")
        out["adversary"][arm] = {"sidedness_mean": float(np.nanmean(side)),
                                 "qi_shift_mean": float(np.nanmean(qi))}
    base = out["within_arm"]["sharpe"].get("baseline")
    degraded = bool(base and base["mean_diff"] < 0 and base["p_holm"] < ALPHA)
    if "full" in arms and "baseline" in arms:
        fam = {}
        for metric in ("sharpe", "sortino"):
            eff = lambda a: _ps(d, a, f"{metric}_learned") - _ps(d, a, f"{metric}_off")
            fam[metric] = _test(eff("full") - eff("baseline"), _scale(d, metric))
        adj = holm_adjust({m: r["p"] for m, r in fam.items()})
        for m, r in fam.items():
            r["p_holm"] = adj[m]
        out["did"]["full_minus_baseline"] = fam
    if not degraded:
        verdict = ("H1 PREMISE FAILS under the learned optimal attack: the undefended baseline "
                   "is not significantly degraded (strongest null available)")
    else:
        did = out["did"].get("full_minus_baseline", {})
        if any(r["mean_diff"] > 0 and r["p_holm"] < ALPHA for r in did.values()):
            verdict = "H1 SUPPORTED (exploratory): baseline degraded and the full model degrades less"
        else:
            verdict = "Baseline degraded; the defence is NOT shown to reduce the attack's effect"
    out["baseline_degraded"] = degraded
    out["verdict"] = verdict
    return out


def ws10b(d: dict) -> dict:
    need = ("adversarial", "detection_noobs", "shuffled_noobs")
    if not all(a in d["per_seed"] for a in need):
        return {"skipped": f"eval file lacks one of {need}"}
    tests = {}
    for label, a, b in (("H_minus_B", "shuffled_noobs", "adversarial"),
                        ("H_minus_G", "shuffled_noobs", "detection_noobs")):
        for metric in ("sharpe_off", "inventory_sd_off"):
            tests[f"{label}:{metric}"] = _test(_ps(d, a, metric) - _ps(d, b, metric),
                                               _scale(d, metric))
    adj = holm_adjust({k: r["p"] for k, r in tests.items()})
    for k, r in tests.items():
        r["p_holm"] = adj[k]
    hb, hg = tests["H_minus_B:sharpe_off"], tests["H_minus_G:sharpe_off"]
    if hb["mean_diff"] > 0 and hb["p_holm"] < ALPHA and not hg["p_holm"] < ALPHA:
        verdict = ("REGULARISATION: shuffled labels reproduce the auxiliary-loss benefit; "
                   "label content is not needed")
    elif hg["mean_diff"] < 0 and hg["p_holm"] < ALPHA:
        verdict = "LABEL CONTENT MATTERS: shuffled labels lose part of the benefit"
    else:
        verdict = "INCONCLUSIVE under the pre-registered rule"
    return {"tests": tests, "verdict": verdict}


def ws10c(d: dict | None, fb: dict | None, fa: dict | None, ref: dict | None = None) -> dict:
    """ref: an earlier eval JSON (e.g. eval_17313) supplying the MLP arms when `d`
    holds only the recurrent ones. Valid because the eval is deterministic per seed:
    every re-run of these arms has reproduced eval_17313 exactly."""
    out = {}
    if d is not None:
        pairs = [("recurrent_baseline", "baseline"), ("recurrent_full", "full")]
        fam = {}
        for a, b in pairs:
            src_b = d if b in d["per_seed"] else ref
            if a in d["per_seed"] and src_b is not None and b in src_b["per_seed"]:
                fam[f"{a}_minus_{b}"] = _test(_ps(d, a, "sharpe_off") - _ps(src_b, b, "sharpe_off"),
                                              _scale(d, "sharpe_off"))
                fam[f"{a}_minus_{b}"]["inventory_sd_off"] = {
                    "recurrent": float(np.nanmean(_ps(d, a, "inventory_sd_off"))),
                    "mlp": float(np.nanmean(_ps(src_b, b, "inventory_sd_off")))}
                fam[f"{a}_minus_{b}"]["mlp_source"] = "same file" if src_b is d else "--mlp-ref"
        adj = holm_adjust({k: r["p"] for k, r in fam.items()}) if fam else {}
        for k, r in fam.items():
            r["p_holm"] = adj[k]
        out["clean_performance"] = fam
    rob = {}
    for side, f in (("bid", fb), ("ask", fa)):
        if f is None:
            continue
        for rec, mlp in (("recurrent_baseline", "baseline"), ("recurrent_full", "full")):
            if rec in f["per_seed"] and mlp in f["per_seed"]:
                eff = lambda a: _ps(f, a, "sharpe_forced") - _ps(f, a, "sharpe_off")
                rob[f"{side}:{rec}_minus_{mlp}"] = _test(eff(rec) - eff(mlp), _scale(f, "sharpe"))
    if rob:
        adj = holm_adjust({k: r["p"] for k, r in rob.items()})
        for k, r in rob.items():
            r["p_holm"] = adj[k]
    out["robustness_did"] = rob
    return out


def format_text(res: dict) -> str:
    lines = ["=== WS10 (Tier 4) pre-registered checks ==="]
    if "ws10a" in res:
        a = res["ws10a"]
        lines.append("[WS10a learned optimal attack]")
        for metric, fam in a["within_arm"].items():
            for arm, r in fam.items():
                lines.append(f"  within {arm:<12} {metric:<8} diff={r['mean_diff']:+.5f} "
                             f"BCa=[{r['bca_low']:+.5f},{r['bca_high']:+.5f}] p={r['p']:.3g} "
                             f"holm={r['p_holm']:.3g}")
        for arm, r in a["adversary"].items():
            lines.append(f"  adversary {arm:<12} sidedness={r['sidedness_mean']:.3f} "
                         f"QI shift={r['qi_shift_mean']:+.3f}")
        for metric, r in a["did"].get("full_minus_baseline", {}).items():
            lines.append(f"  DiD full-baseline {metric:<8} diff={r['mean_diff']:+.5f} "
                         f"p={r['p']:.3g} holm={r['p_holm']:.3g}")
        lines.append(f"  verdict: {a['verdict']}")
    if "ws10b" in res:
        b = res["ws10b"]
        lines.append("[WS10b shuffled-label head]")
        for k, r in b.get("tests", {}).items():
            lines.append(f"  {k:<32} diff={r['mean_diff']:+.5f} p={r['p']:.3g} holm={r['p_holm']:.3g}")
        lines.append(f"  verdict: {b.get('verdict', b.get('skipped'))}")
    if "ws10c" in res:
        c = res["ws10c"]
        lines.append("[WS10c recurrent MM]")
        for k, r in c.get("clean_performance", {}).items():
            lines.append(f"  clean {k:<34} diff={r['mean_diff']:+.5f} p={r['p']:.3g} "
                         f"holm={r['p_holm']:.3g} inv_sd rec/mlp="
                         f"{r['inventory_sd_off']['recurrent']:.2f}/{r['inventory_sd_off']['mlp']:.2f}")
        for k, r in c.get("robustness_did", {}).items():
            lines.append(f"  forced DiD {k:<30} diff={r['mean_diff']:+.5f} p={r['p']:.3g} "
                         f"holm={r['p_holm']:.3g}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--learned")
    ap.add_argument("--eval")
    ap.add_argument("--forced-bid")
    ap.add_argument("--forced-ask")
    ap.add_argument("--mlp-ref", help="eval JSON supplying baseline/full when --eval holds only "
                                      "the recurrent arms (e.g. results/eval_17313.json)")
    ap.add_argument("--out")
    args = ap.parse_args()
    res = {}
    if args.learned:
        res["ws10a"] = ws10a(_load(args.learned))
    ev = _load(args.eval) if args.eval else None
    if ev is not None:
        res["ws10b"] = ws10b(ev)
    if ev is not None or args.forced_bid or args.forced_ask:
        res["ws10c"] = ws10c(ev, _load(args.forced_bid) if args.forced_bid else None,
                             _load(args.forced_ask) if args.forced_ask else None,
                             _load(args.mlp_ref) if args.mlp_ref else None)
    text = format_text(res)
    print(text)
    if args.out:
        with open(f"{args.out}.json", "w") as f:
            json.dump(res, f, indent=2, default=float)
        with open(f"{args.out}.txt", "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}.json and {args.out}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
