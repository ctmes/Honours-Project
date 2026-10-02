"""H3m: does a detection head trained on MATERIAL attacks detect the maximal one-sided attack?

Applies the rules pre-registered in preregistration.json -> amendments ->
h3_material_attack_amendment_2026-10-02 to the eval_forced_attack.py outputs of the
scripted_detection arm (v4s_config4_detection: v4 detection arm, adversary replaced by
a scripted one-sided injection during training, ADV_SCRIPTED).

Primary, per side: one-sample test of the 20 per-seed AUROCs (det_prob, forced vs off)
against 0.5 (stats.one_sample_comparison, plus the RC1 sign-flip), Holm over the two
sides, classified by the RC3 rule (analysis.rc3_detection_check.classify):
    mean >= 0.60 and Holm p < 0.05        -> DETECTS
    95% CI of the mean inside [0.45, 0.55] -> CANNOT DETECT
    otherwise                             -> INCONCLUSIVE
H3m is SUPPORTED if both sides DETECT, NOT SUPPORTED if both CANNOT DETECT, else
INCONCLUSIVE.

Secondary (paired by seed against the v4 detection arm, sign-flip p, Holm over the
family; descriptive, never a gate): the AUROC gain from material training per side,
the clean cost (sharpe_off, sortino_off, per step), clean inventory activity
(inventory_sd_off, the no-trade-collapse indicator), and the forced-attack
difference-in-differences on Sharpe per side (does detecting the attack change how
the attack affects the market maker).

The v4 detection arm comes from the same file if it was evaluated alongside, else from
--ref-bid / --ref-ask (the RC3 re-runs, which carry its det_auroc_forced_vs_off).

    python -m analysis.h3_material_check results/forced_attack_v4s_bid.json \\
        results/forced_attack_v4s_ask.json --ref-bid results/forced_attack_v4_bid_rc3.json \\
        --ref-ask results/forced_attack_v4_ask_rc3.json --out results/h3_material_check
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from analysis.rc3_detection_check import classify
from gymnax_exchange.jaxrl.MARL.adversarial_eval.stats import (
    bca_ci, holm_adjust, one_sample_comparison, sign_flip_permutation,
)

ARM = "scripted_detection"
REF_ARM = "detection"
AUROC = "det_auroc_forced_vs_off"


def _ps(d: dict, arm: str, key: str) -> np.ndarray:
    return np.asarray(d["per_seed"][arm][key], dtype=np.float64)


def _scale(d: dict, metric: str) -> float:
    ppy = float(d.get("_meta", {}).get("periods_per_year") or 0.0)
    return 1.0 / math.sqrt(ppy) if ppy > 0 and metric.startswith(("sharpe", "sortino")) else 1.0


def _paired(diff: np.ndarray, scale: float = 1.0) -> dict:
    diff = diff[np.isfinite(diff)]
    lo, hi = bca_ci(diff)
    return {"n": int(diff.size), "mean_diff": float(diff.mean()) * scale,
            "bca_low": lo * scale, "bca_high": hi * scale, "p": sign_flip_permutation(diff)}


def _ref(d: dict, ref: dict | None) -> dict | None:
    if REF_ARM in d["per_seed"]:
        return d
    if ref is not None and REF_ARM in ref["per_seed"]:
        return ref
    return None


def primary(sides: dict) -> dict:
    cells, pvals = {}, {}
    for side, d in sides.items():
        x = _ps(d, ARM, AUROC)
        x = x[np.isfinite(x)]
        r = one_sample_comparison(x, 0.5)
        lo, hi = bca_ci(x)
        cells[side] = {"n": int(x.size), "mean": float(x.mean()), "bca_low": lo, "bca_high": hi,
                       "test": r.test, "p": r.p_value, "p_sign_flip": sign_flip_permutation(x - 0.5),
                       "cohens_d": r.cohens_d,
                       "det_prob_mean_forced": float(np.nanmean(_ps(d, ARM, "det_prob_mean_forced"))),
                       "det_prob_mean_off": float(np.nanmean(_ps(d, ARM, "det_prob_mean_off")))}
        pvals[side] = r.p_value
    adj = holm_adjust(pvals) if pvals else {}
    for side, c in cells.items():
        c["p_holm"] = adj[side]
        c["verdict"] = classify(c["mean"], c["bca_low"], c["bca_high"], c["p_holm"])
    v = {c["verdict"] for c in cells.values()}
    if len(cells) == 2 and v == {"DETECTS"}:
        overall = "SUPPORTED"
    elif len(cells) == 2 and v == {"CANNOT DETECT"}:
        overall = "NOT SUPPORTED"
    else:
        overall = "INCONCLUSIVE"
    return {"cells": cells, "overall": overall}


def secondary(sides: dict, refs: dict) -> dict:
    fam = {}
    for side, d in sides.items():
        r = _ref(d, refs.get(side))
        if r is None:
            continue
        if AUROC in r["per_seed"][REF_ARM]:
            fam[f"{side}:auroc_gain"] = _paired(_ps(d, ARM, AUROC) - _ps(r, REF_ARM, AUROC))
        eff = lambda f, a: _ps(f, a, "sharpe_forced") - _ps(f, a, "sharpe_off")
        fam[f"{side}:did_sharpe"] = _paired(eff(d, ARM) - eff(r, REF_ARM), _scale(d, "sharpe"))
    # the 'off' condition does not depend on the side, so clean metrics come from one file
    side0 = next(iter(sides))
    d0 = sides[side0]
    r0 = _ref(d0, refs.get(side0))
    clean = {}
    if r0 is not None:
        for m in ("sharpe_off", "sortino_off", "inventory_sd_off"):
            fam[f"clean:{m}"] = _paired(_ps(d0, ARM, m) - _ps(r0, REF_ARM, m), _scale(d0, m))
            clean[m] = {ARM: float(np.nanmean(_ps(d0, ARM, m))) * _scale(d0, m),
                        REF_ARM: float(np.nanmean(_ps(r0, REF_ARM, m))) * _scale(d0, m)}
    adj = holm_adjust({k: t["p"] for k, t in fam.items()}) if fam else {}
    for k, t in fam.items():
        t["p_holm"] = adj[k]
    return {"tests": fam, "clean_means": clean}


def off_consistent(sides: dict) -> bool | None:
    """The off condition is side-independent, so both files must agree on it."""
    if len(sides) < 2:
        return None
    a, b = sides.values()
    x, y = _ps(a, ARM, "sharpe_off"), _ps(b, ARM, "sharpe_off")
    return bool(x.shape == y.shape and np.allclose(x, y, rtol=1e-9, atol=1e-12))


def run(bid: str | None, ask: str | None, ref_bid: str | None, ref_ask: str | None) -> dict:
    load = lambda p: json.load(open(p)) if p else None
    sides = {s: load(p) for s, p in (("bid", bid), ("ask", ask)) if p}
    refs = {s: load(p) for s, p in (("bid", ref_bid), ("ask", ref_ask)) if p}
    for s, d in sides.items():
        if ARM not in d["per_seed"]:
            raise SystemExit(f"{s} file has no '{ARM}' arm")
        side_meta = d.get("_meta", {}).get("side")
        if side_meta and side_meta != s:
            raise SystemExit(f"{s} file was run with --side {side_meta}")
    return {"primary": primary(sides), "secondary": secondary(sides, refs),
            "off_consistent_across_sides": off_consistent(sides),
            "rule": "per side: mean AUROC >= 0.60 & Holm(2) p < 0.05 -> DETECTS; CI in "
                    "[0.45,0.55] -> CANNOT DETECT; both DETECT -> H3m SUPPORTED"}


def format_text(res: dict) -> str:
    p = res["primary"]
    lines = ["=== H3m: detection head trained on material attacks, forced maximal one-sided "
             "attack (EXPLORATORY) ==="]
    for side, c in p["cells"].items():
        lines.append(f"  {side:<4} AUROC={c['mean']:.4f} BCa=[{c['bca_low']:.4f}, {c['bca_high']:.4f}] "
                     f"p={c['p']:.3g} p_holm={c['p_holm']:.3g} p_sf={c['p_sign_flip']:.3g} "
                     f"det_prob forced/off={c['det_prob_mean_forced']:.4f}/{c['det_prob_mean_off']:.4f}"
                     f"  -> {c['verdict']}")
    lines.append(f"  H3m: {p['overall']}")
    if res["off_consistent_across_sides"] is not None:
        lines.append("  off condition identical across sides: "
                     + ("yes" if res["off_consistent_across_sides"] else "!! NO -- check the runs"))
    s = res["secondary"]
    if s["tests"]:
        lines += ["", f"--- secondary: {ARM} minus {REF_ARM}, paired by seed "
                      "(per step for Sharpe/Sortino; sign-flip p, Holm over the family) ---"]
        for k, t in s["tests"].items():
            lines.append(f"  {k:<26} diff={t['mean_diff']:+.4g} BCa=[{t['bca_low']:+.4g}, "
                         f"{t['bca_high']:+.4g}] p={t['p']:.3g} p_holm={t['p_holm']:.3g}")
        for m, v in s["clean_means"].items():
            lines.append(f"  mean {m:<20} {ARM}={v[ARM]:.4g}  {REF_ARM}={v[REF_ARM]:.4g}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bid")
    ap.add_argument("ask")
    ap.add_argument("--ref-bid", default=None, help="file holding the v4 detection arm (bid)")
    ap.add_argument("--ref-ask", default=None, help="file holding the v4 detection arm (ask)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(args.bid, args.ask, args.ref_bid, args.ref_ask)
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
