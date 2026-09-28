"""RC3: can the detection head detect a maximal one-sided attack?

Reads the two eval_forced_attack.py re-runs of the head arms (bid, ask), which now
carry det_auroc_forced_vs_off per seed, and applies the rule pre-registered in
preregistration.json -> amendments -> robustness_checks_amendment_2026-09-28:

  per arm x side: one-sample test of the 20 per-seed AUROCs against 0.5
  (stats.one_sample_comparison, plus the RC1 sign-flip), Holm over the 6 cells.
    mean >= 0.60 and Holm p < 0.05       -> DETECTS: H3's chance result was a
                                            label/treatment artefact
    95% CI of the mean inside [0.45, 0.55] -> CANNOT DETECT even a maximal attack
    otherwise                             -> INCONCLUSIVE

Faithfulness: the re-run must reproduce the existing forced-attack grid exactly on
every metric both files carry (same loop, same PRNG keys). --reference-bid /
--reference-ask default to results/forced_attack_v4_{bid,ask}.json.

    python -m analysis.rc3_detection_check results/forced_attack_v4_bid_rc3.json \\
        results/forced_attack_v4_ask_rc3.json --out results/rc3_detection
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from gymnax_exchange.jaxrl.MARL.adversarial_eval.stats import (
    bca_ci, holm_adjust, one_sample_comparison, sign_flip_permutation,
)

HEAD_ARMS = ("detection", "full", "detection_noobs")
_CHECK_KEYS = ("sharpe_forced", "sharpe_off", "sortino_forced", "sortino_off",
               "cvar_forced", "cvar_off", "inventory_sd_forced", "qi_mean_forced")


def faithfulness(new: dict, ref: dict) -> list[dict]:
    out = []
    for arm in HEAD_ARMS:
        a, b = new["per_seed"].get(arm), ref["per_seed"].get(arm)
        if a is None or b is None:
            continue
        for k in _CHECK_KEYS:
            if k in a and k in b:
                x, y = np.asarray(a[k], float), np.asarray(b[k], float)
                diff = float(np.nanmax(np.abs(x - y))) if x.shape == y.shape else float("inf")
                out.append({"arm": arm, "metric": k, "max_abs_diff": diff,
                            "match": diff <= 1e-6 * max(1.0, float(np.nanmax(np.abs(y))))})
    return out


def classify(mean: float, lo: float, hi: float, p_holm: float) -> str:
    if mean >= 0.60 and p_holm < 0.05:
        return "DETECTS"
    if 0.45 <= lo and hi <= 0.55:
        return "CANNOT DETECT"
    return "INCONCLUSIVE"


def run(bid_path: str, ask_path: str, ref_bid: str | None, ref_ask: str | None) -> dict:
    cells, pvals, faith = {}, {}, {}
    for side, path, ref_path in (("bid", bid_path, ref_bid), ("ask", ask_path, ref_ask)):
        new = json.load(open(path))
        if ref_path:
            faith[side] = faithfulness(new, json.load(open(ref_path)))
        for arm in HEAD_ARMS:
            vals = new["per_seed"].get(arm, {}).get("det_auroc_forced_vs_off")
            if vals is None:
                continue
            x = np.asarray(vals, float)
            x = x[np.isfinite(x)]
            r = one_sample_comparison(x, 0.5)
            p_sf = sign_flip_permutation(x - 0.5)
            lo, hi = bca_ci(x)
            key = f"{arm}/{side}"
            cells[key] = {"arm": arm, "side": side, "n": int(x.size), "mean": float(x.mean()),
                          "bca_low": lo, "bca_high": hi, "test": r.test, "p": r.p_value,
                          "p_sign_flip": p_sf, "cohens_d": r.cohens_d,
                          "det_prob_mean_forced": float(np.nanmean(new["per_seed"][arm]["det_prob_mean_forced"])),
                          "det_prob_mean_off": float(np.nanmean(new["per_seed"][arm]["det_prob_mean_off"]))}
            pvals[key] = r.p_value
    holm = holm_adjust(pvals) if pvals else {}
    for key, c in cells.items():
        c["p_holm"] = holm[key]
        c["verdict"] = classify(c["mean"], c["bca_low"], c["bca_high"], c["p_holm"])
    verdicts = {c["verdict"] for c in cells.values()}
    overall = verdicts.pop() if len(verdicts) == 1 else "MIXED: " + ", ".join(
        f"{k}={c['verdict']}" for k, c in cells.items())
    all_faithful = all(r["match"] for rows in faith.values() for r in rows) if faith else None
    return {"cells": cells, "overall": overall, "faithfulness": faith,
            "all_faithful": all_faithful,
            "rule": "mean>=0.60 & Holm p<0.05 -> DETECTS; CI in [0.45,0.55] -> CANNOT DETECT; "
                    "else INCONCLUSIVE; Holm over 6 arm x side cells"}


def format_text(res: dict) -> str:
    lines = ["=== RC3: detection AUROC, forced maximal one-sided vs clean ==="]
    if res["all_faithful"] is not None:
        lines.append("  faithfulness vs existing forced grid: "
                     + ("ALL MATCH" if res["all_faithful"] else "!! MISMATCH -- do not use"))
    for key, c in res["cells"].items():
        lines.append(f"  {key:<22} AUROC={c['mean']:.4f} BCa=[{c['bca_low']:.4f}, {c['bca_high']:.4f}] "
                     f"p={c['p']:.3g} p_holm={c['p_holm']:.3g} p_sf={c['p_sign_flip']:.3g} "
                     f"det_prob forced/off={c['det_prob_mean_forced']:.4f}/{c['det_prob_mean_off']:.4f}"
                     f"  -> {c['verdict']}")
    lines.append(f"  overall: {res['overall']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bid")
    ap.add_argument("ask")
    ap.add_argument("--reference-bid", default="results/forced_attack_v4_bid.json")
    ap.add_argument("--reference-ask", default="results/forced_attack_v4_ask.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(args.bid, args.ask, args.reference_bid, args.reference_ask)
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
