"""H1 (adversarial robustness) has been tested with the wrong contrast.

`run_evaluation.py`'s inferential contrasts (`compare_configs`, Holm) compare
arms to each other UNDER ATTACK only (`full_vs_baseline` etc. on
`sortino_on`/`sharpe_on`/`cvar_on`/`auroc`). That answers "is arm E better
than arm A while both are being attacked", which is confounded with general
policy quality -- some arms partially collapse to a no-trade equilibrium for
reasons that have nothing to do with the adversary being present. It is NOT a
test of "does the attack change this arm's outcome", which is what H1 actually
claims (does a defence make the MM more robust TO BEING ATTACKED).

Every eval report already carries the `_on`/`_off` pair needed for the
correct test -- it was just never run:

  1. WITHIN-ARM: does attack-on differ from attack-off, for this arm, on its
     own? (paired by seed, same arm). This is the direct causal check of
     whether the adversary does anything at all to a given policy.

  2. CROSS-ARM DIFFERENCE-IN-DIFFERENCES: does the attack's effect
     (on - off) differ between two arms? This is the actual H1 test -- it
     isolates whether a defence changes the SIZE of the attack's effect,
     rather than comparing two arms' absolute quality under one condition.
     Valid as a paired test because `preregistration.json`'s common_adversary
     design pairs every arm's seeds by index against the same adversary.

This was flagged as the correct specification before this script was written
(see docs/critical_review_2026-09-26.md, "the properly-specified H1 test"),
not chosen after inspecting which framing looked better. It is reported
alongside, not instead of, the pre-registered `full_vs_baseline` contrast
already in the eval JSON, which is left untouched.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from analysis.evalreport import EvalReport
from gymnax_exchange.jaxrl.MARL.adversarial_eval.stats import (
    holm_adjust, paired_comparison,
)

METRICS = ("sharpe", "sortino", "cvar")

# In every CONTRAST_LABELS pair "arm_a_vs_arm_b", arm_a carries at least as much
# defence/adversary-exposure as arm_b (e.g. full_vs_baseline: E has both defences,
# A has none; adversarial_vs_baseline: B trained with an adversary, A did not).
# H1 predicts the more-defended arm degrades LESS under attack than the less-
# defended one, i.e. diff_a (on-off) > diff_b (on-off), i.e. DiD > 0.
# adversarial_vs_unconstrained is a different kind of contrast (adversary cost
# model, not MM defence level) and is not directionally scored.
_DIRECTIONAL_CONTRASTS = {
    "adversarial_vs_baseline", "detection_vs_adversarial", "full_vs_regime",
    "regime_vs_adversarial", "full_vs_detection", "full_vs_adversarial",
    "full_vs_baseline",
}

# Six sanity-reference values already hand-verified against results/eval_1270.json
# (sharpe, within-arm on-vs-off, n=20) -- checked on Cohen's d only, printed as
# MATCH/MISMATCH not asserted. The reference p-values were computed with a plain
# paired t-test and are NOT compared here: paired_comparison() correctly screens
# each arm's (on-off) distribution for normality and falls back to Wilcoxon where
# it fails, which is the right thing for this pipeline to do and legitimately
# gives a different p than a naive always-t-test would -- that is not a bug in
# either number, so checking p here would flag a false mismatch on every row.
_SANITY_REFERENCE = {
    ("eval_1270.json", "baseline", "sharpe"): -0.31,
    ("eval_1270.json", "adversarial", "sharpe"): 0.22,
    ("eval_1270.json", "detection", "sharpe"): -0.13,
    ("eval_1270.json", "regime", "sharpe"): -0.21,
    ("eval_1270.json", "full", "sharpe"): 0.19,
    ("eval_1270.json", "unconstrained", "sharpe"): 0.21,
}
_SANITY_TOL_D = 0.02


def within_arm(report: EvalReport, source_name: str) -> list[dict]:
    """Does attack-on differ from attack-off, within each arm, on its own?"""
    rows = []
    for arm in report.factorial_arms():
        for metric in METRICS:
            on_key, off_key = f"{metric}_on", f"{metric}_off"
            if not (report.has(arm, on_key) and report.has(arm, off_key)):
                continue
            on, off = report.per_seed(arm, on_key), report.per_seed(arm, off_key)
            mask = report.finite(on, off)
            if mask.sum() < 2:
                continue
            r = paired_comparison(on[mask], off[mask])
            check = ""
            ref_d = _SANITY_REFERENCE.get((source_name, arm, metric))
            if ref_d is not None:
                ok = abs(r.cohens_d - ref_d) <= _SANITY_TOL_D
                check = "  [MATCH]" if ok else f"  [MISMATCH expected d={ref_d}]"
            rows.append({
                "arm": arm, "metric": metric, "n": r.n,
                "mean_on": r.mean_a, "mean_off": r.mean_b,
                "diff": r.mean_diff, "cohens_d": r.cohens_d,
                "test": r.test, "p_value": r.p_value,
                "ci_low": r.ci_low, "ci_high": r.ci_high,
                "sanity_check": check.strip() or None,
            })
    return rows


def cross_arm_diff_in_diff(report: EvalReport) -> list[dict]:
    """Does the attack's effect (on - off) differ between two arms?

    This is the actual H1 test: for each pre-registered contrast, compare
    arm_a's (on - off) against arm_b's (on - off), paired by seed index.
    """
    rows = []
    for contrast in report.contrasts():
        arm_a, arm_b = contrast.split("_vs_")
        pvals = {}
        row_data = {}
        for metric in METRICS:
            on_key, off_key = f"{metric}_on", f"{metric}_off"
            needed = [(arm_a, on_key), (arm_a, off_key),
                      (arm_b, on_key), (arm_b, off_key)]
            if not all(report.has(a, k) for a, k in needed):
                continue
            diff_a = report.per_seed(arm_a, on_key) - report.per_seed(arm_a, off_key)
            diff_b = report.per_seed(arm_b, on_key) - report.per_seed(arm_b, off_key)
            mask = report.finite(diff_a, diff_b)
            if mask.sum() < 2:
                continue
            r = paired_comparison(diff_a[mask], diff_b[mask])
            pvals[metric] = r.p_value
            row_data[metric] = r
        if report.is_confirmatory and pvals:
            adjusted = holm_adjust(pvals)
        else:
            adjusted = {k: None for k in pvals}
        for metric, r in row_data.items():
            direction = None
            if contrast in _DIRECTIONAL_CONTRASTS:
                direction = "H1 direction" if r.mean_diff > 0 else "WRONG direction"
            rows.append({
                "contrast": contrast, "arm_a": arm_a, "arm_b": arm_b,
                "metric": metric, "n": r.n,
                "diff_in_diff": r.mean_diff, "cohens_d": r.cohens_d,
                "test": r.test, "p_value": r.p_value,
                "p_holm": adjusted.get(metric),
                "ci_low": r.ci_low, "ci_high": r.ci_high,
                "direction": direction,
            })
    return rows


def format_report(report: EvalReport, source_name: str,
                   within: list[dict], did: list[dict]) -> str:
    lines = [
        f"=== H1 attack-effect check: {source_name} ===",
        report.describe(),
        f"is_confirmatory={report.is_confirmatory}  "
        f"[DiD Holm-adjusted only if True; otherwise exploratory estimates]",
        "",
        "--- (1) WITHIN-ARM: attack-on vs attack-off, paired by seed ---",
        "  (the direct test of whether the adversary does anything to this arm)",
    ]
    for r in within:
        tag = f"  {r['sanity_check']}" if r["sanity_check"] else ""
        lines.append(
            f"  {r['arm']:<15} {r['metric']:<8} n={r['n']:<3} "
            f"diff={r['diff']:+8.3g} d={r['cohens_d']:+.3g} {r['test']:<9} "
            f"p={r['p_value']:.3g}{tag}")
    lines += ["", "--- (2) CROSS-ARM DIFFERENCE-IN-DIFFERENCES (the actual H1 test) ---",
              "  (does the attack's effect differ between arms? paired by seed index)"]
    for r in did:
        p_str = f"p={r['p_value']:.3g}"
        if r["p_holm"] is not None:
            p_str += f" (Holm p={r['p_holm']:.3g})"
        tag = f"  <- {r['direction']}" if r["direction"] else ""
        lines.append(
            f"  {r['contrast']:<28} {r['metric']:<8} n={r['n']:<3} "
            f"DiD={r['diff_in_diff']:+8.3g} d={r['cohens_d']:+.3g} "
            f"{r['test']:<9} {p_str}{tag}")
    return "\n".join(lines)


def run(path: str, label: str) -> dict:
    report = EvalReport.load(path)
    source_name = label or path.split("/")[-1]
    within = within_arm(report, source_name)
    did = cross_arm_diff_in_diff(report)
    print(format_report(report, source_name, within, did))
    return {"source": source_name, "within_arm": within, "diff_in_diff": did}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_json", help="path to an eval_*.json report")
    ap.add_argument("--label", default="", help="source name for sanity-check lookup, "
                    "e.g. 'eval_1270.json' (defaults to the filename)")
    ap.add_argument("--out", default=None, help="optional path to write rows as JSON")
    args = ap.parse_args()

    result = run(args.eval_json, args.label)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.out}")
