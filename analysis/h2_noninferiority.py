"""H2 one-sided non-inferiority re-test — a pre-identified specification fix,
not a new statistical method.

H2 asks whether the defended market maker degrades on CLEAN data relative to
baseline. `run_evaluation.py` tests this with a symmetric two-sided TOST
(`tost_paired`, `equivalence_off` in the eval report): equivalence requires
BOTH one-sided nulls to reject, i.e. the paired difference must sit inside
(-margin, +margin). For v3 `full_vs_baseline`, the defended arm is not worse
on clean data, it is a lot BETTER (sortino_off, sharpe_off both positive and
far outside the +/-0.5 margin), so the upper-bound half of the symmetric test
fails by construction and "not shown equivalent" is reported -- even though
the only thing H2 is actually a claim about (no DEGRADATION) already holds.

This script does not add a new test. `tost_paired` already computes the
one-sided lower-bound statistic internally (`p_lower`, H0: diff <= -margin);
this script reads that half out and reports it alone as the non-inferiority
p-value, per NEJM/FDA non-inferiority convention (Wellek 2010; Walker &
Nowacki 2011) for a hypothesis where only one direction of "different"
matters. The upper-bound half (`p_upper`) is reported alongside for
transparency but is not part of the non-inferiority verdict.

This was flagged as the correct test BEFORE this script was written (see
project notes on the v3 result), not chosen after inspecting outcomes here.
It is reported as a labelled supplementary re-analysis alongside, not instead
of, the pre-registered two-sided H2 result in `eval_1179095.json`, which is
left untouched.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from analysis.evalreport import EvalReport
from gymnax_exchange.jaxrl.MARL.adversarial_eval.stats import tost_paired

H2_METRICS = ("sortino_off", "sharpe_off")
DEFENDED_ARMS = ("full", "adversarial")


def noninferiority(report: EvalReport, defended: str, metric: str,
                    margin: float, alpha: float = 0.05) -> dict:
    a, b = report.paired(defended, "baseline", metric)
    mask = report.finite(a, b)
    a, b = a[mask], b[mask]
    t = tost_paired(a, b, margin=margin, alpha=alpha)
    return {
        "defended": defended,
        "baseline": "baseline",
        "metric": metric,
        "n": t.n,
        "mean_diff": t.mean_diff,
        "margin": margin,
        "p_lower_noninferiority": t.p_lower,
        "p_upper_not_used_for_verdict": t.p_upper,
        "two_sided_tost_p": t.p_value,
        "noninferior": bool(t.p_lower < alpha),
        "two_sided_equivalent": t.equivalent,
    }


def run(path: str) -> list[dict]:
    report = EvalReport.load(path)
    margins = report.equivalence_margins()
    if not margins:
        raise SystemExit(f"{path}: no equivalence_margins_resolved block -- "
                          "H2 was not run in this eval, nothing to re-test.")
    rows = []
    for defended in DEFENDED_ARMS:
        if not report.has(defended, H2_METRICS[0]) or not report.has("baseline", H2_METRICS[0]):
            continue
        for metric in H2_METRICS:
            margin = margins.get(metric)
            if margin is None:
                continue
            rows.append(noninferiority(report, defended, metric, margin))
    return rows


def format_rows(rows: list[dict]) -> str:
    lines = ["=== H2 one-sided non-inferiority re-test (supplementary to the "
             "pre-registered two-sided TOST) ===", ""]
    for r in rows:
        verdict = "NON-INFERIOR" if r["noninferior"] else "not shown non-inferior"
        two_sided = "equivalent" if r["two_sided_equivalent"] else "not equivalent"
        lines.append(
            f"{r['defended']}_vs_baseline  {r['metric']:<12} n={r['n']:<3} "
            f"diff={r['mean_diff']:+.3g} margin=+/-{r['margin']:.3g}  "
            f"p_lower={r['p_lower_noninferiority']:.3g} -> {verdict}   "
            f"(two-sided TOST p={r['two_sided_tost_p']:.3g} -> {two_sided})"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("eval_json", help="path to an eval_*.json report, e.g. results/eval_1179095.json")
    ap.add_argument("--out", default=None, help="optional path to write the rows as JSON")
    args = ap.parse_args()

    rows = run(args.eval_json)
    print(format_rows(rows))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.out}")
