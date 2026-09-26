"""H4 (regime conditioning) has never actually been tested.

`rollout.py` computes `sortino_lowvol`, `sortino_highvol` and `regime_gap`
(= |sortino_highvol - sortino_lowvol|) per arm/seed specifically for H4 (see
its "Regime-split Sortino + absolute gap (H4)" comment) -- but
`run_evaluation.py`'s inferential contrasts (`compare_configs`, Holm) only
ever run over `primary_metrics = (sortino_on, sharpe_on, cvar_on, auroc)`.
`regime_gap` is written into every arm's per-seed summary and then never
compared between arms anywhere in the pipeline. H4's own purpose-built metric
has been computed every eval run and never tested. This is not a new metric
invented after seeing results -- it existed for this exact purpose and was
simply never wired into the comparison step.

H4's claim is that regime conditioning REDUCES the gap between high- and
low-volatility performance, so the sign to look for is: gap(no regime arm) -
gap(regime arm) > 0. The 2x2 factorial (detection head x regime) gives two
clean, minimal-pair isolations of the regime factor alone:
  - regime vs adversarial   (regime effect when detection head = off)
  - full  vs detection      (regime effect when detection head = on)

`regime_gap` was never in `primary_metrics`, so this is EXPLORATORY even on
v3 data -- it was not pre-registered, and reporting it now must say so
explicitly, the same as any other post-hoc test. It differs from a chased
result only in that the metric was built for exactly this hypothesis before
any of this session's results existed, not selected after seeing them.
"""

from __future__ import annotations

import argparse

from analysis.evalreport import EvalReport
from gymnax_exchange.jaxrl.MARL.adversarial_eval.stats import paired_comparison

METRICS = ("regime_gap_on", "regime_gap_off", "sortino_lowvol_on", "sortino_highvol_on")
# (no_regime_arm, regime_arm) -- diff is computed as no_regime - regime, so a
# POSITIVE diff on regime_gap means the regime arm has the SMALLER gap (H4 support).
ISOLATIONS = (
    ("adversarial", "regime"),   # regime effect at detection=off
    ("detection", "full"),       # regime effect at detection=on
)


def run(path: str, label: str) -> None:
    report = EvalReport.load(path)
    print(f"=== H4 regime_gap check: {label} ({path}) ===")
    print(f"exploratory={report.meta.get('exploratory')} "
          f"confirmatory={report.meta.get('confirmatory')} "
          f"project_prefix={report.meta.get('project_prefix')}  "
          f"[this test itself is EXPLORATORY regardless -- not in primary_metrics]")
    print()
    for no_regime, regime in ISOLATIONS:
        if not report.has(no_regime, "regime_gap_on") or not report.has(regime, "regime_gap_on"):
            print(f"[skip] {no_regime}_vs_{regime}: arm missing from this report")
            continue
        print(f"--- {no_regime} (no regime) vs {regime} (regime) ---")
        for metric in METRICS:
            if not report.has(no_regime, metric) or not report.has(regime, metric):
                continue
            x, y = report.paired(no_regime, regime, metric)
            mask = report.finite(x, y)
            if mask.sum() < 2:
                print(f"  {metric:<20} insufficient finite pairs ({int(mask.sum())})")
                continue
            r = paired_comparison(x[mask], y[mask])
            tag = ""
            if metric.startswith("regime_gap"):
                tag = "  <- H4 support if diff > 0" if r.mean_diff > 0 else "  <- WRONG DIRECTION for H4"
            print(f"  {metric:<20} n={r.n:<3} diff={r.mean_diff:+8.3g} "
                  f"d={r.cohens_d:+.3g} {r.test:<9} p={r.p_value:.3g} "
                  f"CI=[{r.ci_low:.3g}, {r.ci_high:.3g}]{tag}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("eval_json")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    run(args.eval_json, args.label or args.eval_json)
