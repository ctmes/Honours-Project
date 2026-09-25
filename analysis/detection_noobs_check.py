"""Does the detection benefit require the policy to CONSUME its own detection
probability, or does training the auxiliary BCE loss alone account for it?

v4_config7_detection_noobs (preregistration.json amendments) trains the same
detection head (BCE loss, PCGrad) as v4_config4_detection but sets
prev_detection_in_obs=false, so the policy never sees det_prob. Everything
else -- architecture, cost model, seeds, reward -- matches v4_config4_detection.

If detection_noobs reproduces detection's gain over the no-head arms despite
never consuming the signal, the gain cannot be attributed to the policy
conditioning on suspected attacks (the mechanism contribution 2 claims) and is
better explained as auxiliary-task regularisation of the shared encoder.

This is EXPLORATORY (inherits the v4 gate: project_prefix != "v3"). It is not
part of the pre-registered confirmatory family -- report as an estimate with
CI, not a significance claim, same treatment as every other v4 result.
"""

from __future__ import annotations

import argparse

from analysis.evalreport import EvalReport
from gymnax_exchange.jaxrl.MARL.adversarial_eval.stats import paired_comparison

METRICS = ("sharpe_off", "sortino_off", "cvar_off", "inventory_sd_off")
CONTRASTS = (
    ("detection_noobs", "detection"),   # the decisive one: same head, obs vs no-obs
    ("detection_noobs", "baseline"),    # still beats no-head at all?
    ("detection_noobs", "adversarial"), # ditto, vs the plain co-trained arm
)


def run(path: str) -> None:
    report = EvalReport.load(path)
    print(f"=== detection_noobs mechanism check ({path}) ===")
    print(f"exploratory={report.meta.get('exploratory')} "
          f"confirmatory={report.meta.get('confirmatory')} "
          f"project_prefix={report.meta.get('project_prefix')}")
    print()
    for a, b in CONTRASTS:
        if not report.has(a, METRICS[0]) or not report.has(b, METRICS[0]):
            print(f"[skip] {a}_vs_{b}: arm missing from this report")
            continue
        print(f"--- {a} vs {b} ---")
        for metric in METRICS:
            x, y = report.paired(a, b, metric)
            mask = report.finite(x, y)
            r = paired_comparison(x[mask], y[mask])
            print(f"  {metric:<16} n={r.n:<3} diff={r.mean_diff:+8.3g} "
                  f"d={r.cohens_d:+.3g} {r.test:<9} p={r.p_value:.3g} "
                  f"CI=[{r.ci_low:.3g}, {r.ci_high:.3g}]")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("eval_json", help="path to eval_17313.json or similar")
    args = ap.parse_args()
    run(args.eval_json)
