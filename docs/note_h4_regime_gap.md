# Note: H4's actual test had never been run (for the record)

**The gap.** `rollout.py` computes `sortino_lowvol`, `sortino_highvol` and
`regime_gap = |sortino_highvol - sortino_lowvol|` per arm/seed, with a comment
in the code identifying this as being computed specifically for H4. But
`run_evaluation.py`'s inferential contrasts (`compare_configs`, Holm) only
ever run over `primary_metrics = (sortino_on, sharpe_on, cvar_on, auroc)`.
`regime_gap` was written into every eval's per-seed summary and never
compared between arms. H4's own purpose-built metric existed in every run and
was never tested against the hypothesis it was built for.

**The test.** H4 claims regime conditioning REDUCES the gap between high- and
low-volatility performance. The 2x2 factorial (detection head x regime) gives
two clean, minimal-pair isolations of the regime factor alone:
- `adversarial vs regime` (regime effect with detection head off)
- `detection vs full` (regime effect with detection head on)

Sign convention: diff = gap(no-regime arm) - gap(regime arm); positive
supports H4 (regime arm has the smaller gap). `analysis/h4_regime_gap_check.py`
runs both isolations, on `regime_gap_on`/`_off` and the underlying
`sortino_lowvol`/`sortino_highvol`, using the same `paired_comparison`
machinery as every other contrast in the pipeline.

**This is EXPLORATORY even on v3 data** — `regime_gap` was never in
`primary_metrics`, so this is a post-hoc test regardless of which eval it
runs against. It differs from a chased result only in that there is exactly
one natural candidate metric for this hypothesis (built for it, labelled as
such, before this session existed), and every isolation × condition it
produces is reported here, not just the favourable one.

## Results

| eval | isolation | metric | diff | d | p | direction |
|---|---|---|---|---|---|---|
| v3 | adversarial vs regime | regime_gap_on | -18.3 | -0.42 | 0.076 | **wrong** |
| v3 | adversarial vs regime | regime_gap_off | -18.0 | -0.42 | 0.077 | **wrong** |
| v3 | detection vs full | regime_gap_on | -6.76 | -0.12 | 0.648 | **wrong** |
| v3 | detection vs full | regime_gap_off | -2.74 | -0.05 | 0.841 | **wrong** |
| v4 | adversarial vs regime | regime_gap_on | +2.51 | +0.03 | 0.956 | right, but noise |
| v4 | adversarial vs regime | regime_gap_off | +2.20 | +0.02 | 0.985 | right, but noise |
| v4 | detection vs full | regime_gap_on | -6.12 | -0.24 | 0.330 | **wrong** |
| v4 | detection vs full | regime_gap_off | -6.22 | -0.24 | 0.294 | **wrong** |

Three of four isolations point opposite to H4's hypothesised direction; the
fourth is a near-zero effect indistinguishable from noise. None clear even an
uncorrected p < 0.05, let alone anything Holm-adjusted.

**Bearing on the regime-label leakage** (separately disclosed): the in-sample
threshold used to build `regime_labels.json` leaks the held-out test period's
own volatility into the label, which if it biased anything would bias in
FAVOUR of the regime arms (a cleaner, more separated label distribution).
Getting a null — and a wrong-signed one at that — despite that bias makes the
null more credible, not less.

**Reproduce:**
```bash
python -m analysis.h4_regime_gap_check results/eval_1179095.json --label v3
python -m analysis.h4_regime_gap_check results/eval_1270.json --label v4
```

**Write-up note.** H4 is not "underpowered" in the way H1's `full_vs_baseline`
trend is — that has a real, medium-sized effect the study wasn't powered to
confirm. H4's own metric shows no effect to be underpowered for, and mostly
runs backwards. Report it as a clean null, not a promissory one.
