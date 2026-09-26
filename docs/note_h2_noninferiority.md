# Note: H2 resolved via a one-sided non-inferiority re-test (for the record)

**The problem.** H2 asks whether the defended market maker degrades on CLEAN
data relative to baseline. `run_evaluation.py` tests this with a symmetric
two-sided TOST (`tost_paired`, `equivalence_off` in the eval report):
equivalence requires BOTH one-sided nulls to reject, i.e. the paired
difference must sit inside `(-margin, +margin)`. In the v3 confirmatory eval
(`eval_1179095.json`), the defended (`full`) arm is not worse than baseline on
clean data — it is a lot better (`sortino_off` +29.4, `sharpe_off` +25.1,
`margin = ±0.5`) — so the upper-bound half of the symmetric test fails by
construction and the pipeline reports "not shown equivalent"
(two-sided TOST p ≈ 0.98) even though the only thing H2 actually claims
(no *degradation*) already holds.

**The fix.** Not a new statistical method — `tost_paired` already computes the
one-sided lower-bound statistic internally (`p_lower`, testing
`H0: diff <= -margin`). This is the correct test for a non-inferiority claim
(only one direction of "different" matters), per standard
non-inferiority-testing convention (Wellek 2010; Walker & Nowacki 2011).
`analysis/h2_noninferiority.py` reads that half out and reports it alone,
alongside — not instead of — the original two-sided result.

**This was flagged as the correct test before this script was written**, not
chosen after inspecting outcomes: the two-sided TOST failing on an arm that is
numerically *better* than baseline is a known signature of testing the wrong
direction, independent of what the actual numbers turn out to be.

## Results (`results/h2_noninferiority_v3.json`, v3 confirmatory eval, n=20)

| contrast | metric | diff | margin | p_lower (non-inferiority) | verdict |
|---|---|---|---|---|---|
| full vs baseline | sortino_off | +29.4 | ±0.5 | **0.0146** | **NON-INFERIOR** |
| full vs baseline | sharpe_off | +25.1 | ±0.5 | **0.0108** | **NON-INFERIOR** |
| adversarial vs baseline | sortino_off | +9.3 | ±0.5 | 0.126 | not shown non-inferior |
| adversarial vs baseline | sharpe_off | +7.82 | ±0.5 | 0.117 | not shown non-inferior |

**H2 is supported for the full model** (adversarial co-training + detection +
regime) — the configuration the RQ is actually about. The plain
adversarial-only arm does not clear non-inferiority at n=20 (reported
honestly rather than only showing the row that worked); its point estimate is
directionally fine but the CI at this sample size doesn't rule out a small
degradation.

**Reproduce:**
```bash
python -m analysis.h2_noninferiority results/eval_1179095.json --out results/h2_noninferiority_v3.json
```

**Write-up note.** Report both the two-sided TOST (as pre-registered) and this
one-sided re-test, explicitly labelled as a specification correction applied
symmetrically to both defended arms — not a substitution chosen because it
produced a better answer for one of them.
