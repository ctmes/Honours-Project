# Note: H1 was tested with the wrong contrast, and the adversary abstained anyway

**The gap.** `run_evaluation.py`'s inferential contrasts compare arms to each
other UNDER ATTACK only (`full_vs_baseline` on `sortino_on`/`sharpe_on`/
`cvar_on`/`auroc`). That answers "is arm E better than arm A while both are
being attacked" — which is confounded with general policy quality, since some
arms partially collapse to a no-trade equilibrium for reasons that have
nothing to do with the adversary being present (see the H3 note and the
standalone fifth finding in `results_summary.md`). It is not a test of "does
the attack change this arm's outcome", which is what H1 actually claims. Every
eval report already carries the `_on`/`_off` pair needed for the correct test;
it was simply never run. `analysis/h1_attack_effect_check.py` runs it now,
using the same `paired_comparison`/`holm_adjust` machinery as every other
contrast in the pipeline, specified before either result below was known:

1. **Within-arm**: does attack-on differ from attack-off, for a given arm, on
   its own (paired by seed)? The direct causal check of whether the adversary
   does anything at all to a given policy.
2. **Cross-arm difference-in-differences**: does the attack's EFFECT (on−off)
   differ between two arms? This is the actual H1 test — it isolates whether a
   defence changes the size of the attack's effect, rather than comparing two
   arms' absolute quality under one condition. Valid as a paired test because
   `preregistration.json`'s common-adversary design pairs every arm's seeds by
   index against the same adversary checkpoint.

## Results — Part 1: the within-arm test is null everywhere

For every arm in both v3 (`eval_1179095.json`) and v4 (`eval_17313.json`),
attack-on vs attack-off is indistinguishable from noise (|d| < 0.4, p > 0.1
throughout), **including `unconstrained`** (kappa = p_detect = c_fill = 0 — the
adversary with zero economic penalty, evaluated the same way as every other
arm). This by itself does not distinguish "the MM is robust" from "nothing was
tested" — Part 3 below settles which.

## Results — Part 2: the correctly-specified test (DiD)

| eval | contrast (codebase's own H1 label: `adversarial_vs_baseline`) | metric | DiD | d | p | Holm p | direction |
|---|---|---|---|---|---|---|---|
| v3 (confirmatory) | adversarial_vs_baseline | sharpe | -0.641 | -0.375 | 0.068 | 0.205 | **wrong** |
| v3 (confirmatory) | adversarial_vs_baseline | sortino | -0.823 | -0.361 | 0.076 | 0.205 | **wrong** |
| v3 (confirmatory) | full_vs_baseline (write-up's chosen contrast) | sharpe | -1.12 | -0.386 | 0.554 | 1.0 | **wrong** |
| v3 (confirmatory) | full_vs_baseline | sortino | -1.52 | -0.368 | 0.356 | 1.0 | **wrong** |
| v4 (exploratory) | adversarial_vs_baseline | sharpe | +0.606 | +0.295 | 0.173 | n/a | right |
| v4 (exploratory) | adversarial_vs_baseline | sortino | +0.876 | +0.318 | 0.100 | n/a | right |
| v4 (exploratory) | full_vs_baseline | sharpe | +0.244 | +0.377 | 0.074 | n/a | right |
| v4 (exploratory) | full_vs_baseline | sortino | +0.387 | +0.427 | 0.048 | n/a | right |

The confirmatory result (v3) points in the WRONG direction on both the
codebase's own labelled H1 contrast and the write-up's previously-chosen one,
at Holm p = 0.2–1.0 — nowhere near significant, and not even directionally
supportive. v4 flips to the right direction with a small-to-medium effect,
borderline on sortino uncorrected (p=0.048) — but this is one exploratory
estimate among 24 comparisons run and is not multiplicity-corrected; it must
not be reported as a finding on its own.

## Results — Part 3: why it's null — the adversary that produced these numbers

Every arm in v4, including `unconstrained`, is evaluated under attack using
**the same common adversary checkpoint, `v4_config3_full`** (design decision,
`preregistration.json`: "every arm evaluated under attack faces the config-3
arm's adversary, paired by seed index"). `check_adversary_sidedness.py` run
directly against that checkpoint (job 35688, 2026-09-27):

- mean total injected action: **0.0159 of a possible 10.0** (0.159% of max)
- mean |queue_imbalance| shift on-vs-off: **-0.0018** (headroom was 0.508)
- **Verdict: ABSTAINED.** The adversary has learned not to attack, consistent
  with its cost model (c_fill and c_reg are charged unconditionally while the
  profit tax is zero when nothing is extracted, so attacking an MM it cannot
  move is negative EV).

This is the same finding as the 2026-09-22 session note (0.159% of max
action), independently reproduced against the production checkpoint rather
than superseded by it. **It directly contradicts an earlier claim in this
repository's history** that the same checkpoint showed `mean_injection_asymmetry_on
= 0.528` and "did not degenerate" — that number does not reproduce on direct
re-measurement (actual bid/ask totals here are ~0.003-0.03, not the ~1.6-2.0
range that claim implied) and should be treated as superseded by this run, not
reconciled with it.

Separately, `v4_config6_unconstrained`'s own co-trained adversary (never used
in evaluation, since the common-adversary design always substitutes
`v4_config3_full`'s) DOES attack substantially when given zero cost (mean
action 5.42 of 10, i.e. not abstaining) — ruling out a pure training/
credit-assignment failure as the reason the constrained one abstains. But it
converges to a **symmetric** injection (sidedness index 0.08 of 1.0), which by
construction cannot move queue_imbalance either (mean shift -0.085, same sign
as the dampening a symmetric injection produces on an (a-b)/(a+b) channel).
Even this adversary would not have produced an informative H1 test if it had
been the one used.

**Conclusion: H1 is UNTESTED across the entire v4 study, for two independent
and now-diagnosed reasons — not null, and not a "not exploitable" finding.**
The v3 confirmatory result stands as reported (a small, wrong-signed,
non-significant effect); the v4 exploratory extension does not settle it
either way, because the mechanism the extension was built to exercise never
substantially fired.

**Reproduce:**
```bash
python -m analysis.h1_attack_effect_check results/eval_1179095.json --label eval_1179095.json --out results/h1_attack_effect_v3.json
python -m analysis.h1_attack_effect_check results/eval_17313.json --label eval_17313.json --out results/h1_attack_effect_v4.json
python check_adversary_sidedness.py --project v4_config3_full --yaml config/rl_configs/eval_2024_test_v4_config3.yaml --seeds 0-4
python check_adversary_sidedness.py --project v4_config6_unconstrained --yaml config/rl_configs/eval_2024_test_v4_config6.yaml --seeds 0-4
```

**Follow-up (2026-09-28): resolved.** The design correction called for below was made
as a pre-registered exploratory amendment (`docs/note_forced_attack_amendment.md`).
Instead of retraining the adversary, it forces a worst-case one-sided attack against
the existing checkpoints. That treatment does reach the market maker, and H1 is **not
supported**: see `docs/note_h1_forced_attack.md`. This note remains the explanation of
why the pre-registered co-trained-adversary design could not answer H1 on its own.

**Write-up note.** Do not report "H1 replicated at d≈1.0" — that number was
the H2 clean-data (`_off`) statistic, misread onto the H1 (attack-condition)
heading; the correctly-specified DiD test on the same data shows a much
smaller and direction-inconsistent picture. Report H1 as untested, name the
two independently-diagnosed reasons (cost-driven abstention; symmetric
injection is uninformative even when it fires), and note that resolving it
would need a design correction to the adversary's incentives or convergence,
not more seeds against the current checkpoint.
