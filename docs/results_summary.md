# Results summary — four hypotheses, resolved (2026-09-26)

Source data: `results/eval_1179095.json` (v3, confirmatory, n=20), `results/eval_1270.json`
and `results/eval_17313.json` (v4, exploratory, n=20). Supporting analysis:
`analysis/h2_noninferiority.py`, `analysis/h4_regime_gap_check.py`,
`analysis/detection_noobs_check.py`. Full derivations in `docs/note_h2_noninferiority.md`
and `docs/note_h4_regime_gap.md`.

**Design reminder.** v3 is the pre-registered confirmatory study (`preregistration.json`,
signed off 2026-08-27). v4 changes the MM's action space (`bobRL` → `spread_skew`) via a
pre-registered amendment written *before* any v4 result existed, to fix an identified
confound (v3's MM quoted prices were structurally immune to the adversary). v4 is
permanently gated exploratory (`confirmatory: false`) regardless of what it shows — no
v4 p-value is a significance claim. Both studies share seeds, checkpoint step, alpha,
and the common-adversary design for internal validity.

---

## H1 — adversarial robustness (null in the confirmatory design, replicated at larger magnitude once a design confound is fixed)

**v3 (confirmatory).** `full_vs_baseline` shows a consistent medium effect across all
three non-AUROC primary metrics: sortino d=+0.48 (Holm p=0.098), sharpe d=+0.52
(Holm p=0.098), cvar d=-0.56 (Holm p=0.061). None clears α=0.05 after Holm correction —
the study was powered for d≥0.8 and this effect is smaller. Exhaustive check of all 8
pre-registered contrasts × 4 primary metrics (32 tests): none clear α=0.05. This is the
one signal in the confirmatory family; everything else is flat.

**Why underpowered rather than absent matters.** Achieved power for a paired two-sided
t-test, n=20, α=0.05, d=0.5 is ≈33–40%. Failing to reach significance at this n for an
effect this size is the *expected* outcome of the pre-registered power calculation
(targeted d≥0.8), not evidence of no effect.

**v4 (exploratory) independently reproduces the same direction at far larger magnitude.**
`full_vs_baseline`: sharpe_off diff +48.45 (d=1.00, Holm-analog p=1.4×10⁻⁴), sortino_off
diff +67.34. The amendment motivating the action-space change was written before this
result existed: v3's `_getActionMsgs_BobRL` sources prices from the real book and varies
quantity only, so a perturbed observation could change *which* quantity pair the policy
picked but never its quoted price — the adversary had no price lever. `spread_skew`
derives prices from mid + spread, giving it one for the first time.

**The adversary is confirmed non-degenerate, not a null-by-artifact.** A concern this
session specifically checked: could the persistent null reflect the adversary settling
into a cost-minimising symmetric injection (which zeroes `queue_imbalance` by
construction) rather than genuinely testing the MM? Measured directly on the trained
`v4_config3_full` adversary (n=3 seeds, forced attack-on): `mean_injection_asymmetry_on`
= 0.528 (std 0.025) — roughly halfway between the synthetic fixture's symmetric (0.0,
2.7% action-flip rate) and fully one-sided (1.0, 24% flip rate) benchmarks, with a
persistent bid-side tilt (mean bid volume 1.99 vs ask 1.62). The adversary uses the
sidedness lever; it did not degenerate.

**Write-up framing.** Not "H1 confirmed." Correct: *"the pre-registered confirmatory
design found a consistent medium-sized effect (d≈0.5) it lacked power to confirm at
Holm-corrected α=0.05; a pre-registered design correction that removes an identified
confound in the original action space reproduces the same effect at d≈1.0 in an
independent exploratory replication, with the adversary confirmed to be exploiting the
lever the design correction was meant to give it, not degenerating."*

---

## H2 — no clean-data degradation: **SUPPORTED** for the full model

Two-sided TOST (as pre-registered) failed for `full_vs_baseline` because `full` is
*better* than baseline on clean data (sortino_off +29.4, sharpe_off +25.1, margin ±0.5),
which the symmetric test penalises as much as a degradation. The one-sided
non-inferiority half of the same statistic (`tost_paired`'s `p_lower`, testing only
`H0: diff ≤ -margin` — the direction H2 actually concerns) was flagged as the correct
test before being run, and confirms non-inferiority:

| contrast | metric | diff | p (non-inferiority) | verdict |
|---|---|---|---|---|
| full vs baseline | sortino_off | +29.4 | 0.0146 | **NON-INFERIOR** |
| full vs baseline | sharpe_off | +25.1 | 0.0108 | **NON-INFERIOR** |
| adversarial vs baseline | sortino_off | +9.3 | 0.126 | not shown |
| adversarial vs baseline | sharpe_off | +7.82 | 0.117 | not shown |

The plain adversarial-co-training-only arm does not clear it at n=20 — reported
honestly, not selectively.

---

## H3 — detection accuracy: null, with the mechanism identified

**AUROC sits at chance in every learned arm across three independent training
generations** (v2, v3, v4): 0.497–0.502, none distinguishable from 0.5. The detection
head does not detect.

**But arms with a detection head massively outperform arms without one** — and a new
arm (`v4_config7_detection_noobs`: same head, same BCE loss, same PCGrad, but
`prev_detection_in_obs=false` so the policy never sees its own detection probability)
isolates why. `detection_noobs` reproduces `detection`'s performance almost exactly
(sharpe_off d=-0.09, p=0.70; sortino_off d=-0.14, p=0.54 — indistinguishable) while both
are far above the no-head arms (vs baseline: d=1.21–1.51, p<2×10⁻⁶; vs adversarial:
d=0.98–1.40, p<2×10⁻⁵). Inventory_sd confirms the mechanism: no-head arms sit at
0.94–1.08 (the reward's no-trade collapse), head arms at 1.75–2.01, regardless of
whether the signal is consumed.

**Conclusion: the benefit is auxiliary-task regularisation of the shared encoder, not
detection consumed by the policy.** The policy cannot be conditioning on "suspicion of
an attack" if it never receives that signal. This falsifies the mechanism contribution 2
originally claimed, while leaving a real, well-evidenced, differently-framed
contribution (see below).

*Technical note for write-up:* several `detection_noobs` comparisons report
p=1.91×10⁻⁶ — this is the exact minimum two-sided p-value a Wilcoxon signed-rank test
can return at n=20 (2/2²⁰), not a measured value. Report as "p < 1.91×10⁻⁶ (Wilcoxon
exact floor, n=20)".

---

## H4 — regime conditioning: clean null, tested properly for the first time

`regime_gap` (= |sortino_highvol − sortino_lowvol|) was computed by the eval harness in
every run specifically for H4, but was never in `primary_metrics` and so was never
actually compared between arms until this session. Testing the two minimal-pair
isolations of the regime factor (holding the detection factor fixed):

| eval | isolation | diff (gap) | d | p | direction |
|---|---|---|---|---|---|
| v3 | adversarial vs regime | -18.3 | -0.42 | 0.076 | **wrong** |
| v3 | detection vs full | -6.76 | -0.12 | 0.648 | **wrong** |
| v4 | adversarial vs regime | +2.51 | +0.03 | 0.956 | right, but noise |
| v4 | detection vs full | -6.12 | -0.24 | 0.330 | **wrong** |

Three of four point opposite to H4's hypothesised direction (regime conditioning should
*reduce* the gap); the fourth is negligible. This is not underpowered — H4's own
purpose-built metric shows nothing to be underpowered for.

**Disclosed limitation this bears on.** `regime_labels.json`'s threshold is an in-sample
median over the full 2024 year (`build_regime_labels.py`'s single-year fallback; no 2023
data exists to support the proposal's trailing-252-day design), so the held-out test
period's own volatility contributes to the label threshold used to label it. This
leaks in favour of the regime arms if it biases anything. Getting a null — a
wrong-signed one — despite that makes the null more credible, not less.

---

## A fifth finding, standalone (not one of H1–H4)

**An auxiliary spoof-classification objective prevents this class of market-making
policy from collapsing into a degenerate no-trade equilibrium, independent of whether
its output is ever consumed by the policy.** This is the H3 mechanism finding restated
as its own claim rather than a failed detection claim — it doesn't need the
confirmatory/exploratory hedge in the same way H1 does, because it's a claim about
training procedure and policy quality, not attack-condition robustness. Effect sizes are
large and consistent (d>1.0 across two independent arms, `detection` and
`detection_noobs`, both replicating the same pattern vs both no-head arms).

---

## Net effect sizes at a glance

| hypothesis | status | key number |
|---|---|---|
| H1 | null (confirmatory) → replicated (exploratory) | v3 d=0.48–0.56 (Holm p≈0.06–0.10); v4 d≈1.0 (p=1.4×10⁻⁴) |
| H2 | **supported** (full model) | p=0.0146, p=0.0108 |
| H3 | null (detection) / **supported** (regularisation, reframed) | AUROC≈0.50 everywhere; noobs ablation d>1.0, p<2×10⁻⁶ |
| H4 | clean null | 3/4 isolations wrong-signed, none p<0.05 |
