# Results summary — four hypotheses, resolved (2026-09-27)

Source data: `results/eval_1179095.json` (v3, confirmatory, n=20) and
`results/eval_17313.json` (v4, exploratory, n=20, `partial_run` only in the sense
that it adds the `detection_noobs` arm to the same run captured in the earlier,
now-superseded `results/eval_1270.json` — `full_vs_baseline` is byte-identical
between the two; `eval_17313` is cited throughout as it's the more complete
file). Supporting analysis: `analysis/h1_attack_effect_check.py`,
`analysis/h2_noninferiority.py`, `analysis/h4_regime_gap_check.py`,
`analysis/detection_noobs_check.py`, `check_adversary_sidedness.py`. Full
derivations in `docs/note_h1_attack_effect.md`, `docs/note_h2_noninferiority.md`
and `docs/note_h4_regime_gap.md`.

**Design reminder.** v3 is the pre-registered confirmatory study (`preregistration.json`,
signed off by the candidate 2026-08-27 — **supervisor confirmation from Dr Wen remains
outstanding as of this writing**, and both the auto-liquidate and spread_skew amendments
are marked `candidate_sign_off: PENDING REVIEW` in `preregistration.json`; this should be
resolved before submission). v4 changes the MM's action space (`bobRL` → `spread_skew`)
via a pre-registered amendment written *before* any v4 result existed, to fix an
identified confound (v3's MM quoted prices were structurally immune to the adversary).
v4 is permanently gated exploratory (`confirmatory: false`) regardless of what it shows —
no v4 p-value is a significance claim. Both studies share seeds, checkpoint step, alpha,
and the common-adversary design for internal validity.

**Scope limitation.** All data is a single ticker (AMZN, LOBSTER + Databento) and a
single calendar year (2024: train 2024-01-01..2024-09-30, holdout 2024-10-01..2024-12-31).
Every claim below is scoped to this instrument and period, not to NASDAQ order flow
generally.

**Validity gate, both studies.** The pre-registered `progression_gate` (does the
baseline, undefended MM beat the fixed-policy Avellaneda-Stoikov benchmark by more than
a stated margin?) **FAILS in both v3 and v4**: v3 baseline Sharpe -119.4 vs A-S +4.6,
Sortino -138.7 vs +17.5; v4 baseline Sharpe -30.3 vs A-S +4.6, Sortino -16.4 vs +17.5
(inventory_sd passes in both). Every comparative claim below holds only *among* the
trained arms, which the pre-registration always intended (`interpretation_notes`:
absolute PnL is not a hypothesis) — but the gate failing means the baseline reference
point itself is not a competent trading policy, which is worth the reader knowing
before weighing any "arm X beats baseline" claim. Reproduce: `python -m analysis.tables
results/eval_1179095.json --only gate --stdout` (and the v4 equivalent).

---

## H1 — adversarial robustness: **UNTESTED**, not null and not replicated (corrected 2026-09-27)

**The contrast used to claim "replicated at d≈1.0" was wrong, and once corrected, so is
the mechanism it was said to confirm.** Full derivation in `docs/note_h1_attack_effect.md`;
summary below.

**The v4 "d≈1.0" figure was the H2 clean-data statistic, misread onto the H1
(attack-condition) heading.** `full_vs_baseline`'s `sharpe_off`/`sortino_off` diffs
(+48.45, +67.34) are the `equivalence_off` block — the no-attack comparison — not a
measure of attack response. Worse, the underlying comparison itself (`full` vs
`baseline`, both under attack) conflates general policy quality with attack response:
some arms partially collapse to a no-trade equilibrium for reasons that have nothing to
do with the adversary (see H3 below), which alone can produce a large between-arm gap
under attack with zero causal contribution from the attack.

**The correctly-specified test — within-arm attack-on vs attack-off, and the cross-arm
difference-in-differences — is null-to-wrong-signed everywhere.** Within-arm: every
arm in both v3 and v4, *including* `unconstrained` (kappa = p_detect = c_fill = 0, i.e.
zero economic penalty on the adversary), shows no attack effect (|d| < 0.4, p > 0.1).
Cross-arm DiD, the actual H1 test (isolates whether a defence changes the *size* of the
attack's effect, not just arm quality under attack): v3 (confirmatory) is **wrong-signed**
on both the codebase's own labelled H1 contrast (`adversarial_vs_baseline`: d=-0.38,
Holm p=0.21) and the write-up's previous choice (`full_vs_baseline`: d=-0.39, Holm
p=1.0); v4 (exploratory) flips to the right sign with a small-to-medium, uncorrected and
non-multiplicity-adjusted effect (sortino p=0.048 among 24 comparisons run).

**Why: the adversary that produced every v4 number abstained.** Every v4 arm — including
`unconstrained` — is evaluated under attack against the *same* common adversary
checkpoint, `v4_config3_full` (pre-registered design, for internal validity).
`check_adversary_sidedness.py` run directly against it (2026-09-27): mean injected action
0.0159 of a possible 10.0 (0.159% of max), mean |queue_imbalance| shift on-vs-off -0.0018.
**Verdict: ABSTAINED** — consistent with its cost model (c_fill/c_reg charged
unconditionally, profit tax zero when nothing is extracted, so attacking an MM it cannot
move is negative EV) and independently reproducing the 2026-09-22 session finding on the
production checkpoint. This directly supersedes an earlier claim in this document that
the same checkpoint showed `mean_injection_asymmetry_on = 0.528` and "did not
degenerate" — that number does not reproduce on direct re-measurement and should be
treated as retracted.

A second, independent check on `v4_config6_unconstrained`'s own co-trained adversary
(never used in evaluation) rules out pure credit-assignment failure as the reason: given
zero cost, it *does* attack substantially (mean action 5.42 of 10) — but converges to a
**symmetric** injection (sidedness 0.08 of 1.0), which cannot move queue_imbalance either.
Even this adversary would not have produced an informative test if it had been used.

**Write-up framing.** Not "H1 confirmed," not "H1 replicated," not "H1 null." Correct:
*"H1 is untested across both studies. v3's confirmatory contrast is small,
non-significant, and wrong-signed. v4's exploratory extension — built to give the
adversary a price lever it previously lacked — does not settle H1 either way, because
the mechanism it was built to exercise never substantially fired: the common evaluation
adversary abstains under its cost model, and a cost-free variant that does attack
converges to a pattern (symmetric injection) that cannot poison the one channel the
attack relies on. Resolving H1 needs a design correction to the adversary's training
incentives or convergence behaviour, not more seeds against the current checkpoint."*

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
| H1 | **untested** (adversary abstains; corrected contrast wrong-signed in v3) | v3 DiD d=-0.38 (Holm p=0.21); v4 DiD d=+0.30–0.43 (uncorrected, 1 of 24 comparisons) |
| H2 | **supported** (full model) | p=0.0146, p=0.0108 |
| H3 | null (detection) / **supported** (regularisation, reframed) | AUROC≈0.50 everywhere; noobs ablation d>1.0, p<2×10⁻⁶ |
| H4 | clean null | 3/4 isolations wrong-signed, none p<0.05 |
| validity gate | **FAILS both studies** | baseline Sharpe -119/-30 vs A-S +4.6 (inventory_sd passes) |
