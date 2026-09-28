# Results summary — four hypotheses, resolved (2026-09-28)

Source data: `results/eval_1179095.json` (v3, confirmatory, n=20),
`results/eval_17313.json` (v4, exploratory, n=20, `partial_run` only in the sense
that it adds the `detection_noobs` arm to the same run captured in the earlier,
now-superseded `results/eval_1270.json` — `full_vs_baseline` is byte-identical
between the two; `eval_17313` is cited throughout as it's the more complete
file), and `results/forced_attack_v4_{bid,ask}.json` (v4 checkpoints under a forced
worst-case attack, exploratory, n=20 per arm per side). Supporting analysis:
`analysis/h1_attack_effect_check.py`, `analysis/h2_noninferiority.py`,
`analysis/h4_regime_gap_check.py`, `analysis/detection_noobs_check.py`,
`check_adversary_sidedness.py`, `eval_forced_attack.py`. Full derivations in
`docs/note_h1_attack_effect.md`, `docs/note_h1_forced_attack.md`,
`docs/note_forced_attack_amendment.md`, `docs/note_h2_noninferiority.md` and
`docs/note_h4_regime_gap.md`.

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

## H1 — adversarial robustness: **NOT SUPPORTED** (informative null, 2026-09-28)

H1 was resolved in two stages. First, the pre-registered design, which evaluates every
arm against a co-trained adversary, turned out unable to test H1 at all: the adversary
never delivered a treatment. Second, an exploratory amendment, pre-registered before
its results existed, replaced that adversary with a forced worst-case attack. The
forced attack did reach the market makers, and it shows that neither the defended nor
the undefended ones are materially hurt by it, so the defences have little to reduce.

### Stage 2 (the answer): forced worst-case attack — not supported

Full derivation in `docs/note_h1_forced_attack.md`; design and preconditions in
`docs/note_forced_attack_amendment.md`. `eval_forced_attack.py` overrides the
adversary's action with a fixed, maximal, one-sided injection (all bid levels, or all
ask levels, at `inject_mult = 2.0` × best-quote depth) on every step. It runs against
the existing v4 checkpoints, with no retraining, over 7 arms × 20 seeds × both sides.

**The treatment demonstrably reached the market maker, for the first time in this
project.** The attack rate was 1.000 on every arm, seed and side, and the budget
(1e7) was never exhausted. The spoofed queue_imbalance moved from −0.02 to **+0.40**
(bid) / **−0.44** (ask). Inventory variability rose under attack in most arms
(d ≈ +0.4 to +1.0, p<0.05 in 7 of 14 arm×side cells). Every arm's `off` condition
reproduces `eval_17313.json` exactly (56/56 checks, max|diff| = 0), so forced vs off
differs only in the attack.

**It changes behaviour, not performance.** Every arm's point estimate is within ±6.7
Sharpe, against a clean seed-to-seed spread of 15–66 and a full-vs-baseline gap of ~48.
The widest intervals, for `adversarial` (bid, −15.4 to +6.8) and `unconstrained`
(ask, −18.2 to +0.4), belong to the two noisiest arms (clean sd 60–66); relative to
that noise they are still small effects (|d| < 0.35). The
undefended baseline's Sharpe change is +3.2 [95% CI −2.3, +8.7] on the bid side and
−1.1 [−9.2, +5.1] on the ask side. The defended arms have the tightest bounds (e.g.
detection −0.3 [−0.9, +0.4] / −0.2 [−0.7, +0.2]), but their clean performance is also far
less variable (sd 15–19 vs 40–66). That is the H3 regularisation effect again, not an
attack-specific defence. Within-arm: 3 of 42 tests have p<0.05, against ~2.1 expected
by chance, and none survives Holm (smallest adjusted p = 0.18).

**The defences do not reduce the attack's effect.** In the cross-arm
difference-in-differences (the actual H1 test), signs flip between the bid and ask
sides for most contrasts. The codebase's own H1 contrast, `adversarial_vs_baseline`,
is −7.3 on bid and +6.5 on ask, both non-significant. 5 of 48 DiD tests have p<0.05,
against ~2.4 expected. **Two survive Holm, and both are the same finding pointing
against H1**: on the ask side, `regime_vs_adversarial` shows the regime-conditioned
arm hurt *more* than the adversarially-trained arm without regime conditioning
(Sharpe DiD −10.2, Holm p=0.040; Sortino Holm p=0.015). It does not replicate on the
bid side (+3.1, p=0.60), so it is reported as a side-specific exploratory finding.

**This also explains Stage 1.** A cost-bearing adversary facing a market maker that a
maximal spoof barely hurts gains little by attacking, so abstaining is its rational
best response.

**Write-up framing.** *"H1 is not supported. A forced worst-case spoof demonstrably
reaches the market maker and changes its inventory behaviour, but it degrades no arm's
risk-adjusted performance by more than a few Sharpe units, and the defences do not
reduce that effect. The one multiplicity-surviving difference, specific to the
ask-side attack, shows regime conditioning increasing sensitivity rather than reducing
it."* Not "the defences work". The undefended baseline is just as insensitive. Scope:
observation-space spoofing of the top five displayed levels, on AMZN in 2024, under
the v4 action space.

### Stage 1 (why the pre-registered design could not answer it): co-trained adversary — untested

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

**Stage 1 on its own** leaves H1 untested, not null: v3's confirmatory contrast is
small, non-significant and wrong-signed, and v4's co-trained adversary never
substantially fired. Stage 2 above supplies the treatment that Stage 1 lacked. In the
thesis, Stage 1 is the reason the amendment exists, not a separate verdict on H1.

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

**Related, from the H1 forced-attack run** (`docs/note_h1_forced_attack.md`): the only
multiplicity-surviving attack-sensitivity difference in the whole study involves
regime conditioning. It too is wrong-signed: under the ask-side attack the regime arm
is hurt *more* than the arm without it (Sharpe −4.8 within-arm, DiD vs adversarial
Holm p=0.040). The effect is side-specific and exploratory. It is not evidence about
H4's own volatility-gap claim, but it is consistent with that null. Nowhere in this
study does regime conditioning show a protective effect.

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
| H1 | **not supported** (forced worst-case attack reaches the MM; no arm materially hurt; defences don't reduce it) | Sharpe point estimates under attack within ±6.7 for every arm (widest CI −18, in an arm with clean sd 60); DiD signs flip between sides; only Holm survivor is wrong-signed (regime, ask side, p=0.040) |
| H2 | **supported** (full model) | p=0.0146, p=0.0108 |
| H3 | null (detection) / **supported** (regularisation, reframed) | AUROC≈0.50 everywhere; noobs ablation d>1.0, p<2×10⁻⁶ |
| H4 | clean null | 3/4 isolations wrong-signed, none p<0.05 |
| validity gate | **FAILS both studies** | baseline Sharpe -119/-30 vs A-S +4.6 (inventory_sd passes) |

---

## Addendum 2026-09-28: robustness checks RC1, RC2 and the regime-label leak

Pre-registered before running (`preregistration.json` → amendments →
`robustness_checks_amendment_2026-09-28`). The results of record above are
unchanged; this section reports whether they depend on the analysis choices the
literature audit (`docs/literature_audit_2026-09-28.md`) flagged.

**RC1: exact sign-flip permutation test.** One estimand (the mean paired difference),
no Shapiro-Wilk gate, same Holm families. Source:
`python -m analysis.permutation_check …` → `results/permutation_check.{json,txt,tex}`.
Across 350 comparisons in four files, every recomputed mean difference matches its
source JSON, and 342 verdicts agree at α = 0.05 (8 change). The changes that matter:

| File | Comparison | Pre-registered p (Holm) | Sign-flip p (Holm) | Reading |
|---|---|---|---|---|
| v3 (confirmatory) | full vs baseline, `cvar_on` | 0.061 | **0.031** | The full model's **worse** CVaR (d = −0.56; per-step diff −3.02, BCa [−6.19, −1.23]) would survive Holm under the permutation test. It must be reported: the combined defence has a tail-risk cost the pre-registered test just missed. |
| v4 (exploratory) | DiD full vs baseline, sortino | 0.048 (raw) | 0.069 (raw) | The one "right-signed" v4 H1 result does not survive the gate-free test. This strengthens "H1 untested". |
| forced ask (exploratory) | 6 cells, in both directions | — | — | Exploratory and unadjusted across the grid. None changes a conclusion in `note_h1_forced_attack.md`. The regime-arm ask-side Sharpe drop agrees under both tests. |

The pre-registered test remains the result of record. The v3 CVaR line is reported
beside it, not instead of it.

**RC2: per-step units.** Tables and figures are regenerated with Sharpe/Sortino in
per-step units (`--units per-step`, now the default). Annualised copies are kept
with an `ann_` prefix for the appendix. Scale: per-step = annualised ÷ 3,291.5, so
the v4 baseline Sharpe is −0.0092 per step against A-S +0.0014. Every p-value,
effect size and verdict (gate, TOST, non-inferiority, Holm) is identical, which
`tests/test_robustness_checks.py` pins.

**Regime-label leak** (`python -m analysis.regime_label_leak` →
`results/regime_label_leak.json`). The committed labels use day t's own close
(same-day leak) and a full-year median (threshold leak). Point-in-time rules change
4 (lag only), 2 (training-period threshold only) or 6 (both removed) of the 64
held-out days, i.e. 6–9%. That is too few to account for H4's wrong-signed null.
`build_regime_labels.py --point-in-time` now produces leak-free labels. The
committed `regime_labels.json` is unchanged.

**Pending on Kaya:** RC3 (detection AUROC under the maximal one-sided attack) and
RC4 (encoder effective rank). Go/no-go 2026-10-08.

**Correction to "A fifth finding" (2026-09-28).** The auxiliary-loss effect is a v4
(exploratory) finding. It is **not** replicated at the same size in v3. Minimal
pairs on clean data, paired by seed (per-step units; p from the pre-registered test,
with the sign-flip test agreeing):

| | v3 (confirmatory) | v4 (exploratory) |
|---|---|---|
| detection − adversarial, Sharpe | +0.0007, d = 0.07, p = 0.84 | +0.018, d = 0.95, p = 1.9×10⁻⁶ |
| full − regime, Sharpe | +0.0056, d = 0.40, p = 0.09 | +0.013, d = 0.63, p = 2.6×10⁻⁴ |
| detection − adversarial, inventory SD | +0.33, d = 0.71, p = 0.006 | +0.81, d = 2.45, p = 1.2×10⁻⁹ |

In v3 the head raises inventory activity but not risk-adjusted performance. The
claim in `critical_review_2026-09-26.md` that head arms sit at inventory SD 1.75–2.0
against 0.94–1.09 "in both v3 and v4" holds for v4 only: in v3 every arm is between
2.11 and 2.53. Thesis wording: "a large effect in the exploratory v4 study with a
weak analogue in v3" (`thesis_drafts/05_results_h2_h3_h4.tex`). The v3 action space
(quantities at the touch) gave a better representation little to act on.
