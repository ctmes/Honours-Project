# Project Status Report — Attack-Aware Market Making

**Prepared:** 19 June 2026
**Author:** Colin Melville (23170781)
**Project:** *Attack-Aware Market Making: Adversarial Co-Training with Economically-Constrained Spoofing Agents*
**Supervisors:** Dr Ajmal Mian (primary), Dr Yuanji Wen (co-supervisor) · External advisor: Dr Garrison Gao
**Degree:** Bachelor of Advanced Computer Science (AI) — Honours (24-point project)

---

## 0. How to read this document

This report reconciles three things: (1) what the **proposal** committed us to building and measuring; (2) what the **literature review** established as the gap we are filling; and (3) **where the codebase and the research actually stand today**, verified against the repository on 19 June 2026, not against memory or intention.

It is structured so that you can lift the relevant parts directly into a supervisor conversation:

- **Section 1** is the one-paragraph and one-table version — the meeting headline.
- **Section 2** restates the research question and the three contributions precisely, because the meeting will assume that framing.
- **Section 3** summarises the literature review's argument and the gap, framed the way Dr Wen prefers (research direction and economics, not engineering).
- **Section 4** is the honest engineering status: what is built, what is validated, what is wired but inert, and what has not started.
- **Section 5** maps current status against the proposal's five-phase timeline and is blunt about schedule.
- **Section 6** is the risk register, including three items that need a decision or an action now.
- **Section 7** is the concrete next-steps list in priority order.
- **Section 8** is a meeting playbook: what to lead with for Dr Wen specifically, and the questions to expect.

Where the current implementation diverges from the proposal (data year, seed count, regime activation, reward function, adversary cost settings), this report says so explicitly rather than papering over it. Those divergences are the substance of the conversation you need to have.

---

## 1. Executive summary — the meeting headline

**One paragraph.** The full adversarial co-training system described in the proposal is **built and validated end-to-end at small scale**. All three research contributions — the economically-constrained spoofing adversary, the embedded attack-detection head, and the volatility-regime conditioning channel — exist in code, are wired into the environment, and have run together without crashing on a five-day data slice. Since the last supervisor update, the two things that were blocking the move into the real experimental phase have changed: **Kaya supercomputer access has come through** (Slurm batch scripts are committed and target our allocation), and the **volatility-regime label artifact for the full 2022 year is now generated**. What remains before we can report any *results* is mechanical but real: launch the multi-seed training runs on Kaya, finish wiring the regime labels into the training config (one artifact is still incomplete due to a data-loading bug), and resolve the AMZN-2022-versus-2024 data question the proposal left open. **We are at the start of Phase 3 (co-training and experiments), on the proposal's own schedule, with the framework de-risked and the compute secured. No experimental results exist yet — that is the next milestone, not a finished one.**

**Status at a glance:**

| Area | Proposal commitment | Current state | Verdict |
|---|---|---|---|
| **Literature review** | Due 25 May 2026 | Submitted; reframes adversary as observation-space perturbation (matches code) | ✅ Done |
| **Framework — Challenge 1** (constrained adversary) | Unconstrained in training, budget-constrained at eval | Built; spoofing agent injects depth-scaled orders, telegraph on/off bursts, cost hooks present (`c_fill`, `c_reg`, budget) | ✅ Built, ⚠️ eval-time cost calibration pending |
| **Framework — Challenge 2** (detection head) | BCE auxiliary head on shared encoder, fed back to policy | Built; `use_detection_head=true`, PCGrad gradient surgery active, detection loss decreases at smoke scale | ✅ Built, ❓ real-scale AUROC unproven |
| **Framework — Challenge 3** (regime conditioning) | Binary vol-regime indicator appended to obs | Built in env (`regime_conditioning=true`); full-year labels generated; **but config paths empty and window→date map incomplete → currently inert** | ⚠️ Built but not yet active end-to-end |
| **Eval + statistics harness** | Sortino/Sharpe/CVaR + paired tests, Cohen's d, CI | Built as a dedicated package, unit-tested, validated on a checkpoint | ✅ Built |
| **Compute (Kaya)** | ~1,500–2,000 A100-equiv GPU-hours, "pending" | **Access granted**; Slurm scripts committed for baseline + adversarial jobs | ✅ Unblocked |
| **Training runs (≈20 seeds)** | Phases 1–4 | **First single-seed run on Kaya, 19 Jun (~52 updates, no crash)**; ≈20-seed full-scale sweep not started | ⚠️ Started, not at scale (see §4.7) |
| **Results / analysis** | Phase 4 | None yet | ❌ Not started |
| **Data** | AMZN **2024** LOBSTER | AMZN **2022** prepared (250 days) + 5-day slice; **no 2024**; loader silently drops 44 days | ⚠️ Decision needed |

**Three items that need attention now** (detailed in Section 6):
1. **Security:** the live DataBento API key is committed in `.env` and pushed to GitHub. It needs to be untracked and rotated.
2. **Data decision:** proposal says AMZN 2024; only 2022 is prepared, and the loader currently drops 44 of 250 days. We need to choose 2022 or 2024 and fix the loader.
3. **Regime activation:** Challenge 3 is built but inert — the config does not point at the label files, and the window→date map only covers the 5-day demo.

---

## 2. The research question and the three contributions (restated precisely)

The meeting will assume this framing, so it is worth restating exactly as the proposal commits it.

### 2.1 Research question

> *Can adversarially co-trained market-making agents equipped with an explicit attack-detection head and a binary volatility-regime indicator maintain significantly higher risk-adjusted performance under economically-constrained spoofing attacks, and does this robustness generalise across high- and low-volatility regimes without degrading normal-market performance?*

This is framed as an **exploratory study**. The statistical design is fixed: effect sizes reported with **Cohen's _d_** (standardised mean difference across **≈20 seeds**) and **95% confidence intervals**; **Shapiro–Wilk** tests (α = 0.05) decide whether to use parametric tests or non-parametric fallbacks; a G\*Power calculation targets _d_ ≥ 0.8 at α = 0.05 with ≈20 seeds. The "≈20 seeds" figure is not cosmetic — it is what gives the headline comparisons their statistical power, and it is the single most important number that the *current* training config does not yet honour (it runs one seed).

### 2.2 The system as three contributions ("Challenges")

**Challenge 1 — Economically-constrained adversary (an evaluation-side contribution).**
During co-training the adversary is *unconstrained*: it optimises `R_adv = −R_mm` with no fees, capital costs, or fill penalties, so it produces a maximally hard training signal. The economic constraint is applied *at evaluation*: the adversary's spoofing sequences are filtered through a per-action cost,

```
cost_t = c1 · N_orders + c2 · K_committed + c3 · P_fill
```

where **c1** is the NASDAQ equity taker fee (≈ $0.003/share), **c2** is overnight funding on committed notional, and **c3** is the median per-share disgorgement from SEC/DOJ equity spoofing enforcement actions. Any spoofing sequence whose cumulative cost exceeds budget **B** is replaced with a no-op. The instrument is **AMZN equity** specifically so these costs can be calibrated against real NASDAQ equity fee schedules and real enforcement figures (not futures-market schedules). The design rationale — and this is the part Dr Wen cares about — is that **constraining the adversary at evaluation rather than training preserves internal validity** (baseline and defended agents face the same adversary) **while ensuring external validity** (reported attack intensities reflect what a real spoofer would sustain given fees, capital costs, and disgorgement risk). The action space allows layering: `n ∈ {1, 2, 3}` simultaneous orders per side.

**Challenge 2 — Attack-detection auxiliary head.**
A binary classification head on the market maker's *shared encoder* predicts `ŷ_t ∈ [0,1]` — whether a non-bona-fide order is active in the book — with oracle labels supplied by the simulator's adversary tags. Training jointly optimises

```
L = L_PPO + λ_c · L_BCE(ŷ_t, y_t)
```

The novelty claim (verified against a search of ICAIF, IJCAI, AAAI, AAMAS, NeurIPS-FinRL) is that **no prior work embeds spoof-detection as an auxiliary objective *inside* a DRL market maker's policy network** such that the detection signal is available to the policy at execution. Wang & Wellman treat detection as a standalone framework between separate agents; Byrd uses RL to *suppress* spoofing in a trading agent (the inverse problem). Here `ŷ_t` feeds back into the agent's own state so the policy can discount apparent book imbalance during a suspected attack. The proposal is honest that oracle labels are unavailable in live markets, so reported AUROC is an *upper bound* and the contribution is scoped as a **simulation feasibility demonstration**, not a deployable surveillance claim. There is a kill-switch in the design: if detection AUROC does not clear a minimum-viability threshold by mid-Phase 3, the head is dropped and that is reported as a finding.

**Challenge 3 — Volatility-regime conditioning.**
A binary indicator `z_t ∈ {0,1}` is computed from the 20-day rolling realised volatility of AMZN, thresholded at its **trailing 252-day median** (`z_t = 1` ⇒ high volatility), and appended to the observation vector. It requires no offline training and adds exactly one input dimension. The hypothesis is that regime-dependent depth and liquidity-provision dynamics (Brogaard et al.) mean the relationship between depth signals and price impact is itself non-stationary, so a regime-aware policy should be more robust across high- and low-vol periods. The proposal pre-commits to reporting a **null finding** if conditioning does not measurably improve robustness — which is the scientifically correct stance and worth emphasising to Dr Wen.

### 2.3 Method commitments that constrain everything else

- **Platform:** JaxMARL-HFT (the GPU-accelerated multi-agent LOB simulator, ≈240× CPU speedup), matching Mohl et al. directly.
- **Game structure:** a general-sum two-player Markov game; neither agent observes the other's policy. Joint state = top-10 LOB levels, inventory, VWAP, detection signal, and binary regime indicator.
- **Market-replay assumption (critical):** injected orders modify *apparent* LOB state only; they do **not** causally propagate into prices. This is a deliberate simplification — a fully causal simulator would require modelling the market's response to spoofed volume, a separate and substantial estimation problem with no consensus model. The consequence, made explicit in the lit review, is that we are testing robustness to **state perturbation**, not to *spoofing as the microstructure literature defines it*. This is the single most important caveat in the whole project and must be stated cleanly, not buried.
- **MM reward (Spooner & Savani):** `R_mm = ΔPnL − φ·Var(PnL) − λ|q_t|`, where φ penalises profit volatility and λ penalises inventory.
- **Non-stationarity handling:** IPPO with **alternating policy freezes** — hold one agent fixed while the other adapts — a standard but imperfect mitigation for competitive-MARL cycling; the ablation will sweep freeze-schedule hyperparameters.

### 2.4 Evaluation design (so results are not ambiguous later)

- **Primary metrics:** annualised **Sortino** (preferred — attack windows induce negatively-skewed returns) and **Sharpe** (retained for comparability with Spooner & Savani), measured over *attack-on* windows.
- **Secondary:** **CVaR₀.₁₀** (the 10% threshold chosen deliberately — at ≈20 seeds the 5th-percentile effective sample is ~1 observation, making CVaR₀.₀₅ statistically uninformative).
- **Behavioural:** **quote displacement** (mean absolute deviation of quoted price from fair value under attack vs matched clean conditions) and **peak inventory excursion**.
- **Diagnostic:** detection **AUROC** over attack-on windows.
- **Statistics:** paired _t_-tests with Cohen's _d_ and 95% CI; Wilcoxon signed-rank and bootstrap CIs as non-parametric fallbacks when Shapiro–Wilk rejects normality.
- **Three agent configurations** isolate each contribution: (1) **Baseline** = Avellaneda–Stoikov + vanilla IPPO; (2) **Adversarial IPPO** = co-trained, no detection/regime; (3) **Full model** = adversarial + detection + regime. Config 1 vs 2 isolates adversarial co-training; 2 vs 3 isolates the detection and regime contributions.
- **Progression gate:** baseline IPPO must achieve Sortino/Sharpe within an acceptable margin of Avellaneda–Stoikov, and inventory SD within 2× the A–S bound, *on clean data*, before adversarial co-training is allowed to begin. This gate is the formal entry condition into Phase 3, and we have not yet produced the baseline numbers that pass through it at full scale.

---

## 3. What the literature review established (the research-direction framing)

Dr Wen's stated preference is to hear about **research direction and the economics/microstructure angle**, structured by importance, not as a field survey. The literature review is built exactly that way and submitted. Here is its argument in the form most useful for the meeting.

### 3.1 The four literatures and the single gap

The review synthesises four bodies of work that have developed largely in parallel:

1. **Market microstructure foundations.** Inventory risk (Ho–Stoll, Avellaneda–Stoikov) and adverse selection (Glosten–Milgrom, Kyle) are the two fundamental market-maker risks. The adverse-selection channel is *the mechanism through which spoofing operates*: by injecting flow that mimics the signature of informed trading, the spoofer induces the market maker to update quotes as if facing genuine adverse selection. Order-flow imbalance is an approximately linear, depth-scaled predictor of short-horizon price change (Cont–Kukanov–Stoikov), and crucially the inverse-depth coefficient implies a spoofed order's price impact **varies systematically with market conditions** — the microstructure basis for regime conditioning. Brogaard et al. document that HFT liquidity provision switches under correlated stress, making the depth–impact relationship regime-dependent.

2. **DRL for market making.** Learned agents recover Avellaneda–Stoikov-style inventory skewing benignly, with PPO the algorithmic default. But — and this is the structural vulnerability the whole project targets — **none of this line includes an adversarial counterparty or a detection channel**; the state representations contain no signal distinguishing spoofed depth from bona-fide depth. A DRL market maker conditions directly on LOB state, so a spoof that shifts apparent depth shifts the agent's quotes in the spoofer's intended direction, and this persists even in agents trained under benign conditions.

3. **Spoofing and manipulation.** Spoofing is illegal (Dodd–Frank §747; SEC/FINRA under the Securities Exchange Act; EU MAR/MiFID II) and implicated in the 2010 Flash Crash. The enforcement record (e.g. *SEC v. Lek/Avalon*, >$25M illicit profits) is the only public data on spoofing economics — survivorship-biased, but a usable range anchor. Cartea–Jaimungal–Wang give the canonical per-action cost decomposition (accidental-fill cost + regulatory penalty), which is the lineage our **c1/c2/c3** cost model descends from. The detection literature (Do & Putniņš; Wang & Wellman) characterises spoofing's order-flow signatures but keeps detection *external* to any market-making policy.

4. **Adversarial and robust approaches.** Robust Adversarial RL (Pinto et al.) and action-robust RL (Tessler et al.) are the co-training precedents; an **observation-space-perturbing adversary** (as required under replay) is structurally closest to RARL's disturbance formulation. Spooner & Savani are the primary methodological precedent for adversarial market making, but their adversary perturbs *A–S parameters* (epistemic risk), not the visible LOB, carries no order-level cost structure, gives the market maker no detection channel, and holds the volatility coefficient fixed. The multi-task learning literature (Caruana; Yu et al.'s PCGrad) supplies the machinery for the auxiliary detection head and warns that naive joint training can underperform single-task baselines when gradients interfere — which is exactly why PCGrad is in our training loop.

### 3.2 The gap, in one sentence

> No prior work has jointly examined an **economically-constrained order-placing adversary** (per-action costs in the Cartea–Jaimungal–Wang lineage), **spoof detection embedded as a supervised auxiliary objective consumed by the policy at inference**, and **explicit volatility-regime conditioning** — within a single DRL market-making system.

The individual components (adversarial co-training, supervised auxiliary objectives, contextual RL with regime indicators) are all established techniques. The contribution is the *combination*, motivated structurally rather than by convenience: the adversary generates the training distribution under attack; the detection head provides a representational channel through which the policy can condition on attack-specific features at inference; the regime indicator modulates that response across market states whose depth dynamics are themselves non-stationary.

### 3.3 The honesty clause the supervisor will respect

The review states plainly what the simulator choice costs us: **market-replay simulators cannot produce endogenous price response to injected orders, so adversarial training in such simulators tests robustness to state perturbation rather than causal spoofing, and any economic cost model attached to the adversary must be reconciled with that constraint.** This is not a weakness we are hiding — it is the framing that makes the claims defensible. The final hypothesis is therefore phrased carefully: the integrated system produces a statistically significant improvement in risk-adjusted PnL **under observation-space perturbation** relative to a non-adversarially-trained PPO baseline, **without** significant degradation in inventory-adjusted PnL under unperturbed replay; with secondary hypotheses that detection clears above-chance accuracy on held-out adversarial episodes and that regime conditioning reduces performance variance across vol regimes.

---

## 4. Engineering status — what is actually built (verified 19 June 2026)

This section is deliberately precise about the distinction between *built*, *validated*, *wired-but-inert*, and *not started*, because conflating them is how a project tells its supervisor it is further along than it is.

### 4.1 The adversarial framework — built and smoke-validated

The core training system exists and ran end-to-end:

- **`gymnax_exchange/jaxen/adversarial_marl_env.py`** (~260 lines) — the two-player adversarial environment. The spoofing adversary and the attack-aware market maker share an observation space (`adversarial_lob`).
- **`gymnax_exchange/jaxrl/MARL/ippo_adversarial.py`** (~800 lines) — the IPPO co-training driver with alternating policy freezes, checkpoint save/resume, and the combined PPO+BCE update.
- **`gymnax_exchange/jaxrl/MARL/attack_aware_policy.py`** (~140 lines) — the market-maker network with the embedded detection head.
- **`gymnax_exchange/jaxrl/MARL/pcgrad.py`** (~95 lines) — PCGrad gradient surgery to manage interference between the PPO and detection objectives.

The environment config (`config/env_configs/adversarial_mm.json`) confirms **all three contributions are wired on**: the market maker has `use_detection_head=true`, `regime_conditioning=true`, `prev_detection_in_obs=true` (the detection signal feeds back into the observation), and `pcgrad_enabled=true`; the spoofing agent has the depth-scaled injection (`inject_mult=2.0`), the telegraph attack gate (`attack_on_prob=0.1`, `attack_off_prob=0.1`), a budget (`budget_per_episode`), and the cost hooks (`c_fill`, `c_reg`) present and set to zero for unconstrained training — exactly as the proposal specifies for Challenge 1's *training* phase.

**Validation evidence (smoke scale, 5-day slice):**
- The full loop runs without crashing. Two latent crash bugs were found and fixed during this validation — one in the market-maker update (a `value_and_grad` unpacking error) and one in checkpoint restore (a tree-structure mismatch that meant the resume path had *never* worked before). Both are the kind of bug that would have silently wasted a multi-day Kaya run, so finding them at smoke scale was the point of the exercise.
- The detection-head BCE loss decreases; PCGrad merges the gradients; the freeze alternation and checkpoint save/resume are all confirmed working.

**Two substantive dynamics bugs were diagnosed and fixed**, both of which matter for whether the experiment is even meaningful:
1. **Adversary no-op collapse.** Initially the adversary learned to do nothing. Root cause was a *scaling* bug, not a training-dynamics problem: the adversary's action (a sigmoid mean ≤ 1) was being added as raw shares to a book with ~80–155 share depth, so the maximum injection was <1% of depth — zero effect on the market maker, nonzero cost, so the adversary correctly learned to no-op. Fixed by scaling injection by best-quote depth per proposal §3.2 ("order size = multiple of best-quote depth"). The adversary now injects 100–200% of depth and learns to hurt the market maker.
2. **Detection-label degeneracy.** Once the adversary attacked every step, detection became a trivial "always attack" classifier (all-positive labels). Fixed with a **telegraph (on/off burst) gate** — the attack turns on/off stochastically (~10-step bursts), producing a balanced label rate (~0.5–0.65) so the detection task is non-degenerate. Whether detection *actually learns* (BCE below the 0.69 honest baseline, AUROC > 0.5) at real training scale is an open question for the eval harness — and one of the things the Kaya runs will answer.

### 4.2 The evaluation and statistics harness — built and validated

A dedicated package, `gymnax_exchange/jaxrl/MARL/adversarial_eval/`, implements the proposal's entire measurement plan:
- `metrics.py` — Sortino, Sharpe, CVaR₀.₁₀, quote displacement, peak inventory excursion, detection AUROC.
- `stats.py` — paired _t_-tests, Cohen's _d_, confidence intervals, Shapiro–Wilk → Wilcoxon fallback, bootstrap CIs.
- `aggregate.py`, `rollout.py`, `run_evaluation.py` — per-seed array assembly, checkpoint rollout with forced attack-on/off gating, and the driver.
- `test_eval_core.py`, `test_aggregate.py` — the test suite.

This was unit-tested and validated end-to-end on a real checkpoint (restore + forced attack gate + mixed-stream AUROC + per-seed array assembly). Two items were deliberately deferred: `quote_displacement` needs a mid-price level to be finalised, and `periods_per_year` must be set from the real step→time cadence once the production data cadence is fixed. **Significance for the meeting:** the measurement apparatus is ready *before* the experiments run, which is the correct order and means results can be turned around quickly once training completes.

### 4.3 Challenge 3 (regime conditioning) — built but currently inert

This is the most important "looks done but isn't" item.

- **What exists:** `build_regime_labels.py` and `build_window_to_date.py` are committed. `regime_labels.json` now contains the **full 250-day 2022 label set** (verified: 250 dated entries, e.g. early February flips from regime 0 to regime 1, matching real AMZN 2022 volatility). The environment consumes `regime_conditioning=true`.
- **Why it is inert right now:** two gaps. First, the training config (`config/rl_configs/ippo_adversarial.yaml`) still has `REGIME_LABELS_PATH=""` and `WINDOW_TO_DATE_PATH=""` — so at the training-loop level the regime indicator is fed all-zeros regardless of the env flag. Second, `window_to_date.json` currently contains **only 11 entries** — the 5-day `2022_small` demo mapping, not the full year. The window→date map is what translates a simulator window index into a calendar date so the right regime label can be looked up; without the full-year version, the labels cannot be applied to a full-year run.
- **What blocks the full-year map:** a **data-integrity bug** (Section 6.2) — the LOBSTER loader silently drops 44 of 250 days, so the set of dates actually used is unknown, and the window→date map cannot be built by file order until that is resolved.

So Challenge 3 is genuinely built — the labelling logic is correct and validated — but it is not yet active in an end-to-end run, and saying otherwise would be inaccurate.

### 4.4 Compute — Kaya access secured (new since last update)

This is the most consequential change since the last supervisor conversation. Two Slurm batch scripts are committed:
- `slurm_baseline.sh` — runs the non-adversarial 2-player baseline (`ippo_rnn_JAXMARL.py`), 48-hour wall-time.
- `slurm_adversarial.sh` — runs the adversarial co-training (`ippo_adversarial.py`), 24-hour wall-time.

Both target our allocation (`--account=pmc097`, GPU partition, 1 GPU, 8 CPUs, 64 GB), load the CUDA toolchain, point `PYTHONPATH` at the project on the group filesystem, and run with WandB disabled and unbuffered output for live logging. The proposal named **Pawsey as primary (pending) with a UWA cluster as contingency**; in the event it is the **UWA contingency — Kaya — that came through**, and the experiments are running there. The ~1,500–2,000 A100-equivalent GPU-hours budget is now drawn from Kaya, with the caveat that **Kaya's V100s are slower than the A100-equivalent the budget assumes**, so the real wall-clock cost is materially higher (see §4.7).

### 4.5 Training runs and results — not started

No real training run output exists locally — the only run artifacts are the 7 June smoke tests on the 5-day slice. The committed adversarial config still specifies `TimePeriod: "2022_small"` and a **single seed** (`SEED: [42]`). To execute the proposal's experimental design we need to flip this to the full data period and the ≈20-seed sweep. **There are no results to report yet. This is the defining fact of the current phase.**

### 4.6 A few divergences from the proposal worth naming

- **Reward function.** The config uses `spooner_asym_damped2` — an asymmetric, damped variant of the Spooner reward — rather than the plain `ΔPnL − φ·Var − λ|q|` written in the proposal. This is a reasonable evolution (asymmetric damping handles the negatively-skewed attack returns the Sortino choice is also motivated by), but it is a deviation that should be acknowledged and justified in the methodology chapter.
- **Adversary cost parameters.** `c_fill` and `c_reg` are currently zero (correct for *unconstrained training*), but the *evaluation-time* values (c1 ≈ $0.003/share NASDAQ taker, c2 overnight funding, c3 SEC/DOJ per-share disgorgement) are still placeholders awaiting the Phase 1 calibration. This is a Dr Gao / Dr Wen workstream (financial realism) and a clean thing to raise.
- **Data year.** Proposal commits to AMZN **2024**; only AMZN **2022** is prepared (Section 6.2).

### 4.7 First HPC training run — Kaya, 19 June (supersedes parts of 4.4–4.5)

**A correction this forces.** The HPC the runs actually execute on is **Kaya (the UWA HPC), not Kaya** — the UWA cluster the proposal listed as the *contingency* is what came through, on allocation `pmc097`, with **NVIDIA V100 (16 GB)** GPUs, not A100s. Sections 1 and 4.4 (and the Kaya references in 6–8) should be read with that substitution. This matters concretely: the 16 GB V100 memory ceiling shaped the run config below.

**The milestone.** On 19 June the full adversarial co-training ran on a Kaya V100 and completed **52 consecutive updates**, stopped only by the job wall-time — no crash. This is the **first real (non-smoke) training run**, and it supersedes 4.5's "not started": training has begun, on GPU, with the fixed adversary code. Reaching a clean run took substantial HPC bring-up (CUDA/JAX environment, group-filesystem paths, GPU-memory tuning, checkpoint hygiene), all now resolved.

**Measured throughput — the first hard sizing number.** At `NUM_ENVS=16`, `NUM_STEPS=512`, steady state is **~68 s/update** (≈8,200 transitions/update). It is sobering:
- The committed `TOTAL_TIMESTEPS=5e8` would take **~48 days *per seed*** at this rate, and the committed `NUM_ENVS=64` **runs out of memory on a 16 GB V100** (it had to be cut to ≤32; the stable run used 16). So the Section 10.7 config (`5e8`, `NUM_ENVS=64`) is **not runnable as written on Kaya** and must be revised.
- Realistic path: (i) cut `TOTAL_TIMESTEPS` to the true convergence budget (likely 1e7–5e7 ≈ 1–5 days/seed), (ii) push `NUM_ENVS` back toward 32 for throughput, (iii) run the ≈20 seeds as **parallel jobs across Kaya's 34 V100s** so wall-clock ≈ one seed, (iv) checkpoint every *N* updates, not every update (the per-update orbax save to `/group` is part of that 68 s).

**What the run validated.** `adv_label_rate` held at **0.48–0.52** on every market-maker phase — the telegraph attack gate produces the balanced attacked/clean split Challenge 2 needs, confirming the §4.1 adversary fixes hold beyond smoke scale.

**What the run revealed (a finding, not a failure).** Two of Section 6.5's watch-items are now confirmed empirically:
- **Training is numerically unstable.** The market-maker value loss swings violently and trends *up* (≈41k → 813 → ~282k by update 50); policy entropy collapses then rebounds erratically — the signature of **unnormalised, large-magnitude rewards** driving destabilising gradients. **Reward/value normalisation is now the top engineering priority before the multi-seed runs** — without it the config-1-vs-2-vs-3 comparisons would be dominated by training noise rather than the effects we are measuring. This upgrades 6.5's "may warrant value normalisation" from a watch-item to a required fix.
- **The detection head is not discriminating yet** — BCE sits at ~0.693 (= ln 2, chance) throughout. Expected at 50 updates and likely held back by the instability above; it is the open question the Challenge-2 kill-switch already anticipates, and the first thing to re-check once stability is fixed.

**Net.** The Kaya pipeline is proven end-to-end — data load, GPU execution, the fixed adversary, balanced detection labels, checkpointing — which retires the framework/compute risk. The next gate is no longer "can it run" but "make it train *stably*": add reward normalisation, right-size the timestep budget and `NUM_ENVS` to the V100, and fan the seeds across the cluster.

---

## 5. Where we are against the proposal timeline

The proposal lays out five phases. Mapping today (19 June 2026) onto them:

| Phase | Proposal period | Key deliverables | Status today |
|---|---|---|---|
| **1** | Feb–Mar | JaxMARL-HFT baselines (AMZN 2024); adversary cost calibration (SEC/DOJ) | Framework built; **baselines not yet run at scale**; cost calibration still placeholder |
| **2** | Apr–May | Regime indicator validation; progression gate | Regime *labelling* validated; **progression-gate numbers not yet produced** |
| **3** | **Jun–Jul** | IPPO co-training; detection head; regime integration (2-week Aug buffer) | **We are here.** Components built and smoke-tested; real runs imminent |
| **4** | Aug–Sep | Ablation; out-of-sample validation; **draft due mid-September** | Not started |
| **5** | Oct–Nov | Submission, defence, code release | Not started |

**The honest read.** On the calendar we are exactly where Phase 3 says we should be — at the start of the co-training-and-experiments window — and the framework being fully built and de-risked going *into* that window is genuinely good positioning. But the dependency structure matters: several Phase 1–2 deliverables (full-scale baselines, the progression-gate pass, adversary cost calibration) have *not* been completed and are prerequisites for Phase 3 to produce valid results. In practice Phase 1, 2, and 3 work is now running concurrently rather than sequentially. That is workable because the engineering is done, but it compresses the slack.

The hard external constraint is the **thesis draft due mid-September** (Phase 4) and the **final thesis due 21 November**. That leaves roughly five months, of which the first chunk is consumed by getting clean multi-seed runs through Kaya and the analysis they feed. Dr Wen has previously flagged time risk explicitly; the correct message is not "we are behind" but "the framework risk is now retired, the compute is secured, and the critical path from here is execution — getting the runs done and the results analysed — with the methodology chapter drafted in parallel so writing is not back-loaded into October."

---

## 6. Risk register and items needing a decision

### 6.1 🔴 Security — live API key committed to a public-ish remote (action required)

**Finding (verified today):** `.env` is **tracked in git** and present in `origin/main` (HEAD = `56fa5f8`), containing `DATABENTO_API_KEY=db-AR87…`. Adding `.env` to `.gitignore` (which was done) does **not** untrack a file already committed — so the key is still in the repository and on GitHub. The same applies to `.hydra/` and `checkpoints/` (throwaway run artifacts), which are also still tracked despite being gitignored.

**Nuance:** the key prefix has changed from the value recorded on 7 June (`db-nwxHtt6…`) to `db-AR87…`, which suggests the key was rotated once. But the *new* key has now been committed and pushed in turn, so the exposure is current, not historical.

**Required actions (cannot be done silently — they touch the remote):**
1. **Rotate the DataBento key again** — the only real remediation, because `db-AR87…` is in pushed history.
2. `git rm --cached .env .hydra -r checkpoints` (untrack without deleting locally), commit, and push.
3. Optionally scrub history (e.g. `git filter-repo`) so the key is not recoverable from old commits — worth doing given it is a paid data API.

This is flagged prominently because a leaked paid-data credential is the kind of thing that becomes a real problem (billing, or the provider disabling the account) at the worst possible time during a compute crunch.

### 6.2 🟠 Data — 2022 vs 2024, and a loader that silently drops days (decision required)

Two coupled issues:

- **Year mismatch.** The proposal commits to **AMZN 2024** LOBSTER data (matching Mohl et al.). The repository has **AMZN 2022** (250 raw trading days) and a 5-day `2022_small` slice — **no 2024 data at all**. Either we acquire and convert 2024 (via `download_data.py` + `convert_dbn_to_lobster.py`, which is what the DataBento key is for) or we amend the proposal to use 2022. There are defensible arguments either way: 2022 was a high-volatility year (good for exercising the regime channel), while 2024 is what the proposal and the comparison paper use.
- **Silent day-dropping.** The LOBSTER loader catches processing exceptions per day and returns `None`, filtering those days out — the cached 2022 set has only **206 of 250 days**. This means (a) any "2022" run silently trains on 206 days, not the full year, and (b) the full-year window→date map (needed to activate regime conditioning, Section 4.3) cannot be built until we know which 44 days fail and why. The likely root cause is the DBN→LOBSTER conversion producing malformed data for those days. This needs to be diagnosed (capture the failing dates, fix the conversion or the loader, regenerate the cache) before either data path is solid.

**Recommendation to raise:** decide the year now, because it gates the cost calibration (AMZN-specific), the regime labels (computed on AMZN's realised vol), and the window→date map. My read is that if 2024 acquisition is quick via the DataBento pipeline, match the proposal; if it is not, switch to 2022 explicitly and note it — but do not leave it undecided, because three downstream artifacts depend on it.

### 6.3 🟠 Regime conditioning inert (engineering, not a decision)

Covered in Section 4.3. To activate: fix the data-loading bug → regenerate the full-year `window_to_date.json` → set `REGIME_LABELS_PATH` and `WINDOW_TO_DATE_PATH` in the config. Until then, any run is effectively the "Adversarial IPPO without regime" configuration (config 2 of the ablation), which is itself a legitimate experiment — but it is not the full model.

### 6.4 🟡 Statistical power — single seed vs ≈20 seeds

The committed config runs one seed. The entire statistical design (Cohen's _d_, 95% CIs, the G\*Power _d_ ≥ 0.8 target) assumes ≈20. This is a config change, not a code change, but it has a direct compute-budget implication: 20 seeds × 3 configurations × the data period is the bulk of the 1,500–2,000 GPU-hour estimate, and it needs to be planned against the Kaya allocation and the 24-hour per-job wall-time limit (jobs may need checkpoint-resume chaining to fit).

### 6.5 🟡 Open modelling watch-items

- **Co-training imbalance.** At tiny scale the always-on unconstrained adversary crushes the market maker; the telegraph gate and longer freeze schedules are the mitigations, but balance at real scale is unproven.
- **Unnormalised value/PPO loss magnitudes** — **no longer a watch-item: confirmed at scale on Kaya (19 Jun).** The value loss diverges (≈41k → ~282k over 50 updates) and entropy is erratic. **Reward/value normalisation is now a required fix before the multi-seed runs, not optional** (see §4.7).
- **Detection viability** — whether AUROC clears the kill-switch threshold is a genuine open empirical question, by design.

---

## 7. Next steps, in priority order

1. **Security remediation** (Section 6.1) — rotate the key, untrack `.env`/`.hydra`/`checkpoints`, push. Do this first; it is fast and the exposure is live.
2. **Decide the data year** (2022 vs 2024) and, in the same stroke, **fix the silent day-drop** so a full clean dataset exists. Everything downstream depends on this.
3. **Regenerate the full-year `window_to_date.json`** and **wire the regime paths into the config** to make Challenge 3 active end-to-end.
4. **Run the baseline at full scale and check the progression gate** (Sortino/Sharpe within margin of Avellaneda–Stoikov, inventory SD within 2× the A–S bound). This is the proposal's formal entry condition into adversarial co-training, and it produces the first reportable numbers.
5. **Calibrate the evaluation-time adversary costs** (c1/c2/c3 and budget B) against NASDAQ equity fee schedules and SEC/DOJ disgorgement figures — a Dr Gao / Dr Wen financial-realism workstream that can proceed in parallel with the runs.
6. **Launch the ≈20-seed adversarial sweep** across the three configurations on Kaya, with checkpoint-resume chaining to fit the 24-hour wall-time.
7. **Run the eval harness** over the resulting checkpoints to produce the primary/secondary/behavioural/diagnostic metrics and the paired-test effect sizes.
8. **Draft the methodology chapter now**, in parallel, so the September writing crunch is front-loaded.

Items 1–3 are housekeeping that unblocks everything; items 4–7 are the research; item 8 protects the schedule.

---

## 8. Meeting playbook — what to lead with for Dr Wen

Dr Wen's recorded priorities: she cares about the **research question and direction**, the **economics/microstructure angle** (inventory risk, adverse selection, price impact, spoofing regulation), and **Colin staying on schedule** — *not* the ML engineering details. She has urged moving past the lit review into research. So:

**Lead with the schedule-and-progress story, not the code.** The headline she wants to hear is: *"The lit review is done, the framework is fully built and de-risked, Kaya access came through, and I'm now at the start of the experimental phase — exactly where the timeline says Phase 3 should be. The next milestone is producing the first results."* That directly answers her standing concern about time.

**Then give her the economics hooks she finds interesting:**
- The adversary is **economically constrained at evaluation** using real NASDAQ equity taker fees, overnight funding, and SEC/DOJ disgorgement figures — so reported attack intensities reflect what a real spoofer would actually sustain. This is the internal-vs-external-validity argument she will appreciate, and it is *her* workstream (adversary cost calibration / simulation realism). Flag that the c1/c2/c3 calibration is the next concrete thing on that front.
- The **regime conditioning** rests on Brogaard et al.'s regime-dependent HFT liquidity provision and the Cont–Kukanov–Stoikov inverse-depth price-impact coefficient — i.e. a spoof of fixed size has different impact in calm vs stressed markets. This is a microstructure argument, not an ML one, and it is the §1.4 framing she was "open to as motivation."
- Be candid about the **replay caveat** — we test robustness to *state perturbation*, not causal spoofing — because she is sceptical of overstated claims and will respect the project being precise about its own scope.

**Be honest about the three open items** (security, data year, regime activation) rather than presenting a frictionless picture — but frame them as *decisions and housekeeping on the critical path*, not as the project being stuck. Specifically ask her to weigh in on the **2022-vs-2024 data decision**, since it touches the cost calibration she co-owns.

**What to expect her to push on:**
- *"When will you have results?"* — Honest answer: first baseline/progression-gate numbers within the next short cycle once the data decision and Kaya runs land; full multi-seed adversarial results are the Phase 3 output, with the draft due mid-September as the forcing function.
- *"Is the adversary realistic?"* — Point to the cost model and the explicit replay caveat; note the calibration is the immediate next step on that front.
- *"Are you on track for the thesis?"* — The framework risk is retired and compute is secured; the remaining risk is execution-and-analysis time, which is why the methodology chapter is being drafted in parallel rather than left to October.

---

## 9. Reference material below this line

Sections 1–8 are the meeting-ready core. Everything from here is **deeper reference material** you can draw on if a question goes technical or if you want it for your own clarity: Sections 10–11 are the architecture and the literature in depth; Sections 12–14 are the evaluation protocol, the adversary economics, and the replay caveat; Sections 15–17 are a methodology-chapter outline, an expanded Q&A, and a glossary; Sections 18–19 are the compute plan and the "what changed" delta; Section 20 lists the evidence base. None of it is required reading for the meeting — it is the backing detail behind the headlines above.

---

## 10. Deep dive — system architecture and the training loop

This section is for your own reference and for any technically-inclined question in the meeting (more Dr Mian's register than Dr Wen's). It describes the machine that produces the results, component by component, distinguishing verified implementation from design intent.

### 10.1 The platform underneath everything

The whole project sits on **JaxMARL-HFT** (Mohl et al., ICAIF 2025), which is itself an extension of two prior systems: **JAX-LOB** (Frey et al., ICAIF 2023), a GPU-accelerated limit-order-book matching engine written in JAX, and **JaxMARL**, a library of multi-agent RL environments and algorithms in JAX. The reason this stack exists at all is economic: on-market reinforcement learning is infeasible because PPO needs millions of environment interactions to converge, and every interaction in a live market incurs transaction costs and market impact. So all DRL market-making work runs in simulation, and simulator throughput becomes the binding constraint on how many experiments you can run. JAX-LOB's contribution is to JIT-compile the order-book matching logic and `vmap` it across thousands of parallel environments on a single GPU, reporting large per-message speedups over CPU implementations. JaxMARL-HFT layers heterogeneous multi-agent training on top, reporting up to a 240× end-to-end training speedup and demonstrating two-player IPPO training on a full year of LOB data covering 400 million orders. That 240× is what makes a ≈20-seed, three-configuration sweep with hyperparameter ablation feasible within an Honours timeframe and a Kaya allocation; on a CPU agent-based simulator it would not be.

The cost of that throughput is the **market-replay assumption**, discussed at length in Section 14: the historical order book evolves along its recorded path regardless of what the agents do. This is the structural fact that shapes how the adversary had to be designed.

### 10.2 The two agents and their interaction

The environment (`adversarial_marl_env.py`) hosts a **general-sum two-player Markov game**. Neither agent observes the other's policy parameters; each sees only the shared observation derived from the (possibly perturbed) book state.

- **The market maker (`AdversarialMM`)** is the protagonist. It posts bid/ask quotes to provide liquidity and earn the spread while managing inventory risk. Its action space in the current config is `bobRL` (a best-or-better quoting scheme) with 6 discrete actions including a "do nothing" option; it places up to 2 action messages per step. Its observation space is `adversarial_lob` — the top-10 LOB levels plus inventory, VWAP, the previous-step detection signal (because `prev_detection_in_obs=true`), and the regime indicator.
- **The spoofing adversary (`Spoofing`)** is the antagonist. It does not trade to profit in the conventional sense; it injects non-bona-fide depth into the visible book to distort the market maker's perception of supply/demand. In the current config it operates over 5 spoof levels (`n_spoof_levels=5`), injects order size as a multiple of best-quote depth (`inject_mult=2.0`), and its attacks are gated by a telegraph process (`attack_on_prob=0.1`, `attack_off_prob=0.1`) so that attacks come in bursts rather than continuously.

The interaction is competitive: during co-training the adversary's reward is the negative of the market maker's, `R_adv = −R_mm`, so it is explicitly trying to make the market maker quote badly and accumulate adverse inventory.

### 10.3 The market maker's reward, in detail

The reward follows Spooner & Savani:

```
R_mm,t = ΔPnL_t − φ · Var(PnL_t) − λ · |q_t|
```

Three terms, each doing a specific job:
- **ΔPnL_t** — the change in mark-to-market profit-and-loss this step. This is the spread-capture incentive: post tight quotes, get filled on both sides, earn the spread.
- **−φ · Var(PnL_t)** — a penalty on the *variance* of PnL. This discourages strategies that earn on average but with violent swings, and it specifically penalises slow inventory unwinding. φ is one of the sensitivity-sweep hyperparameters.
- **−λ · |q_t|** — a penalty proportional to the absolute inventory `q_t`. This is the inventory-risk term; it is the mechanism that, in benign DRL market makers, recovers Avellaneda–Stoikov-style quote skewing (the agent shades its quotes to encourage trades that reduce its position). λ is also swept.

As noted in Section 4.6, the *implemented* reward is `spooner_asym_damped2`, an asymmetric, damped variant. The asymmetry matters because spoofing attacks produce *directionally biased* losses (the agent gets picked off on one side), so a symmetric variance penalty would under-weight exactly the kind of damage the project is trying to measure. This is the same reasoning that makes **Sortino** (which penalises only downside deviation) the *primary* metric over Sharpe. The reward variant and the metric choice are therefore consistent design decisions, but the deviation from the proposal's written reward needs a sentence of justification in the methodology chapter.

### 10.4 The detection head and how the signal feeds back

Challenge 2's detection head is the project's clearest architectural novelty, so it is worth being precise about the data flow:

1. The market maker's network has a **shared encoder** that processes the observation into a latent representation.
2. From that shared latent, **two heads** branch: the usual **policy/value heads** (which produce the quoting action and the value estimate) and a **binary detection head** that outputs `ŷ_t ∈ [0,1]` — the predicted probability that a non-bona-fide order is currently active in the book.
3. The detection head is trained against **oracle labels** `y_t` from the simulator: because the simulator *knows* which orders the adversary injected, it can tag each step as attack/no-attack. The loss is binary cross-entropy, `L_BCE(ŷ_t, y_t)`.
4. Crucially, the detection prediction is **fed back into the observation** at the next step (`prev_detection_in_obs=true`). This closes the loop the proposal cares about: the policy doesn't just learn a detector as a side-task, it can *condition its quoting on its own suspicion that an attack is underway*, discounting apparent book imbalance when `ŷ` is high.

The combined objective is `L = L_PPO + λ_c · L_BCE`. Because the PPO loss and the BCE loss share the encoder, their gradients can interfere — pulling the shared representation in conflicting directions. This is the documented failure mode of naive multi-task learning (Yu et al.), and it is why **PCGrad** is in the loop.

### 10.5 PCGrad — why gradient surgery is necessary

PCGrad (Projecting Conflicting Gradients, Yu et al. 2020) addresses the case where two task gradients `g_i`, `g_j` over the shared encoder point in opposing directions (their inner product is negative). When that happens, PCGrad projects `g_i` onto the normal plane of `g_j`:

```
g_i^PC = g_i − (⟨g_i, g_j⟩ / ‖g_j‖²) · g_j
```

and the encoder is updated with the de-conflicted sum rather than the raw sum. In plain terms: when the quoting objective and the detection objective disagree about how to change the shared representation, PCGrad removes the component of one gradient that would actively undo the other, so neither task sabotages the other's learning. The smoke tests confirm PCGrad "merges" the gradients without error; whether it actually *helps* (versus simple gradient summation or magnitude normalisation) is one of the things the ablation can examine, since the lit review explicitly notes that no prior work has evaluated gradient-balancing methods for an *adversarial-RL primary plus a supervised auxiliary detection head*.

### 10.6 IPPO and the non-stationarity problem

Both agents train with **Independent PPO** — each runs its own PPO update treating the other agent as part of the environment. PPO itself is the standard clipped-surrogate policy-gradient method:

```
L^CLIP(θ) = E_t[ min( r_t(θ) · Â_t, clip(r_t(θ), 1−ε, 1+ε) · Â_t ) ]
```

where `r_t(θ)` is the policy ratio and `ε` (0.2 in our config) bounds how far each update can move the policy. The clip is what stops a single large update from collapsing subsequent learning — important when experience from rare events (spoofing bursts) is sparse.

The problem IPPO creates in a *competitive* setting is **non-stationarity**: as the adversary improves, the market maker's environment distribution shifts, and vice versa. Joint training can cycle indefinitely between dominated policies without converging — the lit review notes this is exactly why the cooperative IPPO result (Schroeder de Witt et al.) does not transfer cleanly to zero-sum settings. The mitigation in our system is **alternating policy freezes** (`FREEZE_ALTERNATION=10` updates per phase): hold one agent fixed while the other adapts for 10 updates, then switch. This partially stabilises training but does not eliminate cycling, and poorly-tuned freeze schedules can themselves cause it — which is why freeze-schedule hyperparameters are in the ablation. The smoke tests confirm the freeze alternation mechanism works mechanically; whether it produces *stable* policies at real scale in a market-maker-versus-spoofer game is, per the lit review, genuinely unverified in the literature and is one of the project's empirical contributions.

### 10.7 The training config, annotated

For completeness, the current `ippo_adversarial.yaml` settings and what they imply:
- `NUM_ENVS: 64`, `NUM_STEPS: 6400`, `TOTAL_TIMESTEPS: 5e8` — 500 million environment steps is a *real-scale* budget (not a smoke test). **However, the 19 June Kaya run (§4.7) showed this is not runnable as written on a 16 GB V100: `NUM_ENVS=64` runs out of memory (cut to ≤32), and `5e8` would take ~48 days/seed at the measured ~68 s/update. These three values must be re-sized — lower `NUM_STEPS`, `NUM_ENVS≈32`, and `TOTAL_TIMESTEPS≈1e7–5e7` — before the real sweep.**
- `GAMMA: [0.999999999, 0.99]` — the market maker's discount factor is essentially 1 (it cares about long-horizon cumulative PnL), the adversary's is 0.99 (shorter horizon). This asymmetry is deliberate.
- `GAE_LAMBDA: [0.85, 0.95]`, `UPDATE_EPOCHS: 4`, `NUM_MINIBATCHES: 4`, `CLIP_EPS: 0.2`, `ENT_COEF: [0.01, 0.01]` — standard PPO machinery.
- `FREEZE_ALTERNATION: 10`, `DETECTION_LOSS_COEF: 0.5` (this is λ_c), `PCGRAD_ENABLED: true` — the adversarial-specific knobs.
- `SEED: 42` and `SWEEP_PARAMETERS.SEED.values: [42]` — **the single-seed limitation**; this is the line that must become ≈20 seeds for the statistics to mean anything.
- `REGIME_LABELS_PATH: ""`, `WINDOW_TO_DATE_PATH: ""` — **the regime-inert lines**.

---

## 11. The literature in depth — what each source contributes

Dr Wen wanted the review "structured by importance, not as a field survey," and the submitted review honours that. This appendix lays out the load-bearing references grouped by the four literatures, so you can speak to any of them if asked. The point is not to recite — it is to show that every design decision in the system traces to a specific finding.

### 11.1 Microstructure foundations (the "why this is even a problem" layer)

- **Avellaneda & Stoikov (2008)** — the closed-form HF quoting strategy: quote symmetrically around an *inventory-adjusted reservation price* that skews away from the mid in proportion to inventory, with the spread widening in volatility and as the terminal horizon approaches. This is **our reference benchmark and performance floor** — the IPPO agent must match it on clean data to pass the progression gate. It is also the behaviour benign DRL agents are shown to recover.
- **Ho & Stoll (1981)** — the foundational inventory-risk model A–S extends; reservation prices skew with inventory.
- **Glosten & Milgrom (1985)** and **Kyle (1985)** — the adverse-selection channel: a market maker facing a mix of informed and uninformed flow widens its spread as the perceived share of informed traders rises; Kyle's λ is the price-impact coefficient reflecting order-flow informativeness. **This is the precise mechanism spoofing exploits** — the spoofer injects flow that *mimics the signature of informed trading*, inducing the market maker to update quotes as if facing real adverse selection. Both models assume the market maker cannot distinguish informed from uninformed flow; the detection head is, in effect, an attempt to give it that ability for the manipulative subpopulation.
- **Cont, Kukanov & Stoikov (2014)** — order-flow imbalance (OFI) predicts short-horizon price change approximately linearly, with slope *inversely proportional to depth at the best quote*. **The inverse-depth coefficient is the microstructure basis for regime conditioning**: a spoofed order of fixed size has materially different apparent impact in a thin (calm) versus deep (stressed) book.
- **Brogaard et al. (2018)** — HFTs supply liquidity during single-stock extreme moves but switch to *demanding* liquidity under simultaneous multi-stock stress. **This is the canonical regime-conditional liquidity-provision result** — it is the empirical anchor for the claim that the depth–impact relationship is regime-dependent, and hence that a market maker should condition on regime.
- **Easley, Kiefer, O'Hara (1996) / Easley, López de Prado, O'Hara (2012)** — PIN and VPIN, the operational adverse-selection metrics. The review positions a learned detection head as a discriminative analogue that, unlike VPIN's aggregate volume-imbalance signal, could be trained specifically on the spoofing subpopulation — leaving open whether it can outperform a PIN/VPIN-style aggregate.
- **Khomyn & Putniņš (2021) / Comerton-Forde & Putniņš (2014)** — 97% of limit orders are cancelled before execution (mostly legitimate), and ~1% of closing prices show manipulation evidence. These bound the **base-rate problem**: spoofing is a tiny subpopulation inside a huge cancellation population, which is why detection on the full order stream is hard and why the project scopes detection as a feasibility demonstration, not a deployable surveillance tool.

### 11.2 DRL for market making (the "what's been done and what's missing" layer)

- **Ganesh et al. (2019)** — a DRL market maker recovers A–S-style quote skewing in a multi-agent dealer simulation; cited as evidence DRL learns economically meaningful behaviour, but in a *stationary* environment with no strategic opponent.
- **Guéant & Manziuk (2019)**, **Gašperov & Kostanjčar (2021, 2022)** — scale DRL market making to multi-asset and signal-enriched settings, including Hawkes-process LOB simulators. **None includes an adversarial counterparty or a detection channel** — the common gap across the whole DRL-MM line.
- **Schulman et al. (2017)** — PPO, the algorithmic default. The review notes the PPO-versus-off-policy choice under adversarial non-stationarity is a *convention*, not a settled finding.
- **Sirignano (2019), Sirignano & Cont (2019), Briola/Turiel/Aste, Kolm/Turiel/Westray** — the LOB representation-learning line (spatial nets, universal price-formation features, CNN-LSTM vs MLP comparisons). Relevant to the **shared-encoder design question** — what architecture best supports *both* a quoting policy and an auxiliary detection head is an open design problem none of these works addresses.
- **Schroeder de Witt et al. (2020)** — IPPO matches joint methods on cooperative StarCraft, but this *does not transfer* to competitive zero-sum settings (the non-stationarity/cycling problem). This is the direct justification for the alternating-freeze mitigation.

### 11.3 Spoofing and manipulation (the economics layer Dr Wen cares about most)

- **Cartea, Jaimungal & Wang (2020)** — spoofing as a stochastic optimal-control problem; the **accidental-fill cost + regulatory-penalty decomposition** is the canonical per-action cost structure and the direct lineage of our c1/c2/c3 model. They also decompose spoofer PnL into intended (mid-price trending toward the spoofed direction) and unintended (round-trip on accidental fills).
- **Cartea, Chang & García-Arenas (2023)** — extends to learning agents, calibrated to NASDAQ order-book data; shows an RL market maker can discover manipulation as an *emergent* strategy, and that manipulation rises with inventory-risk tolerance (partly substituting for A–S inventory management). **The limitation the project exploits: their manipulator *is* the market maker, not an external adversary** — the external-adversary-versus-separate-market-maker setting is unexamined.
- **Do & Putniņš (2023)** — the empirical spoofing-signature paper: a comprehensive global sample of prosecuted cases, identifying the most diagnostic features (order-size asymmetry, cancellation-rate spikes, cyclical depth patterns) with high out-of-sample classifier accuracy. **This informs the feature set exposed to the detection head.** The caveat: trained/evaluated on *prosecuted* cases, so accuracy does not translate to acceptable false-positive rates on the full order stream.
- **Wang & Wellman (2020)** — detection as an adversarial learning problem (generator/discriminator/simulator), producing evasion-robust detection rules — **but the detection signal is external to any specific market maker**, not shared with a downstream policy. This is the precise contrast with Challenge 2.
- **Byrd (2022)** — RL to *suppress* spoofing in a trading agent — the **inverse** of detection, and inverting the framing this project takes.
- **Kirilenko et al. (2017)** — the Flash Crash "hot-potato" study; establishes that manipulative/destabilising order flow can damage an automated market. Together with the enforcement record (*SEC v. Lek/Avalon*, >$25M), this supports the *feasibility* premise: spoofing can profit despite detection/prosecution costs — which is what the evaluation-time budget constraint is calibrated to capture.

### 11.4 Adversarial and robust approaches (the methods layer)

- **Spooner & Savani (2020)** — *the* primary methodological precedent: robust market making as a two-player zero-sum game, co-training a market maker against an adversary that perturbs A–S dynamics parameters. The three gaps this project fills directly: their adversary perturbs *parameters* (epistemic risk), not the visible LOB; the market-maker state has *no detection channel*; and the volatility coefficient is *held fixed* (no regime conditioning).
- **Pinto et al. (2017)** — Robust Adversarial RL (RARL), the canonical co-training-as-robustness reference, inspired by H∞ control. An **observation-space-perturbing adversary** (our replay-constrained spoofer) is structurally closest to RARL's disturbance formulation — important framing, because it means the project inherits RARL's theoretical grounding rather than inventing a one-off.
- **Tessler et al. (2019)** — action-robust RL; an adversary perturbing executed actions. Distinguished from our case (which perturbs observations, not actions).
- **Bansal et al. (2018) / Gleave et al. (2020)** — the self-play dialectic: competitive co-training produces complex emergent behaviour (Bansal), but self-play-trained policies remain *brittle to novel adversarial strategies* (Gleave shows adversaries defeating SOTA victims with <3% of training timesteps). This is the bound on what co-training buys: robustness gains are real but limited by adversary-strategy diversity at training time.
- **Robust MDPs (Nilim & El Ghaoui, Iyengar, Wiesemann et al.)** — the alternative paradigm: optimise worst-case return over an *uncertainty set* rather than against a co-trained opponent. Avoids adversary-overfitting at the cost of conservatism; the review notes robust-MDP-vs-co-training has *not been directly compared* in the financial setting.
- **Caruana (1997) / Yu et al. (2020, PCGrad) / Jaderberg et al. (2017, UNREAL) / Shelhamer et al. (2017)** — the auxiliary-task and multi-task-learning machinery. The structural novelty the review pins down: existing auxiliary-task RL is *self-supervised* (pseudo-rewards from the agent's own experience), whereas our detection task is *supervised* (oracle labels). A supervised auxiliary loss combined with a *non-stationary* RL primary loss over a shared encoder is characterised in *neither* literature — which is the gap PCGrad is being tested against.

### 11.5 The synthesis the review lands on

The four literatures establish their pieces independently; **the combination is unexamined**. The review's closing claim is carefully conditional on its search scope (Google Scholar, Semantic Scholar, arXiv; peer-reviewed venues and high-citation preprints; 1981–2025): no prior work jointly examines an economically-constrained order-placing adversary, an embedded supervised detection head consumed by the policy at inference, and explicit regime conditioning. That conditional phrasing is deliberate and is the kind of precision Dr Wen flagged she wants (she was sceptical of unsupported strong claims in §1.3).

---

## 12. The evaluation protocol in full

So that "results" later are unambiguous and pre-registered in spirit, here is the complete measurement plan as it will run through the `adversarial_eval/` harness.

### 12.1 The metric stack and why each is there

| Tier | Metric | Computed over | Rationale |
|---|---|---|---|
| Primary | Annualised **Sortino** | attack-on windows | Attack windows induce negatively-skewed, directionally-biased returns; Sortino penalises only downside deviation, so it is the honest robustness measure |
| Primary | Annualised **Sharpe** | attack-on windows | Retained for comparability with Spooner & Savani and prior work; flagged as potentially flattering because symmetric volatility conflates systematic damage with noise |
| Secondary | **CVaR₀.₁₀** | attack-on windows | Tail-loss exposure; 10% threshold (not 5%) because at ≈20 seeds the 5th-percentile effective sample is ~1 observation |
| Behavioural | **Quote displacement** | attack vs matched clean | Mean absolute deviation of quoted price from fair value; distinguishes *genuine robustness* from *favourable seed variance* by comparing against the same LOB state without attack |
| Behavioural | **Peak inventory excursion** | attack-on windows | Tests for the directional inventory accumulation characteristic of a successfully spoofed agent |
| Diagnostic | Detection **AUROC** | attack-on windows | The kill-switch metric for Challenge 2; below the Phase-1 minimum-viability threshold by mid-Phase 3 ⇒ head dropped and reported |
| Robustness check | **SoftMin Sharpe** | sensitivity sweep | Downweights outlier returns via differentiable softmin; targets episodic outliers rather than sustained mean-drift |

All metrics are computed under **both** the unconstrained and the budget-constrained adversary, and over **attack-off** windows as well, so that the "no degradation on clean data" half of the hypothesis is measured, not assumed.

### 12.2 The statistical decision procedure

For each headline comparison (e.g. Full model vs Adversarial IPPO on Sortino):
1. Collect the metric across ≈20 seeds for each arm → two paired samples.
2. **Shapiro–Wilk** test (α = 0.05) on the paired differences.
3. If normal → **paired _t_-test**, report **Cohen's _d_** and **95% CI**.
4. If non-normal → **Wilcoxon signed-rank** test, report **bootstrap CI**.
5. The G\*Power design targets _d_ ≥ 0.8 at α = 0.05 with ≈20 seeds — i.e. the experiment is powered to detect *large* effects, consistent with the exploratory framing. Effects smaller than that may not be distinguishable, which the write-up must acknowledge.

A **sensitivity sweep** over φ (PnL-variance penalty), λ (inventory penalty), λ_c (detection-loss weight), and B (adversary budget) accompanies the ablation, so conclusions are not artefacts of one hyperparameter setting.

### 12.3 The three configurations and what each comparison proves

| Config | Components | Isolates |
|---|---|---|
| 1. **Baseline** | Avellaneda–Stoikov + vanilla IPPO | The performance floor; must pass the progression gate |
| 2. **Adversarial IPPO** | Co-trained, **no** detection/regime | — |
| 3. **Full model** | Adversarial + detection + regime | — |
| **1 vs 2** | | The value of **adversarial co-training** alone |
| **2 vs 3** | | The added value of the **detection head + regime conditioning** |

This factorisation is what lets the thesis make *component-attributed* claims ("co-training contributes X, the detection/regime additions contribute Y") rather than a single monolithic "the system works" claim. It also means a *negative* result on 2-vs-3 is still publishable as the pre-committed null finding for Challenge 3.

### 12.4 Regime-evaluation windows and the out-of-sample split

- **Regime windows:** the three longest contiguous windows (≥ 20 trading days) of each regime label in the held-out data — so the high-vol-vs-low-vol generalisation claim is tested on substantial, persistent regime stretches, not noisy single-day flips.
- **Out-of-sample:** a held-out *date range on the same instrument* (not a different stock). This is an explicit limitation — single-instrument evaluation bounds the out-of-distribution generalisation claim, and the thesis says so.

---

## 13. The economics of the adversary cost model (Challenge 1 in depth)

This is the part most aligned with Dr Wen's and Dr Gao's interests, and the part where the next concrete non-compute work sits.

The evaluation-time cost is `cost_t = c1·N_orders + c2·K_committed + c3·P_fill`. Each coefficient has a real-world referent that must be *calibrated*, and currently each is a placeholder:

- **c1 — order/transaction cost.** The NASDAQ equity **taker fee**, ≈ $0.003/share. Phase 1 must confirm the applicable fee *tier* (NASDAQ's schedule is volume-banded). This is the per-order cost of placing and (when accidentally filled) removing liquidity.
- **c2 — capital-commitment cost.** The **overnight funding rate** on the notional the spoofer must commit to make its orders credible. A spoof order has to be large enough to move the apparent imbalance, which ties up capital; c2 prices that.
- **c3 — accidental-fill / enforcement cost.** The **median per-share disgorgement** from SEC/DOJ *equity* spoofing enforcement actions (specifically equity, not futures, because the instrument is AMZN). Phase 1 must identify a suitable enforcement action to anchor this. This term prices the regulatory/legal expected cost of getting caught, transported into the per-action shadow-cost structure the lit review describes.

The design logic — and this is the defensible-to-an-economist part — is the **internal/external validity split**: the adversary is *unconstrained during training* (so the defender is hardened against a maximally capable attacker and both baseline and defended agents face the identical training adversary, preserving internal validity), but *budget-constrained at evaluation* (so reported attack intensities reflect what a real spoofer would economically sustain given fees, funding, and disgorgement risk, preserving external validity). A spoofing sequence whose cumulative cost exceeds budget B is replaced with a no-op; B is a sensitivity parameter swept in the ablation.

The honest caveat (Section 14) is that under market replay these costs do not arise *natively* — the simulator does not match accidental fills against a causally-responsive market, so the cost model is **normative**: it encodes what a *real* adversary would face, not what the in-simulator adversary mechanically incurs. The accidental-fill cost can be retained directly (replay *does* match injected orders against historical liquidity at the touch), but the regulatory penalty becomes a hand-set intensity coefficient. Any empirical claim about adversary effectiveness inherits this distinction, and the thesis states it.

---

## 14. The market-replay caveat — why it is defensible, not a flaw

This deserves its own section because it is the most likely point of expert scrutiny and the place where a precise answer signals research maturity.

**The constraint:** JaxMARL-HFT (like JAX-LOB) replays historical order books. The book evolves along its recorded path; the adversary's injected orders modify the *apparent* state the agents observe but do **not** causally propagate into prices. An adversary whose orders cannot move the mid-price is, functionally, a **state-perturbation agent** — regardless of the economic story wrapped around it. Spoofing as the microstructure literature defines it is about inducing *real* price movement through false signals; so adversarial training under pure replay tests robustness to **state perturbation**, not **causal spoofing**.

**Why we accept it:** the fully-causal alternative requires modelling the market's endogenous response to spoofed volume — a separate, substantial estimation problem on which *no consensus model exists*. The agent-based alternatives (ABIDES in interactive mode, PyMarketSim) support endogenous price reaction but require a calibrated agent population (no consensus on the calibration) and incur throughput costs that make a ≈20-seed, three-config, hyperparameter-swept study infeasible. The choice is an explicit **throughput-versus-causal-fidelity trade-off**, and the project chooses throughput, then *addresses the constraint in the formulation of the adversary itself* rather than pretending it isn't there.

**Why the claims survive it:** because the hypotheses are phrased in terms of robustness to *observation-space perturbation*, the replay constraint is *consistent with* the claim rather than undermining it. The project does not claim to defend against causal spoofing in live markets; it claims to (a) build a market maker robust to a structured, economically-grounded observation-space adversary, (b) demonstrate an embedded detection channel is feasible in simulation, and (c) test regime conditioning. The replay assumption bounds the *scope* of the conclusion, and the bound is stated in the limitations. The replay assumption also has *directional* effects worth naming: it **underestimates** adversary impact in thin books (where a real spoof would move prices) and **overestimates** it in deep books — a nuance the thesis flags so the results are read correctly.

**The one-sentence version for the meeting:** *"We test robustness to observation-space perturbation, not causal spoofing — that's a deliberate consequence of choosing a replay simulator for throughput, it's stated as a limitation, and the hypotheses are scoped to match it."* Delivering that unprompted is the single best signal of rigour you can give.

---

## 15. A methodology-chapter outline to start writing from now

Section 7 recommends drafting the methodology chapter in parallel with the runs so writing isn't back-loaded into September/October. Almost all of it can be written *before* any result exists, because it describes the apparatus, not the findings. A suggested skeleton, mapped to material that already exists:

1. **Problem formulation** — the two-player general-sum Markov game; state, action, reward for each agent; the replay assumption stated up front. *(Source: proposal §3.1, Section 10 of this report.)*
2. **The simulation platform** — JaxMARL-HFT / JAX-LOB, the throughput-vs-fidelity trade-off, why replay. *(Source: lit review §2.4, Section 14.)*
3. **The market maker** — network architecture, shared encoder, observation space, the Spooner reward and the asymmetric-damped variant with its justification. *(Source: config, Section 10.3.)*
4. **Challenge 1 — the adversary** — action space, depth-scaled injection, telegraph gating, the cost model with c1/c2/c3 derivations, the unconstrained-training/constrained-evaluation split. *(Source: proposal §3.2, Section 13.)*
5. **Challenge 2 — the detection head** — the auxiliary objective, oracle labelling, the feedback into observation, PCGrad. *(Source: proposal §3.3, Sections 10.4–10.5.)*
6. **Challenge 3 — regime conditioning** — the realised-vol indicator, trailing-median thresholding, the label-generation pipeline. *(Source: proposal §3.4, Section 4.3.)*
7. **Training** — IPPO, alternating freezes, non-stationarity, hyperparameters. *(Source: config, Section 10.6.)*
8. **Evaluation** — the metric stack, the statistical decision procedure, the three configs and ablation, the progression gate, regime windows, out-of-sample split. *(Source: proposal §4, Section 12.)*
9. **Limitations** — replay, single-instrument, oracle labels, binary regime proxy. *(Source: proposal §4 limitations, lit review synthesis.)*

Items 1–9 are all writable now. That is the schedule insurance.

---

## 16. Expanded anticipated Q&A

Beyond the three in Section 8, questions a supervisor (or examiner) is likely to ask, with defensible answers grounded in the current state:

- **"Why 2022 and not 2024 as the proposal says?"** — Honest answer: 2024 isn't prepared yet; 2022 is, and 2022 was a high-volatility year which exercises the regime channel well. This is an open decision (Section 6.2) and I'd value your steer, because it gates the cost calibration and the regime labels.

- **"How do you know the adversary is doing anything?"** — Because we diagnosed and fixed a scaling bug where it was injecting <1% of book depth and correctly learning to no-op; after scaling injection to 100–200% of best-quote depth, it attacks every burst and measurably hurts the market maker's reward at smoke scale. The eval harness's quote-displacement metric (attack vs matched-clean) is the formal test of this at full scale.

- **"Isn't an always-on adversary unrealistic / does it just destroy the market maker?"** — That's exactly why the adversary is unconstrained only in *training* (for a hard signal) and budget-constrained in *evaluation* (for realism), and why attacks are telegraph-gated into bursts rather than continuous. The co-training-imbalance risk at tiny scale is a known watch-item being addressed with the burst gating and longer freezes.

- **"How is detection not just memorising the labels?"** — The detection task was deliberately made non-degenerate: before the telegraph gate, the adversary attacked every step and the detector trivially predicted "always attack." With balanced ~50–65% label rates the task is real, and AUROC over held-out attack-on windows is the test. There's a pre-committed kill-switch: if AUROC doesn't clear the viability threshold, the head is dropped and reported as a finding.

- **"What if regime conditioning does nothing?"** — Then it's reported as a null finding — that's pre-committed in the proposal. The microstructure motivation (Brogaard regime-dependent liquidity; CKS inverse-depth impact) stands as motivation regardless of the empirical outcome.

- **"What's the risk to finishing on time?"** — The framework risk is retired (built and smoke-validated) and compute is secured (Kaya). The remaining risk is execution-and-analysis time, mitigated by drafting the methodology chapter in parallel and by the 2-week August buffer the proposal built into Phase 3.

- **"Has anything actually trained at scale yet?"** — No. The only runs to date are 7 June smoke tests on a 5-day slice. Full-scale baseline and the progression-gate check are the immediate next milestone, and they produce the first reportable numbers.

---

## 17. Glossary

- **A–S (Avellaneda–Stoikov)** — the closed-form inventory-aware HF quoting model used as the benchmark/floor.
- **IPPO** — Independent PPO; each agent runs its own PPO update treating others as environment.
- **PPO** — Proximal Policy Optimisation; clipped-surrogate policy-gradient method, the DRL-MM default.
- **PCGrad** — Projecting Conflicting Gradients; gradient surgery to stop the PPO and detection objectives interfering over the shared encoder.
- **BCE** — Binary cross-entropy; the detection-head loss.
- **AUROC** — Area under the ROC curve; the detection-quality metric and Challenge 2's kill-switch.
- **LOB** — Limit order book.
- **OFI** — Order-flow imbalance (Cont–Kukanov–Stoikov); depth-scaled predictor of short-horizon price change.
- **VPIN/PIN** — (Volume-synchronised) Probability of Informed Trading; aggregate adverse-selection metrics.
- **Sortino / Sharpe** — risk-adjusted return ratios; Sortino penalises only downside (primary metric here).
- **CVaR₀.₁₀** — Conditional Value-at-Risk at the 10% tail; expected loss in the worst 10% of cases.
- **Cohen's _d_** — standardised mean difference; the effect-size measure across seeds.
- **Telegraph gate** — the stochastic on/off process that makes the adversary attack in bursts.
- **Market replay** — the simulator mode where the book follows its historical path; injected orders perturb apparent state but don't move prices causally.
- **Progression gate** — the proposal's formal entry condition into adversarial co-training (baseline must match A–S on clean data first).
- **Kaya** — the supercomputing centre providing the GPU allocation (`account pmc097`).

---

## 18. Compute budget and Kaya execution plan

The proposal budgets **~1,500–2,000 A100-equivalent GPU-hours**. Now that the allocation is live (`account pmc097`), it is worth sketching how that budget maps onto the experiment so the meeting can speak to feasibility, not just intent.

**The cost drivers.** The headline experiment is **3 configurations × ≈20 seeds × the data period**, plus a **hyperparameter sensitivity sweep** over φ, λ, λ_c, and B, plus the **ablation** over freeze schedules and (optionally) PCGrad-on/off. The seed count is the multiplier that turns a single run into a statistically meaningful comparison, and it is also the single biggest consumer of GPU-hours. At `TOTAL_TIMESTEPS = 5e8` per run, 60 headline runs (3×20) alone is a large fraction of the budget before the sweep is counted — so the sweep will likely need to run at a *reduced* seed count (the statistics for the headline claims come from the full 20-seed runs; the sweep is for robustness-of-conclusion, not primary inference).

**The wall-time constraint.** `slurm_adversarial.sh` requests a **24-hour** wall-time and `slurm_baseline.sh` requests 48. If a single full-scale run does not complete within 24 hours, it must use **checkpoint-resume chaining** — submit, checkpoint, resubmit from the checkpoint — which is exactly the code path that had the tree-structure bug fixed during smoke validation. That fix is therefore not incidental housekeeping; it is what makes multi-day runs survivable on a 24-hour queue. This is worth confirming works at scale early, because discovering a resume bug *after* burning a day of queue time is the expensive failure mode.

**Parallelism already in hand.** JaxMARL-HFT's two levels of parallelism (across episodes via `NUM_ENVS=64`, and across agent types via `vmap`) mean a single GPU already runs 64 parallel environments. Seeds, by contrast, are *independent* runs and can be farmed across separate Slurm jobs (job arrays) rather than serialised — so the 20 seeds are an embarrassingly-parallel workload limited by allocation concurrency, not by a serial dependency. This is good news for turnaround: 20 seeds do not take 20× wall-clock if the allocation permits concurrent jobs.

**Sequencing on the critical path.** The sensible order is: (1) one full-scale **baseline** run to validate throughput and the progression gate and to get a real GPU-hour-per-run number; (2) use that number to size the full sweep against the remaining budget; (3) launch the headline 3×20 runs as a job array; (4) the sweep/ablation with whatever budget remains. Step 1 is also the first *reportable result*, so it doubles as a milestone and a planning measurement.

**A planning honesty note.** Until step 1 produces a real per-run GPU-hour figure on Kaya hardware, the 1,500–2,000 estimate is the proposal's a-priori guess. If the real number is higher, the lever to pull is the *sweep* seed count and the *ablation* breadth, not the headline 20 seeds — because the headline statistical claims are what the whole project rests on. Saying this explicitly in the meeting pre-empts the "what if you run out of compute" question.

---

## 19. Delta since the last supervisor update — what is genuinely new

So the meeting can be efficient about what has moved, the concrete changes since the previous update, in decreasing order of significance:

1. **Kaya access granted.** The single most important change. Compute moved from "pending, with a contingency cluster" to "live allocation with committed batch scripts for both baseline and adversarial jobs." This is what unblocks the entire experimental phase.
2. **The framework was completed and de-risked.** The adversarial environment, the IPPO co-training driver, the attack-aware policy with detection head, and PCGrad were all built and smoke-validated end-to-end — and two latent crash bugs (the MM-update unpack and the never-working checkpoint resume) were caught and fixed before they could waste cluster time.
3. **The adversary was made actually adversarial.** The no-op-collapse scaling bug was diagnosed and fixed (injection now scaled to best-quote depth), and the detection-label degeneracy was fixed with the telegraph burst gate. Before these, the adversary either did nothing or made detection trivial; now it attacks in realistic bursts and measurably damages the market maker at smoke scale.
4. **The evaluation harness was built and validated.** The entire measurement-and-statistics apparatus exists as a tested package and ran on a real checkpoint — so results can be turned around fast once training completes.
5. **The full-year regime labels were generated.** `regime_labels.json` now covers all 250 days of 2022 with a regime pattern that matches AMZN's real 2022 volatility, up from the 5-day demo.

What has **not** moved, and is the agenda for the next period: no full-scale runs yet, no results, the regime channel still inert end-to-end, the data-year decision still open, the adversary cost coefficients still placeholders, and the security item still outstanding. Sections 6 and 7 are the action list.

---

## 20. Appendix — evidence base for this report

This report was reconciled against the following, all checked on 19 June 2026:

- **Proposal:** `lit-review/23170781_Honours_Proposal.pdf` (5 pp.) — research question, three contributions, evaluation design, five-phase timeline, compute/supervision plan.
- **Literature review:** `lit-review/Literature_Review_May.pdf` (15 pp.) — four-literature synthesis, the gap statement, the replay/state-perturbation caveat, final hypotheses.
- **Code (verified present and current):** `gymnax_exchange/jaxen/adversarial_marl_env.py`; `gymnax_exchange/jaxrl/MARL/{ippo_adversarial.py, attack_aware_policy.py, pcgrad.py}`; the `adversarial_eval/` package; `config/env_configs/adversarial_mm.json`; `config/rl_configs/ippo_adversarial.yaml`.
- **Artifacts:** `regime_labels.json` (full 250-day 2022, verified); `window_to_date.json` (11 entries — demo slice only, verified); `slurm_baseline.sh`, `slurm_adversarial.sh` (Kaya, account `pmc097`, committed 11 June).
- **Data:** `data/rawLOBSTER/AMZN/2022/` (250 days) and `data/rawLOBSTER/AMZN/2022_small/` (5 days); no 2024 present.
- **Git:** `origin/main` at `56fa5f8`; `.env`, `.hydra/`, and `checkpoints/` still tracked despite `.gitignore`.

*Caveat on remote execution:* this report can only verify the *local* repository and the *committed* Slurm scripts. If training jobs have already been submitted on Kaya, their logs live under `/group/pmc097/cmelville/logs/` on the cluster and are not visible from here — so "training not started" reflects the absence of local run output and should be confirmed against the cluster.
