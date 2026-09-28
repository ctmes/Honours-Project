# Amendment: forced worst-case attack, to make H1 identifiable

> **Terminology (added 2026-09-28).** "Worst-case" in this note's title and the amendment name is historical. A uniform maximal-magnitude injection is one fixed attack, not the worst case: strategically timed or learned attacks can be stronger (Lin et al. 2017; Zhang et al. 2021). The thesis calls it the **maximal one-sided** attack. See docs/literature_audit_2026-09-28.md §3.5.


**Written before any `eval_forced_attack.py` result exists.** Matches the discipline
already applied to every prior design change in this project (`preregistration.json`'s
`amendments`): the decision is recorded before the data that would validate or
embarrass it.

**The problem.** `docs/note_h1_attack_effect.md` established that H1 is untested, not
null, because the co-trained adversary used to evaluate every arm is fundamentally
unidentifiable as a robustness test. Under its cost model (`c_fill`/`c_reg` charged
unconditionally, profit-tax reward zero when nothing is extracted), the constrained
adversary (`v4_config3_full`) rationally abstains (mean action 0.159% of max,
`check_adversary_sidedness.py`). A cost-free variant attacks substantially (54% of max)
but converges to a symmetric injection (sidedness 0.08/1.0), which by construction
can't move `queue_imbalance` — the one observation channel the attack mechanism relies
on. Either way, a null result is indistinguishable from "the MM is robust." No result
from a co-trained adversary can settle H1 under this design.

**The fix.** Stop relying on a co-trained adversary's converged behaviour. Override the
adversary's action with a fixed, maximal, one-sided injection at every step of a
rollout against an *existing, already-trained* market-maker checkpoint — no
retraining. This decouples "is the MM policy sensitive to a worst-case observation
perturbation" (a standard adversarial-ML robustness question, answerable by
construction) from "would a rational, cost-bearing adversary choose to exploit it" (a
game-theoretic equilibrium question, which is what the original design tested and
which turned out to be uninformative regardless of the answer).

**What changes, precisely.** A new standalone script, `eval_forced_attack.py`,
overrides `actions[adv_idx]` immediately before each `env.step` call in an evaluation
rollout (the same seam `check_adversary_sidedness.py` already uses), substituting a
fixed all-bid or all-ask injection vector for whatever the adversary's own trained
policy would have output. `spoofing_agent.py`'s `action_to_injection` is a pure
function of `(action, world_state, budget_remaining, gate)` — a hand-supplied action
passes through it identically to a policy-derived one, so no environment or agent code
changes. Zero changes to `adversarial_marl_env.py`, `spoofing_agent.py`, `rollout.py`,
`run_evaluation.py`, or `run_production_eval.py`.

**Budget and gate (corrected in drafting, before any result).** An earlier draft of
this note claimed the `baseline` arm would make a forced attack a silent no-op, because
its *training* env (`adversarial_mm_v4_config1.json`) sets `budget_per_episode: 0.0` and
`attack_on_prob: 0.0`. That was wrong for evaluation: `eval_2024_test_v4_config1.yaml`
deliberately loads the config-**2** env json instead (its own header explains why —
"the config-1 TRAINING json pins the telegraph off with zero budget and would make
attack-on a no-op"), and every v4 eval env runs with `budget_per_episode = 1e6`. The
project had already closed that trap. The gate is forced by `set_attack_mode("on")`,
exactly as the production eval does. `eval_forced_attack.py` still sets the budget
explicitly on every arm, for a different reason: the trained adversary injected
~0.16% of its maximum, so nothing has ever tested whether 1e6 is enough to *sustain* a
maximal injection across a full episode. That is checked empirically before the grid
(`budget_final_frac` / `budget_exhausted_frac`), and the value is raised if it binds.

**Two axes, pre-specified, not chosen after seeing results.**
- `side`: `bid` and `ask` are both run. Testing only one direction would leave a
  robustness claim resting on an untested assumption of symmetry in the MM's own
  quoting logic.
- `budget`: whether the default override value binds within an episode is an empirical
  question, checked in Phase 2 of the implementation plan before the full grid runs,
  not assumed. If it doesn't bind, that settles "realistic vs generous attacker" for
  this action space; if it does, the value is raised until it doesn't, so the
  sustained-attack condition is genuinely sustained.

**Inference status: EXPLORATORY EXTENSION — NOT CONFIRMATORY.** This is a
supplementary robustness probe, run on existing checkpoints, reported as its own
clearly-labelled analysis. It is not a rerun of, replacement for, or reconciliation
with the v3 confirmatory result (`eval_1179095.json`) or the existing v4 exploratory
result (`eval_17313.json`), which stand as already reported. No p-value from this
analysis is a significance claim in the pre-registered sense; `analysis_decisions_changed:
NONE` in the corresponding `preregistration.json` amendment entry.

**Preconditions before the full grid runs** (mirrors the discipline of the v4
spread_skew amendment's own preconditions): a single-seed smoke test on the `baseline`
arm must show `mean_attack_rate_forced ≈ 1.0` with a non-exhausted budget
(`budget_final_frac` well above 0), and its `off` condition must reproduce
`eval_17313.json`'s `_off` values for that seed (proof this script's loop is faithful
to the production one, so the only difference between conditions is the override);
and the `full` arm's queue-imbalance shift must land in the ballpark of
`check_adversary_lever.py`'s one-sided reference (~+0.50 at `inject_mult=2.0`), with
the sign flipping correctly between `--side bid` and `--side ask`. If either check
fails, the override mechanism is not reaching the environment the way this amendment
assumes, and the grid is not run.

**Preconditions — outcome (2026-09-27, recorded before the grid was submitted).**
Seed-0 smoke tests (Kaya jobs 35836–35838, n_envs=64): `off` reproduced
`eval_17313.json`'s `_off` values exactly (max|diff| = 0, baseline and full); attack
rate 0.998 / 0.998 / 0.964; injection one-sided as intended; mean queue_imbalance
−0.03 → +0.39 (bid) and −0.43 (ask). Budget 1e6 **bound** (exhausted in 0.3% of
env-steps on bid, 3.9% on ask — the source of the ask side's 0.964 attack rate), so per
the rule above it was raised: at **1e7** (job 35875, full arm, ask side, the worst
case) attack rate 1.000, budget never exhausted, 92.5% remaining at episode end. The
grid runs at `--budget-override 1e7`.

**Candidate sign-off:** PENDING REVIEW — drafted 2026-09-27, not yet approved.
