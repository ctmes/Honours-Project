# Note: the market maker's objective function (for Dr Wen)

**As implemented**, the market maker's per-step reward is

> r_t = Ψ_b + Ψ_s + rebate_t + γ·( Ψ_INV − max(0, η·Ψ_INV) )

where Ψ_b = Σ (M̄ − P_b)·Q_b and Ψ_s = Σ (P_a − M̄)·Q_a are the buy- and
sell-side trading PnL against the average mid M̄, Ψ_INV = I_t·(M_t − M_{t−1})
is the mark-to-market PnL on held inventory, rebate_t is the maker rebate on
passive fills (0.15 bps of traded value), and (γ, η) = (0.5, 0.6) weight and
asymmetrically damp the inventory component.

**Provenance.** This is the reward family of the base paper: Mohl et al.
(arXiv:2511.02136, eq. 3) train their market maker on
r_Sp = Ψ_b + Ψ_s + Ψ_INV − (1−λ)·max(0, Ψ_INV), i.e. trading PnL plus
inventory PnL with the *positive* part of the inventory PnL damped, following
Spooner & Savani (2021). Our form is a parametrised member of the same family;
the two additions relative to the paper are (i) the maker-rebate term, which
reflects the economics of passive liquidity provision on NASDAQ, and (ii) the
overall inventory weight γ.

**Economic rationale for the asymmetry.** Damping only the positive part of
the inventory PnL means the agent is not rewarded for lucky directional
inventory appreciation but bears the full cost of adverse moves. This matches
the threat model: a spoofing attack manufactures *directionally biased*
losses (the agent is induced to accumulate inventory on the wrong side), so
the objective should penalise exactly that channel without paying the agent
for taking implicit directional bets. It is the same reasoning behind Sortino
(downside-only deviation) as the primary evaluation metric.

**Deviation from the proposal.** The proposal wrote the objective as
ΔPnL − φ·Var(PnL) − λ|q_t| (Spooner & Savani 2020's form). The implemented
objective supersedes it, aligning with the base paper's eq. (3); the thesis
states this in one sentence in the methodology chapter. Note the damping term
plays the economic role of the variance penalty (both punish PnL swings from
inventory), and inventory risk is additionally bounded mechanically by the
environment's position handling rather than a |q| term.

**A caveat we monitor rather than hide.** Mohl et al. observe that this
reward family is not normalised by traded volume, so near-"never trade"
policies can look attractive; their learned MM trades very infrequently and
loses ~0.2 ticks on average while still beating the Avellaneda–Stoikov
baseline. Accordingly (a) all our hypotheses are comparative (across arms and
vs A–S), never absolute-profitability claims, and (b) evaluation reports a
pre-registered `quote_presence` statistic (fraction of steps with a two-sided
quote) per arm, so a no-trade collapse is visible rather than silently
flattering the risk-adjusted metrics.
