# Note: translating the adversary's economic costs into the replay simulator (for Dr Wen)

**The proposal's cost model.** cost_t = c1·N_orders + c2·K_committed + c3·P_fill,
with c1 a per-order fee, c2 the funding cost of committed capital, c3 an
enforcement/disgorgement cost weighted by fill probability, and a budget B
such that infeasible spoofing sequences become no-ops.

**What the simulator implements today** (following your guidance on the
accidental-fill cost):

1. **Accidental-fill / unwind cost — market-conditional, accruing.** When
   spoofed volume is at risk of being hit, the expected unwind cost is
   charged as: expected filled volume × (half-spread at execution + an
   inverse-depth price-impact term that grows as the book thins). It accrues
   at every step the position is carried, so a slower unwind costs
   proportionally more — the running-penalty semantics you described. The
   fill itself is proxied by a depth-relative probability (the replay book
   cannot generate true counterparty fills), so the penalty accrues on
   *expected* unwanted inventory rather than discrete fills.

2. **Regulatory / enforcement intensity.** A per-unit-volume penalty on
   spoofed size (the Cartea–Jaimungal–Wang "regulatory cost" as a smooth
   intensity), to be calibrated in dollars-per-share from an enforcement
   anchor.

3. **Finite budget B — the binding evaluation constraint.** A per-episode cap
   on total injectable volume. At evaluation the trained adversary is frozen,
   so reward-side costs no longer alter its behaviour — the budget is what
   binds. During training the costs shape what the adversary learns.

**The gap, honestly stated.** The proposal's per-order fee (c1) and
capital-commitment funding (c2) have no direct per-order counterpart: the
implemented costs are per-volume, and the budget is denominated in shares
rather than dollars. Two ways to close this, and this is the wording/design
choice I would like your view on:

- **(A) Implement literally**: add a per-message fee and a funding charge on
  committed notional per step, and restate B in dollar terms. Mechanically
  straightforward; the main work is defending the dollar calibration.
- **(B) Narrow the claim**: describe the adversary as *budget-capped and
  cost-regularised*, with the unwind and enforcement costs calibrated in
  spirit rather than to a fee schedule line-item, and state that c1/c2 are
  subsumed into the budget.

**The three calibration inputs — proposed values (2024 data period):**

1. **Per-order fee (c1): effectively zero on cancelled orders; $0.0030/share
   on the unwind leg.** US equity exchanges charge access (taker) fees on
   *executions*, not on posted-and-cancelled limit orders, so a spoof that
   cancels before fill incurs no direct exchange fee. The binding number is
   the Reg NMS Rule 610 access-fee cap of $0.0030/share, which applied
   throughout 2024 (the SEC's reduction to $0.001, adopted Sept 2024, has a
   Nov 2025 compliance date — after our data period). Proposal: charge
   $0.0030/share on the *unwind* volume of accidental fills (unwinding is
   liquidity-taking), and treat c1 on unexecuted spoofs as zero — which is
   itself an economically meaningful finding: exchange fee schedules impose
   almost no direct cost on layering, consistent with the enforcement-led
   deterrence the c3 term captures.
2. **Funding rate (c2): SOFR ≈ 5.3% p.a. (2024 average), optionally + a
   100–200 bps broker-margin spread.** At intraday holding times this is
   near-negligible (e.g. $1M of resting notional held 60 seconds costs about
   $0.10), which we state rather than hide: capital commitment constrains a
   spoofer through the *budget* (position limits), not through funding cost —
   supporting option (B) above.
3. **Disgorgement anchor (c3): *SEC v. Lek Securities / Avalon FA* final
   judgments (S.D.N.Y. 2019–20).** Court-ordered disgorgement against Avalon
   and its principals was $4,495,564 (plus $131,750 prejudgment interest,
   jointly and severally); Lek Securities paid a $1M penalty plus $525,892
   disgorgement, and its CEO $420,000. Because disgorgement is defined as
   illicit profits, the cleanest calibration is not per-share but a
   *profit-tax*: expected enforcement cost = detection probability x
   (kappa x adversary gross gain), with kappa anchored at ~1 by the Avalon
   judgment (disgorgement ≈ proven profits) and swept over {0.5, 1, 2} in the
   sensitivity analysis to cover penalty multipliers and detection-probability
   uncertainty.

**Context for the replay caveat** (stated in the thesis limitations): under
market replay, injected orders perturb the *observed* book but do not move
prices causally, so all costs are normative — they encode what a real
adversary would face, not what the simulated one mechanically incurs. The
lit review develops this point in Section 3.2.
