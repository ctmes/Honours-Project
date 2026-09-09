"""Schema layer over an eval_*.json report written by run_evaluation.py.

Everything downstream (figures, tables) goes through EvalReport rather than
indexing raw dicts, for three reasons that all bit earlier ad-hoc analysis:

  1. PAIRING. Contrasts are paired by seed INDEX (see run_evaluation's module
     docstring). Any code that slices two arms independently can silently
     mis-pair them if an arm has a different seed count. `paired()` refuses.

  2. NON-FINITE VALUES ARE MEANINGFUL, NOT MISSING. quote_displacement is NaN
     exactly on the seeds that never posted a two-sided quote -- it is undefined
     rather than lost. Dropping those NaNs conditions the statistic on the
     subsample that kept quoting, which is precisely the selection that makes a
     collapsed arm look healthy. `finite()` returns the mask so callers must
     decide, and n_finite is carried into every caption.

  3. PARTIAL RUNS. A run may be missing arms (a dead SLURM task), a contrast
     block (both arms must be present), the gate, or the equivalence block.
     Every accessor degrades to None instead of raising, so a half-finished
     v3 eval still plots what it has.

The report is treated as read-only evidence: no statistic is recomputed here
that run_evaluation.py already computed. Figures show what was tested, not a
second opinion computed at plot time.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

import numpy as np

# Design order from preregistration.json /design. Arms B..E are the 2x2 factorial
# over {detection head, regime conditioning}; F is the unconstrained-adversary
# comparator; 'as' is the fixed-policy Avellaneda-Stoikov benchmark, which is not
# an arm of the factorial and is drawn apart from it everywhere.
ARM_ORDER = ("baseline", "adversarial", "detection", "regime",
             "full", "unconstrained", "as")

ARM_LABELS = {
    "baseline":      "A  baseline",
    "adversarial":   "B  adversarial",
    "detection":     "C  +detection",
    "regime":        "D  +regime",
    "full":          "E  full",
    "unconstrained": "F  unconstrained",
    "as":            "A-S benchmark",
}

# The fixed-policy benchmark: not learned, so it has no training seeds in the
# same sense and no detection head. Excluded from factorial panels by default.
BENCHMARK_ARMS = ("as",)

# Pre-registered confirmatory family (preregistration.json /primary_metrics).
PRIMARY_METRICS = ("sortino_on", "sharpe_on", "cvar_on", "auroc")

# Arms that actually carry a detection head (preregistration.json /design: the
# head is what C adds to B, and E carries it too). Every arm reports an `auroc`
# because the rollout always computes one, so without this list an arm with no
# head -- including the fixed-policy A-S benchmark -- can post a nominally
# significant AUROC and be read as evidence for H3. It is not: H3 is a claim
# about the head, and only these arms can support or refute it.
DETECTION_ARMS = ("detection", "full")

# Contrast labels in the order run_evaluation.py emits them, with the role each
# plays in the design. Used to order forest-plot rows.
CONTRAST_LABELS = {
    "adversarial_vs_baseline":      "B vs A   co-training effect (H1)",
    "detection_vs_adversarial":     "C vs B   detection | regime off",
    "full_vs_regime":               "E vs D   detection | regime on",
    "regime_vs_adversarial":        "D vs B   regime | detection off",
    "full_vs_detection":            "E vs C   regime | detection on",
    "full_vs_adversarial":          "E vs B   joint defence",
    "full_vs_baseline":             "E vs A   full vs undefended",
    "adversarial_vs_unconstrained": "B vs F   adversary cost model",
}

METRIC_LABELS = {
    "sharpe":                     "Sharpe ratio",
    "sortino":                    "Sortino ratio",
    "cvar":                       "CVaR (worst 10%)",
    "peak_inventory":             "Peak |inventory|",
    "inventory_sd":               "Inventory SD",
    "quote_displacement":         "Quote displacement (MAD from fair value)",
    "quote_presence":             "Quote presence (frac. steps two-sided)",
    "mean_attack_rate":           "Attack rate (frac. steps attacked)",
    "mean_injected_volume":       "Mean injected volume",
    "injected_volume_per_attack": "Injected volume per attack step",
    "sortino_lowvol":             "Sortino (low-vol regime)",
    "sortino_highvol":            "Sortino (high-vol regime)",
    "regime_gap":                 "Regime gap  |Sortino_hi - Sortino_lo|",
    "auroc":                      "Detection AUROC",
}

# Metrics where a LOWER value is the better outcome. Used only to annotate axes;
# no figure silently reorients an axis, because a reader comparing two figures
# with flipped axes is worse off than one reading a label.
LOWER_IS_BETTER = ("peak_inventory", "inventory_sd",
                   "quote_displacement", "regime_gap")

# Pre-registered validity gate (interpretation_notes/quote_presence_validity_gate):
# "an arm whose seeds are predominantly quote_presence ~= 0 cannot support an H1
# robustness claim". The preregistration fixes the RULE but not the two numbers it
# needs, so both are exposed as flags and both defaults are stated in every caption:
#   COLLAPSE_CUT       -- what counts as "~= 0"
#   COLLAPSE_MAJORITY  -- what counts as "predominantly"
# On v2 the pooled quote_presence_on distribution over the six learned arms is a
# spike of 68/120 seeds at EXACTLY 0.0, a thin scatter of 9 seeds in (0, 0.1], and
# a quoting mode from 0.28 upward -- so it is bimodal but NOT cleanly separated,
# and the collapsed count does move with the cut (72/120 at 0.001, 73 at 0.01,
# 77 at 0.1). The gate VERDICT is what has to be robust, not the count, so
# collapse_sensitivity() sweeps the cut and every validity figure prints the
# sweep next to the verdict rather than asking the reader to trust one number.
# 0.5 is the plain reading of "predominantly". Neither default was chosen after
# seeing a p-value.
COLLAPSE_CUT = 0.01
COLLAPSE_MAJORITY = 0.5

# Cuts swept by collapse_sensitivity(): spans "exactly zero" to "quoting under a
# tenth of the time", i.e. the whole range over which the word "~= 0" could be
# read in good faith.
COLLAPSE_CUT_SWEEP = (0.0, 0.001, 0.01, 0.05, 0.1)


def metric_label(key: str) -> str:
    """Map a per-seed key to a human axis label; unknown keys pass through."""
    for suffix, cond in (("_on", "attack on"), ("_off", "attack off")):
        if key.endswith(suffix):
            base = key[: -len(suffix)]
            return f"{METRIC_LABELS.get(base, base)} ({cond})"
    return METRIC_LABELS.get(key, key)


class EvalReport:
    """Read-only view of one eval_*.json."""

    def __init__(self, data: dict, source: str = "<dict>"):
        if "per_seed" not in data:
            raise ValueError(
                f"{source}: no 'per_seed' block. This report predates the per-seed "
                "patch in run_evaluation.py and only carries summaries, which are "
                "not enough to plot a distribution. Re-run the evaluation.")
        self._d = data
        self.source = source

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, path: str) -> "EvalReport":
        with open(path) as fh:
            return cls(json.load(fh), source=os.path.basename(path))

    # ------------------------------------------------------------- structure

    @property
    def meta(self) -> dict:
        return self._d.get("_meta", {})

    @property
    def arms(self) -> tuple[str, ...]:
        """Arms present, in design order; anything unrecognised is appended."""
        present = set(self._d["per_seed"])
        known = [a for a in ARM_ORDER if a in present]
        return tuple(known + sorted(present - set(known)))

    def factorial_arms(self) -> tuple[str, ...]:
        """Arms excluding the fixed-policy benchmark."""
        return tuple(a for a in self.arms if a not in BENCHMARK_ARMS)

    @property
    def seeds(self) -> list[int]:
        return list(self.meta.get("seeds", range(self.n_seeds)))

    @property
    def n_seeds(self) -> int:
        for arm in self._d["per_seed"].values():
            for v in arm.values():
                return len(v)
        return 0

    def metrics(self, arm: str | None = None) -> tuple[str, ...]:
        arm = arm or self.arms[0]
        return tuple(sorted(self._d["per_seed"][arm]))

    def has(self, arm: str, metric: str) -> bool:
        return metric in self._d["per_seed"].get(arm, {})

    @property
    def is_confirmatory(self) -> bool:
        """True only for a signed-off, complete run.

        Gates the confirmatory label on figures: an exploratory or partial run
        must not be captioned as if its Holm-adjusted p-values were the
        pre-registered test.
        """
        m = self.meta
        return (bool(m.get("confirmatory")) and bool(m.get("signed_off"))
                and not m.get("partial_run", False))

    # ----------------------------------------------------------- per-seed data

    def per_seed(self, arm: str, metric: str) -> np.ndarray:
        try:
            return np.asarray(self._d["per_seed"][arm][metric], dtype=np.float64)
        except KeyError as exc:
            raise KeyError(
                f"{self.source}: no per-seed {metric!r} for arm {arm!r}. "
                f"Arms: {list(self._d['per_seed'])}") from exc

    def paired(self, arm_a: str, arm_b: str,
               metric: str) -> tuple[np.ndarray, np.ndarray]:
        """Seed-index-paired arrays (a, b); raises if the arms disagree on length.

        Length equality is the only pairing check the JSON supports -- it does not
        record which training seed produced which row. Ordering is guaranteed
        upstream by evaluate_seeds and asserted in the preregistration, not here.
        """
        a, b = self.per_seed(arm_a, metric), self.per_seed(arm_b, metric)
        if a.shape != b.shape:
            raise ValueError(
                f"{self.source}: cannot pair {arm_a}/{arm_b} on {metric}: "
                f"{a.shape} vs {b.shape} seeds. Paired tests need equal, "
                "index-aligned seed lists.")
        return a, b

    @staticmethod
    def finite(*arrays: np.ndarray) -> np.ndarray:
        """Mask of positions finite in EVERY argument (pairwise-complete)."""
        mask = np.ones(np.shape(arrays[0]), dtype=bool)
        for arr in arrays:
            mask &= np.isfinite(np.asarray(arr, dtype=np.float64))
        return mask

    # ------------------------------------------------------- validity gate

    def collapse_mask(self, arm: str, condition: str = "on",
                      cut: float = COLLAPSE_CUT) -> np.ndarray:
        """Mask of seeds that effectively never posted a TWO-SIDED quote.

        Read the emphasis literally. quote_presence counts steps with a bid AND
        an ask posted, so quote_presence = 0 does NOT mean the seed was inactive:
        on v2 every single zero-presence seed still has inventory_sd up to 130
        and peak |inventory| up to 449. Those seeds are trading one-sided and
        ratcheting inventory, which is a different failure from the Mohl et al.
        no-trade optimum and has a different fix. Use inventory_activity() before
        describing any of these seeds as "not trading".
        """
        return self.per_seed(arm, f"quote_presence_{condition}") <= cut

    def inventory_activity(self, condition: str = "on",
                           cut: float = COLLAPSE_CUT) -> dict[str, dict]:
        """Is each arm's zero-presence subgroup inactive, or trading one-sided?

        The discriminator between the two collapse stories. A no-trade optimum
        implies inventory_sd ~ 0 on those seeds; an inventory ratchet implies
        large inventory_sd and peak |inventory| despite no two-sided quote.
        """
        out = {}
        for arm in self.arms:
            if not (self.has(arm, f"quote_presence_{condition}")
                    and self.has(arm, f"inventory_sd_{condition}")):
                continue
            m = self.collapse_mask(arm, condition, cut)
            isd = self.per_seed(arm, f"inventory_sd_{condition}")[m]
            pk = (self.per_seed(arm, f"peak_inventory_{condition}")[m]
                  if self.has(arm, f"peak_inventory_{condition}")
                  else np.full(isd.shape, np.nan))
            isd_f, pk_f = isd[np.isfinite(isd)], pk[np.isfinite(pk)]
            out[arm] = {
                "n_no_two_sided": int(m.sum()),
                "n_of_those_flat": int((isd_f == 0).sum()),
                "max_inventory_sd": float(isd_f.max()) if isd_f.size else float("nan"),
                "max_peak_inventory": float(pk_f.max()) if pk_f.size else float("nan"),
                # True => the no-trade reading is the wrong one for this arm.
                "trading_one_sided": bool(isd_f.size and (isd_f > 0).any()),
            }
        return out

    def collapse_summary(self, condition: str = "on", cut: float = COLLAPSE_CUT,
                         majority: float = COLLAPSE_MAJORITY) -> dict[str, dict]:
        """Per-arm no-two-sided-quote counts and the pre-registered H1 verdict.

        gate_ok False means: whatever this arm's risk metrics say, they are not
        evidence about robustness. The adversary perturbs only the MM's observed
        depth, so it can move outcomes ONLY via the MM's quoting response; an arm
        that mostly does not quote two-sided is invariant to it for a reason that
        has nothing to do with the hypothesis. See preregistration.json
        interpretation_notes/quote_presence_validity_gate.
        """
        out = {}
        for arm in self.arms:
            if not self.has(arm, f"quote_presence_{condition}"):
                continue
            qp = self.per_seed(arm, f"quote_presence_{condition}")
            collapsed = qp <= cut
            n_col, n = int(collapsed.sum()), int(qp.size)
            out[arm] = {
                "n": n,
                "n_collapsed": n_col,
                "frac_collapsed": n_col / n if n else float("nan"),
                "gate_ok": bool(n and (n_col / n) <= majority),
                "mean_qp": float(np.nanmean(qp)),
                "median_qp": float(np.nanmedian(qp)),
                "mean_qp_survivors": (float(np.nanmean(qp[~collapsed]))
                                      if (~collapsed).any() else float("nan")),
            }
        return out

    def collapse_sensitivity(self, condition: str = "on",
                             cuts: Iterable[float] = COLLAPSE_CUT_SWEEP,
                             majority: float = COLLAPSE_MAJORITY) -> dict:
        """How the collapse verdict moves as the "~= 0" cut is varied.

        The preregistration says "predominantly quote_presence ~= 0" without
        fixing a number, so a single cut is a judgement call the reader is
        entitled to audit. Returns {'cuts': [...], 'counts': {arm: [...]},
        'verdicts': {arm: [bool, ...]}, 'stable': {arm: bool}} where `stable`
        means the pass/fail verdict is unchanged across the whole sweep -- the
        only property the argument actually needs.
        """
        cuts = list(cuts)
        counts: dict[str, list[int]] = {}
        verdicts: dict[str, list[bool]] = {}
        for arm in self.arms:
            if not self.has(arm, f"quote_presence_{condition}"):
                continue
            qp = self.per_seed(arm, f"quote_presence_{condition}")
            n = int(qp.size)
            counts[arm] = [int((qp <= c).sum()) for c in cuts]
            verdicts[arm] = [bool(n and (k / n) <= majority) for k in counts[arm]]
        return {
            "cuts": cuts,
            "counts": counts,
            "verdicts": verdicts,
            "stable": {a: len(set(v)) == 1 for a, v in verdicts.items()},
            "n": self.n_seeds,
        }

    # --------------------------------------------------------------- stats

    def contrasts(self) -> tuple[str, ...]:
        """Contrast blocks actually present, in design order."""
        return tuple(k for k in CONTRAST_LABELS if k in self._d)

    def contrast(self, label: str, metric: str | None = None) -> Any:
        blk = self._d.get(label)
        if blk is None or metric is None:
            return blk
        return blk.get(metric)

    def holm(self, label: str, metric: str) -> float | None:
        return self._d.get("holm", {}).get(label, {}).get(metric)

    def summary(self, arm: str, metric: str) -> dict | None:
        return self._d.get("summaries", {}).get(arm, {}).get(metric)

    def equivalence(self) -> dict:
        """{'full_vs_baseline': {metric: tost_dict}}; empty if H2 was not run."""
        return self._d.get("equivalence_off", {}) or {}

    def equivalence_margins(self) -> dict:
        return self._d.get("equivalence_margins_resolved", {}) or {}

    def auroc_above_chance(self) -> dict:
        return self._d.get("auroc_above_chance", {}) or {}

    def progression_gate(self) -> dict | None:
        return self._d.get("progression_gate")

    # -------------------------------------------------------------- helpers

    def describe(self) -> str:
        """Provenance block, printed by every CLI and stamped on every figure."""
        m = self.meta
        gate = self.progression_gate()
        gate_str = ("ABSENT" if gate is None
                    else ("PASS" if gate.get("passed") else "FAIL"))
        return "\n".join([
            f"source              {self.source}",
            f"arms                {', '.join(self.arms)}",
            f"seeds               n={self.n_seeds}",
            f"checkpoint_step     {m.get('checkpoint_step', '?')}",
            f"common_adversary    {m.get('common_adversary', 'self-play (per-arm)')}",
            f"periods_per_year    {m.get('periods_per_year', '?')}",
            f"signed_off          {m.get('signed_off', '?')}",
            f"partial_run         {m.get('partial_run', '?')}",
            f"confirmatory        {self.is_confirmatory}",
            f"progression_gate    {gate_str}",
        ])

    def provenance_stamp(self) -> str:
        """Compact one-liner burned into every figure, so a PNG in a slide deck
        can always be traced back to the run that produced it."""
        m = self.meta
        tag = "CONFIRMATORY" if self.is_confirmatory else "EXPLORATORY/PARTIAL"
        return (f"{self.source} | n={self.n_seeds} seeds | step "
                f"{m.get('checkpoint_step', '?')} | {tag}")


def load_reports(paths: Iterable[str]) -> list[EvalReport]:
    return [EvalReport.load(p) for p in paths]
