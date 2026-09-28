"""Reporting units for the Sharpe-family metrics (RC2).

Every Sharpe/Sortino in the eval JSONs is annualised as
    per_step_ratio * sqrt(periods_per_year),  periods_per_year = 10,833,681,
so a baseline Sharpe of -119 is a per-step ratio of about -0.036. The sqrt(q)
rule is only valid for IID returns (Lo 2002, "The Statistics of Sharpe Ratios");
per-step market-maker PnL is serially correlated through inventory marking, so
the annualised magnitudes are not interpretable as annual Sharpe ratios. The
thesis reports PER-STEP values (preregistration.json -> amendments ->
robustness_checks_amendment_2026-09-28, RC2).

The conversion is exact and loses nothing: the per-env ratio is mean/sd*sqrt(q),
and a constant factor commutes with every mean over envs and seeds. Differences,
CIs, SDs and margins scale by the same factor; p-values, Cohen's d, AUROC, CVaR
and every verdict are unchanged. Nothing is re-run.
"""
from __future__ import annotations

import math

UNITS = ("per-step", "annualised")

# Metric-name prefixes that carry the sqrt(periods_per_year) factor.
_SCALED_PREFIXES = ("sharpe", "sortino", "regime_gap")


def is_scaled(metric: str) -> bool:
    """True for sharpe_*, sortino_* (incl. sortino_lowvol/highvol_*) and regime_gap_*."""
    return metric.startswith(_SCALED_PREFIXES)


def factor(metric: str, periods_per_year: float, units: str = "per-step") -> float:
    """Multiplier that converts a stored (annualised) value into `units`."""
    if units not in UNITS:
        raise ValueError(f"units must be one of {UNITS}, got {units!r}")
    if units == "annualised" or not is_scaled(metric):
        return 1.0
    if not periods_per_year or periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    return 1.0 / math.sqrt(periods_per_year)


def unit_label(metric: str, units: str = "per-step") -> str:
    """Short suffix for axis labels / table headers."""
    if not is_scaled(metric):
        return ""
    return " (per step)" if units == "per-step" else " (annualised)"


# Fields of a stats.ComparisonResult / TostResult / summary that carry the metric's
# units. Everything else (p-values, cohens_d, statistics, n, booleans) is unit-free:
# the t and Wilcoxon statistics are invariant to a positive rescaling of the data.
_VALUE_FIELDS = ("mean_a", "mean_b", "mean_diff", "ci_low", "ci_high", "hl_estimate",
                 "margin", "mean", "std", "median", "ippo", "as")


def _scale_fields(d: dict, f: float) -> None:
    for k in _VALUE_FIELDS:
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            d[k] = v * f


def rescale_report(raw: dict, units: str = "per-step") -> dict:
    """Deep-copied eval JSON (run_evaluation / eval_forced_attack schema) in `units`.

    Converts per_seed, summaries, every "<a>_vs_<b>" contrast block,
    equivalence_off, equivalence_margins_resolved and progression_gate.detail for
    Sharpe-family metrics. p-values, Holm, Cohen's d, AUROC blocks and all
    verdict booleans are left untouched -- they are unit-invariant, which
    tests/test_robustness_checks.py pins by recomputing them.
    """
    import copy
    out = copy.deepcopy(raw)
    meta = out.setdefault("_meta", {})
    if units == "annualised" or meta.get("units") == units:
        return out
    ppy = float(meta.get("periods_per_year") or 0.0)
    if ppy <= 0:
        raise ValueError("report has no positive _meta.periods_per_year; cannot rescale")

    for arm_metrics in out.get("per_seed", {}).values():
        for m, vals in arm_metrics.items():
            if is_scaled(m):
                f = factor(m, ppy, units)
                arm_metrics[m] = [v * f if isinstance(v, (int, float)) and v is not None else v
                                  for v in vals]
    for arm_metrics in out.get("summaries", {}).values():
        for m, s in arm_metrics.items():
            if is_scaled(m) and isinstance(s, dict):
                _scale_fields(s, factor(m, ppy, units))
    for key, blk in out.items():
        if "_vs_" in key and isinstance(blk, dict):
            for m, res in blk.items():
                if is_scaled(m) and isinstance(res, dict):
                    _scale_fields(res, factor(m, ppy, units))
    for blk in out.get("equivalence_off", {}).values():
        for m, res in blk.items():
            if is_scaled(m) and isinstance(res, dict):
                _scale_fields(res, factor(m, ppy, units))
    em = out.get("equivalence_margins_resolved")
    if isinstance(em, dict):
        for m, v in em.items():
            if is_scaled(m) and isinstance(v, (int, float)):
                em[m] = v * factor(m, ppy, units)
    detail = out.get("progression_gate", {}).get("detail", {})
    for m, d in detail.items():
        if is_scaled(m) and isinstance(d, dict):
            _scale_fields(d, factor(m, ppy, units))
    meta["units"] = units
    return out
