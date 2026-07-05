"""
Evaluation metrics for the adversarial market-making study.

Pure NumPy functions operating on collected rollout arrays — no JAX, no env, no
checkpoints — so they can be unit-tested in isolation and reused for both the
real multi-seed runs and synthetic validation.

Metric set follows the proposal §4:
  Primary (risk-adjusted): annualised Sortino, annualised Sharpe
  Secondary (tail):        CVaR at the 10% level
  Behavioural:             quote displacement, peak inventory excursion
  Diagnostic:              detection AUROC

Conventions
-----------
- `returns` are per-step PnL changes (one value per environment step). Sharpe and
  Sortino are annualised by sqrt(periods_per_year); see `periods_per_year` note.
- `periods_per_year` MUST be supplied explicitly. It is the number of return-steps
  in a trading year = steps_per_episode * episodes_per_trading_day * trading_days_per_year.
  Hard-coding a wrong constant silently corrupts every Sharpe/Sortino, so there is
  no default — derive it from the data/episode config and pass it in.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

_EPS = 1e-12


def _flatten(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def sharpe_ratio(returns, periods_per_year: float) -> float:
    """Annualised Sharpe = mean(returns) / std(returns) * sqrt(periods_per_year).

    Uses population-style std with ddof=1 (sample std). Returns nan if the series
    has < 2 points or zero variance.
    """
    r = _flatten(returns)
    if r.size < 2:
        return np.nan
    sd = r.std(ddof=1)
    if sd < _EPS:
        return np.nan
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def softmin_sharpe(returns, periods_per_year: float, temperature: float = 1.0) -> float:
    """Annualised SoftMin Sharpe (Spooner & Savani robustness check).

    A Sharpe variant that downweights toward the *worst* returns via a softmin
    weighting w_i = softmax(-r_i / temperature), so episodic outlier losses
    dominate the ratio rather than the sustained mean. Targets episodic outliers
    rather than the mean-drift spoofing produces; retained for the sensitivity
    sweep, not as a primary metric. `temperature -> inf` recovers vanilla Sharpe;
    smaller temperature concentrates weight on the minimum return. Set the
    temperature to match the Spooner & Savani convention used in the writeup.
    """
    r = _flatten(returns)
    if r.size < 2:
        return np.nan
    # Numerically stable softmin weights over returns.
    z = -r / max(temperature, _EPS)
    w = np.exp(z - z.max())
    w /= w.sum()
    w_mean = float(np.sum(w * r))
    w_var = float(np.sum(w * (r - w_mean) ** 2))
    if w_var < _EPS:
        return np.nan
    return float(w_mean / np.sqrt(w_var) * np.sqrt(periods_per_year))


def sortino_ratio(returns, periods_per_year: float, target: float = 0.0) -> float:
    """Annualised Sortino = mean(returns - target) / downside_deviation * sqrt(ppy).

    Downside deviation uses the root-mean-square of the *negative* deviations from
    `target`, averaged over ALL observations (the standard MAR convention), not only
    the downside ones. Returns nan if there is no downside risk (no losses) or < 2 pts.
    """
    r = _flatten(returns)
    if r.size < 2:
        return np.nan
    excess = r - target
    downside = np.minimum(excess, 0.0)
    dd = np.sqrt(np.mean(downside ** 2))
    if dd < _EPS:
        return np.nan  # no downside risk -> Sortino undefined (not +inf, to avoid skewing aggregates)
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def cvar(returns, alpha: float = 0.10) -> float:
    """Conditional Value-at-Risk (expected shortfall) at level `alpha`.

    Mean of the worst `alpha` fraction of returns (the most negative tail). Returned
    as a signed return value: negative means an expected loss in the tail. With ~20
    seeds the proposal uses alpha=0.10 because the 5th-percentile effective sample is
    ~1 observation.
    """
    r = np.sort(_flatten(returns))
    if r.size == 0:
        return np.nan
    k = max(1, int(np.ceil(alpha * r.size)))
    return float(r[:k].mean())


def quote_displacement(quoted_price, fair_value) -> float:
    """Mean absolute deviation of the quoted price from fair value.

    `quoted_price` and `fair_value` are aligned per-step arrays. The proposal compares
    this under attack vs matched clean conditions; that comparison happens at the stats
    layer — this returns the raw MAD for one condition.
    """
    q = _flatten(quoted_price)
    f = _flatten(fair_value)
    if q.size == 0 or q.size != f.size:
        return np.nan
    return float(np.mean(np.abs(q - f)))


def peak_inventory_excursion(inventory) -> float:
    """Maximum absolute inventory reached over the window (directional-accumulation test)."""
    inv = _flatten(inventory)
    if inv.size == 0:
        return np.nan
    return float(np.max(np.abs(inv)))


def inventory_sd(inventory) -> float:
    """Sample standard deviation of the inventory path (progression-gate criterion).

    The Phase-1 gate bounds vanilla-IPPO inventory SD at a factor of the A-S
    bound on clean data; this is that statistic for one episode/window.
    """
    inv = _flatten(inventory)
    if inv.size < 2:
        return np.nan
    return float(inv.std(ddof=1))


def detection_auroc(probs, labels) -> float:
    """Area under the ROC curve for the detection head.

    Rank-based (Mann-Whitney U) computation with proper tie handling, so no sklearn
    dependency. `labels` are oracle 0/1; `probs` are predicted attack probabilities.
    Returns nan if only one class is present (AUROC undefined) — this is exactly the
    degenerate case an always-on or never-on adversary produces, so callers should
    check for nan rather than treat it as 0.5.
    """
    p = _flatten(probs)
    y = _flatten(labels)
    if p.size == 0 or p.size != y.size:
        return np.nan
    pos = y > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(p)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
