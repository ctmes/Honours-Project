"""
Statistical comparison layer for the adversarial market-making study.

Implements the inference plan from the proposal §2 and §4:
  - Per-metric effect size via paired Cohen's d (standardised mean difference).
  - Normality screen with Shapiro-Wilk (alpha=0.05) on the paired differences.
  - Parametric path: paired t-test + t-based 95% CI when differences are normal.
  - Non-parametric fallback: Wilcoxon signed-rank + percentile-bootstrap 95% CI
    when normality is rejected.

Inputs are per-seed metric arrays (one scalar metric per seed, ~20 seeds), with
configurations compared pairwise on matched seeds (e.g. baseline vs adversarial).
All functions are pure NumPy/SciPy so they unit-test against synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from scipy import stats

_EPS = 1e-12


@dataclass
class ComparisonResult:
    n: int                    # number of paired seeds
    mean_a: float
    mean_b: float
    mean_diff: float          # mean(a - b)
    cohens_d: float           # paired Cohen's d
    normal: bool              # Shapiro-Wilk did NOT reject normality of differences
    test: str                 # "paired_t" or "wilcoxon"
    statistic: float
    p_value: float
    ci_low: float             # 95% CI on mean(a - b)
    ci_high: float
    ci_method: str            # "t" or "bootstrap"

    def as_dict(self) -> dict:
        return asdict(self)


def cohens_d_paired(a, b) -> float:
    """Paired Cohen's d = mean(a - b) / std(a - b, ddof=1).

    Returns nan if < 2 pairs or zero difference-variance.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = a - b
    if d.size < 2:
        return np.nan
    sd = d.std(ddof=1)
    if sd < _EPS:
        return np.nan
    return float(d.mean() / sd)


def bootstrap_ci(diffs, ci: float = 0.95, n_boot: int = 10000,
                 seed: Optional[int] = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `diffs` (paired differences)."""
    d = np.asarray(diffs, dtype=np.float64).ravel()
    if d.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boot_means = d[idx].mean(axis=1)
    lo = (1.0 - ci) / 2.0 * 100.0
    hi = (1.0 + ci) / 2.0 * 100.0
    return (float(np.percentile(boot_means, lo)), float(np.percentile(boot_means, hi)))


def _t_ci(diffs, ci: float = 0.95) -> tuple[float, float]:
    d = np.asarray(diffs, dtype=np.float64).ravel()
    n = d.size
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf((1.0 + ci) / 2.0, df=n - 1)
    return (float(mean - tcrit * se), float(mean + tcrit * se))


def paired_comparison(a, b, alpha: float = 0.05, ci: float = 0.95,
                      boot_seed: Optional[int] = 0) -> ComparisonResult:
    """Compare two configurations on matched seeds.

    `a`, `b` are per-seed metric arrays of equal length. Procedure:
      1. Shapiro-Wilk on (a - b); normal if p >= alpha.
      2. If normal: paired t-test (ttest_rel) + t-based CI.
         Else:      Wilcoxon signed-rank + bootstrap CI.
    Effect size is paired Cohen's d in both cases.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"a and b must have equal length, got {a.size} and {b.size}")
    n = a.size
    d = a - b

    # Normality screen on the differences. Shapiro needs >= 3 points; if fewer, or if
    # the differences are constant, default to the non-parametric path.
    if n >= 3 and d.std(ddof=1) > _EPS:
        _, sw_p = stats.shapiro(d)
        normal = bool(sw_p >= alpha)
    else:
        normal = False

    if normal:
        res = stats.ttest_rel(a, b)
        statistic, p_value = float(res.statistic), float(res.pvalue)
        test = "paired_t"
        ci_low, ci_high = _t_ci(d, ci)
        ci_method = "t"
    else:
        # Wilcoxon is undefined if all differences are zero; guard it.
        if np.allclose(d, 0.0):
            statistic, p_value = np.nan, 1.0
        else:
            res = stats.wilcoxon(a, b)
            statistic, p_value = float(res.statistic), float(res.pvalue)
        test = "wilcoxon"
        ci_low, ci_high = bootstrap_ci(d, ci, seed=boot_seed)
        ci_method = "bootstrap"

    return ComparisonResult(
        n=n,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=float(d.mean()),
        cohens_d=cohens_d_paired(a, b),
        normal=normal,
        test=test,
        statistic=statistic,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_method=ci_method,
    )
