"""
Statistical comparison layer for the adversarial market-making study.

Implements the inference plan from the proposal §2 and §4:
  - Per-metric effect size via paired Cohen's d (standardised mean difference).
  - Normality screen with Shapiro-Wilk (alpha=0.05) on the paired differences.
  - Parametric path: paired t-test + t-based 95% CI when differences are normal.
  - Non-parametric fallback: Wilcoxon signed-rank + percentile-bootstrap 95% CI
    when normality is rejected (with the Hodges-Lehmann estimate reported as the
    location estimand matching the Wilcoxon test).
  - Multiplicity control: Holm step-down adjustment over a pre-registered family
    of primary tests (holm_adjust).
  - Equivalence testing: paired TOST for the "no degradation on clean data"
    hypothesis — absence of a significant difference is NOT evidence of
    equivalence, so H2-style claims must use tost_paired with a pre-specified
    margin, not a failed paired t-test.
  - One-sample tests vs a fixed null (one_sample_comparison), e.g. detection
    AUROC vs the chance level 0.5.

Inputs are per-seed metric arrays (one scalar metric per seed, ~20 seeds), with
configurations compared pairwise on matched seeds (e.g. baseline vs adversarial).
All functions are pure NumPy/SciPy so they unit-test against synthetic data.

Power note (paired t, two-sided, n=20): d_z=0.8 gives ~0.92 power at alpha=0.05
but only ~0.69 at a Holm/Bonferroni-corrected alpha of 0.05/8 — keep the
confirmatory family small (one primary endpoint per hypothesis) and treat the
remaining metrics as estimation-only.
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
    # Hodges-Lehmann pseudo-median of the differences — the location estimand the
    # Wilcoxon signed-rank test actually addresses (nan on the parametric path).
    hl_estimate: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TostResult:
    """Paired two-one-sided-tests (TOST) equivalence result on mean(a - b)."""
    n: int
    mean_diff: float
    margin: float             # symmetric equivalence margin (+/- margin)
    p_lower: float            # H0: diff <= -margin
    p_upper: float            # H0: diff >= +margin
    p_value: float            # max(p_lower, p_upper) — the TOST p-value
    equivalent: bool          # p_value < alpha
    alpha: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class OneSampleResult:
    """One-sample comparison of a per-seed metric against a fixed null value."""
    n: int
    mean: float
    null_value: float
    cohens_d: float           # (mean - null) / sd
    normal: bool
    test: str                 # "t" or "wilcoxon"
    statistic: float
    p_value: float

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

    hl = float("nan")
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
        hl = hodges_lehmann(d)

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
        hl_estimate=hl,
    )


def hodges_lehmann(diffs) -> float:
    """Hodges-Lehmann pseudo-median: median of all Walsh averages (d_i + d_j)/2, i<=j.

    The location estimand matching the Wilcoxon signed-rank test. O(n^2) memory,
    fine for the ~20-seed arrays used here.
    """
    d = np.asarray(diffs, dtype=np.float64).ravel()
    if d.size == 0:
        return float("nan")
    i, j = np.triu_indices(d.size)
    return float(np.median((d[i] + d[j]) / 2.0))


def holm_adjust(pvals: dict) -> dict:
    """Holm step-down adjusted p-values for a family of tests.

    `pvals` maps test name -> raw p-value. Returns the same keys with adjusted
    p-values (monotone, capped at 1). Apply over the PRE-REGISTERED confirmatory
    family only; exploratory metrics should be reported as estimates + CIs, not
    run through this and called significant.
    """
    names = [k for k in pvals]
    p = np.asarray([pvals[k] for k in names], dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(1.0, running_max)
    return {names[i]: float(adjusted[i]) for i in range(m)}


def tost_paired(a, b, margin: float, alpha: float = 0.05) -> TostResult:
    """Paired equivalence test (TOST) for H2-style no-degradation claims.

    Tests whether mean(a - b) lies within (-margin, +margin) via two one-sided
    paired t-tests. Equivalence is declared when BOTH one-sided nulls are
    rejected, i.e. max(p_lower, p_upper) < alpha. The margin is a scientific
    input (an economically negligible degradation), not a statistical one —
    pre-specify it and justify it in the writeup.
    """
    if margin <= 0:
        raise ValueError(f"margin must be positive, got {margin}")
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"a and b must have equal length, got {a.size} and {b.size}")
    d = a - b
    n = d.size
    if n < 2 or d.std(ddof=1) < _EPS:
        return TostResult(n=n, mean_diff=float(d.mean()) if n else float("nan"),
                          margin=margin, p_lower=float("nan"), p_upper=float("nan"),
                          p_value=float("nan"), equivalent=False, alpha=alpha)
    se = d.std(ddof=1) / np.sqrt(n)
    df = n - 1
    t_lower = (d.mean() + margin) / se     # H0: diff <= -margin, H1: diff > -margin
    t_upper = (d.mean() - margin) / se     # H0: diff >= +margin, H1: diff < +margin
    p_lower = float(1.0 - stats.t.cdf(t_lower, df))
    p_upper = float(stats.t.cdf(t_upper, df))
    p_value = max(p_lower, p_upper)
    return TostResult(n=n, mean_diff=float(d.mean()), margin=margin,
                      p_lower=p_lower, p_upper=p_upper, p_value=p_value,
                      equivalent=bool(p_value < alpha), alpha=alpha)


def one_sample_comparison(x, null_value: float, alpha: float = 0.05) -> OneSampleResult:
    """Test a per-seed metric against a fixed null (e.g. detection AUROC vs 0.5).

    Same Shapiro-Wilk gate as paired_comparison, applied to (x - null): t-test
    when normality is not rejected, Wilcoxon signed-rank otherwise. NaN entries
    (e.g. AUROC undefined for a degenerate rollout) are dropped with n reduced.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    d = x - null_value
    sd = d.std(ddof=1) if n >= 2 else float("nan")
    cd = float(d.mean() / sd) if n >= 2 and sd > _EPS else float("nan")

    if n >= 3 and sd > _EPS:
        _, sw_p = stats.shapiro(d)
        normal = bool(sw_p >= alpha)
    else:
        normal = False

    if n < 2:
        return OneSampleResult(n=n, mean=float(x.mean()) if n else float("nan"),
                               null_value=null_value, cohens_d=cd, normal=False,
                               test="none", statistic=float("nan"), p_value=float("nan"))
    if normal:
        res = stats.ttest_1samp(x, null_value)
        return OneSampleResult(n=n, mean=float(x.mean()), null_value=null_value,
                               cohens_d=cd, normal=True, test="t",
                               statistic=float(res.statistic), p_value=float(res.pvalue))
    if np.allclose(d, 0.0):
        return OneSampleResult(n=n, mean=float(x.mean()), null_value=null_value,
                               cohens_d=cd, normal=False, test="wilcoxon",
                               statistic=float("nan"), p_value=1.0)
    res = stats.wilcoxon(d)
    return OneSampleResult(n=n, mean=float(x.mean()), null_value=null_value,
                           cohens_d=cd, normal=False, test="wilcoxon",
                           statistic=float(res.statistic), p_value=float(res.pvalue))
