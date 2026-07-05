"""
Unit tests for the adversarial-eval metrics + statistics core.

Pytest-discoverable AND runnable directly (python test_eval_core.py).
Each metric/stat is checked against a hand-computed answer or a known invariant,
so the core is validated with no JAX, no env, and no trained checkpoints.
"""

import math
import numpy as np

from gymnax_exchange.jaxrl.MARL.adversarial_eval.metrics import (
    sharpe_ratio, softmin_sharpe, sortino_ratio, cvar, quote_displacement,
    peak_inventory_excursion, inventory_sd, detection_auroc,
)
from gymnax_exchange.jaxrl.MARL.adversarial_eval.stats import (
    cohens_d_paired, bootstrap_ci, paired_comparison,
    holm_adjust, tost_paired, one_sample_comparison, hodges_lehmann,
)


def _approx(a, b, tol=1e-6):
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(a - b) <= tol


# ---------------------------------------------------------------- metrics

def test_sharpe_known_value():
    # returns [-1, 2] -> mean 0.5, std(ddof=1)=sqrt(4.5), ppy=1
    assert _approx(sharpe_ratio([-1, 2], 1.0), 0.5 / math.sqrt(4.5))


def test_sharpe_sqrt_ppy_scaling():
    assert _approx(sharpe_ratio([-1, 2], 4.0), 2.0 * sharpe_ratio([-1, 2], 1.0))


def test_sharpe_degenerate():
    assert math.isnan(sharpe_ratio([3, 3, 3], 1.0))
    assert _approx(sharpe_ratio([-1, 1, -1, 1], 1.0), 0.0)


def test_softmin_sharpe_downweights_losses():
    # Softmin weighting concentrates on the worst returns, so with an outlier
    # loss present softmin Sharpe must be below vanilla Sharpe.
    r = [0.5, 0.4, 0.6, -5.0, 0.5, 0.3]
    assert softmin_sharpe(r, 1.0, temperature=0.5) < sharpe_ratio(r, 1.0)


def test_sortino_known_value():
    # returns [-1, 2], target 0 -> mean 0.5, downside=[-1,0], dd=sqrt(0.5)
    assert _approx(sortino_ratio([-1, 2], 1.0), 0.5 / math.sqrt(0.5))


def test_sortino_edge_cases():
    assert math.isnan(sortino_ratio([1, 2, 3], 1.0))          # no downside -> nan
    r = [-1, 2, 3, -0.5, 4]
    assert sortino_ratio(r, 1.0) >= sharpe_ratio(r, 1.0)      # upside vol not penalised


def test_cvar():
    assert _approx(cvar(np.arange(100), 0.10), 4.5)           # worst 10 of 0..99
    assert cvar(np.arange(100), 0.10) <= np.arange(100).mean()


def test_quote_displacement():
    assert _approx(quote_displacement([1, 2, 3], [1, 1, 1]), 1.0)


def test_inventory_stats():
    assert _approx(peak_inventory_excursion([-3, 2, 1]), 3.0)
    assert _approx(inventory_sd([1.0, 3.0]), math.sqrt(2.0))  # std(ddof=1) of [1,3]
    assert math.isnan(inventory_sd([1.0]))


def test_auroc():
    assert _approx(detection_auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
    assert _approx(detection_auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0)
    assert _approx(detection_auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]), 0.5)
    assert math.isnan(detection_auroc([0.2, 0.8], [1, 1]))    # one class -> nan


# ---------------------------------------------------------------- paired stats

def test_cohens_d_paired_known():
    # a-b = [1,2,3], mean 2, std(ddof=1)=1 -> d=2.0
    assert _approx(cohens_d_paired([2, 4, 6], [1, 2, 3]), 2.0)


def test_bootstrap_ci_brackets_mean():
    d = np.array([0.2, 0.5, -0.1, 0.3, 0.4, 0.1, 0.6, 0.0])
    lo, hi = bootstrap_ci(d, seed=0)
    assert lo <= d.mean() <= hi


def test_paired_comparison_normal_path():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 20)
    adv = base + rng.normal(0.0, 0.05, 20) + 0.5   # clear ~normal positive shift
    res = paired_comparison(adv, base)
    assert res.test == "paired_t" and res.ci_method == "t"
    assert res.p_value < 0.05
    assert res.cohens_d > 0
    assert res.ci_low <= res.mean_diff <= res.ci_high
    assert math.isnan(res.hl_estimate)             # HL only reported on the Wilcoxon path


def test_paired_comparison_nonnormal_path():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 20)
    shift = np.full(20, 0.3)
    shift[0] = 50.0                                 # outlier -> Shapiro rejects
    res = paired_comparison(base + shift, base)
    assert res.test == "wilcoxon" and res.ci_method == "bootstrap"
    # HL pseudo-median is robust to the single outlier: close to 0.3, far from mean_diff
    assert abs(res.hl_estimate - 0.3) < 0.2
    assert res.mean_diff > 2.0                      # the mean is dragged by the outlier


def test_paired_comparison_identical_and_errors():
    base = np.random.default_rng(1).normal(0, 1, 20)
    res = paired_comparison(base, base)
    assert _approx(res.mean_diff, 0.0) and _approx(res.p_value, 1.0)
    try:
        paired_comparison([1, 2, 3], [1, 2])
        assert False, "unequal lengths should raise"
    except ValueError:
        pass


# ---------------------------------------------------------------- new: Holm

def test_holm_adjust_known():
    # Classic worked example: p = [0.01, 0.04, 0.03], m=3
    # sorted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone max => 0.03, 0.06, 0.06
    adj = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert _approx(adj["a"], 0.03)
    assert _approx(adj["c"], 0.06)
    assert _approx(adj["b"], 0.06)


def test_holm_adjust_caps_at_one():
    adj = holm_adjust({"a": 0.9, "b": 0.8})
    assert adj["a"] <= 1.0 and adj["b"] <= 1.0


# ---------------------------------------------------------------- new: TOST

def test_tost_equivalent_when_diff_tiny():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 20)
    same = base + rng.normal(0, 0.02, 20)          # negligible difference
    t = tost_paired(same, base, margin=0.2)
    assert t.equivalent and t.p_value < 0.05


def test_tost_not_equivalent_when_diff_large():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 20)
    worse = base - 0.5                             # degradation larger than the margin
    t = tost_paired(worse + rng.normal(0, 0.05, 20), base, margin=0.2)
    assert not t.equivalent


def test_tost_absence_of_significance_is_not_equivalence():
    # Small n + noisy diffs: a paired t-test would NOT reject (p > .05), but TOST
    # must also fail to declare equivalence — the exact fallacy H2 must avoid.
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 6)
    noisy = base + rng.normal(0, 1.5, 6)
    t = tost_paired(noisy, base, margin=0.1)
    assert not t.equivalent


def test_tost_rejects_bad_margin():
    try:
        tost_paired([1, 2], [1, 2], margin=0.0)
        assert False, "margin=0 should raise"
    except ValueError:
        pass


# ---------------------------------------------------------------- new: one-sample

def test_one_sample_auroc_above_chance():
    rng = np.random.default_rng(4)
    aurocs = 0.75 + rng.normal(0, 0.03, 20)        # clearly above 0.5
    r = one_sample_comparison(aurocs, null_value=0.5)
    assert r.p_value < 0.01 and r.mean > 0.5


def test_one_sample_at_chance():
    rng = np.random.default_rng(5)
    aurocs = 0.5 + rng.normal(0, 0.05, 20)
    r = one_sample_comparison(aurocs, null_value=0.5)
    assert r.p_value > 0.05


def test_one_sample_drops_nans():
    x = [0.7, np.nan, 0.72, 0.68, np.nan, 0.71]
    r = one_sample_comparison(x, null_value=0.5)
    assert r.n == 4


# ---------------------------------------------------------------- new: Hodges-Lehmann

def test_hodges_lehmann_symmetric():
    assert _approx(hodges_lehmann([1.0, 2.0, 3.0]), 2.0)


def test_hodges_lehmann_robust_to_outlier():
    d = np.array([0.3] * 19 + [50.0])
    assert abs(hodges_lehmann(d) - 0.3) < 0.5      # pseudo-median ignores the outlier
    assert d.mean() > 2.0                          # ...unlike the mean


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                failed += 1
                print(f"  FAIL  {name}: {e}")
    sys.exit(1 if failed else 0)
