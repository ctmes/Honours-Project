"""
Unit tests for the adversarial-eval metrics + statistics core.

Pure-Python, runnable directly:  python test_eval_core.py
Each metric/stat is checked against a hand-computed answer or a known invariant,
so the core is validated with no JAX, no env, and no trained checkpoints.
"""

import math
import numpy as np

from gymnax_exchange.jaxrl.MARL.adversarial_eval.metrics import (
    sharpe_ratio, sortino_ratio, cvar, quote_displacement,
    peak_inventory_excursion, detection_auroc,
)
from gymnax_exchange.jaxrl.MARL.adversarial_eval.stats import (
    cohens_d_paired, bootstrap_ci, paired_comparison,
)

TOL = 1e-9
_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


def approx(a, b, tol=1e-6):
    if a is None or b is None:
        return False
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(a - b) <= tol


# ---------------------------------------------------------------- metrics
print("metrics:")

# Sharpe: returns [-1, 2] -> mean 0.5, std(ddof=1)=sqrt(4.5)=2.12132, ppy=1
check("sharpe known value", approx(sharpe_ratio([-1, 2], 1.0), 0.5 / math.sqrt(4.5)))
# Sharpe scales with sqrt(ppy)
check("sharpe sqrt(ppy) scaling",
      approx(sharpe_ratio([-1, 2], 4.0), 2.0 * sharpe_ratio([-1, 2], 1.0)))
# Zero-variance -> nan
check("sharpe constant -> nan", math.isnan(sharpe_ratio([3, 3, 3], 1.0)))
# Zero-mean symmetric -> 0
check("sharpe zero-mean -> 0", approx(sharpe_ratio([-1, 1, -1, 1], 1.0), 0.0))

# Sortino: returns [-1, 2], target 0 -> mean 0.5, downside=[-1,0], dd=sqrt(0.5)
check("sortino known value",
      approx(sortino_ratio([-1, 2], 1.0), 0.5 / math.sqrt(0.5)))
# No downside -> nan
check("sortino no-loss -> nan", math.isnan(sortino_ratio([1, 2, 3], 1.0)))
# Sortino >= Sharpe when there is upside volatility (downside dev <= total std)
r = [-1, 2, 3, -0.5, 4]
check("sortino >= sharpe with upside vol",
      sortino_ratio(r, 1.0) >= sharpe_ratio(r, 1.0))

# CVaR: 0..99, alpha 0.10 -> worst 10 = 0..9 -> mean 4.5
check("cvar worst-10pct mean", approx(cvar(np.arange(100), 0.10), 4.5))
# CVaR <= mean (tail no better than average)
check("cvar <= mean", cvar(np.arange(100), 0.10) <= np.arange(100).mean())

# Quote displacement: |q-f| mean
check("quote displacement MAD",
      approx(quote_displacement([1, 2, 3], [1, 1, 1]), 1.0))

# Peak inventory excursion: max |inv|
check("peak inventory excursion", approx(peak_inventory_excursion([-3, 2, 1]), 3.0))

# AUROC
check("auroc perfectly separable -> 1.0",
      approx(detection_auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0))
check("auroc reversed -> 0.0",
      approx(detection_auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0))
check("auroc all-ties -> 0.5",
      approx(detection_auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]), 0.5))
check("auroc one-class -> nan", math.isnan(detection_auroc([0.2, 0.8], [1, 1])))


# ---------------------------------------------------------------- statistics
print("statistics:")

# Cohen's d: a-b = [1,2,3], mean 2, std(ddof=1)=1 -> d=2.0
check("cohens d paired known",
      approx(cohens_d_paired([2, 4, 6], [1, 2, 3]), 2.0))

# Bootstrap CI brackets the sample mean
d = np.array([0.2, 0.5, -0.1, 0.3, 0.4, 0.1, 0.6, 0.0])
lo, hi = bootstrap_ci(d, seed=0)
check("bootstrap CI brackets mean", lo <= d.mean() <= hi)

# Paired comparison, normal differences -> paired t-test path.
rng = np.random.default_rng(1)
base = rng.normal(0, 1, 20)
adv = base + rng.normal(0.0, 0.05, 20) + 0.5   # a clear, ~normal positive shift
res = paired_comparison(adv, base)
check("normal diffs -> paired_t", res.test == "paired_t" and res.ci_method == "t")
check("paired_t detects effect (p<0.05)", res.p_value < 0.05)
check("cohens d positive for positive shift", res.cohens_d > 0)
check("CI brackets mean_diff (t)", res.ci_low <= res.mean_diff <= res.ci_high)

# Heavy-tailed / skewed differences -> Wilcoxon + bootstrap path.
skewed = base.copy()
shift = np.full(20, 0.3)
shift[0] = 50.0   # extreme outlier => Shapiro should reject normality of diffs
res2 = paired_comparison(base + shift, base)
check("non-normal diffs -> wilcoxon", res2.test == "wilcoxon" and res2.ci_method == "bootstrap")

# Identical configs -> zero effect, graceful handling
res3 = paired_comparison(base, base)
check("identical -> mean_diff 0", approx(res3.mean_diff, 0.0))
check("identical -> p_value 1.0", approx(res3.p_value, 1.0))

# Unequal lengths raise
try:
    paired_comparison([1, 2, 3], [1, 2])
    check("unequal lengths raise", False)
except ValueError:
    check("unequal lengths raise", True)


# ---------------------------------------------------------------- summary
print(f"\n{_passed} passed, {_failed} failed")
raise SystemExit(1 if _failed else 0)
