"""
Unit tests for the aggregation + reporting layer.
Runnable directly:  python test_aggregate.py
"""

import math
import numpy as np

from gymnax_exchange.jaxrl.MARL.adversarial_eval.aggregate import (
    summarize_seeds, compare_configs, progression_gate, format_comparison_table,
)

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


# ---------------------------------------------------------------- summarize
print("summarize_seeds:")
s = summarize_seeds({"sharpe": [1.0, 2.0, 3.0], "x": [np.nan, 4.0, 6.0]})
check("mean", abs(s["sharpe"]["mean"] - 2.0) < 1e-9)
check("median", abs(s["sharpe"]["median"] - 2.0) < 1e-9)
check("n", s["sharpe"]["n"] == 3)
check("ignores NaN in mean", abs(s["x"]["mean"] - 5.0) < 1e-9)
check("ignores NaN in n", s["x"]["n"] == 2)


# ---------------------------------------------------------------- compare_configs
print("compare_configs:")
rng = np.random.default_rng(0)
seeds = 20
base = rng.normal(0, 1, seeds)
# Per-seed noise on each metric so the paired differences vary (a constant difference
# has undefined Cohen's d and would route to the non-parametric path — see metrics core).
metrics_by_config = {
    "adversarial": {"sharpe": base + 0.6 + rng.normal(0, 0.1, seeds),
                    "cvar": base - 0.3 + rng.normal(0, 0.1, seeds)},
    "baseline":    {"sharpe": base + rng.normal(0, 0.1, seeds),
                    "cvar": base + rng.normal(0, 0.1, seeds)},
}
res = compare_configs(metrics_by_config, "adversarial", "baseline", ["sharpe", "cvar"])
check("sharpe diff ~ +0.6", abs(res["sharpe"].mean_diff - 0.6) < 0.1)
check("sharpe significant", res["sharpe"].p_value < 0.05)
check("sharpe d positive", res["sharpe"].cohens_d > 0)
check("cvar diff ~ -0.3", abs(res["cvar"].mean_diff + 0.3) < 0.1)
check("paired alignment preserved (n=20)", res["sharpe"].n == 20)


# ---------------------------------------------------------------- progression gate
print("progression_gate:")
as_metrics = {"sharpe": [2.0] * 10, "sortino": [3.0] * 10, "inventory_sd": [5.0] * 10}

ippo_good = {"sharpe": [1.9] * 10, "sortino": [2.9] * 10, "inventory_sd": [6.0] * 10}
g = progression_gate(ippo_good, as_metrics, sharpe_margin=0.2, sortino_margin=0.2)
check("gate passes when close to A-S", g.passed)

ippo_bad_sharpe = {"sharpe": [1.0] * 10, "sortino": [2.9] * 10, "inventory_sd": [6.0] * 10}
g2 = progression_gate(ippo_bad_sharpe, as_metrics, sharpe_margin=0.2, sortino_margin=0.2)
check("gate fails on low sharpe", (not g2.passed) and (not g2.sharpe_ok) and g2.sortino_ok)

ippo_bad_inv = {"sharpe": [1.9] * 10, "sortino": [2.9] * 10, "inventory_sd": [11.0] * 10}
g3 = progression_gate(ippo_bad_inv, as_metrics, sharpe_margin=0.2, sortino_margin=0.2,
                      inv_sd_factor=2.0)
check("gate fails on inventory SD > 2x", (not g3.passed) and (not g3.inventory_ok))


# ---------------------------------------------------------------- table
print("format:")
table = format_comparison_table(res, title="adversarial vs baseline")
check("table contains metric names", "sharpe" in table and "cvar" in table)
check("table contains title", "adversarial vs baseline" in table)
print("\n" + table + "\n")


print(f"{_passed} passed, {_failed} failed")
raise SystemExit(1 if _failed else 0)
