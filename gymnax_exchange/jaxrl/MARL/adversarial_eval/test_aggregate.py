"""
Unit tests for the aggregation + reporting layer.

Pytest-discoverable AND runnable directly (python test_aggregate.py).
"""

import numpy as np

from gymnax_exchange.jaxrl.MARL.adversarial_eval.aggregate import (
    summarize_seeds, compare_configs, progression_gate, format_comparison_table,
)


def _synth_configs(seeds=20, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, seeds)
    # Per-seed noise on each metric so the paired differences vary (a constant
    # difference has undefined Cohen's d and would route to the non-parametric path).
    return {
        "adversarial": {"sharpe": base + 0.6 + rng.normal(0, 0.1, seeds),
                        "cvar": base - 0.3 + rng.normal(0, 0.1, seeds)},
        "baseline":    {"sharpe": base + rng.normal(0, 0.1, seeds),
                        "cvar": base + rng.normal(0, 0.1, seeds)},
    }


# ---------------------------------------------------------------- summarize

def test_summarize_seeds():
    s = summarize_seeds({"sharpe": [1.0, 2.0, 3.0], "x": [np.nan, 4.0, 6.0]})
    assert abs(s["sharpe"]["mean"] - 2.0) < 1e-9
    assert abs(s["sharpe"]["median"] - 2.0) < 1e-9
    assert s["sharpe"]["n"] == 3
    assert abs(s["x"]["mean"] - 5.0) < 1e-9      # NaN ignored in mean
    assert s["x"]["n"] == 2                       # NaN ignored in n


# ---------------------------------------------------------------- compare_configs

def test_compare_configs():
    metrics_by_config = _synth_configs()
    res = compare_configs(metrics_by_config, "adversarial", "baseline", ["sharpe", "cvar"])
    assert abs(res["sharpe"].mean_diff - 0.6) < 0.1
    assert res["sharpe"].p_value < 0.05
    assert res["sharpe"].cohens_d > 0
    assert abs(res["cvar"].mean_diff + 0.3) < 0.1
    assert res["sharpe"].n == 20                  # paired alignment preserved


# ---------------------------------------------------------------- progression gate

_AS = {"sharpe": [2.0] * 10, "sortino": [3.0] * 10, "inventory_sd": [5.0] * 10}


def test_gate_passes_when_close():
    ippo = {"sharpe": [1.9] * 10, "sortino": [2.9] * 10, "inventory_sd": [6.0] * 10}
    g = progression_gate(ippo, _AS, sharpe_margin=0.2, sortino_margin=0.2)
    assert g.passed


def test_gate_fails_on_low_sharpe():
    ippo = {"sharpe": [1.0] * 10, "sortino": [2.9] * 10, "inventory_sd": [6.0] * 10}
    g = progression_gate(ippo, _AS, sharpe_margin=0.2, sortino_margin=0.2)
    assert (not g.passed) and (not g.sharpe_ok) and g.sortino_ok


def test_gate_fails_on_inventory_sd():
    ippo = {"sharpe": [1.9] * 10, "sortino": [2.9] * 10, "inventory_sd": [11.0] * 10}
    g = progression_gate(ippo, _AS, sharpe_margin=0.2, sortino_margin=0.2, inv_sd_factor=2.0)
    assert (not g.passed) and (not g.inventory_ok)


def test_gate_condition_suffixed_keys():
    # The eval pipeline emits condition-suffixed metric names (sharpe_off, ...);
    # the gate must be wireable to them via its key kwargs.
    as_m = {"sharpe_off": [2.0] * 10, "sortino_off": [3.0] * 10, "inventory_sd_off": [5.0] * 10}
    ippo = {"sharpe_off": [1.9] * 10, "sortino_off": [2.9] * 10, "inventory_sd_off": [6.0] * 10}
    g = progression_gate(ippo, as_m, sharpe_margin=0.2, sortino_margin=0.2,
                         sharpe_key="sharpe_off", sortino_key="sortino_off",
                         inv_sd_key="inventory_sd_off")
    assert g.passed


# ---------------------------------------------------------------- table

def test_format_table():
    res = compare_configs(_synth_configs(), "adversarial", "baseline", ["sharpe", "cvar"])
    table = format_comparison_table(res, title="adversarial vs baseline")
    assert "sharpe" in table and "cvar" in table
    assert "adversarial vs baseline" in table


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
