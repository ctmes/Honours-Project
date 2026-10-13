"""analysis.dose_response: the decision rule of dose_response_amendment_2026-10-09,
checked on synthetic ladders where the right answer is known."""

import json
import math

import numpy as np
import pytest

from analysis.dose_response import run, verdict

PPY = 3291.5 ** 2          # per-step = annualised / 3291.5
SEEDS = list(range(20))


def _file(tmp_path, mult, shift_per_step, noise=0.001, seed=0, budget_exhausted=0.0,
          side="bid"):
    rng = np.random.default_rng(seed)
    k = math.sqrt(PPY)
    off = rng.normal(-0.008, 0.004, 20) * k
    forced = off + (shift_per_step + rng.normal(0, noise, 20)) * k
    ps = {"sharpe_off": off, "sharpe_forced": forced,
          "sortino_off": off, "sortino_forced": forced,
          "cvar_off": np.full(20, -1.0), "cvar_forced": np.full(20, -1.0),
          "inventory_sd_off": np.ones(20), "inventory_sd_forced": np.ones(20),
          "quote_displacement_off": np.zeros(20), "quote_displacement_forced": np.zeros(20),
          "qi_absmean_off": np.full(20, 0.5), "qi_absmean_forced": np.full(20, 0.9),
          "mean_bid_volume_injected_forced": np.full(20, 670.0 * mult),
          "budget_final_frac_forced": np.full(20, 0.99),
          "budget_exhausted_frac_forced": np.full(20, budget_exhausted)}
    d = {"per_seed": {"baseline": {k2: v.tolist() for k2, v in ps.items()}},
         "_meta": {"inject_mult": mult, "side": side, "seeds": SEEDS,
                   "periods_per_year": PPY}}
    p = tmp_path / f"dose_response_m{mult:g}.json"
    p.write_text(json.dumps(d))
    return str(p)


MULTS = (2, 10, 50, 200, 1000)


def test_flat_ladder_is_not_harmed(tmp_path):
    paths = [_file(tmp_path, m, 0.0, seed=i) for i, m in enumerate(MULTS)]
    res = run(paths, None)
    assert res["verdict"].startswith("NOT HARMED")


def test_harm_from_a_high_rung_is_found_and_located(tmp_path):
    shifts = {2: 0.0, 10: 0.0, 50: 0.0, 200: -0.02, 1000: -0.05}
    paths = [_file(tmp_path, m, shifts[m], seed=i) for i, m in enumerate(MULTS)]
    res = run(paths, None)
    assert res["verdict"] == "HARMED from inject_mult 200"
    assert res["trend"]["mean_slope_per_decade"] < 0


def test_a_gain_is_never_called_harm(tmp_path):
    paths = [_file(tmp_path, m, +0.03, seed=i) for i, m in enumerate(MULTS)]
    assert not run(paths, None)["verdict"].startswith("HARMED")


def test_wide_noisy_null_is_inconclusive():
    rungs = [{"inject_mult": 2.0, "sharpe": {"holm_p": 0.4, "mean_diff": -0.001,
                                               "bca_low": -0.02}}]
    assert verdict(rungs) == "INCONCLUSIVE"


def test_a_binding_budget_voids_the_ladder(tmp_path):
    paths = [_file(tmp_path, m, 0.0, seed=i, budget_exhausted=0.3 if m == 1000 else 0.0)
             for i, m in enumerate(MULTS)]
    assert run(paths, None)["verdict"].startswith("VOID")


def test_duplicate_rungs_are_refused(tmp_path):
    a = _file(tmp_path, 10, 0.0)
    b = tmp_path / "copy.json"
    b.write_text(open(a).read())
    with pytest.raises(SystemExit):
        run([a, str(b)], None)


def test_ask_side_file_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        run([_file(tmp_path, 10, 0.0, side="ask")], None)


def test_rung_two_must_reproduce_the_existing_forced_run(tmp_path):
    p2 = _file(tmp_path, 2, 0.0)
    d = json.loads(open(p2).read())
    ref = {"per_seed": {"baseline": {"sharpe_forced": d["per_seed"]["baseline"]["sharpe_forced"],
                                      "sharpe_off": d["per_seed"]["baseline"]["sharpe_off"]}},
           "_meta": {"seeds": SEEDS}}
    rp = tmp_path / "ref.json"
    rp.write_text(json.dumps(ref))
    assert run([p2], str(rp))["reproduction_m2"] == "MATCH"
    ref["per_seed"]["baseline"]["sharpe_forced"][3] += 1.0
    rp.write_text(json.dumps(ref))
    assert run([p2], str(rp))["reproduction_m2"].startswith("DIFFERS")
