"""analysis/gate_arms_check.py: the gate and H2 applied to every arm (post-hoc).

Pins the three things that could silently corrupt the extension:
  * the gate criteria come from preregistration.json and are applied by the SAME
    aggregate.progression_gate the eval harness uses, so they cannot drift;
  * the published baseline-only gate verdict is reproduced, and the baseline row of
    the per-arm table equals it;
  * the H2 rows for (full, adversarial) reproduce analysis/h2_noninferiority.py, the
    pre-registered script, so the extension is the same statistic on more arms;
  * extra arms folded in from an eval_forced_attack.py file are refused when the run
    disagrees on seeds / checkpoint step / periods_per_year.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from analysis import gate_arms_check as G

V4 = "results/eval_17313.json"
V3 = "results/eval_1179095.json"
V4S = "results/forced_attack_v4s_bid.json"


def _have(*paths) -> bool:
    import os
    return all(os.path.exists(p) for p in paths)


pytestmark = pytest.mark.skipif(not _have(V4, V3), reason="eval JSONs not present")


def test_gate_criteria_come_from_the_preregistration():
    crit = G.gate_criteria()
    with open("preregistration.json") as f:
        g = json.load(f)["progression_gate"]
    assert crit == {"sharpe_margin": float(g["sharpe_margin"]),
                    "sortino_margin": float(g["sortino_margin"]),
                    "inv_sd_factor": float(g["inv_sd_factor"])}


def test_gate_criteria_fall_back_without_a_preregistration(tmp_path):
    crit = G.gate_criteria(str(tmp_path / "nope.json"))
    assert crit == {"sharpe_margin": 0.5, "sortino_margin": 0.5, "inv_sd_factor": 2.0}


@pytest.mark.parametrize("path", [V3, V4])
def test_baseline_row_reproduces_the_published_gate(path):
    res = G.run(path, [])
    of_record = res["gate_of_record_baseline_only"]
    mine = res["gate_per_arm"]["baseline"]
    assert of_record is not None
    for k in ("passed", "sharpe_ok", "sortino_ok", "inventory_ok"):
        assert bool(mine[k]) == bool(of_record[k]), k
    for metric in ("sharpe", "sortino", "inventory_sd"):
        assert mine["detail"][metric]["ippo"] == pytest.approx(
            of_record["detail"][metric]["ippo"], rel=1e-12)
        assert mine["detail"][metric]["as"] == pytest.approx(
            of_record["detail"][metric]["as"], rel=1e-12)


def test_v4_head_arms_pass_and_no_head_arms_fail():
    """The substantive claim: the gate failure is specific to the arms without the
    auxiliary loss. If this flips, the write-up claim must change with it."""
    g = G.run(V4, [])["gate_per_arm"]
    for arm in ("detection", "full", "detection_noobs"):
        assert g[arm]["passed"] is True, f"{arm} no longer passes the gate"
    for arm in ("baseline", "adversarial", "regime", "unconstrained"):
        assert g[arm]["passed"] is False, f"{arm} now passes the gate"
        assert g[arm]["inventory_ok"] is True  # it fails on return, not on inventory


def test_v3_has_no_competent_arm():
    g = G.run(V3, [])["gate_per_arm"]
    assert all(not v["passed"] for v in g.values())


def test_h2_rows_reproduce_the_preregistered_script():
    from analysis.evalreport import EvalReport
    from analysis.h2_noninferiority import noninferiority
    report = EvalReport.load(V3)
    margins = G.run(V3, [])["h2_margins"]
    mine = G.run(V3, [])["h2_per_arm"]
    for arm in ("full", "adversarial"):
        for metric in ("sortino_off", "sharpe_off"):
            ref = noninferiority(report, arm, metric, float(margins[metric]))
            assert mine[arm]["metrics"][metric]["p_noninferiority"] == pytest.approx(
                ref["p_lower_noninferiority"], rel=1e-12)
            assert mine[arm]["metrics"][metric]["mean_diff"] == pytest.approx(
                ref["mean_diff"], rel=1e-9)
            assert mine[arm]["metrics"][metric]["n"] == ref["n"]


def test_vs_benchmark_is_scale_invariant_in_p_and_d():
    """Per-step reporting must not change any verdict (RC2)."""
    res = G.run(V4, [])
    for key, c in res["vs_benchmark"].items():
        ppy = 10833681.0
        assert c["mean_diff_per_step"] == pytest.approx(
            c["mean_diff"] / np.sqrt(ppy), rel=1e-9), key
        assert np.sign(c["mean_diff_per_step"]) == np.sign(c["mean_diff"]), key


def test_benchmark_arm_is_seed_varying_so_pairing_is_meaningful():
    d = G._load(V4)
    for metric in ("sharpe_off", "sortino_off"):
        x = G._ps(d, "as", metric)
        assert np.nanstd(x) > 0, f"A-S {metric} is constant; pairing would be vacuous"


@pytest.mark.skipif(not _have(V4S), reason="v4s forced-attack json not present")
def test_extra_arm_is_folded_in_with_clean_metrics_only():
    res = G.run(V4, [V4S])
    assert "scripted_detection" in res["gate_per_arm"]
    d = G.merge_extra(G._load(V4), [G._load(V4S)])
    folded = d["per_seed"]["scripted_detection"]
    assert all(k.endswith("_off") for k in folded), "forced-condition keys leaked in"
    assert "sharpe_forced" not in folded


def test_extra_arm_refused_on_mismatched_run_metadata():
    main = G._load(V4)
    bad = {"per_seed": {"ghost": {m: [0.0] * 20 for m in G.GATE_METRICS}},
           "_meta": {"periods_per_year": 1.0, "seeds": list(range(20)),
                     "checkpoint_step": 1002}}
    with pytest.raises(SystemExit, match="periods_per_year"):
        G.merge_extra(main, [bad])


def test_existing_arms_are_never_overwritten_by_an_extra_file():
    main = G._load(V4)
    ppy = main["_meta"]["periods_per_year"]
    shadow = {"per_seed": {"detection": {m: [999.0] * 20 for m in G.GATE_METRICS}},
              "_meta": {"periods_per_year": ppy}}
    d = G.merge_extra(main, [shadow])
    assert G._ps(d, "detection", "sharpe_off").tolist() == \
        G._ps(main, "detection", "sharpe_off").tolist()


def test_report_is_marked_exploratory():
    res = G.run(V4, [])
    assert res["confirmatory"] is False
    assert "POST-HOC" in res["status"]
    assert "POST-HOC" in G.format_text(res)
