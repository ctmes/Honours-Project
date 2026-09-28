"""
Pins the robustness checks added by preregistration.json -> amendments ->
robustness_checks_amendment_2026-09-28 (RC1-RC4), without data or checkpoints.

RC1  exact sign-flip permutation test + BCa CI (stats.py)
RC4  encoder effective rank / srank / dead units (metrics.py), the encoder
     feature extraction and the pre-registered minimal-pair decision rule
     (check_encoder_rank.py)
"""

import numpy as np
import pytest
from scipy import stats as S

from gymnax_exchange.jaxrl.MARL.adversarial_eval.metrics import (
    dead_unit_fraction, effective_rank, srank,
)
from gymnax_exchange.jaxrl.MARL.adversarial_eval.stats import (
    bca_ci, sign_flip_permutation,
)


# ------------------------------------------------------------------ RC1

def test_sign_flip_exact_floor_values():
    assert sign_flip_permutation([1, 2, 3, 4, 5]) == pytest.approx(2 / 32)
    assert sign_flip_permutation(np.arange(1, 21)) == pytest.approx(2 / 2 ** 20)
    assert sign_flip_permutation(np.arange(1, 21), alternative="greater") == pytest.approx(1 / 2 ** 20)


@pytest.mark.parametrize("n", [5, 9, 14])
@pytest.mark.parametrize("alt", ["two-sided", "greater", "less"])
def test_sign_flip_matches_scipy_exact(n, alt):
    d = np.random.default_rng(n).normal(0.4, 1.0, n)
    ref = S.permutation_test((d,), np.mean, permutation_type="samples",
                             n_resamples=np.inf, alternative=alt).pvalue
    assert sign_flip_permutation(d, alternative=alt) == pytest.approx(ref, abs=1e-12)


def test_sign_flip_symmetric_and_degenerate():
    assert sign_flip_permutation([-3.0, -1.0, 1.0, 3.0]) == pytest.approx(1.0)
    assert sign_flip_permutation([0.0, 0.0, 0.0]) == 1.0
    assert sign_flip_permutation([np.nan, 1.0, 2.0, 3.0]) == pytest.approx(2 / 8)


def test_sign_flip_monte_carlo_above_exact_limit():
    d = np.random.default_rng(0).normal(0.0, 1.0, 30)
    p = sign_flip_permutation(d, n_mc=20000)
    ref = S.permutation_test((d,), np.mean, permutation_type="samples",
                             n_resamples=20000, random_state=1).pvalue
    assert p == pytest.approx(ref, abs=0.02)


def test_sign_flip_noninferiority_shift():
    # H0: diff <= -m  <=>  test d + m with alternative="greater"
    d = np.full(20, 10.0) + np.random.default_rng(2).normal(0, 1, 20)
    assert sign_flip_permutation(d + 0.5, alternative="greater") < 1e-5


def test_bca_ci_covers_mean_and_handles_constant():
    d = np.random.default_rng(4).normal(2.0, 1.0, 20)
    lo, hi = bca_ci(d)
    assert lo < d.mean() < hi
    assert bca_ci([3.0] * 10) == (3.0, 3.0)
    assert all(np.isnan(bca_ci([1.0])))


# ------------------------------------------------------------------ RC4

def test_effective_rank_and_srank_reference_values():
    eye = np.eye(128)
    assert effective_rank(eye) == pytest.approx(128.0)
    assert srank(eye, delta=0.01) == 127          # 127/128 >= 0.99 > 126/128
    r1 = np.outer(np.arange(1, 40), np.ones(128))
    assert effective_rank(r1) == pytest.approx(1.0)
    assert srank(r1) == 1
    assert effective_rank(np.zeros((4, 3))) == 0.0 and srank(np.zeros((4, 3))) == 0


def test_rank_invariant_to_row_order_and_dead_units():
    f = np.maximum(np.random.default_rng(5).normal(size=(500, 64)), 0.0)
    f[:, :8] = 0.0
    perm = np.random.default_rng(6).permutation(500)
    assert effective_rank(f) == pytest.approx(effective_rank(f[perm]))
    assert srank(f) == srank(f[perm])
    assert dead_unit_fraction(f) == pytest.approx(8 / 64)


def test_encoder_features_match_full_network_encoder():
    import jax
    import jax.numpy as jnp
    from check_encoder_rank import encoder_features
    from gymnax_exchange.jaxrl.MARL.attack_aware_policy import AttackAwarePolicyNet

    net = AttackAwarePolicyNet(action_dim=6, config={})
    obs = jax.random.normal(jax.random.PRNGKey(0), (32, 45))
    dones = jnp.zeros((32,), dtype=bool)
    hidden = AttackAwarePolicyNet.initialize_carry(32, 8)
    params = net.init(jax.random.PRNGKey(1), hidden, (obs, dones))
    _, inter = net.apply(params, hidden, (obs, dones),
                         capture_intermediates=True, mutable=["intermediates"])
    ref = np.asarray(inter["intermediates"]["SharedEncoder_0"]["__call__"][0])
    got = encoder_features(params, obs)
    assert got.shape == (32, 128)
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


def _per_seed(diff_by_pair):
    rng = np.random.default_rng(7)
    base = rng.normal(40.0, 1.0, 20)
    arms = {"adversarial": base.copy(), "regime": base.copy()}
    arms["detection"] = base + diff_by_pair[0] + rng.normal(0, 0.1, 20)
    arms["full"] = base + diff_by_pair[1] + rng.normal(0, 0.1, 20)
    arms["detection_noobs"] = base + diff_by_pair[2] + rng.normal(0, 0.1, 20)
    return {a: {"erank_fixed": v} for a, v in arms.items()}


def test_rc4_decision_rule():
    from check_encoder_rank import rc4_tests
    assert rc4_tests(_per_seed((5.0, 5.0, 5.0)))["verdict"].startswith("SUPPORTED")
    assert rc4_tests(_per_seed((0.0, 0.0, 0.0)))["verdict"].startswith("NOT SUPPORTED")
    assert rc4_tests(_per_seed((-5.0, -5.0, -5.0)))["verdict"].startswith("NOT SUPPORTED")
    assert rc4_tests(_per_seed((5.0, -5.0, 0.0)))["verdict"].startswith("INCONCLUSIVE")
    res = rc4_tests(_per_seed((5.0, 5.0, 5.0)))
    assert set(res["minimal_pairs"]) == {"detection_vs_adversarial", "full_vs_regime",
                                         "detection_noobs_vs_adversarial"}


# ------------------------------------------------------------------ RC2 units

import json
import math
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_V3 = _ROOT / "results" / "eval_1179095.json"
_V4 = _ROOT / "results" / "eval_17313.json"


def test_units_factor():
    from analysis.units import factor, is_scaled
    ppy = 10833681.0
    assert factor("sharpe_on", ppy) == pytest.approx(1 / math.sqrt(ppy))
    assert factor("sortino_highvol_off", ppy) == pytest.approx(1 / math.sqrt(ppy))
    assert factor("regime_gap_on", ppy) == pytest.approx(1 / math.sqrt(ppy))
    for m in ("cvar_on", "auroc", "inventory_sd_off", "quote_presence_on"):
        assert not is_scaled(m) and factor(m, ppy) == 1.0
    assert factor("sharpe_on", ppy, "annualised") == 1.0
    with pytest.raises(ValueError):
        factor("sharpe_on", ppy, "daily")


@pytest.mark.parametrize("path", [_V3, _V4])
def test_rescale_preserves_every_verdict(path):
    """RC2 must not change a single p-value, effect size or verdict."""
    from analysis.units import rescale_report
    from gymnax_exchange.jaxrl.MARL.adversarial_eval.stats import paired_comparison, tost_paired
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    raw = json.loads(path.read_text())
    ps = rescale_report(raw, "per-step")
    f = 1 / math.sqrt(raw["_meta"]["periods_per_year"])
    assert ps["_meta"]["units"] == "per-step"
    assert raw["_meta"].get("units") is None            # source untouched (deep copy)

    # contrasts: recomputing from rescaled per-seed data gives the same p and d,
    # and the rescaled block's mean_diff equals the recomputed one
    blk = ps["full_vs_baseline"]["sharpe_on"]
    a = np.asarray(ps["per_seed"]["full"]["sharpe_on"])
    b = np.asarray(ps["per_seed"]["baseline"]["sharpe_on"])
    r = paired_comparison(a, b)
    assert r.p_value == pytest.approx(raw["full_vs_baseline"]["sharpe_on"]["p_value"], rel=1e-9)
    assert r.cohens_d == pytest.approx(raw["full_vs_baseline"]["sharpe_on"]["cohens_d"], rel=1e-9)
    assert blk["mean_diff"] == pytest.approx(r.mean_diff, rel=1e-9)
    assert blk["mean_diff"] == pytest.approx(raw["full_vs_baseline"]["sharpe_on"]["mean_diff"] * f)
    assert ps["holm"] == raw["holm"]

    # H2 TOST: same p-values and verdict with the margin scaled alongside the data
    for label, mets in ps["equivalence_off"].items():
        a_arm, b_arm = label.split("_vs_")
        for m, res in mets.items():
            t = tost_paired(ps["per_seed"][a_arm][m], ps["per_seed"][b_arm][m], res["margin"])
            src = raw["equivalence_off"][label][m]
            assert t.p_lower == pytest.approx(src["p_lower"], rel=1e-6, abs=1e-12)
            assert t.p_upper == pytest.approx(src["p_upper"], rel=1e-6, abs=1e-12)
            assert t.equivalent == src["equivalent"]

    # progression gate: every criterion's verdict is reproduced from rescaled numbers
    g = ps["progression_gate"]
    for m in ("sharpe", "sortino"):
        d = g["detail"][m]
        assert (d["ippo"] >= d["as"] - d["margin"]) == g[f"{m}_ok"]
        assert d["margin"] == pytest.approx(0.5 * f)
    # unscaled metrics untouched
    assert ps["per_seed"]["full"]["cvar_on"] == raw["per_seed"]["full"]["cvar_on"]
    assert ps["per_seed"]["full"]["auroc"] == raw["per_seed"]["full"]["auroc"]


def test_rescale_is_idempotent_and_annualised_is_identity():
    from analysis.units import rescale_report
    if not _V4.exists():
        pytest.skip("eval_17313.json not present")
    raw = json.loads(_V4.read_text())
    once = rescale_report(raw)
    assert rescale_report(once) == once
    assert rescale_report(raw, "annualised")["per_seed"] == raw["per_seed"]


# ------------------------------------------------------------------ RC3 rule

def test_rc3_classification_rule():
    from analysis.rc3_detection_check import classify
    assert classify(0.72, 0.66, 0.78, 0.001) == "DETECTS"
    assert classify(0.72, 0.66, 0.78, 0.2) == "INCONCLUSIVE"      # not significant
    assert classify(0.501, 0.47, 0.53, 0.9) == "CANNOT DETECT"
    assert classify(0.53, 0.44, 0.62, 0.3) == "INCONCLUSIVE"


def test_rc3_end_to_end_on_synthetic_files(tmp_path):
    from analysis.rc3_detection_check import run
    rng = np.random.default_rng(0)

    def fake(auroc_mean):
        ps = {}
        for arm in ("detection", "full", "detection_noobs"):
            ps[arm] = {"det_auroc_forced_vs_off": list(rng.normal(auroc_mean, 0.01, 20)),
                       "det_prob_mean_forced": [0.6] * 20, "det_prob_mean_off": [0.4] * 20,
                       "sharpe_forced": list(range(20)), "sharpe_off": list(range(20))}
        return {"per_seed": ps}

    for side, m in (("bid", 0.8), ("ask", 0.8)):
        (tmp_path / f"{side}.json").write_text(json.dumps(fake(m)))
        (tmp_path / f"ref_{side}.json").write_text(json.dumps(fake(m)))
    res = run(str(tmp_path / "bid.json"), str(tmp_path / "ask.json"),
              str(tmp_path / "ref_bid.json"), str(tmp_path / "ref_ask.json"))
    assert res["overall"] == "DETECTS"
    assert res["all_faithful"] is True
    assert len(res["cells"]) == 6
