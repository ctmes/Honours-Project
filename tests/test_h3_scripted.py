"""
Pins the H3m machinery (preregistration.json -> amendments ->
h3_material_attack_amendment_2026-10-02) without data:

  * scripted_injection_action: one side per episode, a fixed material magnitude,
    in the adversary's [bid levels..., ask levels...] encoding;
  * through SpoofingAgent.action_to_injection every gated-on step injects at least
    2.5x the spoofed side's best-quote depth, nothing on the other side, and
    nothing when the gate is off -- so with the pre-registered materiality floor
    of 0.0 the oracle label equals the gate;
  * the generated configs differ from their v4 sources only where intended;
  * analysis.h3_material_check: the pre-registered decision rule.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------ scripted attack

def test_scripted_action_is_one_sided_and_material():
    from gymnax_exchange.jaxrl.MARL.ippo_adversarial import scripted_injection_action
    w = jnp.arange(200)
    a = np.asarray(scripted_injection_action(w, 5))
    assert a.shape == (200, 10)
    bid, ask = a[:, :5], a[:, 5:]
    even = (np.arange(200) % 2 == 0)
    assert np.all(bid[even] > 0) and np.all(ask[even] == 0)
    assert np.all(ask[~even] > 0) and np.all(bid[~even] == 0)
    mag = np.where(even, bid[:, 0], ask[:, 0])
    assert np.all((mag >= 0.25) & (mag < 1.0))
    assert np.allclose(np.where(even[:, None], bid, ask), mag[:, None])   # flat across levels
    # both sides and a spread of magnitudes are covered
    assert mag.min() < 0.35 and mag.max() > 0.9
    # deterministic per window: the same episode always gets the same attack
    assert np.array_equal(a, np.asarray(scripted_injection_action(w, 5)))


def test_scripted_action_through_injection_is_material_and_gated():
    from gymnax_exchange.jaxen.spoofing_agent import SpoofingAgent
    from gymnax_exchange.jaxob.jaxob_config import SpoofingAgentConfig
    from gymnax_exchange.jaxrl.MARL.ippo_adversarial import scripted_injection_action
    cfg = SpoofingAgentConfig()
    agent = SimpleNamespace(cfg=cfg)
    ws = SimpleNamespace(best_bids=jnp.array([[10000, 300]]), best_asks=jnp.array([[10100, 900]]))
    for w in range(40):
        a = scripted_injection_action(jnp.array([w]), cfg.n_spoof_levels)[0]
        on = np.asarray(SpoofingAgent.action_to_injection(agent, a, ws, jnp.float32(1e7), 1.0))
        off = np.asarray(SpoofingAgent.action_to_injection(agent, a, ws, jnp.float32(1e7), 0.0))
        side_depth = 300.0 if w % 2 == 0 else 900.0
        spoofed, other = (on[:5], on[5:]) if w % 2 == 0 else (on[5:], on[:5])
        assert spoofed.sum() >= 2.5 * side_depth - 1e-3 and np.all(other == 0)
        assert np.all(off == 0)
        # oracle label at the pre-registered floor 0.0 (adversarial_marl_env)
        assert on.sum() > 0.0 * 0.5 * (300 + 900) and not off.sum() > 0.0


# ------------------------------------------------------------ configs

def _load_yaml_keys(path):
    out = {}
    for ln in Path(path).read_text().splitlines():
        if ln.startswith('"') and ":" in ln:
            k, v = ln.split(":", 1)
            out[k.strip('"')] = v.split("#")[0].strip()
    return out


def test_h3m_configs_differ_from_v4_detection_only_where_intended():
    env = ROOT / "config/env_configs"
    a = json.loads((env / "adversarial_mm_v4_config4.json").read_text())
    b = json.loads((env / "adversarial_mm_v4s_detection.json").read_text())
    assert b["dict_of_agents_configs"]["Spoofing"].pop("budget_per_episode") == 1.0e7
    a["dict_of_agents_configs"]["Spoofing"].pop("budget_per_episode")
    assert a == b
    assert b["dict_of_agents_configs"]["Spoofing"].get("label_materiality_frac", 0.0) == 0.0

    rl = ROOT / "config/rl_configs"
    src = _load_yaml_keys(rl / "kaya_v4_config4_detection.yaml")
    new = _load_yaml_keys(rl / "kaya_v4s_config4_detection.yaml")
    assert new.pop("ADV_SCRIPTED") == "true"
    assert new.pop("PROJECT") == '"v4s_config4_detection"' and src.pop("PROJECT")
    assert new.pop("ENV_CONFIG") == '"config/env_configs/adversarial_mm_v4s_detection.json"'
    src.pop("ENV_CONFIG")
    assert new == src

    from eval_forced_attack import EXTRA_HEAD_ARMS, resolve_arm
    project, yaml_path = resolve_arm("scripted_detection", "v4")
    assert project == "v4s_config4_detection" and "scripted_detection" in EXTRA_HEAD_ARMS
    ev = _load_yaml_keys(ROOT / yaml_path)
    assert ev["PROJECT"] == '"v4s_config4_detection"'
    assert ev["ENV_CONFIG"] == '"config/env_configs/adversarial_mm_v4s_detection.json"'
    assert ev["TimePeriod"].strip('"') == "2024_test"


# ------------------------------------------------------------ decision rule

def _forced(side, auroc, extra_arm=None, sharpe_shift=0.0):
    rng = np.random.default_rng(0 if side == "bid" else 1)
    n = 20

    def arm(a, shift=0.0):
        return {"det_auroc_forced_vs_off": list(a + rng.normal(0, 0.01, n)),
                "det_prob_mean_forced": [0.5] * n, "det_prob_mean_off": [0.1] * n,
                "sharpe_forced": list(np.linspace(-1, 1, n) + shift),
                "sharpe_off": list(np.linspace(-1, 1, n)),
                "sortino_off": list(np.linspace(-1, 1, n)),
                "inventory_sd_off": [2.0] * n}

    per = {"scripted_detection": arm(auroc, sharpe_shift)}
    if extra_arm is not None:
        per["detection"] = arm(extra_arm)
    return {"per_seed": per, "_meta": {"side": side, "periods_per_year": 100.0}}


def _run(tmp_path, b, a, rb=None, ra=None):
    from analysis.h3_material_check import run
    paths = []
    for name, d in (("b", b), ("a", a), ("rb", rb), ("ra", ra)):
        if d is None:
            paths.append(None)
            continue
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(d))
        paths.append(str(p))
    return run(*paths)


def test_h3m_rule(tmp_path):
    r = _run(tmp_path, _forced("bid", 0.80), _forced("ask", 0.75))
    assert r["primary"]["overall"] == "SUPPORTED"
    assert r["off_consistent_across_sides"] is True
    r = _run(tmp_path, _forced("bid", 0.50), _forced("ask", 0.50))
    assert r["primary"]["overall"] == "NOT SUPPORTED"
    r = _run(tmp_path, _forced("bid", 0.80), _forced("ask", 0.50))
    assert r["primary"]["overall"] == "INCONCLUSIVE"
    # 0.58: clearly above chance but below the 0.60 bar -> not DETECTS
    r = _run(tmp_path, _forced("bid", 0.58), _forced("ask", 0.58))
    assert r["primary"]["overall"] == "INCONCLUSIVE"


def test_h3m_secondary_uses_reference_file_and_scales(tmp_path):
    ref_b, ref_a = _forced("bid", 0.5, extra_arm=0.5), _forced("ask", 0.5, extra_arm=0.5)
    for d in (ref_b, ref_a):
        d["per_seed"].pop("scripted_detection")
    r = _run(tmp_path, _forced("bid", 0.8, sharpe_shift=1.0), _forced("ask", 0.8), ref_b, ref_a)
    t = r["secondary"]["tests"]
    assert t["bid:auroc_gain"]["mean_diff"] == pytest.approx(0.3, abs=0.02)
    assert t["bid:did_sharpe"]["mean_diff"] == pytest.approx(1.0 / 10.0)   # per step: / sqrt(100)
    assert t["clean:sharpe_off"]["mean_diff"] == pytest.approx(0.0)
    # and from the same file when the detection arm was evaluated alongside
    r2 = _run(tmp_path, _forced("bid", 0.8, extra_arm=0.5), _forced("ask", 0.8, extra_arm=0.5))
    assert r2["secondary"]["tests"]["ask:auroc_gain"]["mean_diff"] == pytest.approx(0.3, abs=0.02)


def test_h3m_refuses_swapped_sides(tmp_path):
    with pytest.raises(SystemExit):
        _run(tmp_path, _forced("ask", 0.8), _forced("bid", 0.8))
