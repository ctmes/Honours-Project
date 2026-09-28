"""
Pins the Tier-4 (WS10) machinery without data (preregistration.json ->
amendments -> tier4_extensions_amendment_2026-09-28):

  * make_mm_network / AttackAwareRecurrentNet: the recurrent MM keeps the same
    SharedEncoder, carries memory across steps and resets it on done;
  * observe_victim: the adversary observation grows by the victim's 45 dims;
  * load_mm_params: MM params are copied from a checkpoint exactly, and a
    structural mismatch is refused;
  * analysis.tier4_check: the pre-registered decision rules.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# ------------------------------------------------------------ recurrent MM

def _init(net, B=4, T=3, D=45, H=16):
    obs = jax.random.normal(jax.random.PRNGKey(0), (T, B, D))
    dones = jnp.zeros((T, B), dtype=bool)
    h0 = net.initialize_carry(B, H)
    params = net.init(jax.random.PRNGKey(1), h0, (obs, dones))
    return params, obs, dones, h0


def test_factory_selects_network():
    from gymnax_exchange.jaxrl.MARL.attack_aware_policy import (
        AttackAwarePolicyNet, AttackAwareRecurrentNet, make_mm_network)
    assert isinstance(make_mm_network(6, {}), AttackAwarePolicyNet)
    assert isinstance(make_mm_network(6, {"MM_RECURRENT": False}), AttackAwarePolicyNet)
    assert isinstance(make_mm_network(6, {"MM_RECURRENT": True}), AttackAwareRecurrentNet)


def test_recurrent_net_shapes_and_shared_encoder_name():
    from gymnax_exchange.jaxrl.MARL.attack_aware_policy import make_mm_network
    net = make_mm_network(6, {"MM_RECURRENT": True})
    params, obs, dones, h0 = _init(net)
    h, pi, value, det = net.apply(params, h0, (obs, dones))
    assert h.shape == h0.shape and value.shape == (3, 4) and det.shape == (3, 4)
    assert pi.logits.shape == (3, 4, 6)
    assert "SharedEncoder_0" in params["params"]          # RC4 rank analysis relies on it
    assert not np.allclose(np.asarray(h), 0.0)            # memory actually updates


def test_recurrent_net_uses_memory_and_resets_on_done():
    from gymnax_exchange.jaxrl.MARL.attack_aware_policy import make_mm_network
    net = make_mm_network(6, {"MM_RECURRENT": True})
    params, obs, dones, h0 = _init(net, T=4)
    _, pi_a, _, _ = net.apply(params, h0, (obs, dones))
    # different history, same last observation -> different last-step policy
    obs_b = obs.at[:3].set(obs[:3] * -1.0)
    _, pi_b, _, _ = net.apply(params, h0, (obs_b, dones))
    assert not np.allclose(np.asarray(pi_a.logits[-1]), np.asarray(pi_b.logits[-1]))
    # a done flag at the last step erases the history
    dones_reset = dones.at[3].set(True)
    _, pi_ra, _, _ = net.apply(params, h0, (obs, dones_reset))
    _, pi_rb, _, _ = net.apply(params, h0, (obs_b, dones_reset))
    np.testing.assert_allclose(np.asarray(pi_ra.logits[-1]), np.asarray(pi_rb.logits[-1]),
                               rtol=1e-6, atol=1e-6)


def test_mlp_ignores_hidden_so_update_carry_change_is_inert():
    """ippo_adversarial now passes the rollout carry into the MM update; for the
    memoryless MLP that must not change anything."""
    from gymnax_exchange.jaxrl.MARL.attack_aware_policy import make_mm_network
    net = make_mm_network(6, {})
    params, obs, dones, h0 = _init(net)
    _, pi0, v0, d0 = net.apply(params, h0, (obs, dones))
    _, pi1, v1, d1 = net.apply(params, h0 + 3.0, (obs, dones))
    np.testing.assert_array_equal(np.asarray(pi0.logits), np.asarray(pi1.logits))
    np.testing.assert_array_equal(np.asarray(v0), np.asarray(v1))


# ------------------------------------------------------------ observe_victim

def test_observe_victim_observation_space_and_concat():
    from gymnax_exchange.jaxen.adversarial_marl_env import AdversarialMARLEnv
    from gymnax_exchange.jaxen.spoofing_agent import SpoofingAgent
    from gymnax_exchange.jaxob.jaxob_config import SpoofingAgentConfig, World_EnvironmentConfig
    wc = World_EnvironmentConfig()
    assert SpoofingAgent(SpoofingAgentConfig(), wc).observation_space().shape == (43,)
    assert SpoofingAgent(SpoofingAgentConfig(observe_victim=True), wc).observation_space().shape == (88,)
    adv = jnp.arange(43, dtype=jnp.float32).reshape(1, 43)
    mm = jnp.full((1, 45), 7.0)
    out = AdversarialMARLEnv._with_victim_obs(adv, mm)
    assert out.shape == (1, 88)
    np.testing.assert_array_equal(np.asarray(out[0, :43]), np.arange(43))
    np.testing.assert_array_equal(np.asarray(out[0, 43:]), np.full(45, 7.0))


# ------------------------------------------------------------ MM_INIT_FROM

def _train_state(net, obs_dim, seed):
    import optax
    from flax.training.train_state import TrainState
    x = (jnp.zeros((1, 4, obs_dim)), jnp.zeros((1, 4)))
    params = net.init(jax.random.PRNGKey(seed), net.initialize_carry(4, 8), x)
    return TrainState.create(apply_fn=net.apply, params=params, tx=optax.adam(1e-3))


def test_load_mm_params_copies_exactly_and_refuses_mismatch(tmp_path):
    import orbax.checkpoint as oxcp
    from flax.training import orbax_utils
    from gymnax_exchange.jaxrl.MARL.attack_aware_policy import AdversaryNet, make_mm_network
    from gymnax_exchange.jaxrl.MARL.ippo_adversarial import load_mm_params

    mm_net, adv_net = make_mm_network(6, {}), AdversaryNet(action_dim=10)
    src = [_train_state(mm_net, 45, 1), _train_state(adv_net, 43, 2)]
    ckpt_dir = tmp_path / "checkpoints" / "MARLCheckpoints" / "srcproj" / "seed_3"
    mgr = oxcp.CheckpointManager(str(ckpt_dir), oxcp.PyTreeCheckpointer(),
                                 oxcp.CheckpointManagerOptions(create=True))
    ckpt = {"model": src, "metrics": {"avg_reward": [0.0, 0.0]}}
    mgr.save(7, ckpt, save_kwargs={"save_args": orbax_utils.save_args_from_target(ckpt)})
    mgr.wait_until_finished()

    cfg = {"SEED": 3, "world_config": {"alphatradePath": str(tmp_path)}}
    fresh = _train_state(mm_net, 45, 99)
    loaded = load_mm_params(fresh, "srcproj/seed_{SEED}@7", 0, cfg)
    for a, b in zip(jax.tree.leaves(loaded.params), jax.tree.leaves(src[0].params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    # a recurrent template has a different tree -> refused, not silently mangled
    rec = _train_state(make_mm_network(6, {"MM_RECURRENT": True}), 45, 5)
    with pytest.raises(ValueError):
        load_mm_params(rec, "srcproj/seed_{SEED}@7", 0, cfg)
    with pytest.raises(FileNotFoundError):
        load_mm_params(fresh, "nothere/seed_{SEED}", 0, cfg)


# ------------------------------------------------------------ decision rules

def _learned_file(base_eff, full_eff, n=20, seed=0):
    rng = np.random.default_rng(seed)
    ps = {}
    for arm, eff in (("baseline", base_eff), ("adversarial", base_eff), ("full", full_eff)):
        off = rng.normal(0.0, 1.0, n)
        on = off + eff + rng.normal(0.0, 0.05, n)
        ps[arm] = {"sharpe_off": off.tolist(), "sharpe_learned": on.tolist(),
                   "sortino_off": off.tolist(), "sortino_learned": on.tolist(),
                   "mean_bid_volume_injected_learned": [10.0] * n,
                   "mean_ask_volume_injected_learned": [0.0] * n,
                   "qi_mean_learned": [0.4] * n, "qi_mean_off": [0.0] * n}
    return {"per_seed": ps, "_meta": {"periods_per_year": 1.0}}


def test_ws10a_rules():
    from analysis.tier4_check import ws10a
    r = ws10a(_learned_file(0.0, 0.0))
    assert r["verdict"].startswith("H1 PREMISE FAILS") and not r["baseline_degraded"]
    r = ws10a(_learned_file(-1.0, -0.1))
    assert r["verdict"].startswith("H1 SUPPORTED")
    assert r["adversary"]["baseline"]["sidedness_mean"] == pytest.approx(1.0)
    r = ws10a(_learned_file(-1.0, -1.0))
    assert r["verdict"].startswith("Baseline degraded; the defence is NOT")


def _eval_file(h_minus_b, h_minus_g, n=20, seed=1):
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, 1.0, n)
    g = b + 2.0 + rng.normal(0, 0.05, n)
    h_from_b = b + h_minus_b + rng.normal(0, 0.05, n)
    h = h_from_b if h_minus_g is None else g + h_minus_g + rng.normal(0, 0.05, n)
    mk = lambda x: {"sharpe_off": x.tolist(), "inventory_sd_off": (x + 5).tolist()}
    return {"per_seed": {"adversarial": mk(b), "detection_noobs": mk(g), "shuffled_noobs": mk(h)},
            "_meta": {"periods_per_year": 1.0}}


def test_ws10b_rules():
    from analysis.tier4_check import ws10b
    assert ws10b(_eval_file(2.0, 0.0))["verdict"].startswith("REGULARISATION")
    assert ws10b(_eval_file(0.0, -2.0))["verdict"].startswith("LABEL CONTENT MATTERS")
    assert "skipped" in ws10b({"per_seed": {"adversarial": {}}})
