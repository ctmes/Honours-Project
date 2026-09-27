"""
Pins the two things eval_forced_attack.py's validity rests on, without data or
checkpoints:

  1. the forced action is one-sided in the adversary's own encoding, so it
     injects on exactly one side after action_to_injection;
  2. a zero budget silently zeroes any forced injection, and every v4 EVAL
     config gives the adversary a positive budget -- including the baseline
     arm, whose training env json zeroes it but whose eval yaml deliberately
     loads the config-2 env instead. If that ever regresses, a forced attack on
     the undefended reference arm becomes a no-op without any error.
"""

import jax.numpy as jnp
import pytest

from eval_forced_attack import force_action, set_budget_override
from gymnax_exchange.jaxen.spoofing_agent import SpoofingAgent
from gymnax_exchange.jaxob.jaxob_config import SpoofingAgentConfig, World_EnvironmentConfig


def _world_state(best_vol: int = 50):
    from gymnax_exchange.jaxen.StatesandParams import WorldState
    best = jnp.tile(jnp.array([[10000, best_vol]], dtype=jnp.int32), (5, 1))
    return WorldState(
        ask_raw_orders=jnp.zeros((100, 8), dtype=jnp.int32),
        bid_raw_orders=jnp.zeros((100, 8), dtype=jnp.int32),
        trades=jnp.zeros((10, 8), dtype=jnp.int32),
        init_time=jnp.zeros(2, dtype=jnp.int32),
        window_index=0,
        max_steps_in_episode=6400,
        start_index=0,
        step_counter=100,
        best_bids=best,
        best_asks=best,
        time=jnp.zeros(2, dtype=jnp.int32),
        order_id_counter=0,
        mid_price=jnp.float32(10000),
        delta_time=jnp.float32(1.0),
    )


@pytest.fixture
def agent():
    return SpoofingAgent(cfg=SpoofingAgentConfig(n_spoof_levels=5, inject_mult=2.0),
                         world_config=World_EnvironmentConfig())


@pytest.mark.parametrize("side,hot,cold", [("bid", slice(0, 5), slice(5, 10)),
                                            ("ask", slice(5, 10), slice(0, 5))])
def test_forced_action_injects_one_side_only(agent, side, hot, cold):
    vol = agent.action_to_injection(force_action(side, 5), _world_state(50),
                                    jnp.float32(1e6), gate=jnp.float32(1.0))
    assert float(jnp.min(vol[hot])) == pytest.approx(2.0 * 50)   # inject_mult * depth, every level
    assert float(jnp.max(vol[cold])) == 0.0


def test_zero_budget_silently_zeroes_a_forced_action(agent):
    vol = agent.action_to_injection(force_action("bid", 5), _world_state(50),
                                    jnp.float32(0.0), gate=jnp.float32(1.0))
    assert float(jnp.sum(vol)) == 0.0


@pytest.mark.parametrize("n", range(1, 8))
def test_every_v4_eval_env_gives_the_adversary_a_budget(n):
    import json
    import yaml
    with open(f"config/rl_configs/eval_2024_test_v4_config{n}.yaml") as f:
        env_json = yaml.safe_load(f)["ENV_CONFIG"]
    assert not env_json.endswith("adversarial_mm_v4_config1.json"), (
        "an eval yaml loads the config-1 TRAINING env, whose zero budget makes any "
        "attack -- forced or learned -- a no-op")
    with open(env_json) as f:
        spoof = json.load(f)["dict_of_agents_configs"]["Spoofing"]
    assert spoof["budget_per_episode"] > 0


def test_budget_override_sets_the_value():
    cfg = {"dict_of_agents_configs": {"Spoofing": {"budget_per_episode": 1e6}}}
    set_budget_override(cfg, 5e7)
    assert cfg["dict_of_agents_configs"]["Spoofing"]["budget_per_episode"] == 5e7


def test_budget_override_rejects_non_positive():
    with pytest.raises(ValueError):
        set_budget_override({"dict_of_agents_configs": {"Spoofing": {}}}, 0.0)


def test_force_action_rejects_unknown_side():
    with pytest.raises(ValueError):
        force_action("both", 5)
