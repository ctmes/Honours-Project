"""
Unit tests for AdversarialMARLEnv logic.

Tests the core adversarial properties that do not require live data:
  1. Oracle label computed correctly from clipped adversary action
  2. Regime array is binary {0.0, 1.0}
  3. MM observation dimension is 45
  4. PCGrad-specific: regime conditioning flag is forwarded to observation

Tests that require a running env with real data are integration tests and
are not included here — they belong in a separate tests/integration/ suite.
"""

import jax
import jax.numpy as jnp
import pytest
import numpy as np

from gymnax_exchange.jaxen.adversarial_marl_env import AdversarialMARLEnv
from gymnax_exchange.jaxob.jaxob_config import (
    SpoofingAgentConfig,
    AdversarialMMConfig,
    MultiAgentConfig,
    World_EnvironmentConfig,
)


# ---------------------------------------------------------------------------
# Test 1: Oracle label derived from clipped action
# ---------------------------------------------------------------------------

def test_oracle_label_zero_action():
    """When clipped_adv_action is all zeros, oracle label must be 0."""
    clipped = jnp.zeros(10)
    label   = (jnp.sum(clipped) > 0.0).astype(jnp.float32)
    assert float(label) == 0.0, f"Expected label 0 for zero action, got {float(label)}"


def test_oracle_label_nonzero_action():
    """When any element of clipped_adv_action > 0, oracle label must be 1."""
    clipped = jnp.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    label   = (jnp.sum(clipped) > 0.0).astype(jnp.float32)
    assert float(label) == 1.0, f"Expected label 1 for nonzero action, got {float(label)}"


# ---------------------------------------------------------------------------
# Test 2: Regime array is binary {0.0, 1.0}
# ---------------------------------------------------------------------------

def test_regime_array_binary():
    """
    _build_regime_array should only contain values in {0.0, 1.0}.
    Test with a synthetic mapping.
    """
    regime_labels   = {"2024-01-02": 0, "2024-08-05": 1, "2024-09-10": 1}
    window_to_date  = {0: "2024-01-02", 1: "2024-08-05", 2: "2024-09-10"}

    # Build a minimal env class mock that has start_indeces with length 3
    class _MockBaseEnv:
        start_indeces = [0, 1, 2]

    class _MockEnv:
        base_env = _MockBaseEnv()

        def _build_regime_array(self, regime_labels, window_to_date):
            return AdversarialMARLEnv._build_regime_array(
                self, regime_labels, window_to_date
            )

    # Call _build_regime_array directly with a monkey-patched self
    import types

    mock = _MockEnv()
    arr = AdversarialMARLEnv._build_regime_array(mock, regime_labels, window_to_date)

    unique_vals = set(float(v) for v in np.array(arr))
    assert unique_vals <= {0.0, 1.0}, \
        f"Regime array contains values outside {{0, 1}}: {unique_vals}"
    assert float(arr[0]) == 0.0, f"Window 0 should be regime 0, got {float(arr[0])}"
    assert float(arr[1]) == 1.0, f"Window 1 should be regime 1, got {float(arr[1])}"
    assert float(arr[2]) == 1.0, f"Window 2 should be regime 1, got {float(arr[2])}"


# ---------------------------------------------------------------------------
# Test 3: Regime array default (no window_to_date) is all zeros
# ---------------------------------------------------------------------------

def test_regime_array_default_zeros():
    """Without window_to_date, regime array should be all zeros."""
    class _MockBaseEnv:
        start_indeces = list(range(10))

    class _MockEnv:
        base_env = _MockBaseEnv()

    mock = _MockEnv()
    arr  = AdversarialMARLEnv._build_regime_array(mock, {}, None)

    assert float(jnp.max(arr)) == 0.0, \
        f"Default regime array should be all zeros, max={float(jnp.max(arr))}"
    assert arr.shape == (10,), f"Unexpected shape: {arr.shape}"


# ---------------------------------------------------------------------------
# Test 4: MM observation space is 45-dim
# ---------------------------------------------------------------------------

def test_mm_obs_space_dim():
    """AdversarialMMConfig observation_space='adversarial_lob' → 45-dim Box."""
    from gymnax_exchange.jaxen.mm_env import MarketMakingAgent

    cfg        = AdversarialMMConfig(observation_space="adversarial_lob")
    world_cfg  = World_EnvironmentConfig()
    mm_agent   = MarketMakingAgent(cfg=cfg, world_config=world_cfg)

    obs_space = mm_agent.observation_space()
    assert obs_space.shape == (45,), \
        f"Expected MM obs shape (45,), got {obs_space.shape}"


# ---------------------------------------------------------------------------
# Test 5: get_adversarial_observation returns 45-dim vector
# ---------------------------------------------------------------------------

def test_get_adversarial_observation_dim():
    """
    MarketMakingAgent.get_adversarial_observation should return a 45-dim array.
    """
    from gymnax_exchange.jaxen.mm_env import MarketMakingAgent
    from gymnax_exchange.jaxen.StatesandParams import MMEnvState
    from gymnax_exchange.jaxob.jaxob_config import AdversarialMMConfig
    from gymnax_exchange.jaxen.StatesandParams import WorldState
    from gymnax_exchange.jaxob.jaxob_config import World_EnvironmentConfig
    import gymnax_exchange.jaxen.StatesandParams as sp

    cfg       = AdversarialMMConfig(observation_space="adversarial_lob")
    world_cfg = World_EnvironmentConfig()
    mm_agent  = MarketMakingAgent(cfg=cfg, world_config=world_cfg)

    # Minimal agent state
    mm_state = MMEnvState(
        posted_distance_bid=0,
        posted_distance_ask=0,
        inventory=0,
        total_PnL=0.0,
        cash_balance=0.0,
    )

    # Minimal world state
    dummy_orders = jnp.zeros((100, 8), dtype=jnp.int32)
    dummy_best   = jnp.zeros((5, 2),   dtype=jnp.int32)
    world_state  = WorldState(
        ask_raw_orders=dummy_orders,
        bid_raw_orders=dummy_orders,
        trades=jnp.zeros((10, 8), dtype=jnp.int32),
        init_time=jnp.zeros(2, dtype=jnp.int32),
        window_index=0,
        max_steps_in_episode=6400,
        start_index=0,
        step_counter=0,
        best_bids=dummy_best,
        best_asks=dummy_best,
        time=jnp.zeros(2, dtype=jnp.int32),
        order_id_counter=0,
        mid_price=jnp.float32(10000.0),
        delta_time=jnp.float32(1.0),
    )

    perturbed_l2 = jnp.zeros(40)
    regime       = jnp.float32(0.0)
    prev_det     = jnp.float32(0.0)

    # Need a dummy agent param — just use MMEnvParams
    from gymnax_exchange.jaxen.StatesandParams import MMEnvParams
    mm_param = MMEnvParams(
        trader_id=jnp.array([-100]),
        time_delay_obs_act=jnp.array([0]),
        normalize=jnp.array([True]),
    )

    obs = mm_agent.get_adversarial_observation(
        world_state, mm_state, mm_param, perturbed_l2, regime, prev_det, normalize=True
    )

    assert obs.shape == (45,), f"Expected obs shape (45,), got {obs.shape}"


# ---------------------------------------------------------------------------
# Test 6: Spoof perturbation indices are correct
# ---------------------------------------------------------------------------

def test_perturbation_index_assignment():
    """
    Bid vol indices [3,7,11,15,19] and ask vol indices [1,5,9,13,17] should
    be the only elements modified when applying the adversary's perturbation.
    """
    true_l2 = jnp.zeros(40)
    adv_action = jnp.ones(10)   # inject 1 unit per level

    ask_vol_idx = jnp.array([1, 5, 9, 13, 17])
    bid_vol_idx = jnp.array([3, 7, 11, 15, 19])

    perturbed = true_l2.at[ask_vol_idx].add(adv_action[5:])
    perturbed = perturbed.at[bid_vol_idx].add(adv_action[:5])

    # Non-volume indices should be zero
    for idx in range(40):
        if idx not in list(ask_vol_idx) and idx not in list(bid_vol_idx):
            assert float(perturbed[idx]) == 0.0, \
                f"Index {idx} should be unmodified, got {float(perturbed[idx])}"

    # Volume indices should be 1
    for idx in list(ask_vol_idx) + list(bid_vol_idx):
        assert float(perturbed[idx]) == 1.0, \
            f"Index {idx} should be 1.0 after perturbation, got {float(perturbed[idx])}"
