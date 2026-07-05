"""
Unit tests for SpoofingAgent.

Tests the budget tracking, zero-LOB-message guarantee, and reward polarity.
These tests exercise the Python/JAX methods directly without needing real
LOBSTER data or a running env.
"""

import jax
import jax.numpy as jnp
import pytest

from gymnax_exchange.jaxen.spoofing_agent import SpoofingAgent
from gymnax_exchange.jaxen.StatesandParams import SpoofingAgentState, SpoofingAgentParams
from gymnax_exchange.jaxob.jaxob_config import SpoofingAgentConfig, World_EnvironmentConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return SpoofingAgentConfig(
        n_spoof_levels=5,
        budget_per_episode=500.0,
        c_fill=0.001,
        c_reg=0.0005,
    )


@pytest.fixture
def world_cfg():
    return World_EnvironmentConfig()


@pytest.fixture
def agent(cfg, world_cfg):
    return SpoofingAgent(cfg=cfg, world_config=world_cfg)


def _make_state(budget: float = 500.0, attack_active: float = 1.0) -> SpoofingAgentState:
    return SpoofingAgentState(
        budget_remaining=jnp.array(budget, dtype=jnp.float32),
        volume_injected=jnp.zeros(()),
        prev_mm_reward=jnp.zeros(()),
        prev_detection_prob=jnp.zeros(()),
        attack_active=jnp.array(attack_active, dtype=jnp.float32),
    )


def _make_world_state(best_price: int = 10000, best_vol: int = 50):
    """Minimal WorldState: the depth-scaled injection only reads best_bids/best_asks
    [-1, 1] (best-quote depth), plus raw order arrays for the cost model."""
    from gymnax_exchange.jaxen.StatesandParams import WorldState
    best = jnp.tile(jnp.array([[best_price, best_vol]], dtype=jnp.int32), (5, 1))
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
        mid_price=jnp.float32(best_price),
        delta_time=jnp.float32(1.0),
    )


# ---------------------------------------------------------------------------
# Test 1: Budget never increases and never goes below zero
# ---------------------------------------------------------------------------

def test_budget_monotone(agent, cfg):
    """
    budget_remaining should strictly decrease (or stay equal) each step and
    never become negative regardless of the action injected.
    """
    state = _make_state(budget=500.0)

    budgets = [float(state.budget_remaining)]
    for step in range(10):
        # Inject a large action every step
        vol_step = jnp.array(60.0)   # 6 steps × 60 > 500 → should hit zero
        extras = {"volume_injected_step": vol_step}
        state, done, info = agent.update_state_and_get_done_and_info(
            new_world_state=None,   # not used in state update logic
            agent_state=state,
            extras=extras,
        )
        budgets.append(float(state.budget_remaining))

    # Budget is monotonically non-increasing
    for i in range(1, len(budgets)):
        assert budgets[i] <= budgets[i - 1] + 1e-5, \
            f"Budget increased at step {i}: {budgets[i - 1]} → {budgets[i]}"

    # Budget never goes below zero
    assert all(b >= -1e-6 for b in budgets), \
        f"Budget went negative: {min(budgets)}"


# ---------------------------------------------------------------------------
# Test 2: get_messages returns zero-row LOB arrays
# ---------------------------------------------------------------------------

def test_zero_lob_messages(agent, cfg):
    """
    SpoofingAgent.get_messages must return (0,8)-shaped arrays so the parent
    MARLEnv's vstack-based message aggregation is a no-op.
    """
    state  = _make_state()
    params = SpoofingAgentParams(
        budget_per_episode=jnp.array([500.0])
    )
    action = jnp.ones(10) * 0.5   # arbitrary action

    action_msgs, cancel_msgs, extras = agent.get_messages(
        action=action,
        world_state=_make_world_state(),   # depth-scaled injection reads best-quote depth
        agent_state=state,
        agent_params=params,
    )

    assert action_msgs.shape == (0, 8), \
        f"Expected (0, 8) action messages, got {action_msgs.shape}"
    assert cancel_msgs.shape == (0, 8), \
        f"Expected (0, 8) cancel messages, got {cancel_msgs.shape}"
    assert "clipped_spoof_action" in extras, "Expected clipped_spoof_action in extras"


# ---------------------------------------------------------------------------
# Test 3: Action is clipped to budget
# ---------------------------------------------------------------------------

def test_action_clipped_to_budget(agent):
    """
    clipped_spoof_action elements must not exceed budget_per_level and must
    sum to at most budget_remaining.
    """
    budget = 100.0
    state  = _make_state(budget=budget)
    params = SpoofingAgentParams(budget_per_episode=jnp.array([budget]))

    # Action well above budget (clip(action,0,1)=1 per level, then depth-scaled:
    # with uniform best-quote depth the 10 levels inject equally and are scaled
    # down proportionally to the remaining budget)
    action = jnp.ones(10) * 200.0

    _, _, extras = agent.get_messages(
        action=action,
        world_state=_make_world_state(best_vol=50),   # 2.0 * 50 * 10 levels = 1000 >> budget
        agent_state=state,
        agent_params=params,
    )

    clipped = extras["clipped_spoof_action"]
    budget_per_level = budget / agent.cfg.n_spoof_levels
    assert float(jnp.max(clipped)) <= budget_per_level + 1e-5, \
        f"Clipped action exceeds budget per level: {jnp.max(clipped)}"
    assert float(jnp.sum(clipped)) <= budget + 1e-5, \
        f"Clipped total volume exceeds budget: {jnp.sum(clipped)}"


# ---------------------------------------------------------------------------
# Test 4: Reward costs are non-positive (costs only, MM component added externally)
# ---------------------------------------------------------------------------

def test_reward_costs_nonpositive(agent, world_cfg):
    """
    SpoofingAgent.get_reward returns costs only (negative).
    The -r_mm component is added later by AdversarialMARLEnv.
    With positive injection volume, reward should be <= 0.
    """
    from gymnax_exchange.jaxob import JaxOrderBookArrays as job

    # Create a minimal state with some injected volume
    state = SpoofingAgentState(
        budget_remaining=jnp.array(400.0),
        volume_injected=jnp.array(100.0),   # 100 units already injected
        prev_mm_reward=jnp.zeros(()),
        prev_detection_prob=jnp.zeros(()),
        attack_active=jnp.array(1.0, dtype=jnp.float32),
    )
    params = SpoofingAgentParams(budget_per_episode=jnp.array([500.0]))

    # We need a minimal WorldState with bid_raw_orders
    # Use zeros — bid depth = 0, depth_pressure = vol / 1 = vol
    dummy_bids = jnp.zeros((100, 8), dtype=jnp.int32)
    dummy_asks = jnp.zeros((100, 8), dtype=jnp.int32)
    dummy_best = jnp.zeros((5, 2), dtype=jnp.int32)

    # Build a minimal WorldState
    from gymnax_exchange.jaxen.StatesandParams import WorldState
    world_state = WorldState(
        ask_raw_orders=dummy_asks,
        bid_raw_orders=dummy_bids,
        trades=jnp.zeros((10, 8), dtype=jnp.int32),
        init_time=jnp.zeros(2, dtype=jnp.int32),
        window_index=0,
        max_steps_in_episode=6400,
        start_index=0,
        step_counter=100,
        best_bids=dummy_best,
        best_asks=dummy_best,
        time=jnp.zeros(2, dtype=jnp.int32),
        order_id_counter=0,
        mid_price=jnp.float32(10000.0),
        delta_time=jnp.float32(1.0),
    )

    r_adv, extras = agent.get_reward(
        world_state=world_state,
        agent_state=state,
        agent_params=params,
        trades=jnp.zeros((10, 8), dtype=jnp.int32),
        bestasks=dummy_best,
        bestbids=dummy_best,
        ep_done_time=False,
    )

    assert float(r_adv) <= 1e-6, \
        f"Expected costs-only reward <= 0, got {float(r_adv)}"
    assert "costs_total" in extras, "Expected costs_total in extras"
    assert float(extras["costs_total"]) >= 0.0, \
        f"costs_total should be non-negative, got {float(extras['costs_total'])}"


# ---------------------------------------------------------------------------
# Test 4b: costs_total reaches the CONSUMER path (info dict), not just extras.
#
# AdversarialMARLEnv.step_env builds R_adv = -r_mm - info["agents"][adv]["costs_total"].
# Before 2026-07-04 this key was only in get_reward's extras, never in the info dict
# produced by update_state_and_get_done_and_info, so a .get() default silently zeroed
# the economic-cost term in the adversary's reward. This test pins the contract.
# ---------------------------------------------------------------------------

def test_costs_total_in_info(agent):
    state = _make_state(budget=400.0)
    extras = {
        "reward": jnp.array(-1.0),
        "costs_total": jnp.array(3.5),
        "accidental_fills": jnp.array(1.5),
        "regulatory_cost": jnp.array(2.0),
        "volume_injected_step": jnp.array(10.0),
    }
    # new_world_state is unused by the spoofer's state update; None is safe here.
    new_state, done, info = agent.update_state_and_get_done_and_info(None, state, extras)

    assert "costs_total" in info, \
        "costs_total missing from spoofer info — AdversarialMARLEnv reads it to build R_adv"
    assert float(info["costs_total"]) == pytest.approx(3.5)
    assert float(info["accidental_fills"]) == pytest.approx(1.5)
    assert float(info["regulatory_cost"]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Test 5: Observation space dimension
# ---------------------------------------------------------------------------

def test_obs_space_dim(agent):
    obs_space = agent.observation_space()
    assert obs_space.shape == (43,), \
        f"Expected obs shape (43,), got {obs_space.shape}"

    act_space = agent.action_space()
    assert act_space.shape == (10,), \
        f"Expected action shape (10,), got {act_space.shape}"
