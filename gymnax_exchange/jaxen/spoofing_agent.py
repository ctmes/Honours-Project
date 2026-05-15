"""
Observation-space spoofing adversary.

Perturbs the market maker's visible LOB depth without submitting real orders to the book.
The mid-price is never affected — this is an observation-space-only adversary.
"""

import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Tuple

from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxob.jaxob_config import SpoofingAgentConfig, World_EnvironmentConfig
from gymnax_exchange.jaxen.StatesandParams import (
    SpoofingAgentState, SpoofingAgentParams, WorldState
)
from gymnax.environments import spaces


class SpoofingAgent:
    """
    Observation-space adversary — never participates in the LOB.

    Interface mirrors MarketMakingAgent so MARLEnv can vmap it uniformly.
    The actual LOB-depth perturbation is applied in AdversarialMARLEnv.step_env
    *after* the LOB step. This class handles state tracking, cost computation,
    and obs/action space definitions.

    Observation (43-dim):
        - L2 top-10 bid/ask levels, interleaved [ask_p, ask_v, bid_p, bid_v] × 10  (40)
        - budget_remaining normalised by budget_per_episode                          (1)
        - best_bid_price normalised by 1e6                                           (1)
        - best_ask_price normalised by 1e6                                           (1)

    Action (10-dim continuous, Box [0, 1]):
        - Volumes to inject at top-5 bid levels (indices 0..4)
        - Volumes to inject at top-5 ask levels (indices 5..9)
        Scaled by budget at env level; clipped to remaining budget in get_messages.
    """

    def __init__(self, cfg: SpoofingAgentConfig, world_config: World_EnvironmentConfig):
        self.cfg = cfg
        self.world_config = world_config

    # ------------------------------------------------------------------
    # Params
    # ------------------------------------------------------------------

    def default_params(
        self,
        agent_config: SpoofingAgentConfig,
        trader_id_range_start: int,
        number_of_agents_per_type: int,
    ) -> Tuple[SpoofingAgentParams, int]:
        params = SpoofingAgentParams(
            budget_per_episode=jnp.full(
                (number_of_agents_per_type,), agent_config.budget_per_episode
            )
        )
        # No trader IDs consumed — return range_start unchanged
        return params, trader_id_range_start

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnames=("self", "num_msgs_per_step"))
    def reset_env(
        self,
        agent_param: SpoofingAgentParams,
        key: chex.PRNGKey,
        world_state: WorldState,
        num_msgs_per_step: int,
    ) -> Tuple[chex.Array, SpoofingAgentState]:
        state = SpoofingAgentState(
            budget_remaining=agent_param.budget_per_episode,
            volume_injected=jnp.zeros(()),
            prev_mm_reward=jnp.zeros(()),
            prev_detection_prob=jnp.zeros(()),
        )
        obs = self._build_obs(world_state, state)
        return obs, state

    # ------------------------------------------------------------------
    # Messages — adversary never submits LOB orders
    # ------------------------------------------------------------------

    def get_messages(
        self,
        action: jax.Array,          # shape (10,), values in [0, 1]
        world_state: WorldState,
        agent_state: SpoofingAgentState,
        agent_params: SpoofingAgentParams,
    ) -> Tuple[jax.Array, jax.Array, dict]:
        # Scale action to [0, budget_per_level] and clip to remaining budget
        budget_per_level = agent_state.budget_remaining / jnp.maximum(
            self.cfg.n_spoof_levels, 1
        )
        clipped_action = jnp.clip(action, 0.0, budget_per_level)
        volume_this_step = jnp.sum(clipped_action)

        empty_action = jnp.zeros((0, 8), dtype=jnp.int32)
        empty_cancel = jnp.zeros((0, 8), dtype=jnp.int32)
        extras = {
            "clipped_spoof_action": clipped_action,   # shape (10,) — used by AdversarialMARLEnv
            "volume_injected_step": volume_this_step,
        }
        return empty_action, empty_cancel, extras

    # ------------------------------------------------------------------
    # Reward — costs only; MM-reward component added in AdversarialMARLEnv
    # ------------------------------------------------------------------

    def get_reward(
        self,
        world_state: WorldState,
        agent_state: SpoofingAgentState,
        agent_params: SpoofingAgentParams,
        trades: jax.Array,
        bestasks: jax.Array,
        bestbids: jax.Array,
        ep_done_time: bool,
    ) -> Tuple[jax.Array, dict]:
        # Accidental fill probability proxy: injected vol / (total depth + 1)
        total_bid_depth = jnp.sum(jnp.where(world_state.bid_raw_orders[:, 0] > 0,
                                            world_state.bid_raw_orders[:, 1], 0))
        depth_pressure = agent_state.volume_injected / (total_bid_depth + 1.0)
        accidental_fills = self.cfg.c_fill * agent_state.volume_injected * depth_pressure
        regulatory_cost = self.cfg.c_reg * agent_state.volume_injected

        # Full adversary reward = -r_mm - costs. The -r_mm term is added by
        # AdversarialMARLEnv.step_env after this function returns.
        costs = accidental_fills + regulatory_cost
        r_adv_costs_only = -costs

        extras = {
            "reward": r_adv_costs_only,
            "accidental_fills": accidental_fills,
            "regulatory_cost": regulatory_cost,
            "costs_total": costs,
        }
        return r_adv_costs_only, extras

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update_state_and_get_done_and_info(
        self,
        new_world_state: WorldState,
        agent_state: SpoofingAgentState,
        extras: dict,
    ) -> Tuple[SpoofingAgentState, jax.Array, dict]:
        vol_step = extras.get("volume_injected_step", jnp.zeros(()))
        new_state = SpoofingAgentState(
            budget_remaining=jnp.maximum(0.0, agent_state.budget_remaining - vol_step),
            volume_injected=agent_state.volume_injected + vol_step,
            prev_mm_reward=extras.get("mm_reward_this_step", agent_state.prev_mm_reward),
            # prev_detection_prob is set by the training loop after each step, not here
            prev_detection_prob=agent_state.prev_detection_prob,
        )
        done = jnp.array(False)
        info = {
            "reward": extras.get("reward", jnp.zeros(())),
            "budget_remaining": new_state.budget_remaining,
            "volume_injected": new_state.volume_injected,
        }
        return new_state, done, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _build_obs(self, world_state: WorldState, agent_state: SpoofingAgentState) -> jax.Array:
        l2 = job.get_L2_state(
            world_state.ask_raw_orders,
            world_state.bid_raw_orders,
            10,
            self.world_config,
        )
        # Normalise L2 prices by 1e6, volumes by 1e4
        l2_norm = l2 / jnp.array(
            [1e6, 1e4, 1e6, 1e4] * 10, dtype=jnp.float32
        )
        budget_norm = agent_state.budget_remaining / jnp.maximum(
            self.cfg.budget_per_episode, 1.0
        )
        best_bid_norm = world_state.best_bids[-1, 0] / 1e6
        best_ask_norm = world_state.best_asks[-1, 0] / 1e6
        return jnp.concatenate([
            l2_norm,
            jnp.array([budget_norm, best_bid_norm, best_ask_norm]),
        ])

    def get_observation(
        self,
        world_state: WorldState,
        agent_state: SpoofingAgentState,
        agent_param: SpoofingAgentParams,
        total_messages,     # unused — signature must match MarketMakingAgent
        old_time,           # unused
        old_mid_price,      # unused
        lob_state_before,   # unused
        normalize: bool,
        flatten: bool,
    ) -> jax.Array:
        obs = self._build_obs(world_state, agent_state)
        return obs

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------

    def action_space(self):
        n = self.cfg.n_spoof_levels * 2  # 10 by default
        return spaces.Box(0.0, 1.0, (n,), dtype=jnp.float32)

    def observation_space(self):
        # 40 L2 features + 3 scalars (budget_norm, best_bid_norm, best_ask_norm)
        return spaces.Box(-1000.0, 1000.0, (43,), dtype=jnp.float32)

    def is_terminal(self, world_state: WorldState) -> bool:
        return False
