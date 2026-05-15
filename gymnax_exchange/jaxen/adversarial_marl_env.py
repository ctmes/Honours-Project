"""
AdversarialMARLEnv — extends MARLEnv with three thesis contributions:
  1. Observation-space spoofing adversary (SpoofingAgent)
  2. Oracle adversarial labels passed to the MM's info dict
  3. Volatility-regime conditioning fed to the MM's observation

Architecture
------------
The parent MARLEnv.step_env handles all LOB processing. After the parent
returns, this class:
  a) Applies the adversary's depth perturbation to the MM's visible L2 obs
  b) Injects the oracle label and regime indicator into info
  c) Corrects the adversary's reward with the -r_mm term
  d) Rebuilds the MM's observation using get_adversarial_observation

The actual LOB state is never modified — only the observation returned to the MM.

Regime array
------------
regime_labels: dict mapping date strings ("2024-08-05") → int (0 or 1).
Mapped to window indices at construction time and stored as a static JAX array.
"""

import json
import dataclasses
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import chex
from jax import vmap

from gymnax_exchange.jaxen.marl_env import MARLEnv
from gymnax_exchange.jaxen.StatesandParams import (
    MultiAgentState, MultiAgentParams, SpoofingAgentState
)
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxob.jaxob_config import (
    MultiAgentConfig, SpoofingAgentConfig, AdversarialMMConfig
)


class AdversarialMARLEnv(MARLEnv):
    """
    MARLEnv subclass for adversarial co-training.

    Expects exactly two agent types in multi_agent_config:
      - index 0 or 1: AdversarialMMConfig  (market maker with detection head)
      - index 0 or 1: SpoofingAgentConfig  (observation-space adversary)
    """

    def __init__(
        self,
        key,
        multi_agent_config: MultiAgentConfig,
        regime_labels: dict,         # {"2024-01-02": 0, "2024-08-05": 1, ...}
        window_to_date: dict = None, # {window_idx: "2024-01-02", ...} — if None, all regime=0
    ):
        super().__init__(key, multi_agent_config)

        # Identify MM and adversary indices in instance_list
        self._mm_idx = None
        self._adv_idx = None
        for i, cfg in enumerate(self.list_of_agents_configs):
            if isinstance(cfg, AdversarialMMConfig):
                self._mm_idx = i
            elif isinstance(cfg, SpoofingAgentConfig):
                self._adv_idx = i
        assert self._mm_idx is not None, "AdversarialMMConfig agent not found in multi_agent_config"
        assert self._adv_idx is not None, "SpoofingAgentConfig agent not found in multi_agent_config"

        # Build static regime array: shape (n_windows,) float32
        self._regime_array = self._build_regime_array(
            regime_labels, window_to_date
        )

    # ------------------------------------------------------------------
    # Regime array construction (Python-level, not JIT-traced)
    # ------------------------------------------------------------------

    def _build_regime_array(self, regime_labels: dict, window_to_date: dict) -> jax.Array:
        """Map regime labels to window indices and store as static JAX array."""
        n_windows = len(self.base_env.start_indeces)
        arr = jnp.zeros(n_windows, dtype=jnp.float32)
        if window_to_date is None:
            return arr  # all low-vol by default
        for win_idx, date_str in window_to_date.items():
            label = float(regime_labels.get(date_str, 0))
            arr = arr.at[win_idx].set(label)
        return arr

    # ------------------------------------------------------------------
    # Step — override to inject adversarial extras after LOB step
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: MultiAgentState,
        actions,           # list[jnp.ndarray], one per agent type
        params: MultiAgentParams,
    ):
        """
        Steps:
        1. Parent handles LOB processing, rewards, and state update (phases A–K)
        2. We replace the MM's observation with the adversarial version
        3. We correct the adversary's reward with the -r_mm component
        4. We add oracle labels and regime to info
        """
        # Read adversary action and clip to budget (before parent call so we have it)
        adv_cfg = self.list_of_agents_configs[self._adv_idx]
        adv_action_raw = actions[self._adv_idx]  # shape (n_adv, 10) — may have leading batch dim
        adv_state_before = state.agent_states[self._adv_idx]

        # Handle potential batch dimension (n_agents=1 case → squeeze to (10,))
        adv_action_1d = jnp.squeeze(adv_action_raw, axis=0) if adv_action_raw.ndim > 1 else adv_action_raw
        budget_per_level = adv_state_before.budget_remaining / jnp.maximum(adv_cfg.n_spoof_levels, 1)
        # budget_remaining is shape (1,) when vmapped over 1 agent — take scalar
        budget_per_level_scalar = jnp.squeeze(budget_per_level)
        clipped_adv_action = jnp.clip(adv_action_1d, 0.0, budget_per_level_scalar)

        # ---- Call parent (all LOB logic happens here) ----
        obs_list, new_state, reward_list, dones, info = super().step_env(
            key, state, actions, params
        )

        # ---- Apply spoof perturbation to MM's visible L2 ----
        # L2 layout from get_L2_state: [ask_p_k, ask_v_k, bid_p_k, bid_v_k] × 10
        # Ask vol indices (top-5): 1,  5,  9, 13, 17
        # Bid vol indices (top-5): 3,  7, 11, 15, 19
        true_l2 = job.get_L2_state(
            new_state.world_state.ask_raw_orders,
            new_state.world_state.bid_raw_orders,
            10,
            self.multi_agent_config.world_config,
        )
        ask_vol_idx = jnp.array([1, 5, 9, 13, 17])
        bid_vol_idx = jnp.array([3, 7, 11, 15, 19])
        perturbed_l2 = true_l2.at[ask_vol_idx].add(clipped_adv_action[5:])
        perturbed_l2 = perturbed_l2.at[bid_vol_idx].add(clipped_adv_action[:5])

        # ---- Oracle adversarial label ----
        adv_label = (jnp.sum(clipped_adv_action) > 0.0).astype(jnp.float32)

        # ---- Regime indicator for current window ----
        regime = self._regime_array[new_state.world_state.window_index]

        # ---- prev_detection_prob (stored in adversary state by training loop) ----
        prev_det = jnp.squeeze(adv_state_before.prev_detection_prob)

        # ---- Rebuild MM observation with adversarial extras ----
        mm_instance = self.instance_list[self._mm_idx]
        new_mm_state = new_state.agent_states[self._mm_idx]
        mm_agent_param = params.agent_params[self._mm_idx]

        # vmap over n_mm_agents (typically 1)
        def build_mm_obs_single(mm_state_single, mm_param_single):
            return mm_instance.get_adversarial_observation(
                new_state.world_state,
                mm_state_single,
                mm_param_single,
                perturbed_l2,
                regime,
                prev_det,
                normalize=True,
            )

        new_mm_obs = vmap(build_mm_obs_single)(new_mm_state, mm_agent_param)
        obs_list[self._mm_idx] = new_mm_obs

        # ---- Correct adversary reward: add -r_mm component ----
        mm_reward = reward_list[self._mm_idx]  # shape (n_mm_agents,)
        mm_reward_scalar = jnp.squeeze(mm_reward)
        adv_costs = info["agents"][self._adv_idx].get("costs_total", jnp.zeros(()))
        adv_reward_corrected = jnp.expand_dims(-mm_reward_scalar - adv_costs, axis=0)
        reward_list[self._adv_idx] = adv_reward_corrected

        # ---- Update adversary state with mm reward this step ----
        old_adv_states = new_state.agent_states[self._adv_idx]
        new_adv_states = old_adv_states.replace(
            prev_mm_reward=mm_reward_scalar,
        )
        new_agent_states = list(new_state.agent_states)
        new_agent_states[self._adv_idx] = new_adv_states
        new_state = new_state.replace(agent_states=new_agent_states)

        # ---- Augment info ----
        info["adv_label"] = adv_label
        info["regime"] = regime
        info["volume_injected_step"] = jnp.sum(clipped_adv_action)

        return obs_list, new_state, reward_list, dones, info

    # ------------------------------------------------------------------
    # Reset — initialise MM obs with zero detection prob and correct regime
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key: chex.PRNGKey, params: MultiAgentParams):
        obs_list, state = super().reset_env(key, params)

        # Regime for this episode's window
        regime = self._regime_array[state.world_state.window_index]

        # Build proper MM obs (perturbed_l2 = true L2 at reset, no injection yet)
        true_l2 = job.get_L2_state(
            state.world_state.ask_raw_orders,
            state.world_state.bid_raw_orders,
            10,
            self.multi_agent_config.world_config,
        )
        mm_instance = self.instance_list[self._mm_idx]
        new_mm_state = state.agent_states[self._mm_idx]
        mm_agent_param = params.agent_params[self._mm_idx]

        def build_mm_obs_single(mm_state_single, mm_param_single):
            return mm_instance.get_adversarial_observation(
                state.world_state,
                mm_state_single,
                mm_param_single,
                true_l2,
                regime,
                jnp.zeros(()),   # prev_detection_prob = 0 at start of episode
                normalize=True,
            )

        new_mm_obs = vmap(build_mm_obs_single)(new_mm_state, mm_agent_param)
        obs_list[self._mm_idx] = new_mm_obs

        return obs_list, state

    # ------------------------------------------------------------------
    # Helper: load regime labels from JSON (called before creating env)
    # ------------------------------------------------------------------

    @staticmethod
    def load_regime_labels(json_path: str) -> dict:
        with open(json_path) as f:
            return json.load(f)
