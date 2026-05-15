"""
Evaluation metrics for attack-aware market making.

Four primary metrics:
  1. Annualized Sharpe ratio (MM rewards)
  2. Detection F1 (DetectionHead vs oracle labels)
  3. Regime-stratified Sharpe (low-vol vs high-vol days)
  4. Adversary profitability (mean reward, Sharpe, profitable flag)

Usage:
    python evaluation/eval.py --checkpoint <path> --config <yaml_path>
              [--regime_labels data/regime_labels.json]
              [--window_to_date data/window_to_date.json]
              [--num_eval_episodes 200]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as oxcp
from flax.training.train_state import TrainState
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

def compute_sharpe(rewards: np.ndarray, steps_per_day: int = 6400, annualize: bool = True) -> float:
    """
    Compute (optionally annualized) Sharpe ratio from per-step rewards.

    Args:
        rewards:       1-D array of per-step rewards.
        steps_per_day: number of env steps in one trading day (used for annualization).
        annualize:     if True, scale by sqrt(252 * steps_per_day).

    Returns:
        Sharpe ratio as a float.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    mean_r = rewards.mean()
    std_r  = rewards.std() + 1e-10
    sharpe = mean_r / std_r
    if annualize:
        sharpe *= np.sqrt(252 * steps_per_day)
    return float(sharpe)


# ---------------------------------------------------------------------------
# Detection F1
# ---------------------------------------------------------------------------

def compute_detection_f1(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 for the spoofing detection head.

    Args:
        probs:     1-D array of DetectionHead outputs in [0, 1].
        labels:    1-D array of oracle binary labels {0, 1}.
        threshold: classification threshold.

    Returns:
        dict with keys: precision, recall, f1, accuracy.
    """
    probs  = np.asarray(probs,  dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    preds  = (probs >= threshold).astype(np.float32)

    tp = float(np.sum((preds == 1) & (labels == 1)))
    fp = float(np.sum((preds == 1) & (labels == 0)))
    fn = float(np.sum((preds == 0) & (labels == 1)))
    tn = float(np.sum((preds == 0) & (labels == 0)))

    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-10)

    return {
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "accuracy":  float(accuracy),
        "n_positive": int(tp + fn),
        "n_total":    int(len(labels)),
    }


# ---------------------------------------------------------------------------
# Regime-stratified Sharpe
# ---------------------------------------------------------------------------

def compute_regime_stratified_sharpe(
    rewards: np.ndarray,
    episode_regimes: np.ndarray,
    steps_per_day: int = 6400,
    annualize: bool = True,
) -> Dict[str, float]:
    """
    Compute per-regime annualized Sharpe ratio.

    Args:
        rewards:          1-D array of per-step rewards (all episodes concatenated).
        episode_regimes:  1-D array of length n_episodes; regime ∈ {0, 1} per episode.
        steps_per_day:    steps per episode (to split rewards into episodes).
        annualize:        if True, annualize the Sharpe.

    Returns:
        dict: {"low_vol": float, "high_vol": float, "all": float}
    """
    rewards         = np.asarray(rewards, dtype=np.float64)
    episode_regimes = np.asarray(episode_regimes, dtype=np.int32)
    n_episodes      = len(episode_regimes)
    ep_len          = len(rewards) // n_episodes

    ep_rewards = rewards[:n_episodes * ep_len].reshape(n_episodes, ep_len)

    low_vol_rewards  = ep_rewards[episode_regimes == 0].flatten()
    high_vol_rewards = ep_rewards[episode_regimes == 1].flatten()

    result = {"all": compute_sharpe(rewards, steps_per_day, annualize)}
    if len(low_vol_rewards) > 0:
        result["low_vol"]  = compute_sharpe(low_vol_rewards, steps_per_day, annualize)
    else:
        result["low_vol"]  = float("nan")
    if len(high_vol_rewards) > 0:
        result["high_vol"] = compute_sharpe(high_vol_rewards, steps_per_day, annualize)
    else:
        result["high_vol"] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Adversary profitability
# ---------------------------------------------------------------------------

def compute_adversary_profitability(adv_rewards: np.ndarray) -> Dict[str, float]:
    """
    Assess whether the adversary earns positive returns on average.

    Args:
        adv_rewards: 1-D array of per-step adversary rewards.

    Returns:
        dict: mean_reward, sharpe, profitable (bool), pct_positive_steps.
    """
    adv_rewards = np.asarray(adv_rewards, dtype=np.float64)
    mean_r      = float(adv_rewards.mean())
    sharpe      = compute_sharpe(adv_rewards, annualize=False)
    profitable  = mean_r > 0.0
    pct_pos     = float(np.mean(adv_rewards > 0.0))

    return {
        "mean_reward":       mean_r,
        "sharpe":            sharpe,
        "profitable":        profitable,
        "pct_positive_steps": pct_pos,
    }


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------

def run_evaluation(
    checkpoint_path: str,
    config: dict,
    num_eval_episodes: int = 200,
    regime_labels_path: str | None = None,
    window_to_date_path: str | None = None,
) -> Dict:
    """
    Load a checkpoint, run evaluation episodes, compute all metrics.

    Returns a dict with keys: sharpe, detection, regime_sharpe, adversary.
    """
    from gymnax_exchange.jaxob.jaxob_config import (
        MultiAgentConfig, World_EnvironmentConfig,
        CONFIG_OBJECT_DICT, AdversarialMMConfig, SpoofingAgentConfig,
    )
    from gymnax_exchange.jaxen.adversarial_marl_env import AdversarialMARLEnv
    from gymnax_exchange.jaxrl.MARL.attack_aware_policy import AttackAwarePolicyNet, AdversaryNet
    from dataclasses import fields

    # Rebuild env
    def _make_agent_configs(cfg):
        agent_configs = {}
        for agent_type, agent_cfg_dict in cfg.get("dict_of_agents_configs", {}).items():
            ac_class   = CONFIG_OBJECT_DICT[agent_type]
            fnames     = {f.name for f in fields(ac_class)}
            overrides  = {k: v for k, v in agent_cfg_dict.items() if k in fnames}
            agent_configs[agent_type] = ac_class(**overrides)
        return agent_configs

    agent_configs = _make_agent_configs(config)
    ma_config = MultiAgentConfig(
        number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"],
        dict_of_agents_configs=agent_configs,
        world_config=World_EnvironmentConfig(
            seed=config["SEED"],
            timePeriod=str(config.get("EvalTimePeriod", config["TimePeriod"])),
            **{k: v for k, v in config["world_config"].items()
               if hasattr(World_EnvironmentConfig(), k) and k not in ["seed", "timePeriod"]},
        ),
    )

    regime_labels = {}
    window_to_date = None
    if regime_labels_path:
        with open(regime_labels_path) as f:
            regime_labels = json.load(f)
    if window_to_date_path:
        with open(window_to_date_path) as f:
            window_to_date = {int(k): v for k, v in json.load(f).items()}

    env = AdversarialMARLEnv(
        key=jax.random.PRNGKey(config["SEED"]),
        multi_agent_config=ma_config,
        regime_labels=regime_labels,
        window_to_date=window_to_date,
    )
    mm_idx  = env._mm_idx
    adv_idx = env._adv_idx

    # Rebuild networks and load params from checkpoint
    networks = []
    for i, cfg_i in enumerate(env.list_of_agents_configs):
        act_space = env.action_spaces[i]
        if isinstance(cfg_i, AdversarialMMConfig):
            net = AttackAwarePolicyNet(action_dim=act_space.n, config=config)
        else:
            net = AdversaryNet(action_dim=act_space.shape[0], config=config)
        networks.append(net)

    orbax_checkpointer = oxcp.PyTreeCheckpointer()
    raw_ckpt = orbax_checkpointer.restore(checkpoint_path)
    train_state_params = raw_ckpt["model"]   # list of TrainState (or just params)

    env_params = env.default_params
    rng = jax.random.PRNGKey(0)

    mm_rewards_all   = []
    adv_rewards_all  = []
    det_probs_all    = []
    adv_labels_all   = []
    episode_regimes  = []

    ep_len = config.get("NUM_STEPS", 6400)

    for ep_i in range(num_eval_episodes):
        rng, reset_rng = jax.random.split(rng)
        obs_list, env_state = env.reset(reset_rng, env_params)

        mm_rew_ep  = []
        adv_rew_ep = []
        det_ep     = []
        lbl_ep     = []

        h_states = [
            net.initialize_carry(1, config.get("GRU_HIDDEN_DIM", 256))
            for net in networks
        ]
        dones = [jnp.zeros((1,), dtype=bool) for _ in networks]

        for step in range(ep_len):
            rng, _rng = jax.random.split(rng)
            actions = []
            new_h_states = []

            for i, net in enumerate(networks):
                obs_i = obs_list[i][jnp.newaxis, :]   # (1, obs_dim)
                ac_in = (obs_i[jnp.newaxis, :], dones[i][jnp.newaxis, :])
                h_new, pi, _, det_p = train_state_params[i].apply_fn(
                    train_state_params[i].params, h_states[i], ac_in
                )
                new_h_states.append(h_new)
                action = pi.sample(seed=_rng)
                if i == adv_idx:
                    action = action.reshape(
                        env.multi_agent_config.number_of_agents_per_type[i], -1
                    )
                else:
                    action = action.reshape(
                        env.multi_agent_config.number_of_agents_per_type[i],
                    )
                actions.append(action)
                if i == mm_idx:
                    det_ep.append(float(jnp.mean(det_p)))

            h_states = new_h_states

            rng, step_rng = jax.random.split(rng)
            obs_list, env_state, rewards, done, info = env.step(
                step_rng, env_state, actions, env_params
            )

            mm_rew_ep.append(float(jnp.mean(rewards[mm_idx])))
            adv_rew_ep.append(float(jnp.mean(rewards[adv_idx])))
            lbl_ep.append(float(info.get("adv_label", 0.0)))

            if done["__all__"]:
                break

        mm_rewards_all.extend(mm_rew_ep)
        adv_rewards_all.extend(adv_rew_ep)
        det_probs_all.extend(det_ep)
        adv_labels_all.extend(lbl_ep)

        regime_val = float(env._regime_array[env_state.world_state.window_index])
        episode_regimes.append(int(regime_val))

    mm_rewards_arr  = np.array(mm_rewards_all,  dtype=np.float64)
    adv_rewards_arr = np.array(adv_rewards_all, dtype=np.float64)
    det_probs_arr   = np.array(det_probs_all,   dtype=np.float32)
    adv_labels_arr  = np.array(adv_labels_all,  dtype=np.float32)
    regime_arr      = np.array(episode_regimes, dtype=np.int32)

    results = {
        "sharpe":         compute_sharpe(mm_rewards_arr, steps_per_day=ep_len),
        "detection":      compute_detection_f1(det_probs_arr, adv_labels_arr),
        "regime_sharpe":  compute_regime_stratified_sharpe(
                              mm_rewards_arr, regime_arr, steps_per_day=ep_len
                          ),
        "adversary":      compute_adversary_profitability(adv_rewards_arr),
        "n_episodes":     num_eval_episodes,
    }
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate adversarial MM checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to orbax checkpoint directory")
    parser.add_argument("--config",     required=True, help="Path to YAML RL config")
    parser.add_argument("--regime_labels",   default=None, help="Path to regime_labels.json")
    parser.add_argument("--window_to_date",  default=None, help="Path to window_to_date.json")
    parser.add_argument("--num_eval_episodes", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    from omegaconf import OmegaConf
    raw_cfg = OmegaConf.load(args.config)
    cfg     = OmegaConf.to_container(raw_cfg, resolve=True)

    results = run_evaluation(
        checkpoint_path=args.checkpoint,
        config=cfg,
        num_eval_episodes=args.num_eval_episodes,
        regime_labels_path=args.regime_labels,
        window_to_date_path=args.window_to_date,
    )

    print("\n=== Evaluation Results ===")
    print(f"MM Sharpe (annualized):   {results['sharpe']:.4f}")
    print(f"Detection F1:             {results['detection']['f1']:.4f}")
    print(f"  Precision:              {results['detection']['precision']:.4f}")
    print(f"  Recall:                 {results['detection']['recall']:.4f}")
    print(f"Regime Sharpe (low-vol):  {results['regime_sharpe']['low_vol']:.4f}")
    print(f"Regime Sharpe (high-vol): {results['regime_sharpe']['high_vol']:.4f}")
    print(f"Adversary profitable:     {results['adversary']['profitable']}")
    print(f"Adversary mean reward:    {results['adversary']['mean_reward']:.6f}")
    print(f"Adversary Sharpe:         {results['adversary']['sharpe']:.4f}")
    print(f"Episodes evaluated:       {results['n_episodes']}")
