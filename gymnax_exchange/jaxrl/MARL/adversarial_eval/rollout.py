"""
Rollout layer for the adversarial market-making evaluation.

Loads a trained adversarial checkpoint, runs deterministic eval episodes under a
forced attack-on or attack-off regime, and emits the per-seed metric arrays the
metrics/stats/aggregate layers consume.

Design choices (see harness notes):
  - Deterministic policies at eval: pi.mode() for both MM (argmax) and adversary (mean).
  - Attack-on / attack-off forced via the telegraph config (on_prob/off_prob) AND by
    overriding the initial attack_active after reset, so an episode is cleanly all-on
    or all-off with no first-step contamination.
  - Returns = MM info "reward_delta_pv" (per-step portfolio-value change).
  - One model -> n_envs parallel eval episodes; per-episode metrics are averaged to a
    single value per model (one "seed"); detection AUROC is pooled across the rollout.

This module is import-light (no wandb / training-script side effects); it reconstructs
the env, networks, and TrainState templates the same way the training loop does so an
orbax checkpoint restores cleanly.
"""

from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import functools
from dataclasses import fields

import numpy as np
import jax
import jax.numpy as jnp
import optax
from omegaconf import OmegaConf
from flax.training.train_state import TrainState
import orbax.checkpoint as oxcp

from gymnax_exchange.jaxob.config_io import load_config_from_file
from gymnax_exchange.jaxob.jaxob_config import (
    MultiAgentConfig, World_EnvironmentConfig, CONFIG_OBJECT_DICT,
    SpoofingAgentConfig, AdversarialMMConfig,
)
from gymnax_exchange.jaxen.adversarial_marl_env import AdversarialMARLEnv
from gymnax_exchange.jaxrl.MARL.attack_aware_policy import AttackAwarePolicyNet, AdversaryNet

from gymnax_exchange.jaxrl.MARL.adversarial_eval import metrics as M

_YAML = "config/rl_configs/ippo_adversarial.yaml"


# --------------------------------------------------------------------------- config
def create_agent_configs(config: dict) -> dict:
    """Build agent-config dataclasses from the merged config (mirrors training loop)."""
    agent_configs = {}
    for agent_type, agent_cfg_dict in config.get("dict_of_agents_configs", {}).items():
        cls = CONFIG_OBJECT_DICT[agent_type]
        field_names = {f.name for f in fields(cls)}
        overrides = {k: v for k, v in agent_cfg_dict.items() if k in field_names}
        agent_configs[agent_type] = cls(**overrides)
    return agent_configs


def load_merged_config(yaml_path: str = _YAML) -> dict:
    """Reproduce the train script's yaml + env-JSON merge, without hydra/wandb."""
    cfg = OmegaConf.load(yaml_path)
    env_config = load_config_from_file(cfg["ENV_CONFIG"])
    env_config = OmegaConf.structured(env_config)
    merged = OmegaConf.merge(cfg, env_config)
    return OmegaConf.to_container(merged, resolve=True)


def set_attack_mode(config: dict, attack_mode: str) -> dict:
    """Configure the telegraph gate for evaluation.

    'on'/'off' force the gate fully on/off (for the PnL/risk metrics, computed over
    attack-on vs attack-off windows). 'mixed' leaves the configured telegraph
    probabilities so both classes appear — required for the detection AUROC.
    """
    spoof = config["dict_of_agents_configs"]["Spoofing"]
    if attack_mode == "on":
        spoof["attack_on_prob"], spoof["attack_off_prob"] = 1.0, 0.0
    elif attack_mode == "off":
        spoof["attack_on_prob"], spoof["attack_off_prob"] = 0.0, 1.0
    elif attack_mode == "mixed":
        pass  # keep configured intermittent schedule
    else:
        raise ValueError(f"attack_mode must be 'on', 'off', or 'mixed', got {attack_mode!r}")
    return config


# --------------------------------------------------------------------------- build
def build_eval(config: dict, n_envs: int):
    """Construct env, networks, and restorable TrainState templates (mirrors make_train)."""
    config = dict(config)
    config["NUM_ENVS"] = n_envs
    init_key = jax.random.PRNGKey(config["SEED"])

    agent_configs = create_agent_configs(config)
    ma_config = MultiAgentConfig(
        number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"],
        dict_of_agents_configs=agent_configs,
        world_config=World_EnvironmentConfig(
            seed=config["SEED"], timePeriod=str(config["TimePeriod"]),
            **{k: v for k, v in config["world_config"].items()
               if hasattr(World_EnvironmentConfig(), k) and k not in ["seed", "timePeriod"]},
        ),
    )
    env = AdversarialMARLEnv(key=init_key, multi_agent_config=ma_config,
                             regime_labels={}, window_to_date=None)

    config["NUM_ACTORS_PERTYPE"] = [n * n_envs for n in config["NUM_AGENTS_PER_TYPE"]]
    config["NUM_UPDATES"] = max(1, int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // n_envs))

    def linear_schedule(lr, count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return lr * frac

    networks, train_states = [], []
    for i, instance in enumerate(env.instance_list):
        cfg_i = env.list_of_agents_configs[i]
        act_space = env.action_spaces[i]
        if isinstance(cfg_i, AdversarialMMConfig):
            net = AttackAwarePolicyNet(action_dim=act_space.n, config=config)
        elif isinstance(cfg_i, SpoofingAgentConfig):
            net = AdversaryNet(action_dim=act_space.shape[0], config=config)
        else:
            raise ValueError(f"Unexpected agent config: {type(cfg_i)}")
        networks.append(net)

        obs_dim = env.observation_spaces[i].shape[0]
        init_key, sub = jax.random.split(init_key)
        init_x = (jnp.zeros((1, n_envs, obs_dim)), jnp.zeros((1, n_envs)))
        init_h = net.initialize_carry(n_envs, config["GRU_HIDDEN_DIM"])
        params = net.init(sub, init_h, init_x)

        if config["ANNEAL_LR"][i]:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"][i]),
                             optax.adam(learning_rate=functools.partial(linear_schedule, config["LR"][i]), eps=1e-5))
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"][i]),
                             optax.adam(config["LR"][i], eps=1e-5))
        train_states.append(TrainState.create(apply_fn=net.apply, params=params, tx=tx))

    return env, networks, train_states, config


def restore_checkpoint(config, train_states, project: str, run_name: str = "local_run", step=None):
    """Restore train_states from an orbax checkpoint (matching the saved {model,metrics} tree)."""
    ckpt_dir = (f'{config["world_config"]["alphatradePath"]}/checkpoints/MARLCheckpoints'
                f'/{project}/{run_name}')
    mgr = oxcp.CheckpointManager(ckpt_dir, oxcp.PyTreeCheckpointer())
    if step is None:
        step = mgr.latest_step()
    if step is None:
        raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")
    restored = mgr.restore(step, items={
        "model": train_states,
        "metrics": {"avg_reward": [0.0 for _ in train_states]},
    })
    return restored["model"], step


# --------------------------------------------------------------------------- rollout
def _per_env(x, n_envs):
    return np.asarray(x, dtype=np.float64).reshape(n_envs, -1).mean(axis=1)


def run_rollout(env, networks, train_states, config, attack_mode, rng, n_envs, n_steps):
    """Deterministic rollout under a forced attack regime; returns per-step arrays."""
    mm_idx, adv_idx = env._mm_idx, env._adv_idx
    nper = env.multi_agent_config.number_of_agents_per_type
    env_params = env.default_params

    rng, rk = jax.random.split(rng)
    obs_list, env_state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(rk, n_envs), env_params)

    # Force the initial attack gate to match the eval regime (on/off only;
    # 'mixed' keeps the reset value so the telegraph produces both classes).
    if attack_mode in ("on", "off"):
        init_gate = 1.0 if attack_mode == "on" else 0.0
        aa = env_state.agent_states[adv_idx].attack_active
        ags = list(env_state.agent_states)
        ags[adv_idx] = env_state.agent_states[adv_idx].replace(attack_active=jnp.full_like(aa, init_gate))
        env_state = env_state.replace(agent_states=ags)

    h_states = [networks[i].initialize_carry(n_envs * nper[i], config["GRU_HIDDEN_DIM"])
                for i in range(len(networks))]
    dones = [jnp.zeros((n_envs * nper[i],), dtype=bool) for i in range(len(networks))]

    series = {"ret": [], "inventory": [], "det_prob": [], "adv_label": []}

    for _ in range(n_steps):
        actions, det_mm = [], None
        for i, ts in enumerate(train_states):
            obs_i = obs_list[i].reshape((n_envs * nper[i], -1))
            ac_in = (obs_i[jnp.newaxis, :], dones[i][jnp.newaxis, :])
            h_new, pi, _, det_prob = ts.apply_fn(ts.params, h_states[i], ac_in)
            h_states[i] = h_new
            if i == mm_idx:
                det_mm = det_prob[0]                      # (n_actors,)
            action = pi.mode()                            # deterministic
            action = action.reshape((n_envs, nper[i], -1))
            actions.append(action.squeeze(-2) if action.ndim > 2 else action.squeeze())

        rng, sk = jax.random.split(rng)
        obs_list, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(sk, n_envs), env_state, actions, env_params)

        # Feed MM detection prob back into adversary state (mirrors training loop).
        mm_det = det_mm.reshape(n_envs, -1)[:, :1]
        ags = list(env_state.agent_states)
        ags[adv_idx] = env_state.agent_states[adv_idx].replace(prev_detection_prob=mm_det)
        env_state = env_state.replace(agent_states=ags)

        dones = [done["agents"][i].reshape(n_envs * nper[i]) for i in range(len(networks))]

        series["ret"].append(_per_env(info["agents"][mm_idx]["reward_delta_pv"], n_envs))
        series["inventory"].append(_per_env(info["agents"][mm_idx]["inventory"], n_envs))
        series["det_prob"].append(np.asarray(det_mm).reshape(n_envs, -1).mean(axis=1))
        series["adv_label"].append(np.asarray(info["adv_label"]).reshape(-1))

    return {k: np.stack(v, axis=0) for k, v in series.items()}   # each (n_steps, n_envs)


def rollout_metrics(arrays, periods_per_year):
    """Reduce per-step rollout arrays to a single metric dict for this model/seed.

    Risk/behavioural metrics are computed per eval episode (env) then averaged;
    detection AUROC is pooled across the whole rollout (needs both classes).
    """
    ret, inv = arrays["ret"], arrays["inventory"]          # (T, n_envs)
    n_envs = ret.shape[1]
    sharpe = np.nanmean([M.sharpe_ratio(ret[:, e], periods_per_year) for e in range(n_envs)])
    sortino = np.nanmean([M.sortino_ratio(ret[:, e], periods_per_year) for e in range(n_envs)])
    cvar = np.nanmean([M.cvar(ret[:, e], 0.10) for e in range(n_envs)])
    peak_inv = np.nanmean([M.peak_inventory_excursion(inv[:, e]) for e in range(n_envs)])
    auroc = M.detection_auroc(arrays["det_prob"].ravel(), arrays["adv_label"].ravel())
    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "cvar": float(cvar),
        "peak_inventory": float(peak_inv),
        "auroc": float(auroc),
        "mean_attack_rate": float(arrays["adv_label"].mean()),
    }


def evaluate_checkpoint(project, run_name="local_run", n_envs=8, n_steps=64,
                        periods_per_year=98280.0, seed=0, step=None, yaml_path=_YAML):
    """Convenience: load a checkpoint and return {attack_mode -> metric dict}."""
    out = {}
    # 'on'/'off' give the attack-on/attack-off PnL & risk metrics; 'mixed' gives a
    # both-classes stream for the detection AUROC (single-class rollouts -> AUROC nan).
    for mode in ("on", "off", "mixed"):
        cfg = set_attack_mode(load_merged_config(yaml_path), mode)
        env, nets, ts_template, cfg = build_eval(cfg, n_envs)
        ts, used_step = restore_checkpoint(cfg, ts_template, project, run_name, step)
        arrays = run_rollout(env, nets, ts, cfg, mode, jax.random.PRNGKey(seed), n_envs, n_steps)
        out[mode] = rollout_metrics(arrays, periods_per_year)
        out[mode]["_checkpoint_step"] = used_step
    return out
