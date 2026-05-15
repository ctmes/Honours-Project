"""
Adversarial IPPO training loop.

Alternating-freeze co-training between:
  - Market Maker (MM): AttackAwarePolicyNet (SharedEncoder + PolicyHead + DetectionHead)
  - Spoofing Adversary: AdversaryNet (continuous Gaussian MLP)

Phase alternation (Python-level, NOT inside lax.scan):
  - Phase 0 (update_i // FREEZE_ALTERNATION is even):  update adversary only (PPO)
  - Phase 1 (update_i // FREEZE_ALTERNATION is odd):   update MM only (PCGrad PPO+BCE)

The env is AdversarialMARLEnv, which handles observation-space spoofing,
oracle adversarial labels, and volatility-regime conditioning internally.
"""

import os
import gc
import sys
import json
import time
import logging
import datetime
import functools
from dataclasses import fields
from typing import NamedTuple, Any, Dict, Sequence

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

logging.getLogger("orbax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import distrax
import wandb
import wandb.sdk
import hydra
import orbax.checkpoint as oxcp
from omegaconf import DictConfig, OmegaConf
from flax.training.train_state import TrainState
from flax.training import orbax_utils

from gymnax_exchange.jaxob.config_io import load_config_from_file, save_config_to_file
from gymnax_exchange.jaxob.jaxob_config import (
    MultiAgentConfig,
    World_EnvironmentConfig,
    CONFIG_OBJECT_DICT,
    SpoofingAgentConfig,
    AdversarialMMConfig,
)
from gymnax_exchange.jaxen.adversarial_marl_env import AdversarialMARLEnv
from gymnax_exchange.jaxrl.MARL.attack_aware_policy import AttackAwarePolicyNet, AdversaryNet
from gymnax_exchange.jaxrl.MARL.pcgrad import pcgrad_merge


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    adv_label: jnp.ndarray   # oracle spoofing label — for MM detection BCE loss
    info: Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def batchify(x: jnp.ndarray, num_actors: int):
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, num_envs: int, num_agents: int):
    return x.reshape((num_envs, num_agents, -1))


def create_agent_configs(config: dict) -> dict:
    agent_configs = {}
    if "AGENT_CONFIGS" in config:
        for agent_type, agent_cfg in config["AGENT_CONFIGS"].items():
            agent_config_class = CONFIG_OBJECT_DICT[agent_type]
            config_overrides = {}
            field_names = {f.name for f in fields(agent_config_class)}
            for key, value in config["dict_of_agents_configs"].items():
                if isinstance(value, dict) and key == agent_type:
                    for k, v in value.items():
                        if k in field_names:
                            config_overrides[k] = v
            sweep_overrides = {k: v for k, v in agent_cfg.items()}
            all_overrides = {**config_overrides, **sweep_overrides}
            agent_configs[agent_type] = agent_config_class(**all_overrides)
    else:
        for agent_type, agent_cfg_dict in config.get("dict_of_agents_configs", {}).items():
            agent_config_class = CONFIG_OBJECT_DICT[agent_type]
            field_names = {f.name for f in fields(agent_config_class)}
            overrides = {k: v for k, v in agent_cfg_dict.items() if k in field_names}
            agent_configs[agent_type] = agent_config_class(**overrides)
    return agent_configs


# ---------------------------------------------------------------------------
# Main make_train
# ---------------------------------------------------------------------------

def make_train(config: dict):
    init_key = jax.random.PRNGKey(config["SEED"])
    agent_configs = create_agent_configs(config)

    ma_config = MultiAgentConfig(
        number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"],
        dict_of_agents_configs=agent_configs,
        world_config=World_EnvironmentConfig(
            seed=config["SEED"],
            timePeriod=str(config["TimePeriod"]),
            **{k: v for k, v in config["world_config"].items()
               if hasattr(World_EnvironmentConfig(), k) and k not in ["seed", "timePeriod"]},
        ),
    )

    # Load regime labels
    regime_labels = {}
    window_to_date = None
    if config.get("REGIME_LABELS_PATH"):
        regime_labels = AdversarialMARLEnv.load_regime_labels(config["REGIME_LABELS_PATH"])
    if config.get("WINDOW_TO_DATE_PATH"):
        with open(config["WINDOW_TO_DATE_PATH"]) as f:
            # JSON keys are strings; convert to int
            window_to_date = {int(k): v for k, v in json.load(f).items()}

    env = AdversarialMARLEnv(
        key=init_key,
        multi_agent_config=ma_config,
        regime_labels=regime_labels,
        window_to_date=window_to_date,
    )

    mm_idx  = env._mm_idx
    adv_idx = env._adv_idx

    config["NUM_ACTORS_PERTYPE"] = [
        n * config["NUM_ENVS"] for n in config["NUM_AGENTS_PER_TYPE"]
    ]
    config["NUM_ACTORS_TOTAL"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZES"] = [
        nact * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
        for nact in config["NUM_ACTORS_PERTYPE"]
    ]

    freeze_n     = config.get("FREEZE_ALTERNATION", 10)
    detect_coef  = config.get("DETECTION_LOSS_COEF", 0.5)
    pcgrad_on    = config.get("PCGRAD_ENABLED", True)

    def linear_schedule(lr, count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return lr * frac

    # -----------------------------------------------------------------------
    def train(rng, run: wandb.sdk.wandb_run.Run = None):

        # ---- Init networks and train states --------------------------------
        networks = []
        for i, instance in enumerate(env.instance_list):
            cfg_i = env.list_of_agents_configs[i]
            obs_dim  = env.observation_spaces[i].shape[0]
            act_space = env.action_spaces[i]

            if isinstance(cfg_i, AdversarialMMConfig):
                net = AttackAwarePolicyNet(action_dim=act_space.n, config=config)
            elif isinstance(cfg_i, SpoofingAgentConfig):
                net = AdversaryNet(action_dim=act_space.shape[0], config=config)
            else:
                raise ValueError(f"Unexpected agent config type: {type(cfg_i)}")
            networks.append(net)

        hstates           = []
        train_states      = []
        init_dones_agents = []

        for i, net in enumerate(networks):
            obs_dim = env.observation_spaces[i].shape[0]
            rng, _rng = jax.random.split(rng)

            init_x = (
                jnp.zeros((1, config["NUM_ENVS"], obs_dim)),
                jnp.zeros((1, config["NUM_ENVS"])),
            )
            init_hstate_single = networks[i].initialize_carry(
                config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
            )
            network_params = net.init(_rng, init_hstate_single, init_x)

            if config["ANNEAL_LR"][i]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"][i]),
                    optax.adam(
                        learning_rate=functools.partial(linear_schedule, config["LR"][i]),
                        eps=1e-5,
                    ),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"][i]),
                    optax.adam(config["LR"][i], eps=1e-5),
                )

            train_state = TrainState.create(
                apply_fn=net.apply,
                params=network_params,
                tx=tx,
            )

            init_hstate = networks[i].initialize_carry(
                config["NUM_ACTORS_PERTYPE"][i], config["GRU_HIDDEN_DIM"]
            )
            hstates.append(init_hstate)
            train_states.append(train_state)
            init_dones_agents.append(
                jnp.zeros((config["NUM_ACTORS_PERTYPE"][i],), dtype=bool)
            )

        # ---- Init env ------------------------------------------------------
        rng, _rng = jax.random.split(rng)
        reset_rng  = jax.random.split(_rng, config["NUM_ENVS"])
        env_params = env.default_params
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # ---- Inner: trajectory collection step -----------------------------
        def _env_step(runner_state, unused):
            train_states, env_state, last_obs, last_done, h_states, rng = runner_state

            rng, _rng = jax.random.split(rng)

            actions   = []
            values    = []
            log_probs = []
            det_probs = []   # detection probabilities from MM network

            for i, train_state in enumerate(train_states):
                obs_i = batchify(last_obs[i], config["NUM_ACTORS_PERTYPE"][i])
                ac_in = (
                    obs_i[jnp.newaxis, :],
                    last_done[i][jnp.newaxis, :],
                )
                h_new, pi, value, det_prob = train_state.apply_fn(
                    train_state.params, h_states[i], ac_in
                )
                h_states[i] = h_new
                values.append(value)
                det_probs.append(det_prob)

                action = pi.sample(seed=_rng)
                log_probs.append(pi.log_prob(action))
                action = unbatchify(
                    action, config["NUM_ENVS"],
                    env.multi_agent_config.number_of_agents_per_type[i],
                )
                actions.append(action.squeeze(-2) if action.ndim > 2 else action.squeeze())

            # Step env
            rng, _rng = jax.random.split(rng)
            rng_step  = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, actions, env_params)

            # Update prev_detection_prob in env_state from MM's det_prob this step
            mm_det = jnp.mean(det_probs[mm_idx])   # scalar: mean over actors
            old_adv_s = env_state.agent_states[adv_idx]
            new_adv_s = old_adv_s.replace(prev_detection_prob=mm_det)
            new_agent_states = list(env_state.agent_states)
            new_agent_states[adv_idx] = new_adv_s
            env_state = env_state.replace(agent_states=new_agent_states)

            # Oracle label: shape (num_envs,) — same for all MM actors in an env
            adv_label_raw = info.get("adv_label", jnp.zeros((config["NUM_ENVS"],)))

            done_batch = done
            transitions = []
            for i, train_state in enumerate(train_states):
                done_batch["agents"][i] = batchify(
                    done["agents"][i], config["NUM_ACTORS_PERTYPE"][i]
                ).squeeze()
                obs_batch    = batchify(last_obs[i], config["NUM_ACTORS_PERTYPE"][i])
                action_batch = batchify(actions[i], config["NUM_ACTORS_PERTYPE"][i])

                # Tile adv_label to match NUM_ACTORS_PERTYPE for this agent type
                adv_label_i = jnp.tile(
                    adv_label_raw,
                    env.multi_agent_config.number_of_agents_per_type[i],
                )

                info_i = {
                    "world": info.get("world", {}),
                    "agent": jax.tree.map(
                        lambda x: x.reshape(config["NUM_ACTORS_PERTYPE"][i], -1),
                        info["agents"][i],
                    ),
                }
                transitions.append(Transition(
                    jnp.tile(done["__all__"], config["NUM_AGENTS_PER_TYPE"][i]),
                    last_done[i],
                    action_batch.squeeze(),
                    values[i].squeeze(),
                    batchify(reward[i], config["NUM_ACTORS_PERTYPE"][i]).squeeze(),
                    log_probs[i].squeeze(),
                    obs_batch,
                    adv_label_i,
                    info_i,
                ))

            runner_state = (
                train_states, env_state, obsv, done_batch["agents"], h_states, rng
            )
            return runner_state, transitions

        # ---- Inner: GAE calculation ----------------------------------------
        def _calculate_gae(gamma, gae_lambda, traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.global_done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + gamma * next_value * (1 - done) - value
                gae = delta + gamma * gae_lambda * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        # ---- MM update: PCGrad PPO + BCE -----------------------------------
        def _update_mm(train_state, traj_batch_mm, advantages_mm, targets_mm):
            def _update_epoch(update_state, unused):
                def _update_minibatch(ts, batch_info):
                    init_hstate, traj, adv, tgt = batch_info

                    # --- PPO loss (no detection head) ---
                    def _loss_ppo(params):
                        _, pi, value, _ = ts.apply_fn(
                            params,
                            init_hstate.squeeze(),
                            (traj.obs, traj.done),
                        )
                        log_prob = pi.log_prob(traj.action)
                        value_pred_clipped = traj.value + (value - traj.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_loss = 0.5 * jnp.maximum(
                            jnp.square(value - tgt),
                            jnp.square(value_pred_clipped - tgt),
                        ).mean()
                        gae_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
                        ratio = jnp.exp(log_prob - traj.log_prob)
                        loss_actor = -jnp.minimum(
                            ratio * gae_norm,
                            jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"]) * gae_norm,
                        ).mean()
                        entropy = pi.entropy().mean()
                        approx_kl = ((ratio - 1) - (log_prob - traj.log_prob)).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        total = (
                            loss_actor
                            + config["VF_COEF"][mm_idx] * value_loss
                            - config["ENT_COEF"][mm_idx] * entropy
                        )
                        return total, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                    # --- BCE detection loss ---
                    def _loss_bce(params):
                        _, _, _, det_prob = ts.apply_fn(
                            params,
                            init_hstate.squeeze(),
                            (traj.obs, traj.done),
                        )
                        labels = traj.adv_label
                        eps = 1e-7
                        bce = -jnp.mean(
                            labels * jnp.log(det_prob + eps)
                            + (1 - labels) * jnp.log(1 - det_prob + eps)
                        )
                        return detect_coef * bce, bce

                    grads_ppo, aux_ppo = jax.grad(_loss_ppo, has_aux=True)(ts.params)
                    grads_bce, aux_bce = jax.grad(_loss_bce, has_aux=True)(ts.params)

                    if pcgrad_on:
                        merged = pcgrad_merge(grads_ppo, grads_bce)
                    else:
                        merged = jax.tree.map(lambda a, b: a + b, grads_ppo, grads_bce)

                    ts = ts.apply_gradients(grads=merged)
                    ppo_loss, (vl, al, ent, ratio, kl, cf) = aux_ppo
                    bce_raw = aux_bce
                    return ts, (ppo_loss, vl, al, ent, ratio, kl, cf, bce_raw)

                ts, init_h, traj, adv, tgt, rng_e = update_state
                rng_e, _rng = jax.random.split(rng_e)
                init_h = jnp.reshape(init_h, (1, config["NUM_ACTORS_PERTYPE"][mm_idx], -1))
                batch = (init_h, traj, adv.squeeze(), tgt.squeeze())
                perm = jax.random.permutation(_rng, config["NUM_ACTORS_PERTYPE"][mm_idx])
                shuffled = jax.tree.map(lambda x: jnp.take(x, perm, axis=1), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                        ),
                        1, 0,
                    ),
                    shuffled,
                )
                ts, loss = jax.lax.scan(_update_minibatch, ts, minibatches)
                update_state = (ts, init_h.squeeze(), traj, adv, tgt, rng_e)
                return update_state, loss

            update_state = (
                train_state,
                jnp.zeros((config["NUM_ACTORS_PERTYPE"][mm_idx], config["GRU_HIDDEN_DIM"])),
                traj_batch_mm,
                advantages_mm,
                targets_mm,
                jax.random.PRNGKey(0),
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            return update_state[0], loss_info

        # ---- Adversary update: standard PPO --------------------------------
        def _update_adv(train_state, traj_batch_adv, advantages_adv, targets_adv):
            def _update_epoch(update_state, unused):
                def _update_minibatch(ts, batch_info):
                    init_hstate, traj, adv, tgt = batch_info

                    def _loss_fn(params):
                        _, pi, value, _ = ts.apply_fn(
                            params,
                            init_hstate.squeeze(),
                            (traj.obs, traj.done),
                        )
                        log_prob = pi.log_prob(traj.action)
                        value_pred_clipped = traj.value + (value - traj.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_loss = 0.5 * jnp.maximum(
                            jnp.square(value - tgt),
                            jnp.square(value_pred_clipped - tgt),
                        ).mean()
                        gae_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
                        ratio = jnp.exp(log_prob - traj.log_prob)
                        loss_actor = -jnp.minimum(
                            ratio * gae_norm,
                            jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"]) * gae_norm,
                        ).mean()
                        entropy = pi.entropy().mean()
                        approx_kl = ((ratio - 1) - (log_prob - traj.log_prob)).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        total = (
                            loss_actor
                            + config["VF_COEF"][adv_idx] * value_loss
                            - config["ENT_COEF"][adv_idx] * entropy
                        )
                        return total, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, aux), grads = grad_fn(ts.params)
                    ts = ts.apply_gradients(grads=grads)
                    return ts, (total_loss, *aux)

                ts, init_h, traj, adv, tgt, rng_e = update_state
                rng_e, _rng = jax.random.split(rng_e)
                init_h = jnp.reshape(init_h, (1, config["NUM_ACTORS_PERTYPE"][adv_idx], -1))
                batch = (init_h, traj, adv.squeeze(), tgt.squeeze())
                perm = jax.random.permutation(_rng, config["NUM_ACTORS_PERTYPE"][adv_idx])
                shuffled = jax.tree.map(lambda x: jnp.take(x, perm, axis=1), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                        ),
                        1, 0,
                    ),
                    shuffled,
                )
                ts, loss = jax.lax.scan(_update_minibatch, ts, minibatches)
                update_state = (ts, init_h.squeeze(), traj, adv, tgt, rng_e)
                return update_state, loss

            update_state = (
                train_state,
                jnp.zeros((config["NUM_ACTORS_PERTYPE"][adv_idx], config["GRU_HIDDEN_DIM"])),
                traj_batch_adv,
                advantages_adv,
                targets_adv,
                jax.random.PRNGKey(0),
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            return update_state[0], loss_info

        # ---- Jitted collect step -------------------------------------------
        @jax.jit
        def _collect_trajectories(runner_state):
            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            return runner_state, traj_batch, initial_hstates

        @jax.jit
        def _get_last_val(train_state, hstate, last_obs_i, last_done_i, actor_i):
            last_obs_b = batchify(last_obs_i, actor_i)
            ac_in = (last_obs_b[jnp.newaxis, :], last_done_i[jnp.newaxis, :])
            _, _, last_val, _ = train_state.apply_fn(
                train_state.params, hstate, ac_in
            )
            return last_val.squeeze()

        jitted_update_mm  = jax.jit(_update_mm)
        jitted_update_adv = jax.jit(_update_adv)

        # ---- Checkpoint setup ----------------------------------------------
        agent_type_names = list(env.type_names)
        checkpoint_dir = (
            f'{config["world_config"]["alphatradePath"]}/checkpoints/MARLCheckpoints'
            f'/{config["PROJECT"]}/{(run.name if run.name else run.id) if run else "GENERIC_RUN"}'
        )
        orbax_checkpointer = oxcp.PyTreeCheckpointer()
        options = oxcp.CheckpointManagerOptions(
            max_to_keep=2, create=True, keep_period=max(1, config["NUM_UPDATES"] // 2)
        )
        checkpoint_manager = oxcp.CheckpointManager(
            checkpoint_dir, orbax_checkpointer, options
        )

        # ---- Main training loop (Python-level) ----------------------------
        runner_state = (
            train_states,
            env_state,
            obsv,
            init_dones_agents,
            hstates,
            jax.random.PRNGKey(config["SEED"] + 1),
        )

        for update_i in range(config["NUM_UPDATES"]):
            phase = (update_i // freeze_n) % 2   # 0 = update adv, 1 = update MM
            print(f"Update {update_i + 1}/{config['NUM_UPDATES']}  phase={'adv' if phase==0 else 'mm'}")

            # Collect trajectories (both agents act)
            runner_state, traj_batch, initial_hstates = _collect_trajectories(runner_state)
            train_states_curr, env_state_curr, last_obs_curr, last_dones_curr, hstates_curr, rng_curr = runner_state

            # Compute last values for GAE
            last_vals = []
            for i in range(len(train_states_curr)):
                lv = _get_last_val(
                    train_states_curr[i],
                    hstates_curr[i],
                    last_obs_curr[i],
                    last_dones_curr[i],
                    config["NUM_ACTORS_PERTYPE"][i],
                )
                last_vals.append(lv)

            advantages = []
            targets = []
            for i in range(len(train_states_curr)):
                adv_i, tgt_i = _calculate_gae(
                    config["GAMMA"][i], config["GAE_LAMBDA"][i],
                    traj_batch[i], last_vals[i],
                )
                advantages.append(adv_i)
                targets.append(tgt_i)

            # Selective update
            if phase == 0:
                # Update adversary
                new_ts_adv, loss_adv = jitted_update_adv(
                    train_states_curr[adv_idx],
                    traj_batch[adv_idx],
                    advantages[adv_idx],
                    targets[adv_idx],
                )
                train_states_curr = list(train_states_curr)
                train_states_curr[adv_idx] = new_ts_adv
                loss_mm = None
            else:
                # Update MM
                new_ts_mm, loss_mm = jitted_update_mm(
                    train_states_curr[mm_idx],
                    traj_batch[mm_idx],
                    advantages[mm_idx],
                    targets[mm_idx],
                )
                train_states_curr = list(train_states_curr)
                train_states_curr[mm_idx] = new_ts_mm
                loss_adv = None

            runner_state = (
                train_states_curr,
                env_state_curr,
                last_obs_curr,
                last_dones_curr,
                hstates_curr,
                rng_curr,
            )

            # ---- Logging --------------------------------------------------
            def _log(update_i, phase, traj_batch, loss_mm, loss_adv, run):
                logging_dict = {"update_step": update_i + 1, "phase": phase}

                for agent_index, tr in enumerate(traj_batch):
                    agent_name = agent_type_names[agent_index]
                    logging_dict[f"agent_{agent_name}/avg_reward"] = float(
                        jnp.mean(tr.reward)
                    )
                    logging_dict[f"agent_{agent_name}/avg_adv_label"] = float(
                        jnp.mean(tr.adv_label)
                    )
                    for key, value in tr.info["agent"].items():
                        if isinstance(value, (jnp.ndarray, np.ndarray)) and value.size > 0:
                            flat = np.array(value).flatten()
                            logging_dict[f"agent_{agent_name}/{key}_mean"] = float(np.mean(flat))

                if loss_mm is not None:
                    loss_mm_mean = jax.tree.map(lambda x: float(jnp.mean(x)), loss_mm)
                    # loss_mm is (total, vl, al, ent, ratio, kl, cf, bce) stacked over epochs×minibatches
                    mm_name = agent_type_names[mm_idx]
                    logging_dict[f"agent_{mm_name}/ppo_loss"] = loss_mm_mean[0]
                    logging_dict[f"agent_{mm_name}/value_loss"] = loss_mm_mean[1]
                    logging_dict[f"agent_{mm_name}/actor_loss"] = loss_mm_mean[2]
                    logging_dict[f"agent_{mm_name}/entropy"] = loss_mm_mean[3]
                    logging_dict[f"agent_{mm_name}/bce_loss"] = loss_mm_mean[7]

                if loss_adv is not None:
                    loss_adv_mean = jax.tree.map(lambda x: float(jnp.mean(x)), loss_adv)
                    adv_name = agent_type_names[adv_idx]
                    logging_dict[f"agent_{adv_name}/total_loss"] = loss_adv_mean[0]
                    logging_dict[f"agent_{adv_name}/value_loss"] = loss_adv_mean[1]

                if config["WANDB_MODE"] != "disabled" and run is not None:
                    wandb.log(logging_dict)

                for agent_index in range(len(traj_batch)):
                    agent_name = agent_type_names[agent_index]
                    print(f"  avg_reward_{agent_name}: {logging_dict.get(f'agent_{agent_name}/avg_reward', 'N/A'):.4f}")

            _log(update_i, phase, traj_batch, loss_mm, loss_adv, run)

            # ---- Checkpoint -----------------------------------------------
            ckpt = {
                "model": runner_state[0],
                "metrics": {"avg_reward": [float(jnp.mean(tr.reward)) for tr in traj_batch]},
            }
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(update_i + 1, ckpt, save_kwargs={"save_args": save_args})
            del traj_batch, advantages, targets, last_vals
            gc.collect()

        checkpoint_manager.wait_until_finished()
        return {"runner_state": runner_state}

    return train


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(
    version_base="1.3",
    config_path="../../../config/rl_configs",
    config_name="ippo_adversarial",
)
def main(config):
    try:
        if config.get("ENV_CONFIG") is not None:
            env_config = load_config_from_file(config["ENV_CONFIG"])
        else:
            env_config = MultiAgentConfig()
            save_config_to_file(
                env_config,
                f"config/env_configs/default_config_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
    except Exception as e:
        print(f"Error loading env config: {e}")
        env_config = MultiAgentConfig()

    env_config   = OmegaConf.structured(env_config)
    final_config = OmegaConf.merge(config, env_config)
    config       = OmegaConf.to_container(final_config)

    sweep_parameters = config.get("SWEEP_PARAMETERS", {}) or {}
    is_single_run    = not sweep_parameters or config.get("WANDB_MODE") == "disabled"

    def sweep_fun():
        run = wandb.init(
            entity=config["ENTITY"],
            project=config["PROJECT"],
            tags=["IPPO", "adversarial", "PCGrad"],
            config=config,
            mode=config["WANDB_MODE"],
            allow_val_change=True,
            config_exclude_keys=[] if is_single_run else ["SEED"],
        )
        rng       = jax.random.PRNGKey(wandb.config["SEED"])
        train_fun = make_train(wandb.config)
        out       = train_fun(rng, run)
        del out
        gc.collect()
        jax.clear_caches()
        run.finish()

    if is_single_run:
        sweep_fun()
    else:
        sweep_config = {"method": "grid", "parameters": sweep_parameters}
        sweep_id = wandb.sweep(sweep=sweep_config, project=config["PROJECT"], entity=config["ENTITY"])
        wandb.agent(sweep_id, function=sweep_fun, count=500)

    sys.exit(0)


if __name__ == "__main__":
    main()
