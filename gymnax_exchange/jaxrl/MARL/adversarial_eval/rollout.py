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
    # Regime labels: load from the same config paths the training loop uses. Without
    # this the regime input is a constant 0 at evaluation even for regime-trained
    # models — silently disabling the H4 (regime-conditioning) hypothesis.
    regime_labels = {}
    window_to_date = None
    if config.get("REGIME_LABELS_PATH"):
        regime_labels = AdversarialMARLEnv.load_regime_labels(config["REGIME_LABELS_PATH"])
    if config.get("WINDOW_TO_DATE_PATH"):
        with open(config["WINDOW_TO_DATE_PATH"]) as f:
            window_to_date = {int(k): v for k, v in json.load(f).items()}
    env = AdversarialMARLEnv(key=init_key, multi_agent_config=ma_config,
                             regime_labels=regime_labels, window_to_date=window_to_date)

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
    """Restore train_states from an orbax checkpoint.

    Handles both checkpoint layouts: the current {"model", "metrics", "normalizer"}
    tree (normalizer state added 2026-07-04) and the legacy {"model", "metrics"} one.
    """
    ckpt_dir = (f'{config["world_config"]["alphatradePath"]}/checkpoints/MARLCheckpoints'
                f'/{project}/{run_name}')
    mgr = oxcp.CheckpointManager(ckpt_dir, oxcp.PyTreeCheckpointer())
    if step is None:
        step = mgr.latest_step()
    if step is None:
        raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")
    metrics_template = {"avg_reward": [0.0 for _ in train_states]}
    n_types = len(train_states)
    norm_template = {
        "mean": np.zeros(n_types), "var": np.ones(n_types), "count": np.zeros(n_types),
        "ret_acc": [np.zeros((config["NUM_ACTORS_PERTYPE"][i],)) for i in range(n_types)],
    }
    try:
        restored = mgr.restore(step, items={
            "model": train_states, "metrics": metrics_template, "normalizer": norm_template,
        })
    except Exception:
        restored = mgr.restore(step, items={
            "model": train_states, "metrics": metrics_template,
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

    tick_size = float(env.multi_agent_config.world_config.tick_size)

    series = {"ret": [], "inventory": [], "det_prob": [], "adv_label": [],
              "regime": [], "quote_disp_ticks": []}

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

        # Feed the MM detection prob into the adversary-state carrier BEFORE stepping
        # (mirrors the training loop): the obs built inside env.step then contains
        # det(obs_t) with a one-step lag, and auto-resets zero it for new episodes.
        prev_shape = env_state.agent_states[adv_idx].prev_detection_prob.shape
        mm_det = det_mm.reshape(prev_shape)
        ags = list(env_state.agent_states)
        ags[adv_idx] = env_state.agent_states[adv_idx].replace(prev_detection_prob=mm_det)
        env_state = env_state.replace(agent_states=ags)

        rng, sk = jax.random.split(rng)
        obs_list, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(sk, n_envs), env_state, actions, env_params)

        dones = [done["agents"][i].reshape(n_envs * nper[i]) for i in range(len(networks))]

        mm_info = info["agents"][mm_idx]
        series["ret"].append(_per_env(mm_info["reward_delta_pv"], n_envs))
        series["inventory"].append(_per_env(mm_info["inventory"], n_envs))
        series["det_prob"].append(np.asarray(det_mm).reshape(n_envs, -1).mean(axis=1))
        series["adv_label"].append(np.asarray(info["adv_label"]).reshape(-1))
        series["regime"].append(np.asarray(info["regime"]).reshape(-1))

        # Quote displacement (proposal behavioural metric): |quoted mid - true end mid|
        # in ticks, only on steps where the MM actually posted two-sided quotes.
        pb = _per_env(mm_info["posted_bid_price"], n_envs)
        pa = _per_env(mm_info["posted_ask_price"], n_envs)
        mid = np.asarray(info["world"]["end_mid_price"], dtype=np.float64).reshape(-1)
        quoted_mid = (pb + pa) / 2.0
        valid = (pb > 0) & (pa > 0)
        series["quote_disp_ticks"].append(
            np.where(valid, np.abs(quoted_mid - mid) / tick_size, np.nan))

    return {k: np.stack(v, axis=0) for k, v in series.items()}   # each (n_steps, n_envs)


def _nanmean(vals) -> float:
    v = np.asarray(vals, dtype=np.float64)
    return float(np.nanmean(v)) if v.size and np.isfinite(v).any() else float("nan")


def rollout_metrics(arrays, periods_per_year):
    """Reduce per-step rollout arrays to a single metric dict for this model/seed.

    Risk/behavioural metrics are computed per eval episode (env) then averaged;
    detection AUROC is pooled across the whole rollout (needs both classes).

    Estimand notes for the writeup:
      - Sharpe/Sortino/CVaR are computed on PER-STEP portfolio-value changes over
        the eval horizon and sqrt-annualised; per-step returns are autocorrelated,
        so treat absolute annualised values as indicative — paired cross-config
        comparisons are the supported inference.
      - CVaR is the mean of the worst 10% of per-step returns (step-level tail),
        not an episode- or seed-level tail.
      - AUROC alignment: det(obs_t) can only see the injection of the adversary's
        PREVIOUS action (the obs the env returns at t carries a_t's injection into
        t+1), so probabilities are paired with the previous step's oracle label.
    """
    ret, inv = arrays["ret"], arrays["inventory"]          # (T, n_envs)
    n_envs = ret.shape[1]
    sharpe = _nanmean([M.sharpe_ratio(ret[:, e], periods_per_year) for e in range(n_envs)])
    sortino = _nanmean([M.sortino_ratio(ret[:, e], periods_per_year) for e in range(n_envs)])
    softmin = _nanmean([M.softmin_sharpe(ret[:, e], periods_per_year) for e in range(n_envs)])
    cvar = _nanmean([M.cvar(ret[:, e], 0.10) for e in range(n_envs)])
    peak_inv = _nanmean([M.peak_inventory_excursion(inv[:, e]) for e in range(n_envs)])
    inv_sd = _nanmean([M.inventory_sd(inv[:, e]) for e in range(n_envs)])

    qd = arrays.get("quote_disp_ticks")
    quote_disp = _nanmean(qd) if qd is not None else float("nan")

    # One-step alignment shift (see docstring).
    auroc = M.detection_auroc(arrays["det_prob"][1:].ravel(),
                              arrays["adv_label"][:-1].ravel())

    # Regime-split Sortino + absolute gap (H4). NaN when a regime is absent from
    # the rollout (e.g. regime labels not wired, or a single-regime eval slice).
    sortino_low = sortino_high = regime_gap = float("nan")
    reg = arrays.get("regime")
    if reg is not None and reg.size:
        lows, highs = [], []
        for e in range(n_envs):
            m_high = reg[:, e] > 0.5
            if (~m_high).sum() >= 2:
                lows.append(M.sortino_ratio(ret[~m_high, e], periods_per_year))
            if m_high.sum() >= 2:
                highs.append(M.sortino_ratio(ret[m_high, e], periods_per_year))
        sortino_low, sortino_high = _nanmean(lows), _nanmean(highs)
        if np.isfinite(sortino_low) and np.isfinite(sortino_high):
            regime_gap = abs(sortino_high - sortino_low)

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "softmin_sharpe": float(softmin),
        "cvar": float(cvar),
        "peak_inventory": float(peak_inv),
        "inventory_sd": float(inv_sd),
        "quote_displacement": float(quote_disp),
        "auroc": float(auroc),
        "sortino_lowvol": float(sortino_low),
        "sortino_highvol": float(sortino_high),
        "regime_gap": float(regime_gap),
        "mean_attack_rate": float(arrays["adv_label"].mean()),
    }


def evaluate_checkpoint(project, run_name="local_run", n_envs=8, n_steps=None,
                        periods_per_year=98280.0, seed=0, step=None, yaml_path=_YAML,
                        adv_project=None, adv_run_name=None, adv_step=None):
    """Convenience: load a checkpoint and return {attack_mode -> metric dict}.

    n_steps defaults to the config's NUM_STEPS (the full training episode length)
    so the terminal unwind — often the largest loss event — is inside the eval
    window; a truncated horizon silently drops it from Sortino/CVaR.

    Common-adversary evaluation (internal validity for H1)
    ------------------------------------------------------
    By default BOTH agents are restored from `run_name`, i.e. each MM faces the
    adversary it was co-trained with. Across configs that confounds the H1
    contrast: the config-1 baseline would face an UNTRAINED adversary while the
    defended configs face their own trained ones. Pass `adv_run_name` (and
    usually `adv_project`, e.g. the config-3 arm) to overwrite the adversary's
    parameters from a reference checkpoint so every arm is attacked by the SAME
    adversary. Pair reference seeds with MM seeds by index (seed_i vs seed_i)
    to preserve the paired-across-configs design.
    """
    out = {}
    # 'on'/'off' give the attack-on/attack-off PnL & risk metrics; 'mixed' gives a
    # both-classes stream for the detection AUROC (single-class rollouts -> AUROC nan).
    for mode in ("on", "off", "mixed"):
        cfg = set_attack_mode(load_merged_config(yaml_path), mode)
        env, nets, ts_template, cfg = build_eval(cfg, n_envs)
        ts, used_step = restore_checkpoint(cfg, ts_template, project, run_name, step)
        adv_used_step = None
        if adv_run_name is not None:
            # AdversaryNet has the same architecture in every arm, so the reference
            # checkpoint restores into the same template; only the adversary slot is
            # swapped — the MM under evaluation keeps its own parameters.
            ts_ref, adv_used_step = restore_checkpoint(
                cfg, ts_template, adv_project or project, adv_run_name, adv_step)
            ts = list(ts)
            ts[env._adv_idx] = ts_ref[env._adv_idx]
        steps = int(n_steps) if n_steps is not None else int(cfg["NUM_STEPS"])
        arrays = run_rollout(env, nets, ts, cfg, mode, jax.random.PRNGKey(seed), n_envs, steps)
        out[mode] = rollout_metrics(arrays, periods_per_year)
        out[mode]["_checkpoint_step"] = used_step
        if adv_run_name is not None:
            out[mode]["_adv_checkpoint"] = f"{adv_project or project}/{adv_run_name}@{adv_used_step}"
    return out


def evaluate_fixed_policy(n_envs=8, n_steps=None, periods_per_year=98280.0,
                          seed=0, yaml_path=_YAML):
    """Evaluate a fixed (non-learned) MM policy — no checkpoint restore.

    The MM's behaviour must come from the env config, e.g. the Avellaneda-Stoikov
    arm: action_space="AvSt" with fixed_action_setting=true pins the closed-form
    A-S rule regardless of the (freshly initialised) network output. This is the
    config-1 A-S half of the progression gate. Only the *_off metrics are
    meaningful for the gate; the adversary in 'on'/'mixed' modes is untrained
    (and A-S quotes off the true book, so perturbed observations don't move it),
    and the detection AUROC is chance by construction.
    """
    out = {}
    for mode in ("on", "off", "mixed"):
        cfg = set_attack_mode(load_merged_config(yaml_path), mode)
        env, nets, ts, cfg = build_eval(cfg, n_envs)
        mm_cfg = env.list_of_agents_configs[env._mm_idx]
        if not getattr(mm_cfg, "fixed_action_setting", False):
            raise ValueError(
                "evaluate_fixed_policy requires an env config with "
                "fixed_action_setting=true for the MM (e.g. the AvSt baseline json); "
                f"got fixed_action_setting={getattr(mm_cfg, 'fixed_action_setting', None)}")
        steps = int(n_steps) if n_steps is not None else int(cfg["NUM_STEPS"])
        arrays = run_rollout(env, nets, ts, cfg, mode, jax.random.PRNGKey(seed), n_envs, steps)
        out[mode] = rollout_metrics(arrays, periods_per_year)
        out[mode]["_checkpoint_step"] = None
    return out
