"""Is the trained market maker sensitive to a WORST-CASE spoof? (forced attack)

H1 could not be tested with a co-trained adversary (docs/note_h1_attack_effect.md):
under its cost model the constrained adversary abstains (0.159% of max action), and
a cost-free one converges to a symmetric injection that cannot move
queue_imbalance. Either way a null is indistinguishable from "the MM is robust".

This script removes the adversary's choice from the experiment. Immediately before
every env.step it overwrites actions[adv_idx] with a fixed, maximal, one-sided
injection (all bid levels at 1.0 or all ask levels at 1.0), against an EXISTING
trained MM checkpoint. action_to_injection is a pure function of (action,
world_state, budget_remaining, gate), so the forced action goes through exactly the
same scaling, gating and observation perturbation a policy-derived one would.
No environment, agent, rollout or confirmatory-pipeline code is modified.

    python eval_forced_attack.py --arms baseline --seeds 0 --side bid --budget-override 1e6
    python eval_forced_attack.py --arms all --seeds 0-19 --side ask --budget-override 1e6 \\
        --out results/forced_attack_v4_ask

Budget and gate
---------------
The gate is forced on by set_attack_mode("on"), as in the production eval. Every
v4 eval env already has budget_per_episode=1e6 (the baseline arm's eval yaml loads
the config-2 env json on purpose; its training json has a zero budget and would
make any attack a no-op). The budget is still set explicitly on every arm because
the trained adversary never came near spending it: whether 1e6 sustains a MAXIMAL
injection for a whole episode is unknown, and action_to_injection silently rescales
to whatever budget is left once it runs out. budget_final_frac and
budget_exhausted_frac report whether it bound -- raise --budget-override if so.
It has no default on purpose, so the value is always a recorded choice.

Pairing and the reproduction check
----------------------------------
Each (arm, seed) runs twice with the same rollout PRNG key: 'forced' and 'off'.
Both conditions go through this file's own loop, so the ONLY difference between
them is the override (plus the gate/budget config it needs). The 'off' condition
uses the arm's config exactly as run_production_eval.py does, so with the same
n_envs / n_steps / checkpoint step its per-seed numbers should reproduce the
existing eval JSON's *_off values -- pass --reference-eval to check that, which
validates this loop against the production code path before any forced number is
trusted.

Output keys are suffixed _forced / _off (never _on) so this file can never be
loaded as if it were a co-trained-adversary result. EXPLORATORY, not
confirmatory: preregistration.json -> amendments -> forced_worst_case_attack_amendment.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

# Layout from mm_env.get_adversarial_observation (45-dim): 40 L2 + inventory(40) +
# time_remaining(41) + queue_imbalance(42) + prev_detection_prob(43) + regime(44).
# Asserted against the live observation at runtime, same as check_adversary_sidedness.py.
_QI_IDX = 42
_OBS_DIM = 45

# Mirrors run_production_eval.py's all_arms table (project suffix, eval-config number).
# 'as' is deliberately absent: A-S quotes off the TRUE book, so a perturbed
# observation cannot move it by construction (see rollout.evaluate_fixed_policy).
ARMS = {
    "baseline":        ("config1_baseline", 1),
    "adversarial":     ("config2_adversarial", 2),
    "full":            ("config3_full", 3),
    "detection":       ("config4_detection", 4),
    "regime":          ("config5_regime", 5),
    "unconstrained":   ("config6_unconstrained", 6),
    "detection_noobs": ("config7_detection_noobs", 7),
}

# Same contrast set and orientation as run_evaluation.run_full_evaluation's
# _CANDIDATES, so analysis/evalreport.py's CONTRAST_LABELS render these blocks.
CONTRASTS = [
    ("adversarial_vs_baseline",        "adversarial",     "baseline"),
    ("detection_vs_adversarial",       "detection",       "adversarial"),
    ("full_vs_regime",                 "full",            "regime"),
    ("regime_vs_adversarial",          "regime",          "adversarial"),
    ("full_vs_detection",              "full",            "detection"),
    ("full_vs_adversarial",            "full",            "adversarial"),
    ("full_vs_baseline",               "full",            "baseline"),
    ("adversarial_vs_unconstrained",   "adversarial",     "unconstrained"),
    ("detection_noobs_vs_adversarial", "detection_noobs", "adversarial"),
    ("detection_vs_detection_noobs",   "detection",       "detection_noobs"),
]

REPORTED = ("sharpe", "sortino", "cvar", "peak_inventory", "inventory_sd",
            "quote_displacement", "quote_presence", "mean_attack_rate",
            "mean_injected_volume", "mean_bid_volume_injected",
            "mean_ask_volume_injected", "qi_absmean", "qi_mean")
FORCED_ONLY = ("budget_final_frac", "budget_exhausted_frac")
CONTRAST_METRICS = ("sharpe_forced", "sortino_forced", "cvar_forced")


def force_action(side: str, n_levels: int):
    """Fixed maximal one-sided action in the adversary's own encoding.

    spoofing_agent.action_to_injection reads a[:n_levels] as bid levels and
    a[n_levels:] as ask levels, each clipped to [0, 1]; 1.0 on every level of one
    side is the largest legal one-sided injection.
    """
    import jax.numpy as jnp
    if side == "bid":
        vec = [1.0] * n_levels + [0.0] * n_levels
    elif side == "ask":
        vec = [0.0] * n_levels + [1.0] * n_levels
    else:
        raise ValueError(f"side must be 'bid' or 'ask', got {side!r}")
    return jnp.asarray(vec, dtype=jnp.float32)


def set_budget_override(config: dict, budget: float) -> dict:
    """Same shape as rollout.set_attack_mode: mutate the Spoofing config in place.

    Applied to every arm so the injectable volume is one recorded value, not
    whatever each arm's env json happens to say.
    """
    if budget <= 0:
        raise ValueError(f"budget override must be positive, got {budget}")
    config["dict_of_agents_configs"]["Spoofing"]["budget_per_episode"] = float(budget)
    return config


def _parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def run_rollout_with_override(env, networks, train_states, config, gate, forced,
                              budget, rng, n_envs, n_steps):
    """rollout.run_rollout's loop, copied, with two additions.

    1. If `forced` is not None, actions[adv_idx] is replaced by it right before
       env.step. The adversary's own forward pass still runs and is discarded, so
       the loop stays structurally identical to every other eval loop in the repo.
    2. Two extra series: the MM's observed queue_imbalance (the channel the attack
       exists to move) and the adversary's budget_remaining (is the attack actually
       sustained, or does the budget clamp it to zero part-way through?).

    Any other change to this loop would make 'forced' vs 'off' differ in more than
    the override -- keep it in lock-step with rollout.run_rollout.
    """
    import jax
    import jax.numpy as jnp
    from gymnax_exchange.jaxrl.MARL.adversarial_eval.rollout import _per_env

    mm_idx, adv_idx = env._mm_idx, env._adv_idx
    nper = env.multi_agent_config.number_of_agents_per_type
    env_params = env.default_params

    rng, rk = jax.random.split(rng)
    obs_list, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rk, n_envs), env_params)

    aa = env_state.agent_states[adv_idx].attack_active
    ags = list(env_state.agent_states)
    ags[adv_idx] = env_state.agent_states[adv_idx].replace(
        attack_active=jnp.full_like(aa, gate))
    env_state = env_state.replace(agent_states=ags)

    got = int(obs_list[mm_idx].reshape((n_envs * nper[mm_idx], -1)).shape[-1])
    assert got == _OBS_DIM, (
        f"MM obs is {got}-dim, expected {_OBS_DIM}; queue_imbalance index {_QI_IDX} "
        "is stale -- re-read mm_env.get_adversarial_observation.")

    forced_b = None
    if forced is not None:
        forced_b = jnp.broadcast_to(forced, (n_envs, forced.shape[-1]))

    h_states = [networks[i].initialize_carry(n_envs * nper[i], config["GRU_HIDDEN_DIM"])
                for i in range(len(networks))]
    dones = [jnp.zeros((n_envs * nper[i],), dtype=bool) for i in range(len(networks))]
    tick_size = float(env.multi_agent_config.world_config.tick_size)

    series = {"ret": [], "inventory": [], "det_prob": [], "adv_label": [],
              "regime": [], "quote_disp_ticks": [], "volume_injected": [],
              "bid_volume_injected": [], "ask_volume_injected": [],
              "qi": [], "budget_remaining": []}

    for _ in range(n_steps):
        actions, det_mm = [], None
        for i, ts in enumerate(train_states):
            obs_i = obs_list[i].reshape((n_envs * nper[i], -1))
            ac_in = (obs_i[jnp.newaxis, :], dones[i][jnp.newaxis, :])
            h_new, pi, _, det_prob = ts.apply_fn(ts.params, h_states[i], ac_in)
            h_states[i] = h_new
            if i == mm_idx:
                det_mm = det_prob[0]
            action = pi.mode()
            action = action.reshape((n_envs, nper[i], -1))
            actions.append(action.squeeze(-2) if action.ndim > 2 else action.squeeze())

        if forced_b is not None:
            if actions[adv_idx].shape != forced_b.shape:
                raise RuntimeError(
                    f"adversary action slot is {actions[adv_idx].shape}, forced action "
                    f"is {forced_b.shape} -- the override would not replace it cleanly")
            actions[adv_idx] = forced_b

        prev_shape = env_state.agent_states[adv_idx].prev_detection_prob.shape
        ags = list(env_state.agent_states)
        ags[adv_idx] = env_state.agent_states[adv_idx].replace(
            prev_detection_prob=det_mm.reshape(prev_shape))
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
        series["volume_injected"].append(
            np.asarray(info["volume_injected_step"], dtype=np.float64).reshape(-1))
        series["bid_volume_injected"].append(
            np.asarray(info["bid_volume_injected_step"], dtype=np.float64).reshape(-1))
        series["ask_volume_injected"].append(
            np.asarray(info["ask_volume_injected_step"], dtype=np.float64).reshape(-1))
        series["regime"].append(np.asarray(info["regime"]).reshape(-1))

        pb = _per_env(mm_info["posted_bid_price"], n_envs)
        pa = _per_env(mm_info["posted_ask_price"], n_envs)
        mid = np.asarray(info["world"]["end_mid_price"], dtype=np.float64).reshape(-1)
        valid = (pb > 0) & (pa > 0)
        series["quote_disp_ticks"].append(
            np.where(valid, np.abs((pb + pa) / 2.0 - mid) / tick_size, np.nan))

        series["qi"].append(np.asarray(
            obs_list[mm_idx].reshape((n_envs * nper[mm_idx], -1))[:, _QI_IDX],
            dtype=np.float64).reshape(n_envs, -1).mean(axis=1))
        series["budget_remaining"].append(
            _per_env(info["agents"][adv_idx]["budget_remaining"], n_envs))

    return {k: np.stack(v, axis=0) for k, v in series.items()}


def extra_metrics(arrays: dict, budget: float) -> dict:
    qi = arrays["qi"]
    out = {"qi_absmean": float(np.abs(qi).mean()), "qi_mean": float(qi.mean())}
    b = arrays["budget_remaining"]
    out["budget_final_frac"] = float(b[-1].mean() / budget) if budget > 0 else float("nan")
    out["budget_exhausted_frac"] = float((b < 0.01 * budget).mean()) if budget > 0 else float("nan")
    return out


def evaluate_arm(arm, project, yaml_path, seeds, side, budget, n_envs, n_steps, step, ppy):
    """Return {metric_forced / metric_off: np.array over seeds} for one arm."""
    import jax
    from gymnax_exchange.jaxrl.MARL.adversarial_eval.rollout import (
        build_eval, load_merged_config, restore_checkpoint, rollout_metrics, set_attack_mode,
    )

    rows = {s: {} for s in seeds}
    # 'off' first, then 'forced': one env per condition (the configs differ), freed
    # before the next is built -- each env holds the period's full message array.
    for cond in ("off", "forced"):
        cfg = load_merged_config(yaml_path)
        if cond == "forced":
            cfg = set_attack_mode(cfg, "on")
            cfg = set_budget_override(cfg, budget)
        else:
            cfg = set_attack_mode(cfg, "off")
        env, nets, ts_template, cfg = build_eval(cfg, n_envs)

        nper = env.multi_agent_config.number_of_agents_per_type
        adv_idx = env._adv_idx
        if int(nper[adv_idx]) != 1:
            raise SystemExit(f"nper[adv_idx]={int(nper[adv_idx])}, expected 1 -- the "
                             "forced action is broadcast to (n_envs, 10) and would mis-shape.")
        adv_cfg = env.list_of_agents_configs[adv_idx]
        n_lv = int(adv_cfg.n_spoof_levels)
        if env.action_spaces[adv_idx].shape[0] != 2 * n_lv:
            raise SystemExit(f"adversary action dim {env.action_spaces[adv_idx].shape[0]} "
                             f"!= 2 * n_spoof_levels ({2 * n_lv})")
        if cond == "forced" and float(adv_cfg.budget_per_episode) != float(budget):
            raise SystemExit("budget override did not reach the built env "
                             f"({adv_cfg.budget_per_episode} != {budget})")

        forced = force_action(side, n_lv) if cond == "forced" else None
        gate = 1.0 if cond == "forced" else 0.0
        steps = int(n_steps) if n_steps is not None else int(cfg["NUM_STEPS"])

        for s in seeds:
            ts, used_step = restore_checkpoint(cfg, ts_template, project, f"seed_{s}", step)
            arrays = run_rollout_with_override(
                env, nets, ts, cfg, gate, forced, budget if cond == "forced" else 0.0,
                jax.random.PRNGKey(s), n_envs, steps)
            m = rollout_metrics(arrays, ppy)
            m.update(extra_metrics(arrays, budget if cond == "forced" else 0.0))
            for k in REPORTED:
                rows[s][f"{k}_{cond}"] = m[k]
            if cond == "forced":
                for k in FORCED_ONLY:
                    rows[s][f"{k}_forced"] = m[k]
            rows[s]["_checkpoint_step"] = used_step
            print(f"[forced-attack] arm={arm} cond={cond} seed={s} step={used_step} "
                  f"sharpe={m['sharpe']:+.3f} attack_rate={m['mean_attack_rate']:.4f} "
                  f"bid_vol={m['mean_bid_volume_injected']:.3g} "
                  f"ask_vol={m['mean_ask_volume_injected']:.3g} "
                  f"|QI|={m['qi_absmean']:.4f}"
                  + (f" budget_final={m['budget_final_frac']:.3f}" if cond == "forced" else ""),
                  flush=True)
            del ts, arrays
            gc.collect()

        del env, nets, ts_template, cfg
        jax.clear_caches()
        gc.collect()

    keys = [k for k in rows[seeds[0]] if not k.startswith("_")]
    return {k: np.array([rows[s][k] for s in seeds], dtype=np.float64) for k in keys}


def reference_check(per_seed: dict, seeds: list[int], ref_path: str) -> list[str]:
    """Do this loop's '_off' numbers reproduce the production eval's '_off' numbers?"""
    with open(ref_path) as f:
        ref = json.load(f)
    ref_seeds = ref.get("_meta", {}).get("seeds", list(range(20)))
    lines = [f"--- reproduction check against {ref_path} (off condition) ---"]
    for arm, metrics in per_seed.items():
        if arm not in ref.get("per_seed", {}):
            lines.append(f"  {arm:<16} not in reference file")
            continue
        for k in ("sharpe_off", "sortino_off", "cvar_off", "inventory_sd_off"):
            if k not in metrics or k not in ref["per_seed"][arm]:
                continue
            ref_vals = np.asarray(ref["per_seed"][arm][k], dtype=np.float64)
            mine = metrics[k]
            idx = [ref_seeds.index(s) for s in seeds if s in ref_seeds]
            if len(idx) != len(seeds):
                lines.append(f"  {arm:<16} {k}: seeds not all in reference")
                continue
            diff = np.abs(mine - ref_vals[idx])
            tol = 1e-6 * max(1.0, float(np.nanmax(np.abs(ref_vals[idx]))))
            verdict = "MATCH" if np.nanmax(diff) <= tol else "DIFFERS"
            lines.append(f"  {arm:<16} {k:<18} max|diff|={np.nanmax(diff):.3g}  {verdict}")
    lines.append("  (DIFFERS is expected only if --n-envs / --n-steps / --step differ "
                 "from the reference run; otherwise this loop is not faithful to "
                 "rollout.run_rollout and no forced number should be trusted)")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default="baseline",
                    help=f"comma list or 'all' ({','.join(ARMS)})")
    ap.add_argument("--project-prefix", default="v4", choices=["v3", "v4"])
    ap.add_argument("--seeds", default="0",
                    help="seed indices, e.g. '0-19' (default: 0, a smoke test)")
    ap.add_argument("--side", required=True, choices=["bid", "ask"])
    ap.add_argument("--budget-override", type=float, required=True,
                    help="injectable volume per episode, applied to every arm; "
                         "check budget_final_frac and raise it if the budget binds")
    ap.add_argument("--n-envs", type=int, default=64,
                    help="default 64 = run_production_eval.py's value, so the 'off' "
                         "condition can reproduce the existing eval JSON")
    ap.add_argument("--n-steps", type=int, default=None,
                    help="default: config NUM_STEPS (one full episode)")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step (default: preregistration.json checkpoint_step)")
    ap.add_argument("--reference-eval", default=None,
                    help="eval_*.json whose *_off per-seed values this run's 'off' "
                         "condition should reproduce, e.g. results/eval_17313.json")
    ap.add_argument("--out", default=None,
                    help="output prefix; writes .json and .txt (omit for a stdout-only smoke test)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    with open(root / "preregistration.json") as f:
        prereg = json.load(f)
    step = prereg["checkpoint_step"] if args.step is None else args.step
    ppy = float(prereg["periods_per_year"])
    pfx = args.project_prefix

    arms = list(ARMS) if args.arms == "all" else [a.strip() for a in args.arms.split(",")]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {list(ARMS)}")
    seeds = _parse_seeds(args.seeds)

    def _yaml(n: int) -> str:
        stem = ("eval_2024_test_config%d" % n if pfx == "v3"
                else "eval_2024_test_%s_config%d" % (pfx, n))
        return "config/rl_configs/%s.yaml" % stem

    banner = ("*** EXPLORATORY — FORCED WORST-CASE ATTACK (side=%s, budget=%g). Not a "
              "co-trained adversary, not the confirmatory result. See "
              "docs/note_forced_attack_amendment.md ***" % (args.side, args.budget_override))
    print(banner)
    print(f"arms {arms}  seeds {seeds}  step {step}  n_envs {args.n_envs}  prefix {pfx}\n")

    per_seed = {}
    for arm in arms:
        suffix, n = ARMS[arm]
        per_seed[arm] = evaluate_arm(arm, f"{pfx}_{suffix}", _yaml(n), seeds, args.side,
                                     args.budget_override, args.n_envs, args.n_steps,
                                     step, ppy)

    from gymnax_exchange.jaxrl.MARL.adversarial_eval.aggregate import (
        compare_configs, summarize_seeds,
    )
    from gymnax_exchange.jaxrl.MARL.adversarial_eval.stats import holm_adjust, paired_comparison

    report = {"summaries": {a: summarize_seeds(m) for a, m in per_seed.items()},
              "per_seed": {a: {k: v.tolist() for k, v in m.items()} for a, m in per_seed.items()},
              "holm": {}}
    if len(seeds) >= 3:
        for label, a, b in CONTRASTS:
            if a in per_seed and b in per_seed:
                res = compare_configs(per_seed, a, b, list(CONTRAST_METRICS))
                report[label] = res
                report["holm"][label] = holm_adjust({m: r.p_value for m, r in res.items()})
    report["_meta"] = {
        "attack_construction": "forced_one_sided_override",
        "side": args.side, "budget_override": args.budget_override,
        "project_prefix": pfx, "arms": arms, "seeds": seeds,
        "checkpoint_step": step, "n_envs": args.n_envs, "n_steps": args.n_steps,
        "periods_per_year": ppy,
        "signed_off": False, "confirmatory": False, "exploratory": True,
        "partial_run": True,
        "amendment": "preregistration.json -> amendments -> forced_worst_case_attack_amendment",
    }

    lines = [banner, "", "=== per-arm means (forced vs off) ==="]
    for arm, m in per_seed.items():
        lines.append(f"[{arm}]")
        for k in ("sharpe", "sortino", "cvar", "mean_attack_rate",
                  "mean_bid_volume_injected", "mean_ask_volume_injected", "qi_absmean", "qi_mean"):
            f_, o_ = m.get(f"{k}_forced"), m.get(f"{k}_off")
            lines.append(f"  {k:<26} forced={np.nanmean(f_):+10.4g}  off={np.nanmean(o_):+10.4g}")
        for k in FORCED_ONLY:
            lines.append(f"  {k:<26} forced={np.nanmean(m[f'{k}_forced']):10.4g}")
    if len(seeds) >= 3:
        lines += ["", "=== within-arm: forced vs off, paired by seed ==="]
        for arm, m in per_seed.items():
            for k in ("sharpe", "sortino", "cvar"):
                x, y = m[f"{k}_forced"], m[f"{k}_off"]
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 3:
                    continue
                r = paired_comparison(x[ok], y[ok])
                lines.append(f"  {arm:<16} {k:<8} n={r.n:<3} diff={r.mean_diff:+9.4g} "
                             f"d={r.cohens_d:+.3g} {r.test:<9} p={r.p_value:.3g}")
        lines.append("  (cross-arm DiD: python -m analysis.h1_attack_effect_check "
                     "<out>.json --on-suffix _forced)")
    if args.reference_eval:
        lines += [""] + reference_check(per_seed, seeds, args.reference_eval)
    text = "\n".join(lines)
    print("\n" + text)

    if args.out:
        from run_production_eval import _serialise
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(f"{out}.json", "w") as f:
            json.dump(_serialise(report), f, indent=2, default=str)
        with open(f"{out}.txt", "w") as f:
            f.write(text + "\n")
        print(f"\nwrote {out}.json and {out}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
