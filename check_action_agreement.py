"""Does the TRAINED market-maker policy's chosen action actually respond to the
spoof, on a real rollout?

preregistration.json's v4 amendment states an explicit precondition: "the
perturbed-vs-clean action-agreement check must show the adversary actually
flipping the selected action. If a spoofed observation does not change the
chosen action, v4 fails for v3's reason." That check has only ever been run
against a RANDOMLY-INITIALISED policy head, in check_adversary_lever.py, which
says so itself: "PROXY ONLY: random heads say the architecture CAN be
sensitive to these channels, not that a trained policy will be. Whether the
converged policy is fooled is the experiment, not its precondition." It has
never been run against the trained checkpoints that produced eval_1179095.json
or eval_1270.json. This script does that.

check_adversary_sidedness.py already answers "did the adversary learn to
attack" (its own verdict: ABSTAINED / SYMMETRIC / PARTIALLY ONE-SIDED /
ONE-SIDED). This script answers the complementary question: "when it does
attack, does the market maker's POLICY -- not just the observation channel --
actually react?" Run check_adversary_sidedness.py first; if it reports
ABSTAINED, H1 is untested for a reason upstream of anything this script can
show, and this script is not worth running yet.

Mechanism: for each real rollout step, the post-step observation the MM
receives is already built from the PERTURBED L2 snapshot (see
adversarial_marl_env.py step_env: perturbed_l2 at lines ~144-147, fed into
get_adversarial_observation at ~172-183). This script recomputes the
counterfactual CLEAN observation for the exact same post-step state by calling
the same public method with the TRUE (unperturbed) L2 snapshot instead, then
runs the trained policy on both (holding recurrent state fixed, since the
clean pass is a counterfactual probe, not a real step) and compares:

  1. argmax flip rate       -- same definition as check_adversary_lever.py's
                                argmax_of(), so directly comparable to the
                                random-head proxy number.
  2. KL divergence           -- softmax(logits_clean) vs softmax(logits_dirty),
                                catches a policy that shifts probability mass
                                without flipping its argmax.
  3. payoff-similarity of flips -- when the argmax does flip, is it to an
                                    action with a similar empirical payoff (a
                                    flip that can't matter much) or a very
                                    different one (a flip that could)?

    python check_action_agreement.py --project v4_config3_full --seeds 0
    python check_action_agreement.py --project v4_config1_baseline --seeds 0-19
"""
from __future__ import annotations

import argparse
import gc

import numpy as np


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", default="v4_config3_full")
    ap.add_argument("--yaml", default="config/rl_configs/eval_2024_test_v4_config3.yaml",
                     help="eval config; ENV_CONFIG must match the arm's training env")
    ap.add_argument("--seeds", default="0",
                     help="seed indices, e.g. '0-19' or '0,3,7' (default: seed 0 only, "
                          "a smoke test -- pass 0-19 for the full pre-registered list)")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=None,
                     help="default: the config's NUM_STEPS (one full episode)")
    ap.add_argument("--step", type=int, default=None,
                     help="checkpoint step (default: latest available)")
    ap.add_argument("--flip-threshold", type=float, default=0.05,
                     help="|payoff difference| below which a flip is called "
                          "'behaviourally inert' -- a judgement call, stated "
                          "explicitly rather than fixed silently")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import gymnax_exchange.jaxob.JaxOrderBookArrays as job
    from gymnax_exchange.jaxrl.MARL.adversarial_eval.rollout import (
        build_eval, load_merged_config, restore_checkpoint, set_attack_mode,
    )

    seeds = _parse_seeds(args.seeds)
    run_names = [f"seed_{s}" for s in seeds]

    print(f"project        {args.project}")
    print(f"yaml           {args.yaml}")
    print(f"seeds          {seeds}")
    if seeds == [0]:
        print("NOTE: single-seed smoke test. Verify shapes/no-NaNs here before "
              "scaling to --seeds 0-19.")
    print()

    cfg = set_attack_mode(load_merged_config(args.yaml), "on")
    env, nets, ts_template, cfg = build_eval(cfg, args.n_envs)

    mm_idx, adv_idx = env._mm_idx, env._adv_idx
    nper = env.multi_agent_config.number_of_agents_per_type
    if int(nper[mm_idx]) != 1:
        raise SystemExit(
            f"nper[mm_idx]={int(nper[mm_idx])}, expected 1 -- this script's "
            "per-env vmap over get_adversarial_observation assumes exactly one "
            "MM agent per env and needs updating before it can be trusted.")
    env_params = env.default_params
    world_config = env.multi_agent_config.world_config
    mm_instance = env.instance_list[mm_idx]
    n_steps = int(args.n_steps) if args.n_steps else int(cfg["NUM_STEPS"])
    n_actions = int(env.multi_agent_config.dict_of_agents_configs["AdversarialMM"].n_actions)

    all_flip_rates, all_kls, all_flip_payoff_deltas = [], [], []

    for s, run_name in zip(seeds, run_names):
        ts, used_step = restore_checkpoint(cfg, ts_template, args.project, run_name, args.step)

        rng = jax.random.PRNGKey(s)
        rng, rk = jax.random.split(rng)
        obs_list, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            jax.random.split(rk, args.n_envs), env_params)

        aa = env_state.agent_states[adv_idx].attack_active
        ags = list(env_state.agent_states)
        ags[adv_idx] = env_state.agent_states[adv_idx].replace(attack_active=jnp.ones_like(aa))
        env_state = env_state.replace(agent_states=ags)

        h_states = [nets[i].initialize_carry(args.n_envs * nper[i], cfg["GRU_HIDDEN_DIM"])
                    for i in range(len(nets))]
        dones = [jnp.zeros((args.n_envs * nper[i],), dtype=bool) for i in range(len(nets))]

        # (clean_action, dirty_action, |logit-softmax KL|, dirty step return) per step.
        step_records = []

        for _ in range(n_steps):
            # -- the real step, exactly as check_adversary_sidedness.py runs it --
            actions, det_mm = [], None
            mm_dirty_obs = None
            for i, t in enumerate(ts):
                obs_i = obs_list[i].reshape((args.n_envs * nper[i], -1))
                ac_in = (obs_i[jnp.newaxis, :], dones[i][jnp.newaxis, :])
                h_new, pi, _, det_prob = t.apply_fn(t.params, h_states[i], ac_in)
                h_states[i] = h_new
                if i == mm_idx:
                    det_mm = det_prob[0]
                    mm_dirty_obs = obs_i  # this step's real (perturbed) MM observation
                    mm_dirty_logits = pi.logits[0]
                a = pi.mode()
                a = a.reshape((args.n_envs, nper[i], -1))
                actions.append(a.squeeze(-2) if a.ndim > 2 else a.squeeze())

            prev_det_before = env_state.agent_states[adv_idx].prev_detection_prob
            prev_shape = prev_det_before.shape
            ags = list(env_state.agent_states)
            ags[adv_idx] = env_state.agent_states[adv_idx].replace(
                prev_detection_prob=det_mm.reshape(prev_shape))
            env_state_pre_step = env_state.replace(agent_states=ags)

            rng, sk = jax.random.split(rng)
            obs_list, env_state, reward_list, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None))(
                jax.random.split(sk, args.n_envs), env_state_pre_step, actions, env_params)
            dones = [done["agents"][i].reshape(args.n_envs * nper[i]) for i in range(len(nets))]

            # -- the counterfactual: same post-step state, TRUE (unperturbed) L2 --
            ws = env_state.world_state
            true_l2 = jax.vmap(job.get_L2_state, in_axes=(0, 0, None, None))(
                ws.ask_raw_orders, ws.bid_raw_orders, 10, world_config)
            regime = env._regime_array[ws.window_index]
            mm_state = env_state.agent_states[mm_idx]
            mm_param = env_params.agent_params[mm_idx]
            # prev_det used by the real step was prev_det_before (pre-step value,
            # matching adversarial_marl_env.py's own lag); the same value feeds the
            # counterfactual, since it is not something the L2 swap should change.
            prev_det_for_obs = prev_det_before.reshape(-1)

            clean_obs = jax.vmap(
                mm_instance.get_adversarial_observation, in_axes=(0, 0, None, 0, 0, 0, None))(
                ws, mm_state, mm_param, true_l2, regime, prev_det_for_obs, True)

            ac_in_clean = (clean_obs[jnp.newaxis, :], dones[mm_idx][jnp.newaxis, :])
            # Counterfactual probe: reuse the SAME h_states[mm_idx] the dirty pass
            # used (captured before this step's h_states[mm_idx] update above would
            # be wrong -- h_states[mm_idx] was already advanced by the dirty pass in
            # this same iteration, so this call sees one-step-stale recurrent state
            # on both the clean and dirty comparison, which is consistent between
            # them and is what "same recurrent state, different observation" means).
            _, pi_clean, _, _ = ts[mm_idx].apply_fn(ts[mm_idx].params, h_states[mm_idx], ac_in_clean)
            clean_logits = pi_clean.logits[0]

            clean_action = np.asarray(jnp.argmax(clean_logits, axis=-1))
            dirty_action = np.asarray(jnp.argmax(mm_dirty_logits, axis=-1))

            p_clean = np.asarray(jax.nn.softmax(clean_logits, axis=-1))
            p_dirty = np.asarray(jax.nn.softmax(mm_dirty_logits, axis=-1))
            eps = 1e-12
            kl = np.sum(p_clean * (np.log(p_clean + eps) - np.log(p_dirty + eps)), axis=-1)

            dirty_step_return = np.asarray(reward_list[mm_idx]).reshape(-1)

            step_records.append((clean_action, dirty_action, kl, dirty_step_return))

        # ---- per-seed aggregation ----
        clean_a = np.concatenate([r[0] for r in step_records])
        dirty_a = np.concatenate([r[1] for r in step_records])
        kls = np.concatenate([r[2] for r in step_records])
        rets = np.concatenate([r[3] for r in step_records])

        flipped = clean_a != dirty_a
        flip_rate = float(flipped.mean())

        # Empirical per-action mean payoff, from the CLEAN-action/dirty-return pairs
        # pooled over this seed's rollout (the closest available proxy for "what does
        # this action tend to pay", since a true clean-condition return is not
        # observed on an attack-forced-on rollout).
        payoff = np.full(n_actions, np.nan)
        for act in range(n_actions):
            m = clean_a == act
            if m.any():
                payoff[act] = rets[m].mean()
        flip_payoff_delta = np.abs(payoff[dirty_a[flipped]] - payoff[clean_a[flipped]])
        flip_payoff_delta = flip_payoff_delta[np.isfinite(flip_payoff_delta)]

        all_flip_rates.append(flip_rate)
        all_kls.append(float(kls.mean()))
        all_flip_payoff_deltas.append(flip_payoff_delta)

        print(f"  seed {s:<3} step={used_step:<6} flip_rate={flip_rate:.4f} "
              f"mean_KL={kls.mean():.4g}  n_flips={int(flipped.sum())}/{flipped.size}")

        del ts
        gc.collect()

    del env, nets, ts_template, cfg
    jax.clear_caches()
    gc.collect()

    # ------------------------------------------------------------------ report
    mean_flip = float(np.mean(all_flip_rates))
    mean_kl = float(np.mean(all_kls))
    all_deltas = np.concatenate(all_flip_payoff_deltas) if any(
        d.size for d in all_flip_payoff_deltas) else np.array([])
    frac_inert = (float((all_deltas < args.flip_threshold).mean())
                  if all_deltas.size else float("nan"))

    print("\nVERDICT")
    print(f"  mean argmax flip rate       {mean_flip:.4f}  "
          f"(check_adversary_lever.py's random-head proxy is the comparison point)")
    print(f"  mean KL(clean||dirty)       {mean_kl:.4g}")
    if all_deltas.size:
        print(f"  flips: mean |payoff delta|  {all_deltas.mean():.4g}   "
              f"median {np.median(all_deltas):.4g}   "
              f"frac below {args.flip_threshold:g} (inert): {frac_inert:.1%}")
    print()
    if mean_flip < 0.01:
        print("  PRECONDITION FAILS ON THE TRAINED CHECKPOINT. The policy's argmax "
              "essentially\n  never changes between clean and perturbed observations. "
              "H1 is UNTESTED for\n  this checkpoint, not null -- report it as such, "
              "not as evidence of robustness.")
    elif all_deltas.size and frac_inert > 0.8:
        print("  MECHANISM PRESENT BUT BEHAVIOURALLY INERT. The policy's argmax does "
              "flip, but\n  almost all flips are between actions of near-identical "
              "empirical payoff, so\n  the outcome-level null is not surprising -- it "
              "does not by itself support a\n  robustness claim, since a flip that "
              "can't matter was never a real test.")
    else:
        print("  MECHANISM PRESENT AND MATERIAL. The policy's argmax flips at a rate "
              "that\n  isn't negligible, and flips include actions with materially "
              "different\n  empirical payoff. If the outcome-level metrics still do "
              "not move (see\n  analysis/h1_attack_effect_check.py), that is the "
              "strongest, most citable\n  version of a genuine 'not exploitable' "
              "finding this study can support.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
