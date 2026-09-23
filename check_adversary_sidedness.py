"""What did the TRAINED adversary actually learn to do?

check_adversary_lever.py established that the lever EXISTS. It synthesises a
one-sided spoof and fires it at randomly-initialised policy heads, and says so
in its own docstring: "a PROXY and nothing more ... Whether the converged policy
is fooled is the experiment itself, not a precondition for it." It never touches
a trained checkpoint. This script does.

Why this matters more than it looks
-----------------------------------
Eval 1270 (n=20, all seven arms) found H1 null: the adversary attacks on 99.95%
of permitted steps and moves every outcome metric by less than 0.5 Sharpe
against between-arm gaps of 50+. Three arms got BETTER under attack. But a null
has two incompatible readings and the study as run cannot separate them:

  (a) the market maker is robust to observation-space spoofing, or
  (b) the adversary never learned to attack, so nothing was tested.

(b) is not a remote possibility, it is what the incentives predict. The
adversary's regulatory cost is a profit tax, p_detect * kappa * max(gross, 0),
which is zero when it extracts nothing -- but c_fill (0.001) and c_reg (0.0005)
are charged unconditionally. An adversary that cannot hurt the MM therefore
faces strictly negative expected value from attacking and should rationally stop.
The training logs already show this happening: adv_label_rate was exactly 0.000
in 57.6% of updates. Evaluation then FORCES the gate on (mean_attack_rate_on =
0.9995 against a realised training rate of ~14%), so a policy that learned not
to attack is made to attack anyway, far outside its training distribution.

The discriminating measurement
------------------------------
From spoofing_agent.action_to_injection, injected volume per level is

    a = clip(action, 0, 1) * gate
    bid_vol = a[0:5] * inject_mult * best_bid_depth
    ask_vol = a[5:10] * inject_mult * best_ask_depth

and from mm_env.get_obs the channel the spoof is meant to poison is

    queue_imbalance = (best_bid_vol - best_ask_vol) / (sum + 1e-8)

computed from the PERTURBED best-level volumes. Only level 0 of each side
enters it, so the whole attack reduces to a[0] versus a[5]. With B ~ A this is

    QI ~ (a[0] - a[5]) * mult / (2 + (a[0] + a[5]) * mult)

which is scale-free: it depends on the ASYMMETRY of the action, not on book
depth or absolute injected volume. That is why the lever check's reference
points transfer directly to a trained policy:

    one-sided at inject_mult = 2.0 -> QI 0.000 -> +0.500, flips 24.0% of actions
    symmetric  at inject_mult = 2.0 -> QI stays 0.000,    flips  2.7%

So: measure a[0] - a[5] and the realised QI on a trained checkpoint. If the
adversary converged to roughly symmetric injection, QI under attack is
indistinguishable from clean, reading (b) holds, and H1's null says nothing
about robustness. If it is genuinely one-sided and QI does move to ~0.5 while
outcomes still do not budge, reading (a) holds and the null is a real result.

Either answer is publishable. Not knowing which is not.

    python check_adversary_sidedness.py
    python check_adversary_sidedness.py --project v4_config4_detection --seeds 0-4
"""
from __future__ import annotations

import argparse
import gc

import numpy as np

# queue_imbalance's index in the 45-dim MM observation. Layout from
# mm_env.get_obs: 40 L2 + inventory(40) + time_remaining(41) + queue_imbalance(42)
# + prev_detection_prob(43) + regime(44). Asserted against observation_space()
# at runtime so a layout change fails loudly instead of silently reporting the
# wrong channel.
_QI_IDX = 42
_OBS_DIM = 45


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
    ap.add_argument("--project", default="v4_config3_full",
                    help="checkpoint project holding the trained adversary")
    ap.add_argument("--yaml", default="config/rl_configs/eval_2024_test_v4_config3.yaml",
                    help="eval config; ENV_CONFIG must match the arm's training env")
    ap.add_argument("--seeds", default="0-4",
                    help="seed indices, e.g. '0-4' or '0,3,7' (default first five)")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=None,
                    help="default: the config's NUM_STEPS (one full episode)")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step (default: latest available)")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from gymnax_exchange.jaxrl.MARL.adversarial_eval.rollout import (
        build_eval, load_merged_config, restore_checkpoint, set_attack_mode,
    )

    seeds = _parse_seeds(args.seeds)
    run_names = [f"seed_{s}" for s in seeds]

    print(f"project        {args.project}")
    print(f"yaml           {args.yaml}")
    print(f"seeds          {seeds}\n")

    # rows[mode][seed] = dict of per-seed aggregates
    rows: dict[str, list[dict]] = {"on": [], "off": []}
    inject_mult = None

    for mode in ("on", "off"):
        cfg = set_attack_mode(load_merged_config(args.yaml), mode)
        env, nets, ts_template, cfg = build_eval(cfg, args.n_envs)

        adv_cfg = env.multi_agent_config.dict_of_agents_configs["Spoofing"]
        inject_mult = float(adv_cfg.inject_mult)
        n_lv = int(adv_cfg.n_spoof_levels)

        mm_idx, adv_idx = env._mm_idx, env._adv_idx
        nper = env.multi_agent_config.number_of_agents_per_type
        env_params = env.default_params
        n_steps = int(args.n_steps) if args.n_steps else int(cfg["NUM_STEPS"])

        for s, run_name in zip(seeds, run_names):
            ts, used_step = restore_checkpoint(
                cfg, ts_template, args.project, run_name, args.step)

            rng = jax.random.PRNGKey(s)
            rng, rk = jax.random.split(rng)
            obs_list, env_state = jax.vmap(env.reset, in_axes=(0, None))(
                jax.random.split(rk, args.n_envs), env_params)

            # Verify the obs layout at runtime rather than trusting a constant:
            # _QI_IDX is only meaningful for the 45-dim adversarial_lob obs, and a
            # silently reshaped observation would make this script report the
            # wrong channel with no error at all.
            got = int(obs_list[mm_idx].reshape((args.n_envs * nper[mm_idx], -1)).shape[-1])
            assert got == _OBS_DIM, (
                f"MM obs is {got}-dim, expected {_OBS_DIM}; queue_imbalance index "
                f"{_QI_IDX} is stale -- re-read mm_env.get_obs before trusting this.")

            # Force the gate to the eval regime, exactly as run_rollout does.
            init_gate = 1.0 if mode == "on" else 0.0
            aa = env_state.agent_states[adv_idx].attack_active
            ags = list(env_state.agent_states)
            ags[adv_idx] = env_state.agent_states[adv_idx].replace(
                attack_active=jnp.full_like(aa, init_gate))
            env_state = env_state.replace(agent_states=ags)

            h_states = [nets[i].initialize_carry(args.n_envs * nper[i],
                                                 cfg["GRU_HIDDEN_DIM"])
                        for i in range(len(nets))]
            dones = [jnp.zeros((args.n_envs * nper[i],), dtype=bool)
                     for i in range(len(nets))]

            acts, qis = [], []
            for _ in range(n_steps):
                actions, det_mm = [], None
                for i, t in enumerate(ts):
                    obs_i = obs_list[i].reshape((args.n_envs * nper[i], -1))
                    ac_in = (obs_i[jnp.newaxis, :], dones[i][jnp.newaxis, :])
                    h_new, pi, _, det_prob = t.apply_fn(t.params, h_states[i], ac_in)
                    h_states[i] = h_new
                    if i == mm_idx:
                        det_mm = det_prob[0]
                    a = pi.mode()
                    a = a.reshape((args.n_envs, nper[i], -1))
                    actions.append(a.squeeze(-2) if a.ndim > 2 else a.squeeze())

                # The MM's queue_imbalance for THIS step, i.e. the perturbed value
                # the policy actually conditioned on.
                qis.append(np.asarray(
                    obs_list[mm_idx].reshape((args.n_envs * nper[mm_idx], -1))[:, _QI_IDX]))

                # The adversary's intent BEFORE the telegraph gate. Under mode
                # 'off' the gate zeroes delivery, but the action still says what
                # it WOULD have injected -- which is the thing being measured.
                acts.append(np.asarray(
                    jnp.clip(actions[adv_idx].reshape(args.n_envs, -1), 0.0, 1.0)))

                prev_shape = env_state.agent_states[adv_idx].prev_detection_prob.shape
                ags = list(env_state.agent_states)
                ags[adv_idx] = env_state.agent_states[adv_idx].replace(
                    prev_detection_prob=det_mm.reshape(prev_shape))
                env_state = env_state.replace(agent_states=ags)

                rng, sk = jax.random.split(rng)
                obs_list, env_state, _, done, _ = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None))(
                    jax.random.split(sk, args.n_envs), env_state, actions, env_params)
                dones = [done["agents"][i].reshape(args.n_envs * nper[i])
                         for i in range(len(nets))]

            A = np.concatenate(acts, axis=0)          # (steps*n_envs, 10)
            Q = np.concatenate(qis, axis=0)           # (steps*n_envs,)
            bid, ask = A[:, :n_lv], A[:, n_lv:]
            tot = bid.sum(1) + ask.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                sided = np.where(tot > 1e-9, (bid.sum(1) - ask.sum(1)) / (tot + 1e-12), 0.0)
            rows[mode].append({
                "seed": s, "step": used_step,
                "a0": float(bid[:, 0].mean()), "a5": float(ask[:, 0].mean()),
                "best_asym": float((bid[:, 0] - ask[:, 0]).mean()),
                "abs_best_asym": float(np.abs(bid[:, 0] - ask[:, 0]).mean()),
                "bid_tot": float(bid.sum(1).mean()), "ask_tot": float(ask.sum(1).mean()),
                "sidedness": float(np.abs(sided).mean()),
                "qi_mean": float(Q.mean()), "qi_absmean": float(np.abs(Q).mean()),
            })
            del ts
            gc.collect()

        del env, nets, ts_template, cfg
        jax.clear_caches()
        gc.collect()

    # ------------------------------------------------------------------ report
    print(f"inject_mult    {inject_mult:g}   (a[0], a[5] are clipped actions in [0,1])\n")
    print("1. WHAT THE ADVERSARY INJECTS  (attack forced ON)")
    print("   %-6s %8s %8s %11s %10s %10s %11s" %
          ("seed", "a[0]bid", "a[5]ask", "|a0-a5|", "bid_tot", "ask_tot", "sidedness"))
    for r in rows["on"]:
        print("   %-6d %8.4f %8.4f %11.4f %10.4f %10.4f %11.4f" %
              (r["seed"], r["a0"], r["a5"], r["abs_best_asym"],
               r["bid_tot"], r["ask_tot"], r["sidedness"]))
    m_asym = float(np.mean([r["abs_best_asym"] for r in rows["on"]]))
    m_side = float(np.mean([r["sidedness"] for r in rows["on"]]))

    print("\n2. WHAT THE MARKET MAKER SEES  (queue_imbalance, obs channel %d)" % _QI_IDX)
    print("   %-6s %12s %12s %12s %12s" %
          ("seed", "|QI| off", "|QI| on", "delta", "QI on (signed)"))
    for ron, roff in zip(rows["on"], rows["off"]):
        print("   %-6d %12.4f %12.4f %12.4f %12.4f" %
              (ron["seed"], roff["qi_absmean"], ron["qi_absmean"],
               ron["qi_absmean"] - roff["qi_absmean"], ron["qi_mean"]))
    d_qi = float(np.mean([a["qi_absmean"] - b["qi_absmean"]
                          for a, b in zip(rows["on"], rows["off"])]))

    # Magnitude BEFORE sidedness. A sidedness index is a ratio, so it stays
    # well-defined and can look substantial while the numerator and denominator
    # are both ~0 -- 19% asymmetry of nothing is still nothing. Distinguishing
    # "chose a balanced attack" from "chose not to attack" changes the claim.
    m_mag = float(np.mean([r["bid_tot"] + r["ask_tot"] for r in rows["on"]]))
    m_qi_off = float(np.mean([r["qi_absmean"] for r in rows["off"]]))

    print("\n3. VERDICT")
    print("   mean total action (of 1.0)  %.4f   <- magnitude: is it attacking AT ALL?" % m_mag)
    print("   mean |a[0]-a[5]|            %.4f" % m_asym)
    print("   mean sidedness index        %.4f   (0 = perfectly symmetric, 1 = one side only)" % m_side)
    print("   mean |QI| clean (attack off)%.4f   <- headroom the spoof has to work with" % m_qi_off)
    print("   mean |QI| shift on-vs-off   %+.4f   (lever check: one-sided -> ~+0.50, symmetric -> ~0.00)" % d_qi)
    print()
    if m_mag < 0.05:
        print("   ABSTAINED. The adversary emits a near-zero action (%.4f of a possible" % m_mag)
        print("   %d.0 across %d levels): it has not learned a symmetric attack, it has" % (2 * n_lv, 2 * n_lv))
        print("   learned NOT TO ATTACK. This is what the cost model predicts -- c_fill and")
        print("   c_reg are charged unconditionally while the profit tax is zero when")
        print("   nothing is extracted, so attacking a market maker it cannot move is")
        print("   strictly negative EV and a -> 0 is optimal.")
        print()
        print("   H1 is therefore UNTESTED, not supported or refuted. Report it as 'the")
        print("   constrained adversary did not converge to an attack under this cost")
        print("   model', NOT as 'the market maker is robust'. To test H1 you need an")
        print("   adversary that attacks: re-run this on the cost-free arm")
        print("   (v4_config6_unconstrained, which zeroes c_fill/c_reg/kappa/p_detect).")
        print("   If that one also abstains, the failure is in adversary training or")
        print("   credit assignment, not in the cost model.")
    elif d_qi < 0.05:
        print("   SYMMETRIC. The adversary injects materially (%.4f) but splits it evenly," % m_mag)
        print("   so queue_imbalance -- the channel the attack exists to poison -- does not")
        print("   move. H1's null is uninformative about robustness: reading (b) in this")
        print("   file's header. Report H1 as 'the adversary did not converge to an")
        print("   EFFECTIVE attack', NOT as 'the market maker is robust'.")
    elif d_qi < 0.25:
        print("   PARTIALLY ONE-SIDED. The attack reaches queue_imbalance but well short of")
        print("   the 0.50 a fully one-sided spoof achieves. H1's null is weak evidence of")
        print("   robustness and must be reported with this attenuation stated.")
    else:
        print("   ONE-SIDED. The adversary does move queue_imbalance close to the one-sided")
        print("   reference, so the attack landed and outcomes still did not move. H1's null")
        print("   is then a genuine robustness result and can be reported as one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
