"""Does the spoofing adversary actually have a lever on this action space?

A precondition recorded in preregistration.json -> amendments ->
v4_spread_skew_amendment. v3 failed H1 not because adversarial co-training does
nothing but because the adversary could not reach the thing that matters: the
bobRL ladder takes prices from the REAL book and varies quantity only, so a
perturbed observation could nudge which of five quantity pairs the policy chose
and nothing else (observed d = 0.242, Holm p = 0.939). Running another six-arm
sweep without checking that v4 fixes this would repeat the mistake at full cost.

Two necessary conditions, measured separately:

  1. REACH. Does the adversary's injection survive into the MM's observation
     after normalisation? If the depth channels are squashed flat, nothing
     downstream can respond and the study is over before it starts. This is
     deterministic and decisive.

  2. SENSITIVITY. Given a changed observation, does the selected action change?
     Measured over many randomly-initialised policy heads, which is a PROXY and
     nothing more: it says the architecture can express sensitivity to these
     channels, not that a trained policy will use it. Whether the converged
     policy is fooled is the experiment itself, not a precondition for it.

ONE-SIDED vs SYMMETRIC matters more than adversary strength. mm_env.py computes
queue_imbalance from the PERTURBED depth precisely so the spoof reaches it, and a
symmetric injection leaves imbalance at exactly zero -- the weakest attack there
is. Measured at the configured inject_mult = 2.0: symmetric moves the observation
0.063 and flips a random head 2.7%; bid-only moves it 0.502 (half the obs norm),
takes queue_imbalance 0.000 -> +0.500, and flips 24.0%. Raising inject_mult from
2 to 5 only takes that to 27.7%, so strength is near its plateau and the lever is
about SIDEDNESS. --side defaults to bid for that reason; --side both measures the
floor, not the realistic case.

What makes v4 different from v3 is the third link, and that one is already
proven by tests/test_spread_skew.py: under spread_skew a changed action changes
quoted PRICES (six actions, six distinct bid/ask pairs). Under bobRL it changed
only quantities. Reach x sensitivity x price-effect is the full chain.

    python check_adversary_lever.py
    python check_adversary_lever.py --config config/env_configs/adversarial_mm_v4_config1.json
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

# L2 layout is [ask_price, ask_vol, bid_price, bid_vol] per level, so the volume
# channels the adversary writes into are these (adversarial_marl_env.step_env).
_ASK_VOL_IDX = jnp.array([1, 5, 9, 13, 17])
_BID_VOL_IDX = jnp.array([3, 7, 11, 15, 19])


def _fixtures(cfg, world_cfg, depth):
    from gymnax_exchange.jaxen.StatesandParams import MMEnvState, MMEnvParams, WorldState

    mid, tick = 10000, world_cfg.tick_size
    l2 = jnp.zeros(40, dtype=jnp.float32)
    for lvl in range(5):
        l2 = l2.at[4 * lvl + 0].set(mid + (lvl + 1) * tick)      # ask price
        l2 = l2.at[4 * lvl + 1].set(depth)                        # ask volume
        l2 = l2.at[4 * lvl + 2].set(mid - (lvl + 1) * tick)      # bid price
        l2 = l2.at[4 * lvl + 3].set(depth)                        # bid volume

    empty = jnp.full((100, 8), -1, dtype=jnp.int32)
    bids = empty.at[0].set(jnp.array([mid - tick, depth, 1, -50, 0, 0, 0, 0], dtype=jnp.int32))
    asks = empty.at[0].set(jnp.array([mid + tick, depth, 2, -50, 0, 0, 0, 0], dtype=jnp.int32))
    world = WorldState(
        ask_raw_orders=asks, bid_raw_orders=bids,
        trades=jnp.full((10, 8), -1, dtype=jnp.int32),
        init_time=jnp.zeros(2, dtype=jnp.int32), window_index=0,
        max_steps_in_episode=6400, start_index=0, step_counter=0,
        best_bids=jnp.tile(jnp.array([[mid - tick, depth]], dtype=jnp.int32), (5, 1)),
        best_asks=jnp.tile(jnp.array([[mid + tick, depth]], dtype=jnp.int32), (5, 1)),
        time=jnp.zeros(2, dtype=jnp.int32), order_id_counter=0,
        mid_price=jnp.float32(mid), delta_time=jnp.float32(1.0))
    # scalar inventory: get_adversarial_observation concatenates it with other
    # scalars, and a (1,) here makes the concatenate ranks disagree.
    state = MMEnvState(posted_distance_bid=0, posted_distance_ask=0,
                       inventory=0, total_PnL=0.0, cash_balance=0.0)
    params = MMEnvParams(trader_id=jnp.array([-100]),
                         time_delay_obs_act=jnp.array([0]), normalize=jnp.array([True]))
    return l2, world, state, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/env_configs/adversarial_mm_v4_config1.json")
    ap.add_argument("--depth", type=float, default=100.0,
                    help="resting volume per level; the adversary injects a multiple of this")
    ap.add_argument("--n-inits", type=int, default=400, help="random policy heads to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--side", choices=["bid","ask","both"], default="bid",
                    help="which side to inflate. A real spoofer works ONE side to "
                         "fake directional pressure; injecting both symmetrically "
                         "leaves queue_imbalance untouched and is the weakest "
                         "possible attack.")
    ap.add_argument("--inject-mult", type=float, default=None,
                    help="override the config's inject_mult, to ask how the lever "
                         "scales with adversary strength (a pre-registered axis)")
    args = ap.parse_args()

    from gymnax_exchange.jaxob.config_io import load_config_from_file
    from gymnax_exchange.jaxen.mm_env import MarketMakingAgent
    from gymnax_exchange.jaxrl.MARL.attack_aware_policy import AttackAwarePolicyNet

    env_cfg = load_config_from_file(args.config)
    mm_cfg = env_cfg.dict_of_agents_configs["AdversarialMM"]
    adv_cfg = env_cfg.dict_of_agents_configs["Spoofing"]
    agent = MarketMakingAgent(cfg=mm_cfg, world_config=env_cfg.world_config)

    # Resolved once, before anything is printed or computed, so the report can
    # never name a different multiplier than the one it used.
    mult = float(adv_cfg.inject_mult if args.inject_mult is None else args.inject_mult)
    src = "config" if args.inject_mult is None else "--inject-mult OVERRIDE"

    print(f"config         {args.config}")
    print(f"action_space   {mm_cfg.action_space}  n_actions={mm_cfg.n_actions}")
    print(f"inject_mult    {mult:g}  ({src})   resting depth per level {args.depth:g}\n")

    l2, world, state, params = _fixtures(mm_cfg, env_cfg.world_config, args.depth)

    # Full-strength attack: the adversary's action is clipped to [0,1] then scaled
    # by inject_mult x best-quote depth, so this is the largest perturbation it can
    # legally make -- the most generous case for finding a lever.
    inject = mult * args.depth
    spoofed = l2
    if args.side in ("bid", "both"):
        spoofed = spoofed.at[_BID_VOL_IDX].add(inject)
    if args.side in ("ask", "both"):
        spoofed = spoofed.at[_ASK_VOL_IDX].add(inject)

    def _qi(v):
        return float((v[3] - v[1]) / (v[3] + v[1] + 1e-8))
    print(f"   spoof side     {args.side}   queue_imbalance "
          f"{_qi(l2):+.3f} -> {_qi(spoofed):+.3f}\n")

    kw = dict(normalize=True)
    clean_obs = agent.get_adversarial_observation(
        world, state, params, l2, jnp.float32(0.0), jnp.float32(0.0), **kw)
    dirty_obs = agent.get_adversarial_observation(
        world, state, params, spoofed, jnp.float32(0.0), jnp.float32(0.0), **kw)

    diff = np.asarray(dirty_obs - clean_obs, dtype=np.float64)
    changed = int((np.abs(diff) > 1e-9).sum())
    print("1. REACH -- does the injection survive into the observation?")
    print(f"   observation dim            {clean_obs.shape[0]}")
    print(f"   channels changed           {changed}/{clean_obs.shape[0]}")
    print(f"   max |change|               {np.abs(diff).max():.6g}")
    print(f"   L2 norm of change          {np.linalg.norm(diff):.6g}")
    print(f"   clean obs L2 norm          {np.linalg.norm(np.asarray(clean_obs)):.6g}")
    if changed == 0:
        print("\n   VERDICT: the adversary cannot reach the observation. Do not run the")
        print("   sweep -- normalisation or the obs builder is discarding the injection.")
        return

    # --- 2. sensitivity over random heads ---------------------------------
    # __call__(hidden, x) where x = (obs, dones); hidden is a pass-through carry
    # for the MLP, so its size only has to be self-consistent.
    net = AttackAwarePolicyNet(action_dim=mm_cfg.n_actions, config={})
    hidden = AttackAwarePolicyNet.initialize_carry(1, 256)
    dones = jnp.zeros((1,), dtype=bool)
    x_clean = (clean_obs[None, :], dones)
    x_dirty = (dirty_obs[None, :], dones)

    def argmax_of(variables, x):
        _, pi, _, _ = net.apply(variables, hidden, x)
        return int(jnp.argmax(pi.logits, axis=-1)[0])

    flips, per_action = 0, np.zeros(mm_cfg.n_actions, dtype=int)
    for i in range(args.n_inits):
        variables = net.init(jax.random.PRNGKey(args.seed + i), hidden, x_clean)
        a_clean = argmax_of(variables, x_clean)
        flips += int(a_clean != argmax_of(variables, x_dirty))
        per_action[a_clean] += 1

    rate = flips / args.n_inits
    print(f"\n2. SENSITIVITY -- does a changed observation change the chosen action?")
    print(f"   random policy heads        {args.n_inits}")
    print(f"   argmax flipped             {flips} ({rate:.1%})")
    print(f"   clean-action spread        {per_action.tolist()}")
    print("   PROXY ONLY: random heads say the architecture CAN be sensitive to")
    print("   these channels, not that a trained policy will be. Whether the")
    print("   converged policy is fooled is the experiment, not its precondition.")

    print("\n3. PRICE EFFECT -- covered by tests/test_spread_skew.py: the six")
    print("   actions map to six distinct bid/ask pairs, so an action change moves")
    print("   quoted prices. Under bobRL it moved quantities only, which is why")
    print("   H1 was close to untestable in v3.")

    ok = changed > 0 and flips > 0
    print("\n" + "=" * 70)
    print(("PRECONDITION MET: the adversary reaches the observation and can change "
           "the\nchosen action, and a changed action moves prices. The sweep tests "
           "something.") if ok else
          ("PRECONDITION NOT MET: the injection reaches the observation but never "
           "changes\nthe action even at full strength. v4 would fail for v3's "
           "reason; do not sweep."))
    print("=" * 70)


if __name__ == "__main__":
    main()
