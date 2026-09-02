"""The bobRL action ladder must honour auto_liquidate_threshold.

This exists because it did not. auto_liquidate_threshold was implemented only
inside _getActionMsgs_fixedQuant, while every production arm runs
action_space="bobRL", which dispatches to _getActionMsgs_BobRL. Setting the key
on a bobRL config therefore did nothing at all: no error, no warning, the value
was simply never read. The v2 sweep ran with a passive-only ladder and no
position limit, inventory ratcheted one way across every episode, and 12 of 20
baseline seeds gave up quoting.

A config key that is silently ignored on the path you actually run is the worst
kind of bug in this project - it costs a full sweep to discover and looks like
a failed hypothesis rather than a missing implementation. These tests pin the
behaviour so it cannot regress to a no-op again.
"""

import jax.numpy as jnp
import pytest

from gymnax_exchange.jaxen.mm_env import MarketMakingAgent
from gymnax_exchange.jaxen.StatesandParams import MMEnvState, MMEnvParams, WorldState
from gymnax_exchange.jaxob.jaxob_config import AdversarialMMConfig, World_EnvironmentConfig

_TRADER_ID = -100
_BEST_BID, _BEST_ASK = 9900, 10100
_IOC, _LIMIT = 4, 1


def _agent(threshold, alpha=1.0):
    cfg = AdversarialMMConfig(
        action_space="bobRL", bob_v0=2, n_actions=5,
        observation_space="adversarial_lob",
        auto_liquidate_threshold=threshold, auto_liquidate_alpha=alpha,
    )
    return MarketMakingAgent(cfg=cfg, world_config=World_EnvironmentConfig())


def _world():
    """A two-sided book owned by somebody other than the agent."""
    empty = jnp.full((100, 8), -1, dtype=jnp.int32)
    # columns: P, Q, OID, TID, SEC, NSEC
    bids = empty.at[0].set(jnp.array([_BEST_BID, 10, 1, -50, 0, 0, 0, 0], dtype=jnp.int32))
    asks = empty.at[0].set(jnp.array([_BEST_ASK, 10, 2, -50, 0, 0, 0, 0], dtype=jnp.int32))
    best = jnp.tile(jnp.array([[_BEST_BID, 10]], dtype=jnp.int32), (5, 1))
    return WorldState(
        ask_raw_orders=asks, bid_raw_orders=bids,
        trades=jnp.full((10, 8), -1, dtype=jnp.int32),
        init_time=jnp.zeros(2, dtype=jnp.int32), window_index=0,
        max_steps_in_episode=6400, start_index=0, step_counter=0,
        best_bids=best, best_asks=best, time=jnp.zeros(2, dtype=jnp.int32),
        order_id_counter=0, mid_price=jnp.float32(10000.0),
        delta_time=jnp.float32(1.0),
    )


def _state(inventory):
    return MMEnvState(posted_distance_bid=0, posted_distance_ask=0,
                      inventory=inventory, total_PnL=0.0, cash_balance=0.0)


def _params():
    return MMEnvParams(trader_id=jnp.array([_TRADER_ID]),
                       time_delay_obs_act=jnp.array([0]),
                       normalize=jnp.array([True]))


def _act(threshold, inventory, action=0):
    msgs, extras = _agent(threshold)._getActionMsgs_BobRL(
        jnp.asarray(action), _world(), _state(inventory), _params())
    # columns of action_msgs: type, side, quant, price, order_id, trader_id, ...
    return {"types": msgs[:, 0], "sides": msgs[:, 1], "quants": msgs[:, 2],
            "prices": msgs[:, 3], "extras": extras}


def test_over_threshold_sends_ioc():
    """Long 200 against a limit of 50 must liquidate at market, not post quotes."""
    r = _act(threshold=50, inventory=200)
    assert list(r["types"]) == [_IOC, _IOC], (
        "expected IOC orders once inventory exceeded the position limit, got "
        f"types={list(r['types'])}")


def test_over_threshold_sells_the_whole_long():
    """alpha=1.0 flattens fully: the sell leg carries the entire position."""
    r = _act(threshold=50, inventory=200)
    sell_leg = int(r["quants"][1])          # sides = [-1, 1]; index 1 sells
    assert sell_leg == 200, f"expected to sell all 200, got {sell_leg}"
    assert int(r["quants"][0]) == 0, "should not also be buying while long"


def test_short_position_buys_back():
    """A short must cover on the ask side, not the bid side."""
    r = _act(threshold=50, inventory=-200)
    assert int(r["quants"][0]) == 200, "expected to buy 200 back to cover the short"
    assert int(r["quants"][1]) == 0, "should not also be selling while short"


def test_liquidation_prices_cross_the_book():
    """An IOC that does not cross is just a resting order under another name."""
    r = _act(threshold=50, inventory=200)
    assert int(r["prices"][1]) < _BEST_BID, (
        "sell leg must price through the bid to clear, got "
        f"{int(r['prices'][1])} against best_bid {_BEST_BID}")


def test_under_threshold_still_quotes():
    """Below the limit nothing changes - the ladder posts limit orders as before."""
    r = _act(threshold=50, inventory=10)
    assert list(r["types"]) == [_LIMIT, _LIMIT]
    assert int(r["extras"]["posted_bid_price"]) == _BEST_BID
    assert int(r["extras"]["posted_ask_price"]) == _BEST_ASK


def test_liquidation_is_not_counted_as_a_quote():
    """quote_presence is the study's validity gate.

    posted_bid_price / posted_ask_price are derived from the LADDER quantities,
    which a liquidation step does not change. Reporting them unmodified would
    count a forced unwind as a two-sided quote and inflate the one metric the
    preregistration uses to decide whether a result is interpretable at all.
    """
    r = _act(threshold=50, inventory=200)
    assert int(r["extras"]["posted_bid_price"]) == 0
    assert int(r["extras"]["posted_ask_price"]) == 0


def test_threshold_zero_disables_the_limit():
    """0 means off, matching mm_env.py's `if threshold != 0` and the shipped configs."""
    r = _act(threshold=0, inventory=100000)
    assert list(r["types"]) == [_LIMIT, _LIMIT], (
        "threshold=0 must leave the ladder untouched no matter the inventory")


@pytest.mark.parametrize("inventory", [51, 200, 5000])
def test_limit_binds_at_every_size_above_it(inventory):
    r = _act(threshold=50, inventory=inventory)
    assert list(r["types"]) == [_IOC, _IOC]


def test_boundary_is_strictly_greater_than():
    """mm_env.py uses `abs(inventory) > threshold`, so exactly at the limit still quotes."""
    assert list(_act(threshold=50, inventory=50)["types"]) == [_LIMIT, _LIMIT]
    assert list(_act(threshold=50, inventory=51)["types"]) == [_IOC, _IOC]
