"""The v4 `spread_skew` action space.

v1, v2 and v3 each failed on something outside the function being changed, so
these tests deliberately cover the surroundings as much as the arithmetic:

  * n_actions. spread_skew decodes a 2x3 grid (spread_type = action // 3,
    skew_type = action % 3), so it needs exactly 6. MarketMaking_EnvironmentConfig
    .__post_init__ -- which AdversarialMMConfig inherits -- already forces 6, so
    the constructor guard is defence-in-depth rather than a live fix, and cannot
    be reached through a normal config. It is still worth having: it documents the
    invariant and would catch a future post_init edit. The tests below therefore
    bypass __post_init__ to exercise it, and separately assert that post_init does
    the sanitising, so if that ever stops being true something fails loudly.

  * posted_bid_price / posted_ask_price. These were hardcoded to 0, and
    quote_presence -- the study's validity gate -- is the fraction of steps with
    both above zero. Left alone, every v4 arm would have reported a total quoting
    collapse regardless of behaviour, indistinguishable from the real v2 failure.

  * auto_liquidate_threshold. Implemented only in _getActionMsgs_fixedQuant, so
    on any other dispatch path it is read from config and silently never used.

  * Shapes. Under the live vmap both `inventory` and `action` arrive as (1,), not
    as scalars. A fixture using scalars passes against code that cannot run in
    production; that cost two cluster launches during v3. Every test here runs
    under BOTH shapes.

Book fixture: best_bid 9900, best_ask 10100, tick 100, so mid = 10000 and
current_spread = 200.
"""

import jax.numpy as jnp
import pytest

from gymnax_exchange.jaxen.mm_env import MarketMakingAgent
from gymnax_exchange.jaxen.StatesandParams import MMEnvState, MMEnvParams, WorldState
from gymnax_exchange.jaxob.jaxob_config import AdversarialMMConfig, World_EnvironmentConfig

_TRADER_ID = -100
_BEST_BID, _BEST_ASK = 9900, 10100
_MID, _SPREAD = 10000, 200
_IOC, _LIMIT = 4, 1

_SHAPES = {"scalar": lambda v: v, "vmapped": lambda v: jnp.array([v])}

# spread_type = action // 3 (0 tight, 1 wide); skew_type = action % 3
# (0 bid skew, 1 neutral, 2 ask skew). With tick multiplier_type, skew shifts the
# mid by skew_multiplier (5) ticks of 100, and wide triples the spread.
_EXPECTED = {           # action: (bid_price, ask_price)
    0: (9400, 9600),    # tight, bid skew   mid 9500  +/- 100
    1: (9900, 10100),   # tight, neutral    mid 10000 +/- 100  == the touch
    2: (10400, 10600),  # tight, ask skew   mid 10500 +/- 100
    3: (9200, 9800),    # wide,  bid skew   mid 9500  +/- 300
    4: (9700, 10300),   # wide,  neutral    mid 10000 +/- 300
    5: (10200, 10800),  # wide,  ask skew   mid 10500 +/- 300
}


def _s(x):
    """First element as a Python int; extras carry the rank of their inputs."""
    return int(jnp.ravel(jnp.asarray(x))[0])


def _agent(**over):
    kw = dict(action_space="spread_skew", n_actions=6, num_action_messages_by_agent=2,
              observation_space="adversarial_lob", auto_liquidate_threshold=0,
              auto_liquidate_alpha=1.0, fixed_quant_value=10)
    kw.update(over)
    return MarketMakingAgent(cfg=AdversarialMMConfig(**kw),
                             world_config=World_EnvironmentConfig())


def _world(empty=False):
    """A two-sided book owned by somebody other than the agent."""
    e = jnp.full((100, 8), -1, dtype=jnp.int32)
    if empty:
        bids = asks = e
    else:
        bids = e.at[0].set(jnp.array([_BEST_BID, 10, 1, -50, 0, 0, 0, 0], dtype=jnp.int32))
        asks = e.at[0].set(jnp.array([_BEST_ASK, 10, 2, -50, 0, 0, 0, 0], dtype=jnp.int32))
    best_b = jnp.tile(jnp.array([[_BEST_BID, 10]], dtype=jnp.int32), (5, 1))
    best_a = jnp.tile(jnp.array([[_BEST_ASK, 10]], dtype=jnp.int32), (5, 1))
    return WorldState(
        ask_raw_orders=asks, bid_raw_orders=bids,
        trades=jnp.full((10, 8), -1, dtype=jnp.int32),
        init_time=jnp.zeros(2, dtype=jnp.int32), window_index=0,
        max_steps_in_episode=6400, start_index=0, step_counter=0,
        best_bids=best_b, best_asks=best_a, time=jnp.zeros(2, dtype=jnp.int32),
        order_id_counter=0, mid_price=jnp.float32(float(_MID)),
        delta_time=jnp.float32(1.0),
    )


def _act(action, inventory=0, shape="vmapped", empty=False, **over):
    msgs, extras = _agent(**over)._getActionMsgs_spread_skew(
        jnp.asarray(_SHAPES[shape](action)), _world(empty),
        MMEnvState(posted_distance_bid=0, posted_distance_ask=0,
                   inventory=_SHAPES[shape](inventory), total_PnL=0.0, cash_balance=0.0),
        MMEnvParams(trader_id=jnp.array([_TRADER_ID]),
                    time_delay_obs_act=jnp.array([0]), normalize=jnp.array([True])))
    assert msgs.shape == (2, 8), (
        f"expected 2 messages x 8 fields, got {msgs.shape} -- a liquidation array "
        "picked up the wrong shape and the reshape multiplied the rows")
    assert msgs.dtype == jnp.int32, f"messages must stay int32 for the book, got {msgs.dtype}"
    return {"types": msgs[:, 0], "sides": msgs[:, 1], "quants": msgs[:, 2],
            "prices": msgs[:, 3], "extras": extras}


# --------------------------------------------------------------- action mapping
@pytest.mark.parametrize("shape", list(_SHAPES))
@pytest.mark.parametrize("action", sorted(_EXPECTED))
def test_every_action_prices_as_specified(action, shape):
    r = _act(action, shape=shape)
    want_bid, want_ask = _EXPECTED[action]
    assert (int(r["prices"][0]), int(r["prices"][1])) == (want_bid, want_ask), (
        f"action {action}: expected bid/ask {want_bid}/{want_ask}, got "
        f"{int(r['prices'][0])}/{int(r['prices'][1])}")


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_neutral_tight_quotes_the_touch(shape):
    """Action 1 is the identity case and anchors the whole mapping."""
    r = _act(1, shape=shape)
    assert int(r["prices"][0]) == _BEST_BID
    assert int(r["prices"][1]) == _BEST_ASK


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_wide_is_wider_than_tight_at_every_skew(shape):
    for skew in range(3):
        tight = _act(skew, shape=shape)["prices"]
        wide = _act(3 + skew, shape=shape)["prices"]
        tw = int(tight[1]) - int(tight[0])
        ww = int(wide[1]) - int(wide[0])
        assert ww > tw, f"skew {skew}: wide gap {ww} not wider than tight {tw}"


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_tight_gap_is_the_market_spread_and_wide_is_tripled(shape):
    tight = _act(1, shape=shape)["prices"]
    wide = _act(4, shape=shape)["prices"]
    assert int(tight[1]) - int(tight[0]) == _SPREAD
    assert int(wide[1]) - int(wide[0]) == 3 * _SPREAD   # spread_multiplier = 3.0


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_skew_direction_matches_the_docstring(shape):
    """0 = bid skew shifts DOWN, 2 = ask skew shifts UP. An inline comment in the
    source used to claim the opposite mapping, so pin the real one."""
    bid_skew, neutral, ask_skew = (_act(a, shape=shape)["prices"] for a in (0, 1, 2))
    assert int(bid_skew[0]) < int(neutral[0]) and int(bid_skew[1]) < int(neutral[1])
    assert int(ask_skew[0]) > int(neutral[0]) and int(ask_skew[1]) > int(neutral[1])


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_quotes_straddle_the_skewed_mid(shape):
    for action in sorted(_EXPECTED):
        p = _act(action, shape=shape)["prices"]
        assert int(p[0]) < int(p[1]), f"action {action}: bid must sit below ask"


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_both_legs_are_limit_orders_when_not_liquidating(shape):
    r = _act(1, shape=shape)
    assert list(r["types"]) == [_LIMIT, _LIMIT]
    assert list(r["sides"]) == [1, -1]


# ------------------------------------------------------------- the validity gate
@pytest.mark.parametrize("shape", list(_SHAPES))
@pytest.mark.parametrize("action", sorted(_EXPECTED))
def test_posted_prices_are_reported_not_hardcoded_zero(action, shape):
    """quote_presence is derived from these. Hardcoded zeros would report a total
    quoting collapse in every arm regardless of behaviour."""
    r = _act(action, shape=shape)
    want_bid, want_ask = _EXPECTED[action]
    assert _s(r["extras"]["posted_bid_price"]) == want_bid
    assert _s(r["extras"]["posted_ask_price"]) == want_ask


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_empty_book_quotes_nothing(shape):
    r = _act(1, shape=shape, empty=True)
    assert _s(r["quants"][0]) == 0 and _s(r["quants"][1]) == 0
    assert _s(r["extras"]["posted_bid_price"]) == 0
    assert _s(r["extras"]["posted_ask_price"]) == 0
    assert bool(jnp.ravel(jnp.asarray(r["extras"]["empty_book"]))[0]) is True


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_empty_book_flag_is_computed_not_hardcoded_false(shape):
    assert bool(jnp.ravel(jnp.asarray(_act(1, shape=shape)["extras"]["empty_book"]))[0]) is False


# ---------------------------------------------------------------- position limit
@pytest.mark.parametrize("shape", list(_SHAPES))
def test_over_threshold_sends_ioc(shape):
    r = _act(1, inventory=200, shape=shape, auto_liquidate_threshold=50)
    assert list(r["types"]) == [_IOC, _IOC]


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_long_is_sold_and_short_is_bought(shape):
    long_ = _act(1, inventory=200, shape=shape, auto_liquidate_threshold=50)
    assert _s(long_["quants"][1]) == 200 and _s(long_["quants"][0]) == 0
    short = _act(1, inventory=-200, shape=shape, auto_liquidate_threshold=50)
    assert _s(short["quants"][0]) == 200 and _s(short["quants"][1]) == 0


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_liquidation_crosses_the_book(shape):
    r = _act(1, inventory=200, shape=shape, auto_liquidate_threshold=50)
    assert int(r["prices"][1]) < _BEST_BID, "sell leg must price through the bid"


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_liquidation_is_not_counted_as_a_quote(shape):
    r = _act(1, inventory=200, shape=shape, auto_liquidate_threshold=50)
    assert _s(r["extras"]["posted_bid_price"]) == 0
    assert _s(r["extras"]["posted_ask_price"]) == 0


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_under_threshold_still_quotes(shape):
    r = _act(1, inventory=10, shape=shape, auto_liquidate_threshold=50)
    assert list(r["types"]) == [_LIMIT, _LIMIT]
    assert _s(r["extras"]["posted_bid_price"]) == _BEST_BID


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_threshold_zero_disables_the_limit(shape):
    r = _act(1, inventory=100000, shape=shape, auto_liquidate_threshold=0)
    assert list(r["types"]) == [_LIMIT, _LIMIT]


# ------------------------------------------------------------------ config guards
def test_post_init_sanitises_n_actions_to_six():
    """The first line of defence. If this ever stops holding, the guard below is
    what stands between a truncated action space and a silently wasted sweep."""
    for passed in (5, 6, 7):
        cfg = AdversarialMMConfig(action_space="spread_skew", n_actions=passed)
        assert cfg.n_actions == 6, f"post_init left n_actions={cfg.n_actions}"
        assert cfg.num_action_messages_by_agent == 2


def _unsanitised(**over):
    """A config with post_init's correction undone, to reach the guard.

    The dataclass is frozen, so post_init itself uses object.__setattr__; the same
    door is the only way to construct the broken config a future edit might allow.
    """
    cfg = AdversarialMMConfig(action_space="spread_skew")
    for k, v in over.items():
        object.__setattr__(cfg, k, v)
    return cfg


def test_n_actions_guard_rejects_a_truncated_action_space():
    """At n_actions=5 the policy can never emit action 5, so "wide spread, ask
    skew" is unreachable and the arm trains on five sixths of its space."""
    for bad in (5, 7):
        with pytest.raises(ValueError, match="n_actions"):
            MarketMakingAgent(cfg=_unsanitised(n_actions=bad),
                              world_config=World_EnvironmentConfig())
    MarketMakingAgent(cfg=_unsanitised(n_actions=6),
                      world_config=World_EnvironmentConfig())   # must not raise


def test_message_count_guard_rejects_a_shape_mismatch():
    """order_ids is built with num_action_messages_by_agent and stacked against
    2-element arrays, so anything else is an unreadable error inside jnp.stack."""
    with pytest.raises(ValueError, match="num_action_messages_by_agent"):
        MarketMakingAgent(cfg=_unsanitised(num_action_messages_by_agent=4),
                          world_config=World_EnvironmentConfig())


def test_unknown_multiplier_type_fails_loudly():
    """'spread' is marked WRONG in jaxob_config; anything unrecognised used to leave
    skewed_mid undefined and surface as a NameError inside tracing."""
    with pytest.raises(ValueError, match="multiplier_type"):
        _act(1, multiplier_type="nonsense")
