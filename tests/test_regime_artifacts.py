"""
Regime-artifact alignment tests (H4 wiring).

The regime obs channel is only real if three artifacts agree:
  regime_labels.json          date -> {0,1}
  window_to_date_<period>.json  window_index -> date
  the loader cache             defines how many windows exist

A mismatched or missing artifact fails SILENTLY at runtime (the env feeds a
constant-zero regime and config-3 degenerates into config-2), so these tests
assert the failure loudly instead. The window-map tests skip until the maps
are generated (build_window_to_date.py after the period caches are built).
"""
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = os.path.join(ROOT, "regime_labels.json")
TRAIN_MAP = os.path.join(ROOT, "window_to_date_2024_train.json")
TEST_MAP = os.path.join(ROOT, "window_to_date_2024_test.json")
QUARTER_MAPS = {
    "q1": os.path.join(ROOT, "window_to_date_2024_q1.json"),
    "q2": os.path.join(ROOT, "window_to_date_2024_q2.json"),
    "q3": os.path.join(ROOT, "window_to_date_2024_q3.json"),
}
QUARTER_RANGES = {
    "q1": ("2024-01-01", "2024-03-31"),
    "q2": ("2024-04-01", "2024-06-30"),
    "q3": ("2024-07-01", "2024-09-30"),
}


def _load(path):
    with open(path) as f:
        return json.load(f)


def test_labels_cover_2024_and_are_binary():
    labels = _load(LABELS)
    assert len(labels) >= 250, "expected a full trading year of labels"
    assert set(labels.values()) == {0, 1}, "labels must contain BOTH regimes"
    assert all(d.startswith("2024-") for d in labels), (
        "regime_labels.json must be the 2024 label set (window maps are 2024)")


@pytest.mark.parametrize("map_path,name",
                         [(TRAIN_MAP, "train"), (TEST_MAP, "test")]
                         + [(p, n) for n, p in QUARTER_MAPS.items()])
def test_window_map_aligns_with_labels(map_path, name):
    if not os.path.exists(map_path):
        pytest.skip(f"{os.path.basename(map_path)} not generated yet "
                    "(run build_window_to_date.py after the period cache build)")
    labels = _load(LABELS)
    wmap = _load(map_path)
    assert len(wmap) > 0
    # Every window's date must have a label — a date missing from the labels
    # silently becomes regime 0 in AdversarialMARLEnv._build_regime_array.
    missing = sorted({d for d in wmap.values() if d not in labels})
    assert not missing, f"{name}: window dates missing from regime_labels.json: {missing}"
    # Window indices must be a dense 0..n-1 range (the env indexes an array).
    idx = sorted(int(k) for k in wmap)
    assert idx == list(range(len(idx))), f"{name}: window indices not dense 0..n-1"
    # The mapped regime sequence must contain BOTH classes, otherwise the
    # regime channel is constant and H4 is untestable on this split.
    mapped = {labels[d] for d in wmap.values()}
    assert mapped == {0, 1}, (
        f"{name}: regime array would be constant ({mapped}) — "
        "H4 cannot be evaluated on this split")


def test_train_and_test_maps_do_not_overlap():
    if not (os.path.exists(TRAIN_MAP) and os.path.exists(TEST_MAP)):
        pytest.skip("window maps not generated yet")
    train_dates = set(_load(TRAIN_MAP).values())
    test_dates = set(_load(TEST_MAP).values())
    assert not (train_dates & test_dates), (
        "train/test date leakage: " + ", ".join(sorted(train_dates & test_dates)))
    assert max(train_dates) < "2024-10-01" <= min(test_dates), (
        "expected the Oct-1 holdout boundary between train and test")


def test_quarter_maps_partition_the_training_pool():
    """Chained-training invariants: Q1/Q2/Q3 are disjoint, stay inside their
    calendar ranges, never touch the Q4 holdout, and together cover EXACTLY the
    master training pool (no day silently dropped by the slicer)."""
    if not all(os.path.exists(p) for p in QUARTER_MAPS.values()):
        pytest.skip("quarter maps not generated yet (slice_period_cache.py + "
                    "build_window_to_date.py)")
    if not os.path.exists(TRAIN_MAP):
        pytest.skip("master train map missing")
    q_dates = {}
    for name, path in QUARTER_MAPS.items():
        wmap = _load(path)
        idx = sorted(int(k) for k in wmap)
        assert idx == list(range(len(idx))), f"{name}: window indices not dense"
        dates = set(wmap.values())
        lo, hi = QUARTER_RANGES[name]
        assert min(dates) >= lo and max(dates) <= hi, (
            f"{name}: dates escape {lo}..{hi}")
        q_dates[name] = dates
    assert not (q_dates["q1"] & q_dates["q2"]) and not (q_dates["q2"] & q_dates["q3"]) \
        and not (q_dates["q1"] & q_dates["q3"]), "quarters overlap"
    union = q_dates["q1"] | q_dates["q2"] | q_dates["q3"]
    train_dates = set(_load(TRAIN_MAP).values())
    assert union == train_dates, (
        f"quarters do not exactly cover the training pool: "
        f"missing={sorted(train_dates - union)} extra={sorted(union - train_dates)}")


# ---------------------------------------------------------------------------
# Point-in-time labelling (literature audit 2026-09-28 sec 3.13)
# ---------------------------------------------------------------------------

CSV = os.path.join(ROOT, "regime_labels.csv")


def test_default_rule_reproduces_committed_labels():
    import pandas as pd
    from build_regime_labels import label_regimes
    if not os.path.exists(CSV):
        pytest.skip("regime_labels.csv not present")
    df = pd.read_csv(CSV)
    new = label_regimes(df[["date", "close"]])
    assert (new["regime"].to_numpy() == df["regime"].to_numpy()).all()


def test_point_in_time_label_ignores_today_and_future_closes():
    import numpy as np
    import pandas as pd
    from build_regime_labels import label_regimes
    rng = np.random.default_rng(0)
    n = 120
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    dates = pd.date_range("2024-01-02", periods=n, freq="B").strftime("%Y-%m-%d")
    df = pd.DataFrame({"date": dates, "close": close})
    base = label_regimes(df, point_in_time=True)["regime"].to_numpy()
    for t in (45, 70, 100):
        shocked = df.copy()
        shocked.loc[t:, "close"] = shocked.loc[t:, "close"] * np.exp(
            rng.normal(0, 0.2, n - t).cumsum())
        got = label_regimes(shocked, point_in_time=True)["regime"].to_numpy()
        # day t's label uses closes through t-1 only
        assert (got[: t + 1] == base[: t + 1]).all(), f"label at/before {t} moved"
    # the default rule DOES move day t's label on a same-day shock somewhere
    moved = False
    for t in range(30, n):
        shocked = df.copy()
        shocked.loc[t, "close"] *= 1.5
        if label_regimes(shocked)["regime"].to_numpy()[t] != label_regimes(df)["regime"].to_numpy()[t]:
            moved = True
            break
    assert moved, "expected the default (same-day) rule to be sensitive to day t's close"


def test_leak_summary_is_small_on_the_test_period():
    import pandas as pd
    from analysis.regime_label_leak import leak_summary
    if not os.path.exists(CSV):
        pytest.skip("regime_labels.csv not present")
    res = leak_summary(pd.read_csv(CSV))
    assert res["n_test_days"] == 64
    # the numbers quoted in the thesis limitations
    assert res["variants"]["lag"]["changed_test"] == 4
    assert res["variants"]["train_thr"]["changed_test"] == 2
    assert res["variants"]["point_in_time"]["changed_test"] == 6
