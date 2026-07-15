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


def _load(path):
    with open(path) as f:
        return json.load(f)


def test_labels_cover_2024_and_are_binary():
    labels = _load(LABELS)
    assert len(labels) >= 250, "expected a full trading year of labels"
    assert set(labels.values()) == {0, 1}, "labels must contain BOTH regimes"
    assert all(d.startswith("2024-") for d in labels), (
        "regime_labels.json must be the 2024 label set (window maps are 2024)")


@pytest.mark.parametrize("map_path,name", [(TRAIN_MAP, "train"), (TEST_MAP, "test")])
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
