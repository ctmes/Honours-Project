"""Unit tests for the analysis layer (analysis/evalreport.py, figures.py, tables.py).

These guard the properties that a wrong figure would silently violate:

  1. Pairing is by seed index and is refused when seed counts disagree.
  2. Non-finite values are surfaced, never silently dropped.
  3. The validity gate implements the pre-registered rule, and its sensitivity
     sweep reports when the verdict depends on the cut.
  4. The ratchet/no-trade discriminator distinguishes the two cases.
  5. Partial reports degrade to skipped figures instead of raising.
  6. LaTeX escaping round-trips, and a table cell never leaks raw markup into
     the plain-text view.

A synthetic report is used rather than a fixture file so the tests state their
own preconditions and do not depend on anything in D:/tmp.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

from analysis.evalreport import (
    COLLAPSE_CUT, EvalReport, metric_label,
)
from analysis import figures as F
from analysis import tables as T


_METRICS = ("sortino", "sharpe", "cvar", "peak_inventory", "inventory_sd",
            "quote_displacement", "quote_presence", "mean_attack_rate",
            "mean_injected_volume", "injected_volume_per_attack",
            "sortino_lowvol", "sortino_highvol", "regime_gap")


def _arm(n=8, quote_presence=None, inventory_sd=None, rng=None):
    """One arm's per-seed block with the real metric key set."""
    rng = rng or np.random.default_rng(0)
    qp = np.full(n, 0.8) if quote_presence is None else np.asarray(quote_presence,
                                                                   dtype=float)
    isd = np.full(n, 5.0) if inventory_sd is None else np.asarray(inventory_sd,
                                                                  dtype=float)
    block = {}
    for m in _METRICS:
        for cond in ("on", "off"):
            if m == "quote_presence":
                v = qp
            elif m == "inventory_sd":
                v = isd
            elif m == "quote_displacement":
                # Undefined exactly where there is no two-sided quote -- the
                # real report's NaN pattern.
                v = np.where(qp <= COLLAPSE_CUT, np.nan, 0.7)
            else:
                v = rng.normal(size=n)
            block[f"{m}_{cond}"] = list(map(float, v))
    block["auroc"] = list(map(float, rng.uniform(0.45, 0.55, size=n)))
    return block


def _report(arms=("baseline", "adversarial"), n=8, **per_arm):
    data = {
        "per_seed": {a: per_arm.get(a) or _arm(n) for a in arms},
        "summaries": {},
        "holm": {},
        "_meta": {"signed_off": True, "partial_run": False, "confirmatory": True,
                  "arms": list(arms), "seeds": list(range(n)),
                  "checkpoint_step": 1002},
    }
    return EvalReport(data, source="synthetic.json")


# ---------------------------------------------------------------- structure

def test_missing_per_seed_is_rejected_with_actionable_message():
    with pytest.raises(ValueError, match="per_seed"):
        EvalReport({"summaries": {}}, source="old.json")


def test_arms_follow_design_order_not_dict_order():
    r = _report(arms=("full", "baseline", "adversarial"))
    assert r.arms == ("baseline", "adversarial", "full")


def test_unknown_arm_is_kept_not_dropped():
    r = _report(arms=("baseline", "mystery"))
    assert "mystery" in r.arms


def test_factorial_arms_excludes_the_fixed_policy_benchmark():
    r = _report(arms=("baseline", "as"))
    assert r.factorial_arms() == ("baseline",)
    assert "as" in r.arms


def test_confirmatory_requires_signed_off_and_complete():
    r = _report()
    assert r.is_confirmatory
    r._d["_meta"]["partial_run"] = True
    assert not r.is_confirmatory


# ------------------------------------------------------------------ pairing

def test_paired_returns_index_aligned_arrays():
    r = _report()
    a, b = r.paired("adversarial", "baseline", "sortino_on")
    assert a.shape == b.shape == (8,)


def test_paired_refuses_mismatched_seed_counts():
    """The failure this exists to prevent: silently mis-pairing two arms."""
    r = _report()
    r._d["per_seed"]["adversarial"] = _arm(n=5)
    with pytest.raises(ValueError, match="cannot pair"):
        r.paired("adversarial", "baseline", "sortino_on")


def test_per_seed_missing_metric_names_the_arms():
    r = _report()
    with pytest.raises(KeyError, match="nope"):
        r.per_seed("baseline", "nope")


# ------------------------------------------------------------- non-finite

def test_finite_mask_is_pairwise_complete():
    a = np.array([1.0, np.nan, 3.0, np.inf])
    b = np.array([1.0, 2.0, np.nan, 4.0])
    assert list(EvalReport.finite(a, b)) == [True, False, False, False]


def test_quote_displacement_nan_pattern_is_preserved_not_filled():
    qp = np.array([0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    r = _report(baseline=_arm(quote_presence=qp))
    qd = r.per_seed("baseline", "quote_displacement_on")
    assert np.isnan(qd[:2]).all()
    assert np.isfinite(qd[2:]).all()


# --------------------------------------------------------- validity gate

def test_collapse_mask_uses_the_cut_inclusively():
    qp = np.array([0.0, COLLAPSE_CUT, COLLAPSE_CUT + 1e-9, 0.5, 1.0, 1.0, 1.0, 1.0])
    r = _report(baseline=_arm(quote_presence=qp))
    assert list(r.collapse_mask("baseline")) == [True, True, False, False,
                                                 False, False, False, False]


def test_gate_fails_only_on_a_majority():
    """The pre-registered rule is 'predominantly', i.e. a strict majority."""
    half = np.array([0.0] * 4 + [0.9] * 4)
    most = np.array([0.0] * 5 + [0.9] * 3)
    r = _report(arms=("baseline", "adversarial"),
                baseline=_arm(quote_presence=half),
                adversarial=_arm(quote_presence=most))
    s = r.collapse_summary()
    assert s["baseline"]["gate_ok"] is True       # exactly half is not "predominantly"
    assert s["adversarial"]["gate_ok"] is False


def test_sensitivity_sweep_flags_a_verdict_that_depends_on_the_cut():
    # Four seeds exactly 0, one just above 0 -> 4/8 at cut 0 (pass),
    # 5/8 once the cut reaches it (fail).
    qp = np.array([0.0] * 4 + [0.005] + [0.9] * 3)
    r = _report(baseline=_arm(quote_presence=qp))
    s = r.collapse_sensitivity()
    assert s["counts"]["baseline"][0] == 4
    assert s["counts"]["baseline"][-1] == 5
    assert s["stable"]["baseline"] is False


def test_sensitivity_sweep_reports_stability_when_there_is_no_ambiguity():
    qp = np.array([0.0] * 6 + [0.9] * 2)
    r = _report(baseline=_arm(quote_presence=qp))
    assert r.collapse_sensitivity()["stable"]["baseline"] is True


# ------------------------------------------- ratchet vs no-trade optimum

def test_inventory_activity_detects_a_ratchet():
    """Zero two-sided quoting but large inventory => one-sided trading."""
    qp = np.array([0.0] * 6 + [0.9] * 2)
    isd = np.array([120.0] * 6 + [5.0] * 2)
    r = _report(baseline=_arm(quote_presence=qp, inventory_sd=isd))
    act = r.inventory_activity()["baseline"]
    assert act["n_no_two_sided"] == 6
    assert act["n_of_those_flat"] == 0
    assert act["trading_one_sided"] is True
    assert act["max_inventory_sd"] == 120.0


def test_inventory_activity_detects_a_genuine_no_trade_optimum():
    qp = np.array([0.0] * 6 + [0.9] * 2)
    isd = np.array([0.0] * 6 + [5.0] * 2)
    r = _report(baseline=_arm(quote_presence=qp, inventory_sd=isd))
    act = r.inventory_activity()["baseline"]
    assert act["n_of_those_flat"] == 6
    assert act["trading_one_sided"] is False


# ---------------------------------------------------------------- labels

@pytest.mark.parametrize("key,expect", [
    ("sortino_on", "attack on"),
    ("sharpe_off", "attack off"),
    ("auroc", "AUROC"),
])
def test_metric_label_carries_the_condition(key, expect):
    assert expect in metric_label(key)


def test_metric_label_passes_through_unknown_keys():
    assert metric_label("brand_new_metric") == "brand_new_metric"


# --------------------------------------------------------------- figures

def test_every_figure_renders_on_a_complete_report(tmp_path):
    r = _report(arms=("baseline", "adversarial", "full", "as"))
    # No contrast/gate/equivalence blocks: those figures must skip, not raise.
    written = F.build(r, str(tmp_path))
    assert written, "expected at least the per-seed figures to render"
    assert all(os.path.getsize(p) > 0 for p in written)


def test_figures_skip_rather_than_raise_when_blocks_are_absent(tmp_path):
    r = _report()
    assert F.fig_forest(r) is None
    assert F.fig_equivalence(r) is None
    assert F.fig_progression_gate(r) is None
    assert F.fig_paired_slopes(r) is None


def test_figure_numbering_is_stable_under_only(tmp_path):
    """A figure cited by number in the thesis must not be renumbered by --only."""
    r = _report(arms=("baseline", "adversarial"))
    F.build(r, str(tmp_path), only=["inventory"])
    names = os.listdir(tmp_path)
    assert any(n.startswith("09_inventory") for n in names), names


def test_arm_palette_is_fixed_per_arm_identity():
    """Colour follows the entity, never its position: dropping an arm must not
    repaint the survivors."""
    before = dict(F.ARM_COLORS)
    r = _report(arms=("baseline", "full"))
    F.fig_validity(r)
    assert F.ARM_COLORS == before


def test_unknown_figure_name_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="unknown figure"):
        F.build(_report(), str(tmp_path), only=["nope"])


# ------------------------------------------------------- H3 detection head

def test_only_arms_with_a_head_are_credited_for_h3():
    """Every arm reports an AUROC; only the two with a head bear on H3."""
    from analysis.evalreport import DETECTION_ARMS
    assert set(DETECTION_ARMS) == {"detection", "full"}
    assert "as" not in DETECTION_ARMS and "baseline" not in DETECTION_ARMS


def test_auroc_table_does_not_bold_a_headless_arm(tmp_path):
    r = _report(arms=("baseline", "detection"))
    # A strongly significant result on the HEADLESS arm must stay unbolded.
    r._d["auroc_above_chance"] = {
        "baseline": {"n": 8, "mean": 0.62, "p_value": 1e-6, "test": "t",
                     "cohens_d": 2.0},
        "detection": {"n": 8, "mean": 0.61, "p_value": 1e-6, "test": "t",
                      "cohens_d": 2.0},
    }
    rows = {row[0]: row for row in T.tbl_auroc(r).rows}
    assert rows["A  baseline"][1] == "no"
    assert r"\textbf" not in rows["A  baseline"][-1]
    assert rows["C  +detection"][1] == "yes"
    assert r"\textbf" in rows["C  +detection"][-1]


# ---------------------------------------------------------------- tables

def test_latex_escaping_covers_the_specials_that_occur():
    assert T._esc("a_b") == r"a\_b"
    assert T._esc("50%") == r"50\%"
    assert T._esc("A&B") == r"A\&B"


def test_delatex_round_trips_scientific_notation():
    assert T._delatex(T._num(7.63e-5)) == "7.63e-5"
    assert T._delatex(T._num(1234.0)) == "1,234"


def test_delatex_leaves_no_raw_markup_in_the_text_view():
    r = _report(arms=("baseline", "adversarial"))
    for key in ("validity", "inventory", "summary"):
        fn, _ = T.TABLES[key]
        tbl = fn(r)
        if tbl is None:
            continue
        text = tbl.to_text()
        for token in ("\\textbf", "\\times", "\\geq", "\\leq", "\\_", "$"):
            assert token not in text, f"{key}: {token!r} leaked into text view"


def test_num_reports_non_finite_rather_than_printing_nan():
    assert T._num(float("nan")) == "n/a"
    assert T._num(float("inf")) == "n/a"
    assert T._num(None) == "--"


def test_summary_table_reports_per_row_n_for_partially_defined_metrics():
    """quote_displacement has a smaller n than its neighbours; the table must
    show that rather than implying a common n."""
    qp = np.array([0.0] * 5 + [0.9] * 3)
    r = _report(arms=("baseline",), baseline=_arm(quote_presence=qp))
    tbl = T.tbl_summary(r, metrics=("quote_presence_on", "quote_displacement_on"))
    ns = {row[1]: row[2] for row in tbl.rows}
    assert ns["Quote presence (frac. steps two-sided) (attack on)"] == "8"
    assert ns["Quote displacement (MAD from fair value) (attack on)"] == "3"


def test_tables_skip_rather_than_raise_when_blocks_are_absent():
    r = _report()
    assert T.tbl_contrasts(r) is None
    assert T.tbl_equivalence(r) is None
    assert T.tbl_gate(r) is None
    assert T.tbl_auroc(r) is None


def test_latex_table_is_balanced():
    r = _report(arms=("baseline", "adversarial"))
    tex = T.tbl_validity(r).to_latex()
    assert tex.count(r"\begin{table}") == tex.count(r"\end{table}") == 1
    assert tex.count(r"\begin{tabular}") == tex.count(r"\end{tabular}") == 1
    body = [l for l in tex.splitlines() if l.strip().endswith(r"\\")]
    widths = {l.count("&") for l in body}
    assert len(widths) == 1, f"ragged table: column counts {widths}"


# ------------------------------------------------------- end-to-end shape

def test_round_trip_through_json_matches_the_real_report_encoding(tmp_path):
    """The real reports are written with json.dump, including NaN tokens."""
    r = _report(arms=("baseline",),
                baseline=_arm(quote_presence=np.array([0.0] * 4 + [0.9] * 4)))
    path = tmp_path / "eval_synth.json"
    path.write_text(json.dumps(r._d))
    back = EvalReport.load(str(path))
    assert back.n_seeds == 8
    assert np.isnan(back.per_seed("baseline", "quote_displacement_on")[:4]).all()
