"""Thesis figures from an eval_*.json report.

    python -m analysis.figures D:/tmp/runs/results/eval_1151370.json -o figures/
    python -m analysis.figures results/eval_v3.json -o figures/ --only validity,forest

Design rules that are not negotiable here, because the v2 data breaks the usual
defaults:

  BIMODAL DATA GETS SHOWN, NOT AVERAGED. quote_presence is a spike at 0 plus a
  quoting mode above 0.28; sortino has seed SDs 2-200x the seed means. A bar of
  means with an error bar is actively misleading on this data, so every arm-level
  panel plots ALL 20 seeds and adds the median as a rule, never a mean bar.

  COLLAPSE IS ENCODED TWICE. Seeds that stopped quoting are drawn with a
  different MARKER as well as a different fill, so the distinction survives
  greyscale printing and colour-vision deficiency.

  NON-FINITE IS REPORTED, NOT DROPPED. Any panel whose metric is undefined on
  some seeds prints the finite count in the panel, because those NaNs are the
  collapsed seeds and silently dropping them is the selection effect that makes a
  dead arm look healthy.

  EVERY FIGURE CARRIES ITS PROVENANCE. Source file, seed count, checkpoint step
  and whether the run was confirmatory are stamped in the footer, so a figure
  pasted into a slide is still traceable.

Palette: validated with the dataviz skill's six checks against the light/print
surface (the only surface a thesis figure is rendered on) -- all pass, worst
adjacent CVD dE 9.5 (deutan). Do not re-order or substitute hues without
re-running that validator.
"""

from __future__ import annotations

import argparse
import os
from typing import Callable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analysis.evalreport import (
    ARM_LABELS, CONTRAST_LABELS, COLLAPSE_CUT, COLLAPSE_MAJORITY,
    DETECTION_ARMS, PRIMARY_METRICS, EvalReport, metric_label,
)

# Validated categorical palette (see module docstring). Assignment is by ARM
# IDENTITY and is fixed -- a figure that drops an arm must not repaint the rest.
ARM_COLORS = {
    "baseline":      "#005A8F",
    "adversarial":   "#B34700",
    "detection":     "#007A59",
    "regime":        "#C28500",
    "full":          "#B35C8A",
    "unconstrained": "#3D91C7",
    # The A-S benchmark is not a categorical series competing for identity: it is
    # the fixed-policy reference line. Deliberately achromatic so it never reads
    # as "arm G", and always direct-labelled on the categorical axis.
    "as":            "#444444",
}

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"
RULE = "#333333"

# Marker shapes carry collapse status independently of colour, so the split
# survives greyscale print and CVD.
MARK_QUOTING = "o"
MARK_COLLAPSED = "x"


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "font.family": "sans-serif",
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.edgecolor": "#b0b0b0",
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.4,
    })


def _stamp(fig, report: EvalReport, note: str | None = None,
           x: float = 0.005, ha: str = "left") -> None:
    """Provenance footer. Every figure gets one.

    `x`/`ha` move it out of the way on figures that also carry a bottom legend.
    """
    txt = report.provenance_stamp()
    if note:
        txt = f"{txt}\n{note}"
    fig.text(x, 0.002, txt, fontsize=6, color=MUTED,
             ha=ha, va="bottom", linespacing=1.5)


def _jitter(n: int, width: float, seed: int) -> np.ndarray:
    """Deterministic jitter -- figures must be byte-reproducible across runs."""
    return np.random.default_rng(seed).uniform(-width, width, size=n)


def _fmt_p(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "p n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _arm_rows(report: EvalReport, arms: Sequence[str]) -> list[tuple[int, str]]:
    """Top-to-bottom row order for categorical-axis panels."""
    return [(len(arms) - 1 - i, a) for i, a in enumerate(arms)]


def _strip_by_arm(ax, report: EvalReport, metric: str, arms: Sequence[str],
                  cut: float = COLLAPSE_CUT, condition: str = "on",
                  split_collapsed: bool = True, seed: int = 0,
                  headroom: float = 0.0) -> int:
    """Per-seed strip plot down a categorical arm axis; returns n non-finite.

    Median drawn as a short rule. No mean, no error bar: on bimodal data both
    describe a value no seed actually took.
    """
    n_nonfinite = 0
    for row, arm in _arm_rows(report, arms):
        v = report.per_seed(arm, metric)
        fin = np.isfinite(v)
        n_nonfinite += int((~fin).sum())
        collapsed = (report.collapse_mask(arm, condition, cut)
                     if split_collapsed and report.has(arm, f"quote_presence_{condition}")
                     else np.zeros_like(fin))
        y = row + _jitter(v.size, 0.16, seed + row)
        colour = ARM_COLORS.get(arm, "#666666")
        # 'x' is an unfilled marker: matplotlib ignores edgecolors on it, so it
        # takes `color` while the filled marker takes an explicit face/edge pair.
        if (fin & ~collapsed).any():
            m = fin & ~collapsed
            ax.scatter(v[m], y[m], s=26, marker=MARK_QUOTING,
                       facecolors=colour, edgecolors=colour,
                       linewidths=1.1, alpha=0.85, zorder=3)
        if (fin & collapsed).any():
            m = fin & collapsed
            ax.scatter(v[m], y[m], s=26, marker=MARK_COLLAPSED, color=colour,
                       linewidths=1.1, alpha=0.85, zorder=3)
        if fin.any():
            med = float(np.median(v[fin]))
            ax.plot([med, med], [row - 0.3, row + 0.3], color=RULE,
                    lw=1.8, zorder=4, solid_capstyle="butt")
    ax.set_yticks([r for r, _ in _arm_rows(report, arms)])
    ax.set_yticklabels([ARM_LABELS.get(a, a) for _, a in _arm_rows(report, arms)])
    # Headroom reserves an empty band above the top row for an in-axes legend,
    # so the legend never lands on data regardless of what the values are.
    ax.set_ylim(-0.6, len(arms) - 0.4 + headroom)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    return n_nonfinite


def _collapse_legend(ax, loc: str = "upper right") -> None:
    """Legend for the two-sided/one-sided split.

    Labels say "two-sided", never "quoting" or "trading": a zero-presence seed is
    still trading (see EvalReport.inventory_activity), and mislabelling that here
    would propagate the wrong failure diagnosis into every caption.
    """
    ax.legend(handles=[
        Line2D([], [], marker=MARK_QUOTING, ls="none", color=MUTED,
               markerfacecolor=MUTED, markersize=5, label="posts two-sided quotes"),
        Line2D([], [], marker=MARK_COLLAPSED, ls="none", color=MUTED,
               markersize=5, label=f"no two-sided quote (qp<={COLLAPSE_CUT:g})"),
        Line2D([], [], color=RULE, lw=1.8, label="median"),
    ], loc=loc)


# --------------------------------------------------------------------------
# Fig 1 - the validity gate. Read this before any risk metric in the report.
# --------------------------------------------------------------------------

def fig_validity(report: EvalReport, cut: float = COLLAPSE_CUT,
                 majority: float = COLLAPSE_MAJORITY):
    """Quote presence per seed, plus the sensitivity of the gate to the cut.

    The pre-registered gate (interpretation_notes/quote_presence_validity_gate)
    says an arm whose seeds are predominantly quote_presence ~= 0 cannot support
    an H1 robustness claim, because an observation-space adversary cannot move an
    MM that is not quoting. Panel (a) is the evidence; panel (b) exists because
    "~= 0" and "predominantly" are not numbers, so the verdict has to be shown to
    survive the whole plausible range rather than one chosen cut.
    """
    arms = list(report.arms)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.0, 0.52 * len(arms) + 2.6),
        gridspec_kw={"width_ratios": [1.55, 1.0]})

    _strip_by_arm(ax1, report, "quote_presence_on", arms, cut=cut, seed=11,
                  headroom=0.95)
    ax1.axvline(cut, color="#B34700", ls=":", lw=1.2, zorder=2)
    ax1.set_xlabel("Quote presence (fraction of steps with a two-sided quote), attack on")
    ax1.set_title("(a) Every seed, every arm", loc="left")
    ax1.set_xlim(-0.05, 1.05)

    summ = report.collapse_summary("on", cut, majority)
    for row, arm in _arm_rows(report, arms):
        s = summ.get(arm)
        if s:
            ax1.text(1.02, row, f"{s['n_collapsed']}/{s['n']}",
                     fontsize=7.5, va="center", ha="left",
                     color="#B34700" if not s["gate_ok"] else "#007A59",
                     transform=ax1.get_yaxis_transform(), clip_on=False)
    ax1.text(1.02, len(arms) - 0.28, "no two-sided\nquote", fontsize=7,
             va="center", ha="left", color=MUTED, linespacing=1.4,
             transform=ax1.get_yaxis_transform(), clip_on=False)
    _collapse_legend(ax1, loc="upper right")

    sens = report.collapse_sensitivity("on", majority=majority)
    cuts, n = sens["cuts"], sens["n"]
    for arm in arms:
        if arm not in sens["counts"]:
            continue
        frac = np.array(sens["counts"][arm], dtype=float) / max(n, 1)
        ax2.plot(range(len(cuts)), frac, marker="o", markersize=4,
                 color=ARM_COLORS.get(arm, "#666666"),
                 ls="--" if arm == "as" else "-",
                 label=ARM_LABELS.get(arm, arm))
    ax2.axhline(majority, color=RULE, ls=":", lw=1.2)
    ax2.text(len(cuts) - 1, majority + 0.02,
             f'gate threshold ("predominantly" = {majority:g})',
             fontsize=7, ha="right", va="bottom", color=RULE)
    ax2.set_xticks(range(len(cuts)))
    ax2.set_xticklabels([f"{c:g}" for c in cuts])
    ax2.set_xlabel('Cut defining "quote_presence ~= 0"')
    ax2.set_ylabel("Fraction of seeds with no two-sided quote")
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_title("(b) Is the verdict an artefact of the cut?", loc="left")
    ax2.grid(axis="y")
    ax2.set_axisbelow(True)
    ax2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    unstable = [a for a, ok in sens["stable"].items() if not ok]
    note = ("Gate verdict is stable across the whole cut sweep for every arm."
            if not unstable else
            "Gate verdict FLIPS with the cut for: " + ", ".join(unstable)
            + " -- report these as boundary cases, not as passes.")
    act = report.inventory_activity("on", cut)
    ratchet = [a for a, v in act.items()
               if v["trading_one_sided"] and v["n_no_two_sided"]]
    if ratchet:
        note += ("\nZero-presence seeds are NOT inactive: they still trade one-sided "
                 "(max inventory SD "
                 f"{max(act[a]['max_inventory_sd'] for a in ratchet):,.0f}, "
                 "see the inventory figure). Not a no-trade optimum.")
    fig.suptitle("Validity gate: is the market maker posting two-sided quotes?",
                 x=0.005, ha="left", fontsize=12, fontweight="bold")
    _stamp(fig, report, note)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    return fig


# --------------------------------------------------------------------------
# Fig 2 - was the attack actually delivered, and did it do anything?
# --------------------------------------------------------------------------

def fig_attack_delivery(report: EvalReport, cut: float = COLLAPSE_CUT):
    """Attack rate, injected volume, and the on-minus-off response per seed.

    Exists because of the eval_1121599 ambiguity: an attack-on rollout in which
    the adversary injected NOTHING was indistinguishable from attack-off, so
    every *_on equalling its *_off looked like robustness. Panels (a) and (b)
    establish the attack WAS delivered; panel (c) then shows the response to it,
    against quote presence. A seed at zero response and zero quote presence is
    invariant because the attack has nothing to act on, not because it is robust.

    Note the asymmetry panel (c) makes visible: zero response implies zero
    two-sided presence, but the converse fails for a few seeds. Those are seeds
    with a tiny non-zero presence, or with one-sided quoting that still moves
    inventory. Do not read the panel as "presence 0 => response 0".
    """
    arms = list(report.arms)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 0.5 * len(arms) + 2.8))
    ax1, ax2, ax3 = axes

    _strip_by_arm(ax1, report, "mean_attack_rate_on", arms, cut=cut, seed=21)
    ax1.set_xlabel("Attack rate, attack-on windows")
    ax1.set_title("(a) Attack delivered", loc="left")
    ax1.set_xlim(-0.05, 1.05)
    off_max = max((float(np.nanmax(report.per_seed(a, "mean_attack_rate_off")))
                   for a in arms if report.has(a, "mean_attack_rate_off")),
                  default=float("nan"))
    if np.isfinite(off_max):
        ax1.text(0.02, len(arms) - 0.5,
                 f"attack-off windows: max rate {off_max:.3g} across all arms",
                 fontsize=7, color=MUTED, va="top",
                 transform=ax1.get_yaxis_transform())

    nnf = _strip_by_arm(ax2, report, "mean_injected_volume_on", arms, cut=cut, seed=22)
    ax2.set_xlabel("Mean injected volume per step, attack on")
    ax2.set_title("(b) Magnitude of injection", loc="left")
    # symlog (not log) because a seed can legitimately sit at exactly 0; the left
    # limit is pinned at 0 so the meaningless negative half of the symlog axis
    # does not eat half the panel.
    ax2.set_xscale("symlog", linthresh=1.0)
    ax2.set_xlim(left=0)
    if nnf:
        ax2.text(0.99, 0.01, f"{nnf} non-finite seed-values omitted",
                 fontsize=7, color="#B34700", ha="right", va="bottom",
                 transform=ax2.transAxes)

    # (c) response vs presence, pooled over arms, colour = arm identity.
    any_pts = False
    for arm in arms:
        if not (report.has(arm, "sortino_on") and report.has(arm, "sortino_off")
                and report.has(arm, "quote_presence_on")):
            continue
        on, off = report.paired(arm, arm, "sortino_on")[0], report.per_seed(arm, "sortino_off")
        qp = report.per_seed(arm, "quote_presence_on")
        d = on - off
        m = np.isfinite(d) & np.isfinite(qp)
        if m.any():
            any_pts = True
            ax3.scatter(qp[m], d[m], s=26, marker=MARK_QUOTING,
                        facecolors=ARM_COLORS.get(arm, "#666666"),
                        edgecolors="white", linewidths=0.5, alpha=0.85,
                        label=ARM_LABELS.get(arm, arm), zorder=3)
    ax3.axhline(0.0, color=RULE, lw=1.0, ls="-")
    ax3.axvline(cut, color="#B34700", ls=":", lw=1.2)
    ax3.set_xlabel("Quote presence (attack on)")
    ax3.set_ylabel("Sortino(attack on) - Sortino(attack off)")
    ax3.set_title("(c) Response to the attack vs presence", loc="left")
    ax3.grid()
    ax3.set_axisbelow(True)
    if any_pts:
        ax3.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    # Count, don't assert. On v2 most zero-presence seeds respond EXACTLY 0, but
    # a handful in the regime/full/unconstrained arms do not, so a flat "these
    # cannot respond" caption would be contradicted by points in its own panel.
    n_zero, n_zero_flat = 0, 0
    for arm in arms:
        if not (report.has(arm, "quote_presence_on") and report.has(arm, "sortino_on")):
            continue
        m = report.per_seed(arm, "quote_presence_on") <= cut
        d = report.per_seed(arm, "sortino_on") - report.per_seed(arm, "sortino_off")
        n_zero += int(m.sum())
        n_zero_flat += int((m & np.isfinite(d) & (d == 0)).sum())
    ax3.text(0.01, 0.02,
             f"{n_zero_flat}/{n_zero} seeds left of the cut respond exactly 0:\n"
             "with no two-sided quote there is little for an\n"
             "observation-space attack to act on",
             fontsize=7, color=MUTED, transform=ax3.transAxes, va="bottom",
             linespacing=1.4)

    fig.suptitle("Attack delivery and response",
                 x=0.005, ha="left", fontsize=12, fontweight="bold")
    _stamp(fig, report)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    return fig


# --------------------------------------------------------------------------
# Fig 3 - the confirmatory family, as a forest plot.
# --------------------------------------------------------------------------

def fig_forest(report: EvalReport, metrics: Sequence[str] = PRIMARY_METRICS):
    """Paired contrasts with CIs, one panel per pre-registered primary metric.

    One panel per metric rather than one standardised panel, because the metrics
    live on incompatible scales (Sortino ~ 50, CVaR ~ 300, AUROC ~ 0.5) and a
    single standardised axis would hide the effect sizes that actually matter for
    the writeup. Holm-adjusted p is annotated per row; the raw p is deliberately
    NOT shown, so an uncorrected exploratory p cannot be lifted out of a figure
    and promoted to a significance claim.
    """
    labels = [c for c in report.contrasts()]
    metrics = [m for m in metrics
               if any(report.contrast(c, m) is not None for c in labels)]
    if not labels or not metrics:
        return None

    fig, axes = plt.subplots(1, len(metrics), sharey=True,
                             figsize=(3.6 * len(metrics) + 1.6,
                                      0.46 * len(labels) + 2.8))
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, metrics):
        for i, lab in enumerate(labels):
            row = len(labels) - 1 - i
            c = report.contrast(lab, metric)
            if c is None:
                continue
            est = c.get("mean_diff", np.nan)
            lo, hi = c.get("ci_low", np.nan), c.get("ci_high", np.nan)
            p_holm = report.holm(lab, metric)
            sig = p_holm is not None and np.isfinite(p_holm) and p_holm < 0.05
            colour = "#B34700" if sig else RULE
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [row, row], color=colour, lw=1.6,
                        solid_capstyle="butt", zorder=3)
            ax.scatter([est], [row], s=42,
                       marker="D" if c.get("test") == "paired_t" else "o",
                       facecolors=colour if sig else "white",
                       edgecolors=colour, linewidths=1.4, zorder=4)
            ax.text(1.01, row, _fmt_p(p_holm), fontsize=7,
                    va="center", ha="left", color=colour,
                    transform=ax.get_yaxis_transform(), clip_on=False)
        ax.axvline(0.0, color=RULE, lw=1.0)
        ax.set_title(metric_label(metric), loc="left", fontsize=9)
        ax.set_xlabel("paired difference (A - B)")
        ax.grid(axis="x")
        ax.set_axisbelow(True)

    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels([CONTRAST_LABELS.get(l, l) for l in reversed(labels)],
                            fontsize=8)
    axes[0].set_ylim(-0.7, len(labels) - 0.3)

    fig.legend(handles=[
        Line2D([], [], marker="o", ls="none", mfc="white", mec=RULE,
               markersize=6, label="Wilcoxon signed-rank (non-normal)"),
        Line2D([], [], marker="D", ls="none", mfc="white", mec=RULE,
               markersize=6, label="paired t"),
        Line2D([], [], color="#B34700", lw=1.6,
               label="Holm-adjusted p < 0.05"),
        Line2D([], [], color=RULE, lw=1.6, label="95% CI (bootstrap)"),
    ], loc="lower left", bbox_to_anchor=(0.005, 0.0), ncol=4)

    tag = ("Confirmatory family, Holm-adjusted within contrast."
           if report.is_confirmatory else
           "EXPLORATORY / PARTIAL RUN -- p-values are not the pre-registered test.")
    fig.suptitle("Pre-registered contrasts", x=0.005, ha="left",
                 fontsize=12, fontweight="bold")
    # Bottom-left is taken by the legend on this figure.
    _stamp(fig, report, tag, x=0.995, ha="right")
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    return fig


# --------------------------------------------------------------------------
# Fig 4 - what the forest plot's means are made of.
# --------------------------------------------------------------------------

def fig_paired_slopes(report: EvalReport, contrast: str = "adversarial_vs_baseline",
                      metrics: Sequence[str] = PRIMARY_METRICS,
                      cut: float = COLLAPSE_CUT):
    """Per-seed paired lines for one contrast.

    A forest plot shows the average of the pairing; this shows the pairing. With
    n=20 and seed SDs several times the seed means, the question "is this a
    consistent shift or two seeds dragging a mean" is answerable only here.
    Lines are split by whether the seed collapsed, because a collapsed pair
    contributes a difference of ~0 to the test for a reason that has nothing to
    do with the hypothesis.
    """
    if contrast not in report.contrasts():
        return None
    arm_a, arm_b = _contrast_arms(contrast, report)
    if arm_a is None:
        return None
    metrics = [m for m in metrics if report.has(arm_a, m) and report.has(arm_b, m)]
    if not metrics:
        return None

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(2.9 * len(metrics) + 1.2, 4.4))
    axes = np.atleast_1d(axes)
    n_collapsed_pairs = 0
    tied = {}          # metric -> how many both-collapsed pairs differ by exactly 0

    for ax, metric in zip(axes, metrics):
        a, b = report.paired(arm_a, arm_b, metric)
        col_a = report.collapse_mask(arm_a, "on", cut) if report.has(arm_a, "quote_presence_on") else np.zeros(a.size, bool)
        col_b = report.collapse_mask(arm_b, "on", cut) if report.has(arm_b, "quote_presence_on") else np.zeros(b.size, bool)
        both_collapsed = col_a & col_b
        n_collapsed_pairs = int(both_collapsed.sum())
        d = a - b
        tied[metric] = int((both_collapsed & np.isfinite(d) & (d == 0)).sum())
        for i in range(a.size):
            if not (np.isfinite(a[i]) and np.isfinite(b[i])):
                continue
            dead = both_collapsed[i]
            ax.plot([0, 1], [b[i], a[i]],
                    color="#bdbdbd" if dead else RULE,
                    lw=0.9 if dead else 1.2,
                    ls=":" if dead else "-",
                    alpha=0.75, zorder=2)
            pair = [ARM_COLORS.get(arm_b, RULE), ARM_COLORS.get(arm_a, RULE)]
            if dead:
                ax.scatter([0, 1], [b[i], a[i]], s=16, marker=MARK_COLLAPSED,
                           color=pair, linewidths=1.0, zorder=3)
            else:
                ax.scatter([0, 1], [b[i], a[i]], s=16, marker=MARK_QUOTING,
                           facecolors=pair, edgecolors=pair,
                           linewidths=1.0, zorder=3)
        for x, arm in ((0, arm_b), (1, arm_a)):
            v = report.per_seed(arm, metric)
            if np.isfinite(v).any():
                med = float(np.median(v[np.isfinite(v)]))
                ax.plot([x - 0.16, x + 0.16], [med, med],
                        color=ARM_COLORS.get(arm, RULE), lw=2.4, zorder=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([ARM_LABELS.get(arm_b, arm_b),
                            ARM_LABELS.get(arm_a, arm_a)], fontsize=8)
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(metric_label(metric), loc="left", fontsize=9)
        ax.grid(axis="y")
        ax.set_axisbelow(True)

    # Count the exact ties rather than asserting them: two seeds that both fail
    # to quote two-sided are still two DIFFERENT training runs, so their
    # difference is not zero by construction -- it is zero empirically on the
    # risk metrics and not on AUROC, which does not depend on quoting.
    worst = max(tied.values()) if tied else 0
    note = (f"{n_collapsed_pairs}/{report.n_seeds} seed pairs post no two-sided "
            f"quote in EITHER arm (dotted). Up to {worst} of those differ by "
            f"exactly zero, so the paired tests carry an effective n nearer "
            f"{report.n_seeds - worst} than {report.n_seeds}: "
            + ", ".join(f"{metric_label(m)} {t} tied" for m, t in tied.items()) + ".")
    fig.suptitle(f"Seed-level pairing: {CONTRAST_LABELS.get(contrast, contrast)}",
                 x=0.005, ha="left", fontsize=12, fontweight="bold")
    _stamp(fig, report, note)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    return fig


def _contrast_arms(contrast: str, report: EvalReport) -> tuple[str | None, str | None]:
    """Recover (arm_a, arm_b) from a contrast label, checking both are present."""
    if "_vs_" not in contrast:
        return None, None
    a, b = contrast.split("_vs_", 1)
    if a in report.arms and b in report.arms:
        return a, b
    return None, None


# --------------------------------------------------------------------------
# Fig 5 - H2 equivalence on clean data.
# --------------------------------------------------------------------------

def fig_equivalence(report: EvalReport):
    """TOST results against the pre-registered margins.

    H2 is a no-degradation claim, so it is tested by equivalence, not by failing
    to reject a difference. The shaded band is the pre-registered margin; a claim
    of equivalence requires the effect to sit inside it with a significant TOST,
    and neither is inferred from a large p on the difference test.

    The conventional 90% CI overlay is absent because the report stores the TOST
    decision, not the interval -- tost_paired would have to return it.
    """
    eq = report.equivalence()
    if not eq:
        return None
    rows = [(comp, metric, res)
            for comp, metrics in eq.items() for metric, res in metrics.items()]
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(9.0, 0.55 * len(rows) + 2.6))
    max_margin = max(abs(float(r[2].get("margin", 0.0))) for r in rows) or 1.0

    for i, (comp, metric, res) in enumerate(rows):
        row = len(rows) - 1 - i
        margin = float(res.get("margin", np.nan))
        diff = float(res.get("mean_diff", np.nan))
        equivalent = bool(res.get("equivalent", False))
        if np.isfinite(margin):
            ax.add_patch(plt.Rectangle((-margin, row - 0.32), 2 * margin, 0.64,
                                       facecolor="#007A59", alpha=0.10,
                                       edgecolor="none", zorder=1))
            for s in (-1, 1):
                ax.plot([s * margin, s * margin], [row - 0.32, row + 0.32],
                        color="#007A59", lw=1.0, ls="--", zorder=2)
        colour = "#007A59" if equivalent else "#B34700"
        ax.scatter([diff], [row], s=52, marker="D", facecolors=colour,
                   edgecolors="white", linewidths=0.8, zorder=4)
        ax.text(1.01, row,
                f"{'equivalent' if equivalent else 'NOT equivalent'}  "
                f"{_fmt_p(res.get('p_value'))}",
                fontsize=7.5, va="center", ha="left", color=colour,
                transform=ax.get_yaxis_transform(), clip_on=False)

    ax.axvline(0.0, color=RULE, lw=1.0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f"{comp.replace('_vs_', ' vs ')}\n{metric_label(metric)}"
         for comp, metric, _ in reversed(rows)], fontsize=8)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Mean paired difference on clean (attack-off) data")
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    # The effects are orders of magnitude outside the margins on v2, so an
    # auto-scaled axis would render the margin band as an invisible hairline.
    # Keep the band legible and say so, rather than silently clipping a point.
    lo = min(float(r[2].get("mean_diff", 0.0)) for r in rows)
    hi = max(float(r[2].get("mean_diff", 0.0)) for r in rows)
    span = max(abs(lo), abs(hi), max_margin * 1.5)
    ax.set_xlim(-span * 1.25, span * 1.25)

    fig.suptitle("H2 -- no degradation on clean data (TOST)",
                 x=0.005, ha="left", fontsize=12, fontweight="bold")
    _stamp(fig, report,
           f"Shaded band = pre-registered equivalence margin "
           f"({', '.join(f'{k}={v:g}' for k, v in report.equivalence_margins().items())}).")
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    return fig


# --------------------------------------------------------------------------
# Fig 6 - H3 detection.
# --------------------------------------------------------------------------

def fig_auroc(report: EvalReport, cut: float = COLLAPSE_CUT):
    """Per-seed detection AUROC against the 0.5 chance line."""
    arms = [a for a in report.arms if report.has(a, "auroc")]
    if not arms:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 0.5 * len(arms) + 2.4))
    _strip_by_arm(ax, report, "auroc", arms, cut=cut, seed=61, headroom=0.95)
    ax.axvline(0.5, color=RULE, lw=1.2)
    ax.text(0.5, len(arms) - 0.2, " chance", fontsize=7.5, color=RULE,
            ha="left", va="center")

    above = report.auroc_above_chance()
    n_head, n_nohead_sig = 0, 0
    for row, arm in _arm_rows(report, arms):
        res = above.get(arm)
        has_head = arm in DETECTION_ARMS
        n_head += has_head
        if res:
            p = res.get("p_value")
            sig = p is not None and np.isfinite(p) and p < 0.05
            # Significance is highlighted ONLY on arms that carry a head. Every
            # arm reports an auroc, so colouring a headless arm red would invite
            # exactly the misreading this figure is supposed to prevent.
            n_nohead_sig += bool(sig and not has_head)
            ax.text(1.02, row,
                    f"{_fmt_p(p)}  (vs 0.5)" + ("" if has_head else "   no head"),
                    fontsize=7, va="center", ha="left",
                    color="#B34700" if (sig and has_head) else MUTED,
                    transform=ax.get_yaxis_transform(), clip_on=False)
    # Grey out the rows without a head so the eye goes to the two that matter.
    for lbl, (_, arm) in zip(reversed(ax.get_yticklabels()),
                             reversed(_arm_rows(report, arms))):
        lbl.set_color(INK if arm in DETECTION_ARMS else "#9a9a9a")
    ax.set_xlabel("Detection AUROC on the mixed attack/clean stream")
    _collapse_legend(ax, loc="upper right")
    fig.suptitle("H3 -- is the detection head above chance?",
                 x=0.005, ha="left", fontsize=12, fontweight="bold")
    note = ("One-sample test vs 0.5, per arm, not Holm-adjusted across arms. "
            f"Only the {n_head} arm(s) in bold carry a detection head; the rest "
            "report an AUROC because the rollout always computes one, and cannot "
            "speak to H3.")
    if n_nohead_sig:
        note += (f" {n_nohead_sig} headless arm(s) reach nominal p<0.05 -- "
                 f"expected under {len(above)} uncorrected tests, and not a finding.")
    _stamp(fig, report, note)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    return fig


# --------------------------------------------------------------------------
# Fig 7 - H4 regime conditioning.
# --------------------------------------------------------------------------

def fig_regime(report: EvalReport, cut: float = COLLAPSE_CUT):
    """Low- vs high-volatility Sortino per seed, and the resulting regime gap.

    regime_gap is |Sortino_high - Sortino_low| (rollout.py), an ABSOLUTE
    difference: it measures how much the regime matters, not which regime is
    better, so a smaller gap is the H4-consistent direction and the sign is not
    recoverable from it. Panel (a) keeps the sign visible.
    """
    arms = [a for a in report.arms
            if report.has(a, "sortino_lowvol_on") and report.has(a, "sortino_highvol_on")]
    if not arms:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 0.5 * len(arms) + 2.8))

    for row, arm in _arm_rows(report, arms):
        lo = report.per_seed(arm, "sortino_lowvol_on")
        hi = report.per_seed(arm, "sortino_highvol_on")
        colour = ARM_COLORS.get(arm, "#666666")
        y = row + _jitter(lo.size, 0.15, 71 + row)
        m = np.isfinite(lo) & np.isfinite(hi)
        for i in np.flatnonzero(m):
            ax1.plot([lo[i], hi[i]], [y[i], y[i]], color=colour,
                     lw=0.8, alpha=0.55, zorder=2)
        ax1.scatter(lo[m], y[m], s=20, marker="o", facecolors="none",
                    edgecolors=colour, linewidths=1.0, zorder=3)
        ax1.scatter(hi[m], y[m], s=20, marker="s", facecolors=colour,
                    edgecolors=colour, linewidths=1.0, zorder=3)
    ax1.set_yticks([r for r, _ in _arm_rows(report, arms)])
    ax1.set_yticklabels([ARM_LABELS.get(a, a) for _, a in _arm_rows(report, arms)])
    ax1.set_ylim(-0.6, len(arms) - 0.4 + 0.95)   # headroom for the legend
    ax1.axvline(0.0, color=RULE, lw=1.0)
    ax1.set_xlabel("Sortino ratio, attack on")
    ax1.set_title("(a) Per seed, by volatility regime", loc="left")
    ax1.grid(axis="x")
    ax1.set_axisbelow(True)
    ax1.legend(handles=[
        Line2D([], [], marker="o", ls="none", mfc="none", mec=MUTED,
               markersize=5, label="low-vol"),
        Line2D([], [], marker="s", ls="none", mfc=MUTED, mec=MUTED,
               markersize=5, label="high-vol"),
    ], loc="upper right")

    _strip_by_arm(ax2, report, "regime_gap_on", arms, cut=cut, seed=72,
                  headroom=0.95)
    ax2.set_xlabel("Regime gap  |Sortino_high - Sortino_low|  (lower = less regime-sensitive)")
    ax2.set_title("(b) Regime gap", loc="left")
    _collapse_legend(ax2, loc="upper right")

    fig.suptitle("H4 -- regime conditioning",
                 x=0.005, ha="left", fontsize=12, fontweight="bold")
    _stamp(fig, report, "Estimation only: H4 metrics are not in the "
                        "Holm-adjusted confirmatory family.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    return fig


# --------------------------------------------------------------------------
# Fig 8 - the Phase-1 progression gate.
# --------------------------------------------------------------------------

def fig_progression_gate(report: EvalReport):
    """The three pre-registered gate criteria, IPPO against the A-S benchmark.

    The gate is a go/no-go on whether vanilla IPPO is a credible market maker at
    all. It is drawn as three separate panels rather than one normalised chart
    because the criteria are on different scales and two are 'within a margin of'
    while the third is 'within a factor of'.
    """
    gate = report.progression_gate()
    if not gate or not gate.get("detail"):
        return None
    detail = gate["detail"]
    keys = [k for k in ("sharpe", "sortino", "inventory_sd") if k in detail]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.5 * len(keys) + 0.8, 3.9))
    axes = np.atleast_1d(axes)

    for ax, key in zip(axes, keys):
        d = detail[key]
        ippo, as_v = float(d.get("ippo", np.nan)), float(d.get("as", np.nan))
        ok = bool(gate.get(f"{key}_ok", False))
        # The gate criterion is a comparison of MEANS, so the mean is drawn as
        # the tested quantity -- but as a rule over the seed cloud, not as a bar.
        # On this data the seed SD is several times the mean (Sharpe: mean -8.8,
        # seeds spanning -93 to +76), and a bar implies a precision that is not
        # there while hiding the overlap that actually explains the verdict.
        mkey = f"{key}_off"          # the gate is judged on clean data
        for x, arm, mean_v in ((0, "baseline", ippo), (1, "as", as_v)):
            colour = ARM_COLORS.get(arm, RULE)
            if report.has(arm, mkey):
                v = report.per_seed(arm, mkey)
                v = v[np.isfinite(v)]
                ax.scatter(x + _jitter(v.size, 0.14, 81 + x), v, s=20,
                           facecolors=colour, edgecolors="white", linewidths=0.5,
                           alpha=0.8, zorder=3)
            ax.plot([x - 0.28, x + 0.28], [mean_v, mean_v], color=colour,
                    lw=2.8, zorder=5, solid_capstyle="butt")
            ax.annotate(f"mean {mean_v:,.3g}", xy=(x, mean_v),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK, zorder=6,
                        bbox=dict(boxstyle="square,pad=0.15", fc="white",
                                  ec="none", alpha=0.85))
        thr, sense = None, None
        if "margin" in d and np.isfinite(as_v):
            thr, sense = as_v - float(d["margin"]), ">="
        elif "factor" in d and np.isfinite(as_v):
            thr, sense = as_v * float(d["factor"]), "<="
        if thr is not None:
            ax.axhline(thr, color="#B34700", ls="--", lw=1.1, zorder=4)
            # Sat on the dashed line and read as struck through; lift it clear
            # and give it an opaque backing so it stays legible over the grid.
            ax.annotate(f"need {sense} {thr:,.3g}", xy=(1.42, thr),
                        xytext=(0, 4), textcoords="offset points",
                        fontsize=7, color="#B34700", va="bottom", ha="left",
                        bbox=dict(boxstyle="square,pad=0.15", fc="white",
                                  ec="none", alpha=0.85))
        ax.axhline(0.0, color=RULE, lw=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["IPPO", "A-S"], fontsize=8)
        ax.set_xlim(-0.6, 1.9)
        ax.set_title(f"{key}   {'PASS' if ok else 'FAIL'}", loc="left",
                     color="#007A59" if ok else "#B34700")
        ax.grid(axis="y")
        ax.set_axisbelow(True)

    passed = bool(gate.get("passed"))
    fig.suptitle(f"Phase-1 progression gate: {'PASSED' if passed else 'FAILED'}",
                 x=0.005, ha="left", fontsize=12, fontweight="bold",
                 color="#007A59" if passed else "#B34700")
    _stamp(fig, report,
           "Points are seeds on clean (attack-off) data; the rule is the mean, "
           "which is the quantity the gate criterion is defined on. Dashed line "
           "is the pre-registered criterion.")
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    return fig


# --------------------------------------------------------------------------
# Fig 9 - the inventory ratchet, and whether the v3 fix removed it.
# --------------------------------------------------------------------------

def fig_inventory(report: EvalReport, cut: float = COLLAPSE_CUT,
                  auto_liquidate_threshold: float | None = None):
    """Inventory dispersion and peak excursion, split by two-sided quoting.

    This figure exists because the obvious reading of quote_presence = 0 -- the
    Mohl et al. no-trade optimum -- is WRONG for this data, and the two readings
    have opposite fixes. On v2 every zero-presence seed still carries inventory
    SD up to 130 and peak |inventory| up to 449: those seeds are not refusing to
    trade, they are quoting one-sided and ratcheting inventory in one direction.

    That distinction is the whole justification for the v3 amendment
    (auto_liquidate_threshold 0 -> 50). So this is the figure that says whether
    the amendment worked: if it did, the x=0 subgroup's peak inventory collapses
    onto the threshold line instead of running to several hundred. Pass
    --auto-liquidate-threshold to draw that line.
    """
    arms = [a for a in report.arms if report.has(a, "inventory_sd_on")]
    if not arms:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 0.5 * len(arms) + 3.0))

    _strip_by_arm(ax1, report, "inventory_sd_on", arms, cut=cut, seed=91,
                  headroom=0.95)
    ax1.set_xlabel("Inventory SD over the episode, attack on")
    ax1.set_title("(a) Inventory dispersion", loc="left")
    _collapse_legend(ax1, loc="upper right")

    if report.has(arms[0], "peak_inventory_on"):
        _strip_by_arm(ax2, report, "peak_inventory_on", arms, cut=cut, seed=92,
                      headroom=0.95)
        ax2.set_xlabel("Peak |inventory| reached, attack on")
        ax2.set_title("(b) Directional accumulation", loc="left")
        if auto_liquidate_threshold:
            ax2.axvline(auto_liquidate_threshold, color="#B34700",
                        ls="--", lw=1.2, zorder=2)
            ax2.annotate(f"auto-liquidate at {auto_liquidate_threshold:g}",
                         xy=(auto_liquidate_threshold, len(arms) - 0.6),
                         xytext=(4, 0), textcoords="offset points",
                         fontsize=7, color="#B34700", va="center", ha="left")

    act = report.inventory_activity("on", cut)
    ratchet = {a: v for a, v in act.items()
               if v["n_no_two_sided"] and v["trading_one_sided"]}
    if ratchet:
        worst = max(ratchet.values(), key=lambda v: v["max_peak_inventory"])
        note = (
            f"Of the seeds with NO two-sided quote, "
            f"{sum(v['n_no_two_sided'] - v['n_of_those_flat'] for v in ratchet.values())}"
            f" still trade (inventory SD > 0), reaching peak |inventory| "
            f"{worst['max_peak_inventory']:,.0f}. This is an inventory ratchet, "
            "NOT the no-trade optimum -- the two have different fixes.")
    else:
        note = ("Seeds with no two-sided quote are also flat (inventory SD = 0): "
                "consistent with a no-trade optimum rather than a ratchet.")

    fig.suptitle("Is 'no two-sided quote' a no-trade optimum or an inventory ratchet?",
                 x=0.005, ha="left", fontsize=12, fontweight="bold")
    _stamp(fig, report, note)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    return fig


# --------------------------------------------------------------------------
# registry + CLI
# --------------------------------------------------------------------------

FIGURES: dict[str, tuple[Callable, str]] = {
    "validity":   (fig_validity, "quote presence per seed + gate cut sensitivity"),
    "attack":     (fig_attack_delivery, "attack delivered, and the response to it"),
    "forest":     (fig_forest, "pre-registered contrasts with CIs and Holm p"),
    "paired":     (fig_paired_slopes, "seed-level pairing for the H1 contrast"),
    "equivalence": (fig_equivalence, "H2 TOST against pre-registered margins"),
    "auroc":      (fig_auroc, "H3 detection AUROC vs chance"),
    "regime":     (fig_regime, "H4 regime conditioning"),
    "gate":       (fig_progression_gate, "Phase-1 progression gate"),
    "inventory":  (fig_inventory, "inventory ratchet vs no-trade optimum"),
}


# Extra keyword arguments accepted per figure from the CLI.
_FIG_KWARGS: dict[str, tuple[str, ...]] = {
    "inventory": ("auto_liquidate_threshold",),
}


def build(report: EvalReport, outdir: str, only: Sequence[str] | None = None,
          formats: Sequence[str] = ("png",), prefix: str = "",
          **kwargs) -> list[str]:
    """Render figures; returns the paths written.

    Skips (with a note) any figure whose inputs are absent, so a partial run
    still produces what it can. Figure numbering follows the registry order, not
    the subset requested, so `--only` cannot silently renumber a figure that is
    already cited by number in the thesis text.
    """
    _setup_style()
    os.makedirs(outdir, exist_ok=True)
    order = {name: i for i, name in enumerate(FIGURES, start=1)}
    names = list(only) if only else list(FIGURES)
    written = []
    for name in names:
        if name not in FIGURES:
            raise SystemExit(f"unknown figure {name!r}; known: {', '.join(FIGURES)}")
        fn, desc = FIGURES[name]
        extra = {k: kwargs[k] for k in _FIG_KWARGS.get(name, ())
                 if kwargs.get(k) is not None}
        fig = fn(report, **extra)
        i = order[name]
        if fig is None:
            print(f"  [skip] {name:12s} inputs absent from this report ({desc})")
            continue
        for ext in formats:
            path = os.path.join(outdir, f"{prefix}{i:02d}_{name}.{ext}")
            fig.savefig(path)
            written.append(path)
        plt.close(fig)
        print(f"  [ok]   {name:12s} {desc}")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", help="eval_*.json written by run_evaluation.py")
    ap.add_argument("-o", "--outdir", default="figures")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset: " + ", ".join(FIGURES))
    ap.add_argument("--format", default="png",
                    help="comma-separated: png, pdf (pdf for LaTeX inclusion)")
    ap.add_argument("--prefix", default="", help="filename prefix, e.g. 'v3_'")
    ap.add_argument("--auto-liquidate-threshold", type=float, default=None,
                    help="draw the env's auto_liquidate_threshold on the "
                         "inventory figure (50 for the v3 arms)")
    args = ap.parse_args(argv)

    report = EvalReport.load(args.report)
    print(report.describe())
    print()
    if not report.is_confirmatory:
        print("  NOTE: this report is not a signed-off complete run. Figures are\n"
              "        stamped EXPLORATORY/PARTIAL and must not be captioned as\n"
              "        the pre-registered confirmatory analysis.\n")
    only = args.only.split(",") if args.only else None
    paths = build(report, args.outdir, only, tuple(args.format.split(",")),
                  args.prefix,
                  auto_liquidate_threshold=args.auto_liquidate_threshold)
    print(f"\n{len(paths)} file(s) -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
