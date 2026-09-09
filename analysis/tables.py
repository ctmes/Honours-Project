"""LaTeX + plain-text results tables from an eval_*.json report.

    python -m analysis.tables D:/tmp/runs/results/eval_1151370.json -o tables/
    python -m analysis.tables results/eval_v3.json --only contrasts --stdout

Emitted tables mirror the figures one-for-one, so the results chapter can quote a
number and point at the panel it came from:

    validity     -> fig 1  (no-two-sided-quote counts, cut sensitivity)
    inventory    -> fig 9  (ratchet vs no-trade discriminator)
    summary      -> per-arm descriptives for every reported metric
    contrasts    -> fig 3  (confirmatory family, raw and Holm-adjusted p)
    equivalence  -> fig 5  (H2 TOST)
    auroc        -> fig 6  (H3)
    gate         -> fig 8  (progression gate)

Reporting conventions, all forced by this data rather than by taste:

  MEDIAN AND IQR, NOT MEAN AND SD. Seed distributions here are bimodal and the
  SD routinely exceeds the mean by an order of magnitude (baseline sortino_on:
  mean 0.26, SD 52.7). The mean is still printed, because the pre-registered
  gate and the TOST margins are defined on means -- but it is never printed
  alone.

  n IS PRINTED PER CELL, NOT PER TABLE. quote_displacement is undefined on the
  seeds that never quoted two-sided, so its column has a different n from its
  neighbours. A single "n = 20" in the caption would be false for that row.

  RAW AND ADJUSTED p SIDE BY SIDE. The confirmatory claim is the Holm-adjusted
  column. The raw column is shown so the adjustment is auditable, and is marked
  as not a significance claim in the caption.
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import numpy as np

from analysis.evalreport import (
    ARM_LABELS, CONTRAST_LABELS, COLLAPSE_CUT, COLLAPSE_MAJORITY,
    DETECTION_ARMS, PRIMARY_METRICS, EvalReport, metric_label,
)

# Metrics described in the per-arm summary table, in reporting order.
SUMMARY_METRICS = (
    "quote_presence_on", "sortino_on", "sharpe_on", "cvar_on", "auroc",
    "inventory_sd_on", "peak_inventory_on", "regime_gap_on",
)


def _esc(s: str) -> str:
    """Escape the LaTeX specials that actually occur in these labels."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


_DELATEX = [
    (r"\textbackslash{}", "\\"), (r"\textasciitilde{}", "~"),
    (r"\textasciicircum{}", "^"),
    (r"\geq", ">="), (r"\leq", "<="), (r"\times", "x"), (r"\,", " "),
    (r"\%", "%"), (r"\_", "_"), (r"\&", "&"), (r"\#", "#"), (r"\$", "$"),
    (r"\{", "{"), (r"\}", "}"),
]


def _delatex(s: str) -> str:
    """Render a LaTeX cell as plain text for the console view.

    Order matters: the multi-character escapes are undone before the single
    backslash ones, or `\\textbackslash{}` would be half-consumed by the `\\{`
    rule and leave debris in the terminal output.
    """
    import re
    # Scientific notation first: it contains \times, which a later rule rewrites.
    s = re.sub(r"\\times10\^\{(-?\d+)\}", r"e\1", s)
    for a, b in _DELATEX:
        s = s.replace(a, b)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    return s.replace("$", "")


def _num(x, sig: int = 3) -> str:
    """Format to `sig` significant figures as a LaTeX cell.

    Extreme values become proper math (`$7.63\\times10^{-5}$`) rather than
    programmer notation; `_delatex` turns that back into `7.63e-5` for the
    console view.
    """
    if x is None:
        return "--"
    x = float(x)
    if not np.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    if abs(x) >= 1e5 or abs(x) < 1e-3:
        mant, exp = f"{x:.{sig - 1}e}".split("e")
        return f"${mant}\\times10^{{{int(exp)}}}$"
    return f"{x:,.{max(0, sig - 1 - int(np.floor(np.log10(abs(x)))))}f}"


def _p(x) -> str:
    if x is None or not np.isfinite(float(x)):
        return "--"
    x = float(x)
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


class Table:
    """A rendered table in both target formats."""

    def __init__(self, key: str, caption: str, header: Sequence[str],
                 rows: Sequence[Sequence[str]], label: str | None = None,
                 note: str | None = None):
        self.key, self.caption, self.header = key, caption, list(header)
        self.rows = [list(r) for r in rows]
        self.label = label or f"tab:{key}"
        self.note = note

    def to_latex(self) -> str:
        cols = "l" + "r" * (len(self.header) - 1)
        out = [r"\begin{table}[htbp]", r"  \centering",
               f"  \\caption{{{_esc(self.caption)}}}",
               f"  \\label{{{self.label}}}",
               f"  \\begin{{tabular}}{{{cols}}}", r"    \toprule",
               "    " + " & ".join(self.header) + r" \\", r"    \midrule"]
        out += ["    " + " & ".join(r) + r" \\" for r in self.rows]
        out += [r"    \bottomrule", r"  \end{tabular}"]
        if self.note:
            out.append(r"  \begin{minipage}{\linewidth}\vspace{4pt}\footnotesize "
                       + _esc(self.note) + r"\end{minipage}")
        out += [r"\end{table}", ""]
        return "\n".join(out)

    def to_text(self) -> str:
        hdr = [_delatex(h) for h in self.header]
        rows = [[_delatex(c) for c in r] for r in self.rows]
        w = [max(len(hdr[i]), *(len(r[i]) for r in rows)) if rows else len(hdr[i])
             for i in range(len(hdr))]
        line = "  ".join("-" * x for x in w)
        out = [self.caption, line,
               "  ".join(h.ljust(w[i]) if i == 0 else h.rjust(w[i])
                         for i, h in enumerate(hdr)), line]
        out += ["  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i])
                          for i, c in enumerate(r)) for r in rows]
        out.append(line)
        if self.note:
            out.append(self.note)
        return "\n".join(out) + "\n"


# --------------------------------------------------------------------------

def tbl_validity(report: EvalReport, cut: float = COLLAPSE_CUT,
                 majority: float = COLLAPSE_MAJORITY) -> Table | None:
    summ = report.collapse_summary("on", cut, majority)
    if not summ:
        return None
    sens = report.collapse_sensitivity("on", majority=majority)
    rows = []
    for arm, s in summ.items():
        counts = "/".join(str(c) for c in sens["counts"].get(arm, []))
        rows.append([
            _esc(ARM_LABELS.get(arm, arm)),
            str(s["n"]),
            f"{s['n_collapsed']}",
            _num(s["frac_collapsed"], 2),
            _num(s["median_qp"], 3),
            _num(s["mean_qp_survivors"], 3),
            counts,
            r"\textbf{FAIL}" if not s["gate_ok"] else "pass",
        ])
    unstable = [a for a, ok in sens["stable"].items() if not ok]
    # Prose deliberately avoids '%' and '<=': notes are escaped for LaTeX, and
    # spelling the comparison out reads better in the caption than markup would.
    note = ("Quote presence is the fraction of steps with a two-sided quote posted. "
            f"An arm fails the pre-registered validity gate when more than "
            f"{majority * 100:g} per cent of its seeds sit at quote presence of "
            f"{cut:g} or below; such an arm cannot support an H1 robustness claim, "
            "because the adversary perturbs only the market maker's observed "
            "depth. The sweep column gives the count at cuts "
            + ", ".join(f"{c:g}" for c in sens["cuts"]) + ". ")
    note += ("The verdict is unchanged across the whole sweep for every arm."
             if not unstable else
             "The verdict is NOT stable across the sweep for: "
             + ", ".join(ARM_LABELS.get(a, a) for a in unstable)
             + "; treat these as boundary cases.")
    return Table("validity", "Validity gate: two-sided quoting by arm.",
                 ["Arm", "n", "No 2-sided", "Frac.", "Median qp",
                  "Mean qp (rest)", "Sweep", "Gate"], rows, note=note)


def tbl_inventory(report: EvalReport, cut: float = COLLAPSE_CUT) -> Table | None:
    act = report.inventory_activity("on", cut)
    if not act:
        return None
    rows = []
    for arm, v in act.items():
        # With no such seeds there is nothing to classify. Printing "flat" here
        # would read as "this arm sits at a no-trade optimum", the opposite of
        # what an arm with zero non-quoting seeds is doing.
        if not v["n_no_two_sided"]:
            reading = "--"
        elif v["trading_one_sided"]:
            reading = "ratchet"
        else:
            reading = "flat"
        rows.append([
            _esc(ARM_LABELS.get(arm, arm)),
            str(v["n_no_two_sided"]),
            str(v["n_of_those_flat"]),
            _num(v["max_inventory_sd"], 4),
            _num(v["max_peak_inventory"], 4),
            reading,
        ])
    note = ("Restricted to the seeds with no two-sided quote. If those seeds were "
            "sitting at a no-trade optimum their inventory SD would be zero; a "
            "large inventory SD and peak excursion instead means they are quoting "
            "one-sided and accumulating directionally, which is a different "
            "failure with a different fix. Columns are maxima over that subgroup.")
    return Table("inventory",
                 "Are the non-quoting seeds inactive, or ratcheting inventory?",
                 ["Arm", "No 2-sided", "of those flat", "max inv. SD",
                  "max peak $|$inv$|$", "Reading"], rows, note=note)


def tbl_summary(report: EvalReport,
                metrics: Sequence[str] = SUMMARY_METRICS) -> Table | None:
    metrics = [m for m in metrics if any(report.has(a, m) for a in report.arms)]
    if not metrics:
        return None
    rows = []
    for arm in report.arms:
        for metric in metrics:
            if not report.has(arm, metric):
                continue
            v = report.per_seed(arm, metric)
            f = v[np.isfinite(v)]
            if not f.size:
                continue
            q1, q3 = np.percentile(f, [25, 75])
            rows.append([
                _esc(ARM_LABELS.get(arm, arm)),
                _esc(metric_label(metric)),
                str(f.size),
                _num(np.median(f)), _num(q1), _num(q3),
                _num(f.mean()), _num(f.std(ddof=1) if f.size > 1 else np.nan),
            ])
    note = ("Median and inter-quartile range lead because these seed distributions "
            "are bimodal; the mean and SD follow because the pre-registered gate "
            "and equivalence margins are defined on means. n is per row: metrics "
            "undefined on non-quoting seeds have a smaller n than their neighbours.")
    return Table("summary", "Per-arm seed distributions.",
                 ["Arm", "Metric", "n", "Median", "Q1", "Q3", "Mean", "SD"],
                 rows, note=note)


def tbl_contrasts(report: EvalReport,
                  metrics: Sequence[str] = PRIMARY_METRICS) -> Table | None:
    labels = report.contrasts()
    if not labels:
        return None
    rows = []
    for lab in labels:
        for metric in metrics:
            c = report.contrast(lab, metric)
            if c is None:
                continue
            holm = report.holm(lab, metric)
            sig = holm is not None and np.isfinite(holm) and holm < 0.05
            rows.append([
                _esc(CONTRAST_LABELS.get(lab, lab)),
                _esc(metric_label(metric)),
                str(c.get("n", "--")),
                _num(c.get("mean_diff")),
                f"[{_num(c.get('ci_low'))}, {_num(c.get('ci_high'))}]",
                _num(c.get("cohens_d"), 2),
                _esc(str(c.get("test", "--"))),
                _p(c.get("p_value")),
                (r"\textbf{" + _p(holm) + "}") if sig else _p(holm),
            ])
    tag = ("" if report.is_confirmatory else
           " THIS RUN IS NOT A SIGNED-OFF COMPLETE EVALUATION: the adjusted column "
           "is not the pre-registered confirmatory test.")
    note = ("Paired contrasts over seeds, paired by seed index. Holm adjustment is "
            "within each contrast's confirmatory family of "
            f"{len(metrics)} metrics. The raw p column is shown so the adjustment "
            "is auditable and is not itself a significance claim; only the "
            "adjusted column supports one. CIs are bootstrap." + tag)
    return Table("contrasts", "Pre-registered contrasts.",
                 ["Contrast", "Metric", "n", "Diff.", "95\\% CI", "$d$",
                  "Test", "$p$ raw", "$p$ Holm"], rows, note=note)


def tbl_equivalence(report: EvalReport) -> Table | None:
    eq = report.equivalence()
    if not eq:
        return None
    rows = []
    for comp, metrics in eq.items():
        for metric, r in metrics.items():
            rows.append([
                _esc(comp.replace("_vs_", " vs ")),
                _esc(metric_label(metric)),
                str(r.get("n", "--")),
                _num(r.get("mean_diff")),
                _num(r.get("margin")),
                _p(r.get("p_lower")), _p(r.get("p_upper")), _p(r.get("p_value")),
                "yes" if r.get("equivalent") else r"\textbf{no}",
            ])
    note = ("Two one-sided tests against pre-registered margins. Equivalence "
            "requires BOTH one-sided tests to reject; a large p on a difference "
            "test is not evidence of equivalence. Margins are a scientific input "
            "fixed before the data were seen, not derived from them.")
    return Table("equivalence", "H2: no degradation on clean data (TOST).",
                 ["Comparison", "Metric", "n", "Mean diff.", "Margin",
                  "$p$ lower", "$p$ upper", "$p$ TOST", "Equivalent"],
                 rows, note=note)


def tbl_auroc(report: EvalReport) -> Table | None:
    above = report.auroc_above_chance()
    if not above:
        return None
    rows, n_nohead_sig = [], 0
    for arm, r in above.items():
        p = r.get("p_value")
        sig = p is not None and np.isfinite(p) and p < 0.05
        has_head = arm in DETECTION_ARMS
        n_nohead_sig += bool(sig and not has_head)
        rows.append([
            _esc(ARM_LABELS.get(arm, arm)),
            "yes" if has_head else "no",
            str(r.get("n", "--")),
            _num(r.get("mean"), 4),
            _num(r.get("cohens_d"), 2),
            _esc(str(r.get("test", "--"))),
            (r"\textbf{" + _p(p) + "}") if (sig and has_head) else _p(p),
        ])
    note = ("One-sample test of AUROC against chance (0.5) on the mixed "
            "attack/clean stream, per arm, not adjusted across arms. Every arm "
            "reports an AUROC because the rollout always computes one, but only "
            "the arms with a detection head can support or refute H3; a nominal "
            "result on a headless arm is not a finding.")
    if n_nohead_sig:
        note += (f" {n_nohead_sig} headless arm(s) reach nominal significance "
                 f"here, which is unremarkable across {len(above)} uncorrected "
                 "tests and is left unbolded for that reason.")
    return Table("auroc", "H3: detection AUROC against chance.",
                 ["Arm", "Head", "n", "Mean AUROC", "$d$", "Test", "$p$ vs 0.5"],
                 rows, note=note)


def tbl_gate(report: EvalReport) -> Table | None:
    gate = report.progression_gate()
    if not gate or not gate.get("detail"):
        return None
    rows = []
    for key, d in gate["detail"].items():
        # These two are built as LaTeX on purpose, so they must NOT go through
        # _esc() -- that would escape the backslashes and print the markup.
        if "margin" in d:
            crit = f"$\\geq$ {_num(float(d['as']) - float(d['margin']))}"
            spec = f"within {_num(d['margin'])} of A-S"
        elif "factor" in d:
            crit = f"$\\leq$ {_num(float(d['as']) * float(d['factor']))}"
            spec = f"within {_num(d['factor'], 2)}$\\times$ A-S"
        else:
            crit, spec = "--", "--"
        ok = bool(gate.get(f"{key}_ok", False))
        rows.append([_esc(key), _num(d.get("ippo")), _num(d.get("as")),
                     spec, crit,
                     "pass" if ok else r"\textbf{FAIL}"])
    note = ("Means over seeds on clean (attack-off) data, which is the quantity "
            "the criteria are defined on. Overall gate: "
            + ("PASSED." if gate.get("passed") else "FAILED."))
    return Table("gate", "Phase-1 progression gate: vanilla IPPO vs the A-S benchmark.",
                 ["Criterion", "IPPO", "A-S", "Requirement", "Threshold", "Result"],
                 rows, note=note)


TABLES = {
    "validity":    (tbl_validity, "two-sided quoting and the validity gate"),
    "inventory":   (tbl_inventory, "ratchet vs no-trade discriminator"),
    "summary":     (tbl_summary, "per-arm seed distributions"),
    "contrasts":   (tbl_contrasts, "pre-registered contrasts, raw and Holm p"),
    "equivalence": (tbl_equivalence, "H2 TOST"),
    "auroc":       (tbl_auroc, "H3 detection vs chance"),
    "gate":        (tbl_gate, "Phase-1 progression gate"),
}


def build(report: EvalReport, outdir: str | None, only: Sequence[str] | None = None,
          to_stdout: bool = False, prefix: str = "") -> list[str]:
    names = list(only) if only else list(TABLES)
    written = []
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    for name in names:
        if name not in TABLES:
            raise SystemExit(f"unknown table {name!r}; known: {', '.join(TABLES)}")
        fn, desc = TABLES[name]
        tbl = fn(report)
        if tbl is None:
            print(f"  [skip] {name:12s} inputs absent from this report ({desc})")
            continue
        if to_stdout:
            print()
            print(tbl.to_text())
        if outdir:
            path = os.path.join(outdir, f"{prefix}{name}.tex")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(tbl.to_latex())
            written.append(path)
            print(f"  [ok]   {name:12s} {desc}")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report")
    ap.add_argument("-o", "--outdir", default=None,
                    help="write .tex files here (omit with --stdout for a dry run)")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset: " + ", ".join(TABLES))
    ap.add_argument("--stdout", action="store_true",
                    help="also print a plain-text rendering")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args(argv)

    if not args.outdir and not args.stdout:
        args.stdout = True
    report = EvalReport.load(args.report)
    print(report.describe())
    print()
    only = args.only.split(",") if args.only else None
    paths = build(report, args.outdir, only, args.stdout, args.prefix)
    if args.outdir:
        print(f"\n{len(paths)} file(s) -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
