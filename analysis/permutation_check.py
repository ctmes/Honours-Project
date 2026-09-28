"""RC1: does any reported verdict depend on the pre-registered test's normality gate?

The pre-registered procedure (stats.paired_comparison) screens each set of paired
differences with Shapiro-Wilk, then runs a paired t-test (t CI) or a Wilcoxon
signed-rank test (percentile-bootstrap CI). Pre-testing normality distorts the
conditional error rate of whichever test is chosen (Rochon et al. 2011, 2012), and
the Wilcoxon p and the bootstrap CI address different estimands, which is why a CI
can exclude zero while p > 0.05 (docs/note_h1_forced_attack.md).

This script re-runs EVERY comparison an eval JSON already reports, on the same
per-seed arrays, with one estimand and no gate: an exact sign-flip permutation
test on the mean paired difference (2**20 patterns at n = 20) and a BCa bootstrap
CI on the same mean. Holm is applied within exactly the families the source used.
The pre-registered test remains the result of record; this reports whether any
verdict at alpha = 0.05 would change (preregistration.json -> amendments ->
robustness_checks_amendment_2026-09-28, RC1).

Families checked, where the file has them:
  contrasts      every "<a>_vs_<b>" block (pre-registered primaries; Holm per block)
  noninferiority equivalence_off: one-sided H0 diff <= -margin (and the TOST upper side)
  auroc          auroc_above_chance: one-sample vs 0.5
  within_arm     attack vs off, per arm (sharpe, sortino, cvar; as h1_attack_effect_check)
  did            cross-arm difference-in-differences (as h1_attack_effect_check;
                 Holm per contrast only for a confirmatory file)

Sharpe/Sortino differences are reported in per-step units (RC2); p-values are
unit-free.

    python -m analysis.permutation_check results/eval_1179095.json results/eval_17313.json \\
        results/forced_attack_v4_bid.json results/forced_attack_v4_ask.json \\
        -o results/permutation_check
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from analysis import units as U
from analysis.evalreport import EvalReport
from gymnax_exchange.jaxrl.MARL.adversarial_eval.stats import (
    bca_ci, holm_adjust, paired_comparison, sign_flip_permutation,
)

ALPHA = 0.05
H1_METRICS = ("sharpe", "sortino", "cvar")


def _on_suffix(rep: EvalReport) -> str:
    """Attack-condition suffix: _on (co-trained), _forced (maximal one-sided) or
    _learned (learned optimal attack on a frozen MM, eval_learned_attack.py)."""
    arm = rep.factorial_arms()[0]
    for suf in ("_forced", "_learned"):
        if rep.has(arm, f"sharpe{suf}"):
            return suf
    return "_on"


def _row(family, label, metric, d, p_pre, p_perm, ppy, **extra) -> dict:
    f = U.factor(metric, ppy)
    lo, hi = bca_ci(d)
    return {"family": family, "label": label, "metric": metric, "n": int(d.size),
            "mean_diff": float(d.mean()) * f, "bca_low": lo * f, "bca_high": hi * f,
            "p_preregistered_test": p_pre, "p_sign_flip": p_perm,
            "units": "per-step" if U.is_scaled(metric) else "native", **extra}


def _verdict_pair(r: dict, p_pre_key: str, p_perm_key: str) -> None:
    pre, perm = r.get(p_pre_key), r.get(p_perm_key)
    if pre is None or perm is None or not np.isfinite(pre) or not np.isfinite(perm):
        r["verdict_agrees"] = None
        return
    r["sig_preregistered"] = bool(pre < ALPHA)
    r["sig_sign_flip"] = bool(perm < ALPHA)
    r["verdict_agrees"] = r["sig_preregistered"] == r["sig_sign_flip"]


def check_contrasts(rep: EvalReport, raw: dict, ppy: float) -> list[dict]:
    rows = []
    holm_src = raw.get("holm", {})
    for label, blk in raw.items():
        if "_vs_" not in label or not isinstance(blk, dict):
            continue
        a_arm, b_arm = label.split("_vs_")
        fam_rows, perm_p = [], {}
        for metric, res in blk.items():
            if not (isinstance(res, dict) and "p_value" in res):
                continue
            if not (rep.has(a_arm, metric) and rep.has(b_arm, metric)):
                continue
            a, b = rep.paired(a_arm, b_arm, metric)
            m = rep.finite(a, b)
            d = a[m] - b[m]
            if d.size < 2:
                continue
            p = sign_flip_permutation(d)
            perm_p[metric] = p
            row = _row("contrasts", label, metric, d, res["p_value"], p, ppy,
                       preregistered_test=res.get("test"),
                       mean_diff_matches_source=bool(np.isclose(
                           d.mean(), res["mean_diff"], rtol=1e-9, atol=1e-12)),
                       p_holm_preregistered=holm_src.get(label, {}).get(metric))
            fam_rows.append(row)
        adj = holm_adjust(perm_p) if perm_p else {}
        for row in fam_rows:
            row["p_holm_sign_flip"] = adj.get(row["metric"])
            _verdict_pair(row, "p_holm_preregistered", "p_holm_sign_flip")
            rows.append(row)
    return rows


def check_noninferiority(rep: EvalReport, raw: dict, ppy: float) -> list[dict]:
    rows = []
    for label, blk in raw.get("equivalence_off", {}).items():
        a_arm, b_arm = label.split("_vs_")
        for metric, res in blk.items():
            if not (rep.has(a_arm, metric) and rep.has(b_arm, metric)):
                continue
            a, b = rep.paired(a_arm, b_arm, metric)
            m = rep.finite(a, b)
            d = a[m] - b[m]
            margin = float(res["margin"])
            p_lo = sign_flip_permutation(d + margin, alternative="greater")
            p_up = sign_flip_permutation(d - margin, alternative="less")
            row = _row("noninferiority", label, metric, d, res["p_lower"], p_lo, ppy,
                       margin=margin * U.factor(metric, ppy),
                       p_upper_preregistered=res["p_upper"], p_upper_sign_flip=p_up,
                       tost_equivalent_preregistered=bool(res["equivalent"]),
                       tost_equivalent_sign_flip=bool(max(p_lo, p_up) < ALPHA))
            _verdict_pair(row, "p_preregistered_test", "p_sign_flip")
            rows.append(row)
    return rows


def check_auroc(rep: EvalReport, raw: dict) -> list[dict]:
    rows = []
    for arm, res in raw.get("auroc_above_chance", {}).items():
        if not rep.has(arm, "auroc"):
            continue
        x = rep.per_seed(arm, "auroc")
        x = x[np.isfinite(x)]
        if x.size < 2:
            continue
        d = x - 0.5
        row = _row("auroc", arm, "auroc", d, res.get("p_value"),
                   sign_flip_permutation(d), 0.0)
        row["mean_auroc"] = float(x.mean())
        _verdict_pair(row, "p_preregistered_test", "p_sign_flip")
        rows.append(row)
    return rows


def check_within_arm(rep: EvalReport, ppy: float) -> list[dict]:
    on, rows = _on_suffix(rep), []
    for arm in rep.factorial_arms():
        for metric in H1_METRICS:
            k_on, k_off = f"{metric}{on}", f"{metric}_off"
            if not (rep.has(arm, k_on) and rep.has(arm, k_off)):
                continue
            x, y = rep.per_seed(arm, k_on), rep.per_seed(arm, k_off)
            m = rep.finite(x, y)
            if m.sum() < 3:
                continue
            r = paired_comparison(x[m], y[m])
            d = x[m] - y[m]
            row = _row("within_arm", arm, k_on, d, r.p_value, sign_flip_permutation(d), ppy,
                       preregistered_test=r.test)
            _verdict_pair(row, "p_preregistered_test", "p_sign_flip")
            rows.append(row)
    return rows


def check_did(rep: EvalReport, raw: dict, ppy: float) -> list[dict]:
    on, rows = _on_suffix(rep), []
    for label in raw:
        if "_vs_" not in label or label == "equivalence_off":
            continue
        a_arm, b_arm = label.split("_vs_")
        fam, perm_p, pre_p = [], {}, {}
        for metric in H1_METRICS:
            keys = [(a_arm, f"{metric}{on}"), (a_arm, f"{metric}_off"),
                    (b_arm, f"{metric}{on}"), (b_arm, f"{metric}_off")]
            if not all(rep.has(a, k) for a, k in keys):
                continue
            da = rep.per_seed(a_arm, f"{metric}{on}") - rep.per_seed(a_arm, f"{metric}_off")
            db = rep.per_seed(b_arm, f"{metric}{on}") - rep.per_seed(b_arm, f"{metric}_off")
            m = rep.finite(da, db)
            if m.sum() < 3:
                continue
            r = paired_comparison(da[m], db[m])
            d = da[m] - db[m]
            p = sign_flip_permutation(d)
            perm_p[metric], pre_p[metric] = p, r.p_value
            fam.append(_row("did", label, metric, d, r.p_value, p, ppy,
                            preregistered_test=r.test))
        confirm = rep.is_confirmatory
        adj_perm = holm_adjust(perm_p) if (confirm and perm_p) else {}
        adj_pre = holm_adjust(pre_p) if (confirm and pre_p) else {}
        for row in fam:
            if confirm:
                row["p_holm_preregistered"] = adj_pre.get(row["metric"])
                row["p_holm_sign_flip"] = adj_perm.get(row["metric"])
                _verdict_pair(row, "p_holm_preregistered", "p_holm_sign_flip")
            else:
                _verdict_pair(row, "p_preregistered_test", "p_sign_flip")
            rows.append(row)
    return rows


def run(path: str) -> dict:
    raw = json.load(open(path))
    rep = EvalReport(raw, source=os.path.basename(path))
    ppy = float(rep.meta.get("periods_per_year", 0.0) or 0.0)
    rows = (check_contrasts(rep, raw, ppy) + check_noninferiority(rep, raw, ppy)
            + check_auroc(rep, raw) + check_within_arm(rep, ppy) + check_did(rep, raw, ppy))
    flips = [r for r in rows if r.get("verdict_agrees") is False]
    mismatched = [r for r in rows if r.get("mean_diff_matches_source") is False]
    by_family = {}
    for r in rows:
        f = by_family.setdefault(r["family"], {"n": 0, "agree": 0, "flip": 0})
        f["n"] += 1
        f["agree"] += r.get("verdict_agrees") is True
        f["flip"] += r.get("verdict_agrees") is False
    return {"source": path, "confirmatory": rep.is_confirmatory,
            "on_suffix": _on_suffix(rep), "rows": rows, "by_family": by_family,
            "flips": flips, "mean_diff_mismatches": mismatched}


def format_text(res: dict) -> str:
    tag = "CONFIRMATORY" if res["confirmatory"] else "exploratory"
    lines = [f"=== RC1 sign-flip check: {res['source']} ({tag}) ===",
             "family           n   agree  flip"]
    for fam, c in res["by_family"].items():
        lines.append(f"  {fam:<14} {c['n']:>3}   {c['agree']:>5} {c['flip']:>5}")
    if res["mean_diff_mismatches"]:
        lines.append(f"  !! {len(res['mean_diff_mismatches'])} contrast(s) whose recomputed "
                     "mean_diff does not match the source JSON -- investigate before use")
    if res["flips"]:
        lines.append("  verdict changes at alpha=0.05 (pre-registered -> sign-flip):")
        for r in res["flips"]:
            pre = r.get("p_holm_preregistered", r["p_preregistered_test"])
            perm = r.get("p_holm_sign_flip", r["p_sign_flip"])
            lines.append(f"    {r['family']:<14} {r['label']:<32} {r['metric']:<16} "
                         f"diff={r['mean_diff']:+.4g}  p {pre:.3g} -> {perm:.3g}")
    else:
        lines.append("  no verdict changes at alpha=0.05")
    return "\n".join(lines)


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def format_tex(results: list[dict]) -> str:
    """Summary table: one row per file x family, plus any flips listed below it."""
    out = [r"\begin{tabular}{llrrr}", r"\toprule",
           r"Source & Family & Tests & Agree & Change \\", r"\midrule"]
    for res in results:
        name = _tex_escape(os.path.basename(res["source"]).replace(".json", ""))
        for fam, c in res["by_family"].items():
            out.append(f"{name} & {_tex_escape(fam)} & {c['n']} & {c['agree']} & {c['flip']} \\\\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out.append(r"\end{tabular}")
    flips = [(res, r) for res in results for r in res["flips"]]
    if flips:
        out += ["", "% Verdict changes (pre-registered p -> sign-flip p):"]
        for res, r in flips:
            pre = r.get("p_holm_preregistered", r["p_preregistered_test"])
            perm = r.get("p_holm_sign_flip", r["p_sign_flip"])
            out.append(f"% {os.path.basename(res['source'])} {r['family']} {r['label']} "
                       f"{r['metric']}: {pre:.3g} -> {perm:.3g}")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--out", default=None,
                    help="output prefix: writes <out>.json, <out>.txt and <out>.tex")
    args = ap.parse_args()
    results = [run(p) for p in args.paths]
    text = "\n\n".join(format_text(r) for r in results)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(f"{out}.json", "w") as f:
            json.dump(results, f, indent=2, default=float)
        with open(f"{out}.txt", "w") as f:
            f.write(text + "\n")
        with open(f"{out}.tex", "w") as f:
            f.write(format_tex(results))
        print(f"\nwrote {out}.json, {out}.txt, {out}.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
