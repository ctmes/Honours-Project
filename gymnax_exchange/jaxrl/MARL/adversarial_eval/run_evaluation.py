"""
Top-level evaluation driver: checkpoints -> per-seed metrics -> contrasts + report.

Glue between the rollout layer (rollout.py) and the analysis layer
(metrics / stats / aggregate). For each config the experiment has a set of
independently-trained seeds (checkpoints); each is rolled out under attack-on,
attack-off, and a mixed stream, reduced to a per-seed metric dict, and stacked
into the {config -> {metric -> array-over-seeds}} structure the contrasts consume.

Metric keys carry their condition so the proposal hypotheses are testable:
  *_on   : attack-on windows  (H1: primary robustness hypothesis)
  *_off  : attack-off windows (H2: no-degradation-under-clean — tested with TOST,
           not with a failed significance test)
  auroc  : detection discrimination on the mixed stream (H3: tested vs 0.5)
  regime_gap_* / sortino_lowvol_* / sortino_highvol_* : H4 (regime conditioning)

Inference discipline
--------------------
`primary_metrics` is the PRE-REGISTERED confirmatory family: raw p-values are
Holm-adjusted within each contrast and reported alongside. Everything else in
the summaries is estimation-only (effect sizes + CIs) — do not promote an
uncorrected exploratory p-value to a significance claim in the writeup.

Pairing assumption: seeds are paired by INDEX across configs (same training-seed
list in the same order, same rollout RNG seeds). Keep `run_names` ordered by
training seed identically for every config or the paired tests are invalid.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from gymnax_exchange.jaxrl.MARL.adversarial_eval.rollout import (
    evaluate_checkpoint, evaluate_fixed_policy,
)
from gymnax_exchange.jaxrl.MARL.adversarial_eval.stats import (
    holm_adjust, tost_paired, one_sample_comparison,
)
from gymnax_exchange.jaxrl.MARL.adversarial_eval.aggregate import (
    compare_configs, progression_gate, summarize_seeds, format_comparison_table,
)

_RISK_METRICS = ("sharpe", "sortino", "softmin_sharpe", "cvar",
                 "peak_inventory", "inventory_sd", "quote_displacement",
                 "sortino_lowvol", "sortino_highvol", "regime_gap")

# Confirmatory family (Holm-adjusted). Keep this SMALL: at n=20 seeds, paired-t
# power for d_z=0.8 is ~0.92 at alpha=0.05 but ~0.69 at alpha=0.05/8.
_DEFAULT_PRIMARY = ("sortino_on", "sharpe_on", "cvar_on", "auroc")


def evaluate_seeds(project=None, run_names: Sequence[str] | None = None, n_envs=8,
                   n_steps=None, periods_per_year=98280.0,
                   seeds: Sequence[int] | None = None, step=None,
                   yaml_path: str | None = None,
                   adv_project: str | None = None,
                   adv_run_names: Sequence[str] | None = None,
                   adv_step=None,
                   fixed_policy: bool = False,
                   n_seeds: int | None = None) -> dict[str, np.ndarray]:
    """Roll out every seed (checkpoint run) of one config -> {metric -> array-over-seeds}.

    `run_names` are the checkpoint run directories for this config's seeds (all under the
    same `project`), ordered by training seed — the SAME order for every config, because
    downstream paired tests pair by index. With seed-derived checkpoint naming these are
    ["seed_0", "seed_1", ...]. `seeds` are the rollout RNG seeds (defaults to
    range(n) so eval randomness is also matched across configs).

    `yaml_path` selects the ARM-MATCHED eval config (e.g. eval_2024_test_config2.yaml):
    the env must zero the same obs channels the arm zeroed in training, or the restored
    network sees inputs from a different distribution than it was trained on.

    Common adversary (H1 internal validity): pass `adv_project` + `adv_run_names`
    (ordered like `run_names`) to attack every arm with the same reference adversary —
    typically the config-3 arm's adversaries, paired by seed index. Omit for the
    self-play default (each MM vs its own co-trained adversary).

    A-S baseline: `fixed_policy=True` skips checkpoint restore entirely (env config
    must pin the MM via fixed_action_setting, see evaluate_fixed_policy); give
    `n_seeds` (or `seeds`) instead of `run_names`.
    """
    kw = {} if yaml_path is None else {"yaml_path": yaml_path}
    if fixed_policy:
        if seeds is None:
            seeds = list(range(n_seeds if n_seeds is not None else 20))
        runs = [(None, sd, None) for sd in seeds]
    else:
        if run_names is None:
            raise ValueError("run_names is required unless fixed_policy=True")
        if seeds is None:
            seeds = list(range(len(run_names)))
        if adv_run_names is not None and len(adv_run_names) != len(run_names):
            raise ValueError("adv_run_names must align 1:1 with run_names "
                             f"(got {len(adv_run_names)} vs {len(run_names)})")
        runs = [(rn, sd, adv_run_names[i] if adv_run_names is not None else None)
                for i, (rn, sd) in enumerate(zip(run_names, seeds))]
    rows = []
    for run_name, sd, adv_rn in runs:
        if fixed_policy:
            res = evaluate_fixed_policy(n_envs, n_steps, periods_per_year, seed=sd, **kw)
        else:
            res = evaluate_checkpoint(project, run_name, n_envs, n_steps,
                                      periods_per_year, seed=sd, step=step,
                                      adv_project=adv_project, adv_run_name=adv_rn,
                                      adv_step=adv_step, **kw)
        row = {}
        for m in _RISK_METRICS:
            row[f"{m}_on"] = res["on"][m]
            row[f"{m}_off"] = res["off"][m]
        row["auroc"] = res["mixed"]["auroc"]
        rows.append(row)
    # transpose list-of-dicts -> dict-of-arrays
    return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0]}


def run_full_evaluation(
    configs: Mapping[str, dict],
    primary_metrics: Sequence[str] = _DEFAULT_PRIMARY,
    equivalence_margins: Mapping[str, float] | None = None,
    gate_as: str | None = None,
    gate_ippo: str | None = None,
    gate_kwargs: dict | None = None,
) -> dict:
    """Evaluate every config and emit the proposal's contrasts + gate + hypothesis tests.

    `configs` maps config_name -> kwargs for evaluate_seeds (must include `project`,
    `run_names`). Contrasts: 'adversarial' vs 'baseline' (isolates co-training) and
    'full' vs 'adversarial' (isolates detection + regime) when those names are present.

    `equivalence_margins` maps *_off metric name -> TOST margin (e.g.
    {"sortino_off": 0.25}) for the H2 no-degradation claim; compared full-vs-baseline
    (and adversarial-vs-baseline when present). Margins are a pre-specified scientific
    input — an economically negligible degradation — not derived from the data.
    """
    metrics_by_config = {name: evaluate_seeds(**kw) for name, kw in configs.items()}

    report = {"summaries": {c: summarize_seeds(m) for c, m in metrics_by_config.items()}}

    contrasts = []
    if "adversarial" in metrics_by_config and "baseline" in metrics_by_config:
        contrasts.append(("adversarial_vs_baseline", "adversarial", "baseline"))
    if "full" in metrics_by_config and "adversarial" in metrics_by_config:
        contrasts.append(("full_vs_adversarial", "full", "adversarial"))
    if "full" in metrics_by_config and "baseline" in metrics_by_config:
        contrasts.append(("full_vs_baseline", "full", "baseline"))

    report["holm"] = {}
    for label, a, b in contrasts:
        avail = [m for m in primary_metrics
                 if m in metrics_by_config[a] and m in metrics_by_config[b]
                 and np.isfinite(metrics_by_config[a][m]).all()
                 and np.isfinite(metrics_by_config[b][m]).all()]
        results = compare_configs(metrics_by_config, a, b, avail)
        report[label] = results
        # Holm within this contrast's confirmatory family.
        report["holm"][label] = holm_adjust({m: r.p_value for m, r in results.items()})

    # H2 — equivalence on clean data (TOST), defended configs vs baseline.
    if equivalence_margins:
        report["equivalence_off"] = {}
        for defended in ("full", "adversarial"):
            if defended in metrics_by_config and "baseline" in metrics_by_config:
                tosts = {}
                for metric, margin in equivalence_margins.items():
                    if (metric in metrics_by_config[defended]
                            and metric in metrics_by_config["baseline"]):
                        tosts[metric] = tost_paired(
                            metrics_by_config[defended][metric],
                            metrics_by_config["baseline"][metric],
                            margin=margin,
                        )
                report["equivalence_off"][f"{defended}_vs_baseline"] = tosts

    # H3 — detection above chance: AUROC vs 0.5, per config that has the head.
    report["auroc_above_chance"] = {
        c: one_sample_comparison(m["auroc"], null_value=0.5)
        for c, m in metrics_by_config.items()
        if "auroc" in m and np.isfinite(m["auroc"]).any()
    }

    # Phase-1 progression gate on CLEAN data (attack-off condition).
    if gate_as and gate_ippo and gate_as in metrics_by_config and gate_ippo in metrics_by_config:
        gk = {
            "sharpe_margin": 0.5, "sortino_margin": 0.5,
            # The pipeline emits condition-suffixed keys; the gate is defined on
            # clean data, so wire it to the *_off metrics.
            "sharpe_key": "sharpe_off", "sortino_key": "sortino_off",
            "inv_sd_key": "inventory_sd_off",
        }
        gk.update(gate_kwargs or {})
        report["progression_gate"] = progression_gate(
            metrics_by_config[gate_ippo], metrics_by_config[gate_as], **gk)

    return report


def format_report(report: dict) -> str:
    lines = []
    for contrast in ("adversarial_vs_baseline", "full_vs_adversarial", "full_vs_baseline"):
        if contrast in report:
            lines.append(format_comparison_table(report[contrast], title=contrast))
            holm = report.get("holm", {}).get(contrast)
            if holm:
                lines.append("  Holm-adjusted p (confirmatory family): "
                             + "  ".join(f"{m}={p:.3g}" for m, p in holm.items()))
            lines.append("")
    if "equivalence_off" in report:
        for label, tosts in report["equivalence_off"].items():
            for metric, t in tosts.items():
                verdict = "EQUIVALENT" if t.equivalent else "not shown equivalent"
                lines.append(f"TOST {label} {metric}: diff={t.mean_diff:.4g} "
                             f"margin=±{t.margin:.4g} p={t.p_value:.3g} -> {verdict}")
        lines.append("")
    if "auroc_above_chance" in report:
        for cfg_name, r in report["auroc_above_chance"].items():
            lines.append(f"AUROC vs 0.5 [{cfg_name}]: mean={r.mean:.3f} "
                         f"test={r.test} p={r.p_value:.3g} (n={r.n})")
        lines.append("")
    if "progression_gate" in report:
        g = report["progression_gate"]
        lines.append(f"progression gate: {'PASS' if g.passed else 'FAIL'}  {g.detail}")
    return "\n".join(lines)
