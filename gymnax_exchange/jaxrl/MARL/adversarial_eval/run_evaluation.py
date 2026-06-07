"""
Top-level evaluation driver: checkpoints -> per-seed metrics -> contrasts + report.

Glue between the rollout layer (rollout.py) and the analysis layer
(metrics / stats / aggregate). For each config the experiment has a set of
independently-trained seeds (checkpoints); each is rolled out under attack-on,
attack-off, and a mixed stream, reduced to a per-seed metric dict, and stacked
into the {config -> {metric -> array-over-seeds}} structure the contrasts consume.

Metric keys carry their condition so both proposal hypotheses are testable:
  *_on   : attack-on windows  (primary robustness hypothesis)
  *_off  : attack-off windows (no-degradation-under-clean hypothesis)
  auroc  : detection discrimination on the mixed stream (secondary hypothesis)
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from gymnax_exchange.jaxrl.MARL.adversarial_eval.rollout import evaluate_checkpoint
from gymnax_exchange.jaxrl.MARL.adversarial_eval.aggregate import (
    compare_configs, progression_gate, summarize_seeds, format_comparison_table,
)

_RISK_METRICS = ("sharpe", "sortino", "cvar", "peak_inventory")


def evaluate_seeds(project, run_names: Sequence[str], n_envs=8, n_steps=64,
                   periods_per_year=98280.0, seeds: Sequence[int] | None = None,
                   step=None) -> dict[str, np.ndarray]:
    """Roll out every seed (checkpoint run) of one config -> {metric -> array-over-seeds}.

    `run_names` are the checkpoint run directories for this config's seeds (all under the
    same `project`). `seeds` are the rollout RNG seeds (defaults to range(len(run_names))).
    """
    if seeds is None:
        seeds = list(range(len(run_names)))
    rows = []
    for run_name, sd in zip(run_names, seeds):
        res = evaluate_checkpoint(project, run_name, n_envs, n_steps,
                                  periods_per_year, seed=sd, step=step)
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
    primary_metrics: Sequence[str] = ("sharpe_on", "sortino_on", "cvar_on", "auroc"),
    gate_as: str | None = None,
    gate_ippo: str | None = None,
    gate_kwargs: dict | None = None,
) -> dict:
    """Evaluate every config and emit the proposal's contrasts + (optional) gate.

    `configs` maps config_name -> kwargs for evaluate_seeds (must include `project`,
    `run_names`). Contrasts: 'adversarial' vs 'baseline' (isolates co-training) and
    'full' vs 'adversarial' (isolates detection + regime) when those names are present.
    """
    metrics_by_config = {name: evaluate_seeds(**kw) for name, kw in configs.items()}

    report = {"summaries": {c: summarize_seeds(m) for c, m in metrics_by_config.items()}}

    if "adversarial" in metrics_by_config and "baseline" in metrics_by_config:
        report["adversarial_vs_baseline"] = compare_configs(
            metrics_by_config, "adversarial", "baseline", primary_metrics)
    if "full" in metrics_by_config and "adversarial" in metrics_by_config:
        report["full_vs_adversarial"] = compare_configs(
            metrics_by_config, "full", "adversarial", primary_metrics)

    if gate_as and gate_ippo and gate_as in metrics_by_config and gate_ippo in metrics_by_config:
        gk = gate_kwargs or {"sharpe_margin": 0.5, "sortino_margin": 0.5}
        report["progression_gate"] = progression_gate(
            metrics_by_config[gate_ippo], metrics_by_config[gate_as], **gk)

    return report


def format_report(report: dict) -> str:
    lines = []
    for contrast in ("adversarial_vs_baseline", "full_vs_adversarial"):
        if contrast in report:
            lines.append(format_comparison_table(report[contrast], title=contrast))
            lines.append("")
    if "progression_gate" in report:
        g = report["progression_gate"]
        lines.append(f"progression gate: {'PASS' if g.passed else 'FAIL'}  {g.detail}")
    return "\n".join(lines)
