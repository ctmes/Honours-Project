"""
Aggregation + reporting for the adversarial market-making study.

Turns per-seed metric arrays into the comparisons the proposal §4 specifies:

  Configs:  (1) Baseline: A-S and vanilla IPPO
            (2) Adversarial IPPO: co-trained, no detection head / regime
            (3) Full model: adversarial IPPO + detection head + regime indicator
  Contrasts: 1 vs 2 isolates adversarial co-training
             2 vs 3 isolates the detection + regime contributions

Also implements the Phase-1 progression gate: baseline IPPO must reach Sortino and
Sharpe within a margin of A-S, and inventory SD within a factor of the A-S bound,
on clean data before co-training begins.

Data model
----------
`metrics_by_config` maps a config name -> {metric_name -> np.ndarray of per-seed values},
with seeds aligned by index across configs (paired comparisons rely on this alignment).
All functions are pure NumPy and unit-testable on synthetic per-seed data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping, Sequence

import numpy as np

from stats import paired_comparison, ComparisonResult


def summarize_seeds(per_metric: Mapping[str, Sequence[float]]) -> dict:
    """Per-metric mean / std / median / n across seeds (NaNs ignored)."""
    out = {}
    for name, vals in per_metric.items():
        v = np.asarray(vals, dtype=np.float64).ravel()
        finite = v[np.isfinite(v)]
        out[name] = {
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "std": float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "n": int(finite.size),
        }
    return out


def compare_configs(
    metrics_by_config: Mapping[str, Mapping[str, Sequence[float]]],
    config_a: str,
    config_b: str,
    metric_names: Sequence[str] | None = None,
) -> dict[str, ComparisonResult]:
    """Paired comparison of config_a vs config_b for each metric (a - b)."""
    a_metrics = metrics_by_config[config_a]
    b_metrics = metrics_by_config[config_b]
    if metric_names is None:
        metric_names = sorted(set(a_metrics) & set(b_metrics))
    results = {}
    for m in metric_names:
        results[m] = paired_comparison(a_metrics[m], b_metrics[m])
    return results


@dataclass
class GateResult:
    passed: bool
    sharpe_ok: bool
    sortino_ok: bool
    inventory_ok: bool
    detail: dict

    def as_dict(self) -> dict:
        return asdict(self)


def progression_gate(
    ippo: Mapping[str, Sequence[float]],
    avellaneda_stoikov: Mapping[str, Sequence[float]],
    sharpe_margin: float,
    sortino_margin: float,
    inv_sd_factor: float = 2.0,
    sharpe_key: str = "sharpe",
    sortino_key: str = "sortino",
    inv_sd_key: str = "inventory_sd",
) -> GateResult:
    """Phase-1 gate: vanilla IPPO close enough to A-S on clean data to proceed.

    Criteria (means across seeds):
      sharpe_ippo  >= sharpe_AS  - sharpe_margin
      sortino_ippo >= sortino_AS - sortino_margin
      invSD_ippo   <= inv_sd_factor * invSD_AS
    """
    def _mean(d, k):
        v = np.asarray(d[k], dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        return float(np.mean(v)) if v.size else float("nan")

    sh_i, sh_a = _mean(ippo, sharpe_key), _mean(avellaneda_stoikov, sharpe_key)
    so_i, so_a = _mean(ippo, sortino_key), _mean(avellaneda_stoikov, sortino_key)
    iv_i, iv_a = _mean(ippo, inv_sd_key), _mean(avellaneda_stoikov, inv_sd_key)

    sharpe_ok = sh_i >= sh_a - sharpe_margin
    sortino_ok = so_i >= so_a - sortino_margin
    inventory_ok = iv_i <= inv_sd_factor * iv_a

    return GateResult(
        passed=bool(sharpe_ok and sortino_ok and inventory_ok),
        sharpe_ok=bool(sharpe_ok),
        sortino_ok=bool(sortino_ok),
        inventory_ok=bool(inventory_ok),
        detail={
            "sharpe": {"ippo": sh_i, "as": sh_a, "margin": sharpe_margin},
            "sortino": {"ippo": so_i, "as": so_a, "margin": sortino_margin},
            "inventory_sd": {"ippo": iv_i, "as": iv_a, "factor": inv_sd_factor},
        },
    )


def format_comparison_table(results: Mapping[str, ComparisonResult],
                            title: str = "") -> str:
    """Human-readable table of a {metric -> ComparisonResult} mapping."""
    lines = []
    if title:
        lines.append(title)
    header = (f"{'metric':<22}{'mean_a':>12}{'mean_b':>12}{'diff':>12}"
              f"{'d':>8}{'test':>10}{'p':>10}{'95% CI':>26}")
    lines.append(header)
    lines.append("-" * len(header))
    for m, r in results.items():
        ci = f"[{r.ci_low:.4g}, {r.ci_high:.4g}]"
        lines.append(
            f"{m:<22}{r.mean_a:>12.4g}{r.mean_b:>12.4g}{r.mean_diff:>12.4g}"
            f"{r.cohens_d:>8.3g}{r.test:>10}{r.p_value:>10.3g}{ci:>26}"
        )
    return "\n".join(lines)
