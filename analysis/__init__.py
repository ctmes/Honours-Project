"""Analysis layer: eval_*.json -> thesis figures and tables.

Deliberately separate from gymnax_exchange/: nothing here imports jax or touches
a checkpoint. These modules read the JSON that run_evaluation.py already wrote,
so the whole results chapter can be regenerated on a laptop in seconds without
the cluster, and a figure bug never costs a re-rollout.
"""

from analysis.evalreport import EvalReport, ARM_ORDER, ARM_LABELS, PRIMARY_METRICS

__all__ = ["EvalReport", "ARM_ORDER", "ARM_LABELS", "PRIMARY_METRICS"]
