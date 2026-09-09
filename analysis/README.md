# `analysis/` — eval JSON to thesis figures and tables

Turns an `eval_*.json` written by `gymnax_exchange/jaxrl/MARL/adversarial_eval/run_evaluation.py`
into the results chapter's figures and LaTeX tables. Nothing here imports jax or
touches a checkpoint.

Built and validated against `eval_1151370.json` (the v2 sweep, 7 arms x 20 seeds).
The v3 reports have the same schema, so this should run unchanged the moment they land.

## Usage

```bash
# everything, PNG for reading + PDF for \includegraphics
python -m analysis.figures results/eval_1151370.json -o figures/ \
    --format png,pdf --auto-liquidate-threshold 50

# LaTeX tables (omit -o and it prints a plain-text version instead)
python -m analysis.tables results/eval_1151370.json -o tables/
python -m analysis.tables results/eval_1151370.json --only contrasts --stdout

# one figure while iterating
python -m analysis.figures results/eval_v3.json -o figures/ --only validity
```

`--prefix v3_` namespaces the output when comparing two runs side by side.

## What gets produced

| # | Figure | Table | Answers |
|---|--------|-------|---------|
| 1 | `validity` | `validity` | Is the MM posting two-sided quotes at all? Includes the cut-sensitivity sweep. |
| 2 | `attack` | — | Was the attack delivered, and did anything respond to it? |
| 3 | `forest` | `contrasts` | The pre-registered contrasts, with Holm-adjusted p. |
| 4 | `paired` | — | The seed-level pairing behind contrast 3, and the effective n. |
| 5 | `equivalence` | `equivalence` | H2, by TOST against the pre-registered margins. |
| 6 | `auroc` | `auroc` | H3, restricted to the arms that actually have a detection head. |
| 7 | `regime` | — | H4 regime conditioning. |
| 8 | `gate` | `gate` | The Phase-1 progression gate. |
| 9 | `inventory` | `inventory` | Is "no two-sided quote" a no-trade optimum or an inventory ratchet? |
| — | — | `summary` | Per-arm seed distributions for every reported metric. |

Figure numbers are fixed by the registry, not by what `--only` selects, so a
figure cited by number in the thesis text keeps that number.

## Reading the v2 output — three traps this code is built to avoid

**1. `quote_presence = 0` does not mean "not trading."** It counts steps with a
bid *and* an ask. On v2, every single zero-presence seed still reaches inventory
SD ~130 and peak |inventory| ~449. Those seeds are quoting one-sided and
ratcheting inventory — a different failure from the Mohl et al. no-trade optimum,
with a different fix. Figure 9 and `EvalReport.inventory_activity()` are the
discriminator; the label everywhere is "no two-sided quote", never "not trading".

**2. n = 20 overstates the power of the paired tests.** In the B-vs-A contrast,
9 of 20 seed pairs post no two-sided quote in either arm, and 8 of those differ by
*exactly* zero on the risk metrics. The effective n is nearer 12. Figure 4 prints
this per metric.

**3. Every arm reports an AUROC, including the ones with no detection head.**
Only `detection` and `full` carry one (`evalreport.DETECTION_ARMS`). On v2 the
fixed-policy A-S benchmark posts a nominally significant AUROC (p = 0.011) purely
as noise across 7 uncorrected tests. Figure 6 and the `auroc` table refuse to
bold or highlight a headless arm for that reason.

## Design rules

- **Distributions, not bars of means.** Seed distributions here are bimodal and
  the SD routinely exceeds the mean by an order of magnitude. Every arm-level
  panel plots all seeds with a median rule. The one exception is the progression
  gate, where the criterion is itself defined on a mean — and there the mean is
  drawn as a rule over the seed cloud, not as a bar.
- **Non-finite is reported, never dropped.** `quote_displacement` is NaN exactly
  on the non-quoting seeds; dropping those silently conditions the statistic on
  the seeds that kept quoting. Panel and per-row `n` make that visible.
- **Collapse is encoded by marker shape as well as colour**, so it survives
  greyscale printing and colour-vision deficiency.
- **Palette is validated, not eyeballed.** The six-arm categorical palette passes
  all six checks of the `dataviz` skill validator against the light/print surface
  (worst adjacent CVD ΔE 9.5, deutan). Colour follows arm identity, so dropping an
  arm never repaints the others. Re-run the validator before changing any hue.
- **Provenance on every figure.** Source file, seed count, checkpoint step, and
  whether the run was signed-off and complete. A report that is not confirmatory
  is stamped `EXPLORATORY/PARTIAL`, and the forest plot says so in place of the
  confirmatory caption.
- **Raw p is shown next to adjusted p in tables, never in figures.** The
  adjustment stays auditable without letting an uncorrected exploratory p be
  lifted out of a figure and promoted to a significance claim.

## When the v3 JSON lands

```bash
python -m analysis.figures results/eval_<jobid>.json -o figures/v3 \
    --format png,pdf --prefix v3_ --auto-liquidate-threshold 50
python -m analysis.tables  results/eval_<jobid>.json -o tables/v3 --prefix v3_
```

Read figure 1 and figure 9 first, in that order. Figure 1 says whether the arms
clear the pre-registered validity gate; figure 9 says whether the
`auto_liquidate_threshold = 50` amendment actually removed the ratchet — if it
did, the peak-inventory cloud collapses onto the threshold line instead of
running out to ~450. Nothing in figures 3–7 is interpretable as evidence about
H1–H4 until figure 1 passes.

## Tests

`tests/test_analysis.py` (35 tests) covers pairing refusal on mismatched seed
counts, the NaN-preservation contract, the gate rule and its sensitivity sweep,
the ratchet/no-trade discriminator, graceful skipping on partial reports, stable
figure numbering, and LaTeX escaping round-trips. They build a synthetic report
rather than reading a fixture, so they state their own preconditions.
