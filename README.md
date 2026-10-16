# Can Spoofing Fool a Learned Market Maker?

Code for the Honours thesis *Can Spoofing Fool a Learned Market Maker? Adversarial Evaluation on a Deep Order Book* (Colin Melville, University of Western Australia, 2026).

The project trains a reinforcement-learning market maker on real NASDAQ limit order book data (AMZN, 2024) and tests whether a spoofing adversary can degrade it by placing fake depth in the order book the market maker observes. It also tests two defences: an auxiliary detection head that predicts whether the book is being spoofed, and conditioning on the volatility regime.

The code extends JaxMARL-HFT (Mohl et al., 2025). The original framework is described in [Upstream framework](#upstream-framework) below.

## Hypotheses

The design and all analysis decisions are pre-registered in [preregistration.json](preregistration.json), including every later amendment and the date it was recorded.

| | Hypothesis |
|---|---|
| H1 | A spoofing attack degrades the market maker's risk-adjusted performance (Sharpe, Sortino, CVaR). |
| H2 | The defences do not degrade performance on clean, unattacked data (non-inferiority test). |
| H3 | The detection head identifies attacked steps better than chance (AUROC above 0.5). |
| H4 | Regime conditioning narrows the performance gap between high- and low-volatility days. |

The thesis reports the results. This repository holds the code, configurations and raw evaluation output.

## What this project adds

| Component | Location |
|---|---|
| Spoofing adversary: injects fake depth into the observed book, with a budget, fill costs and a detection penalty | [gymnax_exchange/jaxen/spoofing_agent.py](gymnax_exchange/jaxen/spoofing_agent.py) |
| Adversarial environment: wraps the market-making environment with the attack, attack labels and regime labels | [gymnax_exchange/jaxen/adversarial_marl_env.py](gymnax_exchange/jaxen/adversarial_marl_env.py) |
| Market-maker network with a shared encoder, policy head and detection head | [gymnax_exchange/jaxrl/MARL/attack_aware_policy.py](gymnax_exchange/jaxrl/MARL/attack_aware_policy.py) |
| PCGrad, to combine the PPO and detection gradients | [gymnax_exchange/jaxrl/MARL/pcgrad.py](gymnax_exchange/jaxrl/MARL/pcgrad.py) |
| Training loop: market maker and adversary trained in alternating phases | [gymnax_exchange/jaxrl/MARL/ippo_adversarial.py](gymnax_exchange/jaxrl/MARL/ippo_adversarial.py) |
| Evaluation: rollouts, metrics, bootstrap statistics, Holm correction | [gymnax_exchange/jaxrl/MARL/adversarial_eval/](gymnax_exchange/jaxrl/MARL/adversarial_eval/) |
| Analysis of evaluation output into thesis figures and tables | `analysis/` |
| Data download, conversion and caching | top-level `download_*.py`, `convert_dbn_to_lobster.py`, `build_*.py` |
| Cluster job scripts (Slurm, Pawsey Kaya) | top-level `slurm_*.sh`, `submit_chain.sh` |
| Tests | [tests/](tests/) |

## Setup

```bash
conda create -n honours python=3.10
conda activate honours
pip install "jax[cuda12]"        # or plain "jax" for CPU
pip install -r requirements.txt
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

Training and evaluation for the thesis ran on CPU nodes. The GPU path works but was not used for the final runs.

## Data

The data is AMZN market-by-order data from DataBento (NASDAQ TotalView-ITCH), converted to LOBSTER format. It is not included because it is licensed and large (about 22 GB for the training period).

1. Put your DataBento key in a `.env` file at the repo root as `DATABENTO_API_KEY=...`. The file is gitignored.
2. Download and convert:
   ```bash
   python download_data.py --year 2024 --estimate-only   # check the cost first
   python download_data.py --year 2024
   python download_data.py --year 2023                   # warm-up year for the regime labels
   python convert_dbn_to_lobster.py
   ```
3. Build the loader caches for the two periods. `2024_train` covers 2 January to 30 September 2024 (188 days). `2024_test` is held out and covers 1 October to 31 December 2024 (64 days).
   ```bash
   python build_period_cache.py --period 2024_train   # build_period_cache_lowmem.py if RAM is short
   python build_period_cache.py --period 2024_test
   ```
4. The window-to-date maps (`window_to_date_2024_*.json`) and the regime labels (`regime_labels.json`) are committed. `build_window_to_date.py` and `build_regime_labels.py` regenerate them.

## Training

Each experimental arm is one config in `config/rl_configs/`, trained for 20 seeds (1002 updates each). The `v4` configs are the final experiment, using the `spread_skew` action space. Configs without a version prefix, or with `v3`, belong to earlier versions and are kept because preregistration.json refers to them.

| Arm | Config |
|---|---|
| Baseline: no adversary | `kaya_v4_config1_baseline` |
| Adversarial: constrained adversary, no defences | `kaya_v4_config2_adversarial` |
| Detection head and regime conditioning | `kaya_v4_config3_full` |
| Detection head only | `kaya_v4_config4_detection` |
| Regime conditioning only | `kaya_v4_config5_regime` |
| Adversary without costs, no defences | `kaya_v4_config6_unconstrained` |
| Detection head, output not fed back to the policy | `kaya_v4_config7_detection_noobs` |
| As above, with shuffled labels (random-label control) | `kaya_v4_config8_shuffled_noobs` |
| As above, with true labels and without PCGrad | `kaya_v4_config9_nopcgrad_noobs` |
| Detection head trained on a scripted, large one-sided attack | `kaya_v4s_config4_detection` |
| Learned attack against a frozen market maker | `kaya_v4opt_{baseline,adversarial,full}` |
| Recurrent market maker | `kaya_v4r_config{1,3}_*` |

[make_tier4_configs.py](make_tier4_configs.py) generates the last five rows from the `v4` configs.

One seed, run locally:

```bash
python gymnax_exchange/jaxrl/MARL/ippo_adversarial.py \
    --config-name=kaya_v4_config3_full SEED=0 \
    TimePeriod=2024_train WINDOW_TO_DATE_PATH=window_to_date_2024_train.json \
    TOTAL_TIMESTEPS=8208384
```

On the cluster, `sbatch --array=0-19%10 slurm_cpu_sweep.sh kaya_v4_config3_full` runs all 20 seeds. Checkpoints are written to `checkpoints/` (gitignored).

## Evaluation

All evaluation runs on the held-out `2024_test` period.

| Script | What it does |
|---|---|
| [run_production_eval.py](run_production_eval.py) | Main evaluation: every arm with the attack on, the attack off, and an Avellaneda-Stoikov baseline. Produces the pre-registered tests. Use `--project-prefix v4` for the final experiment. |
| [eval_forced_attack.py](eval_forced_attack.py) | Replaces the adversary's action with a fixed, maximal one-sided spoof on the bid or ask side. `--inject-mult` scales the attack for the dose-response test. |
| [eval_learned_attack.py](eval_learned_attack.py) | Evaluates attacks learned against a frozen market maker. |
| `check_*.py`, [diagnose_collapse.py](diagnose_collapse.py) | Diagnostics: whether the policy's actions respond to the attack, what the adversary learned, encoder feature rank, and why a market maker stops trading. |

Each script's docstring gives example commands, and the matching `slurm_*.sh` script shows the exact arguments used for the thesis.

## Results

`results/` holds the raw output behind every number in the thesis. Each run writes an `.json` with all statistics and a `.txt` summary.

| File | Contents |
|---|---|
| `eval_1179095` | Version 3 evaluation: the confirmatory, pre-registered test |
| `eval_17313` | Final (version 4) evaluation, all main arms |
| `eval_48162`, `eval_55350`, `eval_87826` | Shuffled-label control, recurrent market maker, and no-PCGrad arm |
| `forced_attack_v4_{bid,ask}` | Maximal one-sided attack on the version 4 arms |
| `forced_attack_v4s_*`, `h3m_generalisation_*` | Detection head trained on scripted attacks: in-family and out-of-family tests |
| `learned_attack_v4`, `forced_attack_tier4_*` | Learned attack, and the recurrent market maker under the maximal attack |
| `dose_response_m*` | Maximal attack at 2, 10, 50, 200 and 1000 times best-quote depth |
| `qi_fair_value_*`, `shallow_book_summary` | How far a rational quote should move per unit of queue imbalance, for AMZN and four shallower-book stocks |
| `gate_arms_*`, `permutation_check`, `rc3_detection` | Robustness checks |

To regenerate the figures and tables from these files:

```bash
python -m analysis.figures results/eval_17313.json -o figures/ --format png,pdf
python -m analysis.tables results/eval_17313.json -o tables/
python -m analysis.h1_attack_effect_check results/forced_attack_v4_bid.json --on-suffix _forced
```

## Tests

```bash
pytest tests/ gymnax_exchange/jaxrl/MARL/adversarial_eval/ --basetemp=.pytest_tmp
```

## Upstream framework

JaxMARL-HFT is a JAX framework for multi-agent reinforcement learning in high-frequency trading. It is built on the [JAX-LOB](https://github.com/KangOxford/jax-lob) order book simulator and extends [JaxMARL](https://github.com/FLAIROx/JaxMARL). It provides GPU-parallel market-making, execution and directional-trading agents on LOBSTER data. The upstream agents can still be trained, for example:

```bash
python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
    --config-name="ippo_rnn_JAXMARL_2player" WANDB_MODE="disabled"
```

Set `alphatradePath`, `dataPath`, `stock` and `timePeriod` in the chosen `config/env_configs/*.json` first. The `Makefile` and `Dockerfile` are from upstream (x86_64 only).

### Citation

```bibtex
@inproceedings{mohl2025jaxmarlhft,
  title={JaxMARL-HFT: GPU-Accelerated Large-Scale Multi-Agent Reinforcement Learning for High-Frequency Trading},
  author={Mohl, Valentin and Frey, Sascha and Leyland, Reuben and Li, Kang and Nigmatulin, George and Cucuringu, Mihai and Zohren, Stefan and Foerster, Jakob and Calinescu, Anisoara},
  booktitle={Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF)},
  pages={18--26},
  year={2025},
  doi={10.1145/3768292.3770416}
}
```

## Disclaimer

This software is for research and education only. It is not intended for live trading or any real-money use, and comes with no warranty.

## License

Apache License 2.0. See [LICENSE](LICENSE).
