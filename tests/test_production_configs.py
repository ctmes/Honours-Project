"""
Wiring tests for the production sweep + eval configs (pure file checks, no JAX).

The three arms must differ ONLY in the intended flags, or the 1-vs-2 / 2-vs-3
contrasts stop isolating what the proposal says they isolate. These tests pin
that invariant, plus the config mistakes that already burned cluster time once:
the divergent GAMMA (0.999999999), empty regime paths, and a training config
whose sizing silently exceeds the 24h wall / V100 memory.
"""
import json
import os

import pytest
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_YAMLS = {
    "config1": "config/rl_configs/kaya_config1_baseline.yaml",
    "config2": "config/rl_configs/kaya_config2_adversarial.yaml",
    "config3": "config/rl_configs/kaya_config3_full.yaml",
}
EVAL_YAMLS = {
    "config1": "config/rl_configs/eval_2024_test_config1.yaml",
    "config2": "config/rl_configs/eval_2024_test_config2.yaml",
    "config3": "config/rl_configs/eval_2024_test_config3.yaml",
    "as": "config/rl_configs/eval_2024_test_as.yaml",
}
ENV_JSONS = {
    "config1": "config/env_configs/adversarial_mm_cluster_config1.json",
    "config2": "config/env_configs/adversarial_mm_cluster_config2.json",
    "config3": "config/env_configs/adversarial_mm_cluster.json",
    "as": "config/env_configs/adversarial_mm_cluster_as.json",
}

MM_ABLATION_FLAGS = ("use_detection_head", "regime_conditioning", "prev_detection_in_obs")


def _yaml(rel):
    return OmegaConf.to_container(OmegaConf.load(os.path.join(ROOT, rel)))


def _json(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return json.load(f)


# --------------------------------------------------------------------- YAMLs
def test_training_arms_differ_only_in_env_config_and_project():
    cfgs = {k: _yaml(p) for k, p in TRAIN_YAMLS.items()}
    keys = set().union(*(c.keys() for c in cfgs.values()))
    allowed_diff = {"ENV_CONFIG", "PROJECT"}
    for key in keys - allowed_diff:
        vals = {k: c.get(key) for k, c in cfgs.items()}
        assert len({json.dumps(v) for v in vals.values()}) == 1, (
            f"training arms must share hyperparameters; '{key}' differs: {vals}")


@pytest.mark.parametrize("name,path", list(TRAIN_YAMLS.items()) + list(EVAL_YAMLS.items()))
def test_yaml_wiring(name, path):
    cfg = _yaml(path)
    # The near-undiscounted MM gamma drove the 19-Jun value-loss divergence.
    assert cfg["GAMMA"][0] == 0.999, f"{path}: MM GAMMA must be 0.999"
    # Regime channel must be wired — empty paths silently zero the regime.
    assert cfg["REGIME_LABELS_PATH"], f"{path}: REGIME_LABELS_PATH is empty"
    assert cfg["WINDOW_TO_DATE_PATH"], f"{path}: WINDOW_TO_DATE_PATH is empty"
    # The env json it points to must exist.
    assert os.path.exists(os.path.join(ROOT, cfg["ENV_CONFIG"])), (
        f"{path}: ENV_CONFIG {cfg['ENV_CONFIG']} does not exist")
    # Integer number of updates (the LR anneal and phase alternation assume it).
    total, steps, envs = cfg["TOTAL_TIMESTEPS"], cfg["NUM_STEPS"], cfg["NUM_ENVS"]
    assert total % (steps * envs) == 0, f"{path}: TOTAL_TIMESTEPS not an integer #updates"


@pytest.mark.parametrize("name", ["config1", "config2", "config3"])
def test_train_and_eval_periods(name):
    train, ev = _yaml(TRAIN_YAMLS[name]), _yaml(EVAL_YAMLS[name])
    assert train["TimePeriod"] != ev["TimePeriod"], "eval must be out-of-sample"
    assert ev["TimePeriod"] == "2024_test"
    assert "2024_test" in ev["WINDOW_TO_DATE_PATH"]
    # Chained training: the yaml carries phase 1 (Q1); q2/q3 are slurm overrides.
    assert train["TimePeriod"] == "2024_q1"
    assert train["TimePeriod"] in train["WINDOW_TO_DATE_PATH"], (
        "window map must match the phase TimePeriod")


@pytest.mark.parametrize("name", ["config1", "config2", "config3"])
def test_chain_phase_math(name):
    """Q1->Q2->Q3 chain: 334 updates/phase, LR anneal pinned to the chain total.

    slurm_sweep.sh hard-codes the cumulative phase budgets (2736128 / 5472256 /
    8208384); the yaml must carry phase 1 and an ANNEAL_TOTAL_UPDATES equal to
    the full chain, or phase 1 anneals the LR to zero by update 334.
    """
    cfg = _yaml(TRAIN_YAMLS[name])
    per_update = cfg["NUM_STEPS"] * cfg["NUM_ENVS"]
    assert cfg["TOTAL_TIMESTEPS"] // per_update == 334
    assert cfg["ANNEAL_TOTAL_UPDATES"] == 3 * 334
    # The slurm-side cumulative budgets must be exact multiples too.
    for cum in (2736128, 5472256, 8208384):
        assert cum % per_update == 0
    assert 8208384 // per_update == cfg["ANNEAL_TOTAL_UPDATES"]


def test_eval_config1_uses_attack_capable_env_with_zeroed_channels():
    # Config-1's TRAINING json disables the adversary entirely; its EVAL json
    # must re-enable the attack while keeping the MM obs channels zeroed —
    # i.e. the config-2 env json.
    ev = _yaml(EVAL_YAMLS["config1"])
    env = _json(ev["ENV_CONFIG"])
    mm, spoof = env["dict_of_agents_configs"]["AdversarialMM"], env["dict_of_agents_configs"]["Spoofing"]
    assert all(mm[f] is False for f in MM_ABLATION_FLAGS)
    assert spoof["budget_per_episode"] > 0, "eval env must be able to attack"


# ------------------------------------------------------------------ env JSONs
def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if k == "_comment":
            continue
        if isinstance(v, dict):
            out.update(_flat(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def _diff(a, b):
    fa, fb = _flat(a), _flat(b)
    return {k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}


def test_config1_json_isolates_attack_and_channels():
    diff = _diff(_json(ENV_JSONS["config1"]), _json(ENV_JSONS["config3"]))
    expected = {
        "dict_of_agents_configs.AdversarialMM.use_detection_head",
        "dict_of_agents_configs.AdversarialMM.regime_conditioning",
        "dict_of_agents_configs.AdversarialMM.prev_detection_in_obs",
        "dict_of_agents_configs.Spoofing.attack_on_prob",
        "dict_of_agents_configs.Spoofing.attack_off_prob",
        "dict_of_agents_configs.Spoofing.budget_per_episode",
    }
    assert diff == expected, f"config1 vs config3 unexpected diffs: {diff ^ expected}"
    spoof = _json(ENV_JSONS["config1"])["dict_of_agents_configs"]["Spoofing"]
    assert spoof["attack_on_prob"] == 0.0 and spoof["attack_off_prob"] == 1.0
    assert spoof["budget_per_episode"] == 0.0


def test_config2_json_isolates_detection_and_regime():
    diff = _diff(_json(ENV_JSONS["config2"]), _json(ENV_JSONS["config3"]))
    expected = {f"dict_of_agents_configs.AdversarialMM.{f}" for f in MM_ABLATION_FLAGS}
    assert diff == expected, f"config2 vs config3 unexpected diffs: {diff ^ expected}"


def test_as_json_pins_the_closed_form_policy():
    mm = _json(ENV_JSONS["as"])["dict_of_agents_configs"]["AdversarialMM"]
    assert mm["action_space"] == "AvSt"
    assert mm["fixed_action_setting"] is True
    assert 0 <= mm["fixed_action"] <= 7, "fixed_action indexes the 8 gamma values"
    spoof = _json(ENV_JSONS["as"])["dict_of_agents_configs"]["Spoofing"]
    assert spoof["budget_per_episode"] > 0
