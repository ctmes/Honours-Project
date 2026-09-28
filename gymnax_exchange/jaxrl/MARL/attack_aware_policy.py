"""
Attack-Aware Policy Networks for adversarial co-training.

AttackAwarePolicyNet  — MM network: SharedEncoder → PolicyHead + DetectionHead
AdversaryNet          — Spoofing adversary: MLP → continuous Gaussian action

Both return a 4-tuple (hidden, pi, value, detection_prob) so the adversarial
training loop can use a uniform interface. For the adversary, detection_prob is
always jnp.zeros(()) since it has no detection head.

The `hidden` argument is accepted and returned as a pass-through (compatible with
ScannedRNN interface) but never updated — these are stateless MLPs.
"""

import functools

import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Dict


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    hidden_dims: Sequence[int] = (256, 256, 128)

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=orthogonal(jnp.sqrt(2)),
                         bias_init=constant(0.0))(x)
            x = nn.relu(x)
        return x  # shape: (..., 128)


class PolicyHead(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0),
                         bias_init=constant(0.0))(x)
        pi = distrax.Categorical(logits=logits)
        return pi, jnp.squeeze(value, axis=-1)


class DetectionHead(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        logit = nn.Dense(1, kernel_init=orthogonal(0.01),
                         bias_init=constant(0.0))(x)
        return jnp.squeeze(nn.sigmoid(logit), axis=-1)  # scalar in [0, 1]


# ---------------------------------------------------------------------------
# Market Maker: AttackAwarePolicyNet
# ---------------------------------------------------------------------------

class AttackAwarePolicyNet(nn.Module):
    """
    Stateless MLP for the adversarially-trained market maker.

    Returns a 4-tuple (hidden, pi, value, detection_prob) to match
    the adversarial training loop interface.  `hidden` is passed through
    unchanged — it exists only for API compatibility with ScannedRNN.

    Observation expected: 45-dim adversarial_lob obs (perturbed L2 + extras).
    Action space: discrete (n discrete actions produced by PolicyHead).
    """
    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        encoded = SharedEncoder()(obs)
        pi, value = PolicyHead(self.action_dim)(encoded)
        detection_prob = DetectionHead()(encoded)
        return hidden, pi, value, detection_prob

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size))


class _ScannedGRU(nn.Module):
    """GRU scanned over the leading (time) axis, reset to zeros where done."""

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        carry = jnp.where(resets[:, jnp.newaxis], jnp.zeros_like(carry), carry)
        new_carry, y = nn.GRUCell(features=carry.shape[-1])(carry, ins)
        return new_carry, y


class AttackAwareRecurrentNet(nn.Module):
    """
    Recurrent variant of AttackAwarePolicyNet (WS10c, MM_RECURRENT=true).

    SharedEncoder -> GRU (memory over the episode) -> PolicyHead. The
    DetectionHead stays on the feed-forward encoder output, so the encoder --
    the object of the RC4 rank analysis -- is identical in structure to the
    memoryless network and only the policy path gains memory. Motivation:
    history-conditioned policies are more robust to observation attacks
    (Zhang et al. 2021, ATLA); the v1-v4 market maker is memoryless.

    Carry: (batch, GRU_HIDDEN_DIM), zeroed where `dones` is true.
    """
    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        encoded = SharedEncoder()(obs)
        hidden, memory = _ScannedGRU()(hidden, (encoded, dones))
        pi, value = PolicyHead(self.action_dim)(memory)
        detection_prob = DetectionHead()(encoded)
        return hidden, pi, value, detection_prob

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size))


def make_mm_network(action_dim: int, config: Dict) -> nn.Module:
    """MM network for this config: recurrent iff config["MM_RECURRENT"] is true.

    Used by both the training loop and the eval harness so a recurrent
    checkpoint is always restored into a recurrent template.
    """
    if bool((config or {}).get("MM_RECURRENT", False)):
        return AttackAwareRecurrentNet(action_dim=action_dim, config=config)
    return AttackAwarePolicyNet(action_dim=action_dim, config=config)


# ---------------------------------------------------------------------------
# Adversary: AdversaryNet (continuous Gaussian)
# ---------------------------------------------------------------------------

class AdversaryNet(nn.Module):
    """
    Continuous-action adversary network.

    Outputs a 10-dim action via MultivariateNormalDiag with learned log_std.
    Mean is passed through sigmoid so it lives in [0, 1]; the env scales by
    the episode budget.

    Returns the same 4-tuple (hidden, pi, value, detection_prob) for a uniform
    interface with AttackAwarePolicyNet.  detection_prob is always 0.0.
    """
    action_dim: int = 10
    config: Dict = None

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        h = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)),
                     bias_init=constant(0.0))(obs)
        h = nn.relu(h)
        h = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)),
                     bias_init=constant(0.0))(h)
        h = nn.relu(h)

        mean = nn.sigmoid(
            nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                     bias_init=constant(0.0))(h)
        )
        log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.action_dim,),
        )
        pi = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))

        value = nn.Dense(1, kernel_init=orthogonal(1.0),
                         bias_init=constant(0.0))(h)
        value = jnp.squeeze(value, axis=-1)

        detection_prob = jnp.zeros(obs.shape[:-1])  # zeros, matching batch shape
        return hidden, pi, value, detection_prob

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size))
