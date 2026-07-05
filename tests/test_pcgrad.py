"""
Unit tests for PCGrad (gradient surgery).

Four tests covering the core mathematical properties:
1. Anti-parallel gradients → projected result is orthogonal to g_detect
2. Orthogonal gradients    → g_policy is returned unchanged
3. Aligned gradients       → g_policy is returned unchanged
4. jax.jit compatibility   → project_gradient compiles without error
"""

import jax
import jax.flatten_util  # explicit import — newer JAX no longer exposes this implicitly
import jax.numpy as jnp
import pytest
from gymnax_exchange.jaxrl.MARL.pcgrad import project_gradient, pcgrad_merge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_flat_pytree(values):
    """Wrap a list of scalars as a pytree dict for testing."""
    return {"a": jnp.array(values[:2]), "b": jnp.array(values[2:])}


# ---------------------------------------------------------------------------
# Test 1: Anti-parallel gradients — projected grad ⊥ g_other
# ---------------------------------------------------------------------------

def test_antiparallel_projection():
    """
    When g_main and g_other point in exactly opposite directions,
    after projection g_main should be orthogonal to g_other (dot ≈ 0).
    """
    g_main  = make_flat_pytree([1.0, 0.0, 0.0, 0.0])
    g_other = make_flat_pytree([-1.0, 0.0, 0.0, 0.0])

    projected = project_gradient(g_main, g_other)

    flat_proj, _ = jax.flatten_util.ravel_pytree(projected)
    flat_other, _ = jax.flatten_util.ravel_pytree(g_other)
    dot = jnp.dot(flat_proj, flat_other)
    assert float(jnp.abs(dot)) < 1e-5, f"Expected near-zero dot product, got {dot}"


# ---------------------------------------------------------------------------
# Test 2: Orthogonal gradients — g_main unchanged
# ---------------------------------------------------------------------------

def test_orthogonal_unchanged():
    """
    When g_main and g_other are orthogonal (dot == 0), g_main should be
    returned exactly as-is (no projection applied).
    """
    g_main  = make_flat_pytree([1.0, 0.0, 0.0, 0.0])
    g_other = make_flat_pytree([0.0, 1.0, 0.0, 0.0])

    projected = project_gradient(g_main, g_other)

    flat_proj, _ = jax.flatten_util.ravel_pytree(projected)
    flat_main, _ = jax.flatten_util.ravel_pytree(g_main)
    assert jnp.allclose(flat_proj, flat_main, atol=1e-6), \
        f"Orthogonal gradients should be unchanged; got {flat_proj} vs {flat_main}"


# ---------------------------------------------------------------------------
# Test 3: Aligned gradients — g_main unchanged
# ---------------------------------------------------------------------------

def test_aligned_unchanged():
    """
    When g_main and g_other point in the same direction (dot > 0), no
    projection should be applied.
    """
    g_main  = make_flat_pytree([2.0, 3.0, 0.0, 0.0])
    g_other = make_flat_pytree([1.0, 1.5, 0.0, 0.0])   # same direction, scaled

    projected = project_gradient(g_main, g_other)

    flat_proj, _ = jax.flatten_util.ravel_pytree(projected)
    flat_main, _ = jax.flatten_util.ravel_pytree(g_main)
    assert jnp.allclose(flat_proj, flat_main, atol=1e-6), \
        f"Aligned gradients should be unchanged; got {flat_proj} vs {flat_main}"


# ---------------------------------------------------------------------------
# Test 4: jax.jit compatibility
# ---------------------------------------------------------------------------

def test_jit_compatible():
    """project_gradient should compile and run under jax.jit."""
    g_main  = make_flat_pytree([1.0, -1.0, 0.5, -0.5])
    g_other = make_flat_pytree([-1.0, 1.0, -0.5, 0.5])

    jitted_project = jax.jit(project_gradient)
    result = jitted_project(g_main, g_other)

    flat_result, _ = jax.flatten_util.ravel_pytree(result)
    assert flat_result.shape == (4,), f"Unexpected output shape: {flat_result.shape}"


# ---------------------------------------------------------------------------
# Test 5: pcgrad_merge leaves non-encoder keys intact
# ---------------------------------------------------------------------------

def test_pcgrad_merge_structure():
    """
    pcgrad_merge on trees shaped like real jax.grad output — BOTH trees contain
    every submodule key, with zero leaves where that loss does not touch the
    module (jax.grad guarantees this; pcgrad_merge relies on it and rejects
    asymmetric key sets).

    Checks:
    - Policy head grad preserved exactly (policy + 0)
    - Detection head grad preserved exactly (0 + detect)
    - Exactly anti-parallel encoder grads mutually project to zero
    """
    grads_policy = {
        "params": {
            "SharedEncoder_0": {"kernel": jnp.array([[1.0, 0.0], [0.0, 1.0]])},
            "PolicyHead_0":    {"kernel": jnp.array([[2.0, 0.0]])},
            "DetectionHead_0": {"kernel": jnp.zeros((1, 2))},   # BCE-only module: zero under PPO loss
        }
    }
    grads_detect = {
        "params": {
            "SharedEncoder_0": {"kernel": jnp.array([[-1.0, 0.0], [0.0, -1.0]])},  # anti-parallel
            "PolicyHead_0":    {"kernel": jnp.zeros((1, 2))},   # PPO-only module: zero under BCE loss
            "DetectionHead_0": {"kernel": jnp.array([[3.0, 0.0]])},
        }
    }

    merged = pcgrad_merge(grads_policy, grads_detect)

    # Policy head preserved exactly (policy grad + zero detect grad)
    assert jnp.allclose(
        merged["params"]["PolicyHead_0"]["kernel"],
        grads_policy["params"]["PolicyHead_0"]["kernel"],
    )
    # Detection head preserved exactly (zero policy grad + detect grad)
    assert jnp.allclose(
        merged["params"]["DetectionHead_0"]["kernel"],
        grads_detect["params"]["DetectionHead_0"]["kernel"],
    )
    # Exactly anti-parallel encoder gradients each project to zero, so their sum is zero
    assert jnp.allclose(merged["params"]["SharedEncoder_0"]["kernel"], 0.0, atol=1e-6)


def test_pcgrad_merge_rejects_asymmetric_trees():
    """Trees with mismatched key sets are NOT valid inputs (real jax.grad never
    produces them); pcgrad_merge should fail loudly rather than merge silently."""
    grads_policy = {"params": {
        "SharedEncoder_0": {"kernel": jnp.ones((2, 2))},
        "PolicyHead_0": {"kernel": jnp.ones((1, 2))},
    }}
    grads_detect = {"params": {
        "SharedEncoder_0": {"kernel": jnp.ones((2, 2))},
        "DetectionHead_0": {"kernel": jnp.ones((1, 2))},
    }}
    with pytest.raises(KeyError):
        pcgrad_merge(grads_policy, grads_detect)
