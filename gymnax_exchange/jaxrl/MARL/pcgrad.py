"""
PCGrad (Gradient Surgery) for the attack-aware market maker.

Reference: Yu et al. (2020), "Gradient Surgery for Multi-Task Learning".

Only the SharedEncoder params are modified; PolicyHead and DetectionHead
gradients are passed through unchanged.  This preserves task-specific
gradient information while resolving conflicts in the shared representation.

Usage (inside the MM update step):
    grads_policy = jax.grad(loss_policy)(params)
    grads_detect = jax.grad(loss_detect)(params)
    merged = pcgrad_merge(grads_policy, grads_detect)
    train_state = train_state.apply_gradients(grads=merged)
"""

import jax
import jax.numpy as jnp
from typing import Any

PyTree = Any


def project_gradient(g_main: PyTree, g_other: PyTree) -> PyTree:
    """
    Project g_main onto the normal plane of g_other when they conflict.

    If dot(g_main, g_other) < 0, subtracts the component of g_main along
    g_other.  Otherwise returns g_main unchanged.

    Both pytrees must have the same structure (same leaves, same shapes).
    This function is jit-compatible.
    """
    flat_main, unravel = jax.flatten_util.ravel_pytree(g_main)
    flat_other, _ = jax.flatten_util.ravel_pytree(g_other)

    dot = jnp.dot(flat_main, flat_other)
    norm_sq = jnp.dot(flat_other, flat_other) + 1e-12

    # Subtract the projection only when the gradients conflict (dot < 0)
    projected_flat = flat_main - (dot / norm_sq) * flat_other
    result_flat = jnp.where(dot < 0.0, projected_flat, flat_main)

    return unravel(result_flat)


def pcgrad_merge(
    grads_policy: PyTree,
    grads_detect: PyTree,
    encoder_key: str = "SharedEncoder_0",
) -> PyTree:
    """
    Merge policy and detection gradients using PCGrad on the shared encoder.

    - SharedEncoder params: apply mutual PCGrad projection, then sum.
    - PolicyHead / remaining params: use grads_policy only.
    - DetectionHead params: use grads_detect only.

    Args:
        grads_policy: gradient pytree from the PPO policy loss.
        grads_detect: gradient pytree from the BCE detection loss.
        encoder_key: Flax param key for the shared encoder submodule.
                     Flax names submodules as ClassName_N; with a single
                     SharedEncoder the key is "SharedEncoder_0".

    Returns:
        Merged gradient pytree with the same structure as grads_policy.
    """
    params_dict = grads_policy["params"]
    det_dict = grads_detect["params"]

    # Project policy encoder grad away from detection encoder grad and vice-versa
    enc_p = params_dict[encoder_key]
    enc_d = det_dict[encoder_key]

    enc_p_proj = project_gradient(enc_p, enc_d)
    enc_d_proj = project_gradient(enc_d, enc_p)

    # Sum the two projected encoder gradients
    enc_merged = jax.tree.map(lambda a, b: a + b, enc_p_proj, enc_d_proj)

    # Rebuild merged params dict:
    #   - encoder: projected + summed
    #   - everything from grads_policy that is NOT the detection head
    #   - detection head from grads_detect
    merged_params = dict(params_dict)
    merged_params[encoder_key] = enc_merged

    # Copy over detection head grads (keys present in det_dict but not policy)
    for key, val in det_dict.items():
        if key not in params_dict:
            merged_params[key] = val

    return {"params": merged_params}
