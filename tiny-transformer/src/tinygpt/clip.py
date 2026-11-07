"""Global-norm gradient clipping (as in torch.nn.utils.clip_grad_norm_)."""
from __future__ import annotations

import numpy as np


def global_norm(grads):
    """L2 norm over all gradient tensors, computed in float64."""
    return float(np.sqrt(sum(float(np.sum(np.square(g, dtype=np.float64))) for g in grads.values())))


def clip_grad_norm(grads, max_norm, eps=1e-6):
    """Return (clipped_grads, pre_clip_norm). Scales all grads by max_norm/(norm+eps) if norm > max_norm."""
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    norm = global_norm(grads)
    if norm <= max_norm:
        return grads, norm
    scale = max_norm / (norm + eps)
    return {k: (g * scale).astype(g.dtype) for k, g in grads.items()}, norm
