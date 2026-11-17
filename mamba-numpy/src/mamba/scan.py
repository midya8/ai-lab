"""Linear recurrences h_t = a_t * h_{t-1} + b_t, the engine of every state-space model.

Two exact implementations:

* ``scan_sequential``: the obvious loop over time (O(T) steps, O(1) extra memory).
* ``scan_parallel``: a Hillis-Steele associative scan. The pair (a, b) composes as
  (a1, b1) o (a2, b2) = (a1*a2, a2*b1 + b2), which is associative, so the whole
  sequence resolves in ceil(log2 T) vectorised passes (O(T log T) work).

The backward pass of a recurrence is another recurrence run in reverse time,
``reverse_scan``, so autodiff through an SSM costs one more scan, not a loop of Jacobians.
All functions scan along axis 1 and broadcast over every other axis.
"""
from __future__ import annotations

import numpy as np


def scan_sequential(a, b, h0=None):
    """h_t = a_t * h_{t-1} + b_t along axis 1. h_{-1} = h0 (zeros if None)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        a = np.broadcast_to(a, b.shape)
    T = b.shape[1]
    h = np.empty_like(b)
    prev = np.zeros_like(b[:, 0]) if h0 is None else h0
    for t in range(T):
        prev = a[:, t] * prev + b[:, t]
        h[:, t] = prev
    return h


def scan_parallel(a, b, h0=None):
    """Same result as ``scan_sequential`` via a log-depth associative scan."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        a = np.broadcast_to(a, b.shape)
    A = np.array(a, copy=True)
    Bv = np.array(b, copy=True)
    if h0 is not None:
        Bv[:, 0] = Bv[:, 0] + A[:, 0] * h0  # fold the initial state into step 0
    T = b.shape[1]
    shift = 1
    while shift < T:
        newA = A.copy()
        newB = Bv.copy()
        newA[:, shift:] = A[:, shift:] * A[:, :-shift]
        newB[:, shift:] = A[:, shift:] * Bv[:, :-shift] + Bv[:, shift:]
        A, Bv = newA, newB
        shift *= 2
    return Bv


def scan(a, b, h0=None, method="sequential"):
    if method == "sequential":
        return scan_sequential(a, b, h0)
    if method == "parallel":
        return scan_parallel(a, b, h0)
    raise ValueError(f"unknown scan method {method!r}")


def reverse_scan(a, dh, method="sequential"):
    """Gradient recurrence g_t = dh_t + a_{t+1} * g_{t+1} (g_{T-1} = dh_{T-1}).

    ``g_t`` is the total derivative of the loss with respect to ``h_t`` when
    ``dh_t`` is the direct derivative and ``h_{t+1} = a_{t+1} h_t + b_{t+1}``.
    """
    a = np.asarray(a)
    dh = np.asarray(dh)
    if a.shape != dh.shape:
        a = np.broadcast_to(a, dh.shape)
    # in reversed time the coefficient on step s is a_{T-s}; step 0 has none
    c = np.concatenate([np.zeros_like(a[:, :1]), a[:, :0:-1]], axis=1)
    r = scan(c, dh[:, ::-1], None, method)
    return r[:, ::-1]
