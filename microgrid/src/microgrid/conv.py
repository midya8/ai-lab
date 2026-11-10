"""conv2d and max_pool2d for microgrid Tensors (NCHW layout, valid padding + optional zero pad)."""
import numpy as np
from .tensor import Tensor


def _windows(x, kh, kw, stride):
    n, c, h, w = x.shape
    oh, ow = (h - kh) // stride + 1, (w - kw) // stride + 1
    s = x.strides
    return np.lib.stride_tricks.as_strided(
        x, (n, c, oh, ow, kh, kw), (s[0], s[1], s[2] * stride, s[3] * stride, s[2], s[3]), writeable=False), oh, ow


def conv2d(x, w, b=None, stride=1, pad=0):
    """x: (N,C,H,W), w: (F,C,kh,kw), b: (F,) or None -> (N,F,oh,ow)."""
    x = Tensor._wrap(x); w = Tensor._wrap(w)
    parents = (x, w) + ((b,) if b is not None else ())
    xp = np.pad(x.data, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else x.data
    xp = np.ascontiguousarray(xp)
    f, c, kh, kw = w.shape
    win, oh, ow = _windows(xp, kh, kw, stride)
    y = np.einsum("ncijkl,fckl->nfij", win, w.data)
    if b is not None:
        y = y + b.data[None, :, None, None]
    out = Tensor(y, parents)

    def bw():
        g = out.grad
        w.grad += np.einsum("nfij,ncijkl->fckl", g, win)
        if b is not None:
            b.grad += g.sum(axis=(0, 2, 3))
        dxp = np.zeros_like(xp)
        for k in range(kh):
            for l in range(kw):
                dxp[:, :, k:k + stride * oh:stride, l:l + stride * ow:stride] += \
                    np.einsum("nfij,fc->ncij", g, w.data[:, :, k, l])
        x.grad += dxp[:, :, pad:dxp.shape[2] - pad, pad:dxp.shape[3] - pad] if pad else dxp
    out._backward = bw
    return out


def max_pool2d(x, k=2, stride=None):
    """Non-overlapping-by-default max pooling; gradient goes to the first max in each window."""
    x = Tensor._wrap(x)
    stride = stride or k
    xc = np.ascontiguousarray(x.data)
    win, oh, ow = _windows(xc, k, k, stride)
    flat = win.reshape(*win.shape[:4], k * k)
    arg = flat.argmax(-1)
    out = Tensor(flat.max(-1), (x,))

    def bw():
        g = out.grad
        n, c = g.shape[:2]
        ni, ci, ii, jj = np.indices((n, c, oh, ow))
        r = ii * stride + arg // k
        q = jj * stride + arg % k
        np.add.at(x.grad, (ni, ci, r, q), g)
    out._backward = bw
    return out
