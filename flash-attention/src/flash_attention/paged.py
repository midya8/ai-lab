"""A paged KV cache (the vLLM idea) with block tables, prefix sharing and copy-on-write.

Keys/values live in a fixed pool of ``block_size``-token blocks. Each sequence owns a *block
table* - a list of physical block ids - so sequences can grow without a contiguous reservation,
freed blocks are reusable immediately, and forked sequences (beam search, shared system prompt)
share their prefix blocks by reference count. Attention runs directly over the blocks: each block
is one key tile of the online softmax, merged exactly with :func:`merge_partials`.
"""
from __future__ import annotations

import numpy as np

from . import quant
from .flash import attend_partial, finalize, merge_partials


class OutOfBlocks(RuntimeError):
    pass


class PagedKVCache:
    def __init__(self, num_blocks: int, block_size: int, n_kv_heads: int, head_dim: int,
                 dtype=np.float64):
        self.block_size, self.n_kv_heads, self.head_dim = block_size, n_kv_heads, head_dim
        self.compute_dtype = np.dtype(dtype).type
        self._init_storage(num_blocks, dtype)
        self.free_blocks = list(range(num_blocks - 1, -1, -1))
        self.refcount = np.zeros(num_blocks, dtype=int)
        self.tables: dict[int, list[int]] = {}
        self.lengths: dict[int, int] = {}
        self._next_id = 0
        self.copies = 0       # copy-on-write events, for tests and accounting

    # -- storage hooks (overridden by QuantPagedKVCache) --------------------------------------
    def _init_storage(self, num_blocks, dtype):
        self.k = np.zeros((num_blocks, self.block_size, self.n_kv_heads, self.head_dim), dtype=dtype)
        self.v = np.zeros_like(self.k)

    def _copy_block(self, src, dst):
        self.k[dst], self.v[dst] = self.k[src], self.v[src]

    def _store(self, b, slot, k, v):
        self.k[b, slot], self.v[b, slot] = k, v

    def _load(self, b, valid):
        """Keys and values of the first ``valid`` slots of block ``b``: (valid, n_kv_heads, d)."""
        return self.k[b, :valid], self.v[b, :valid]

    def nbytes(self) -> int:
        """Bytes of the whole pool (allocated up front, whether or not blocks are in use)."""
        return int(self.k.nbytes + self.v.nbytes)

    def bytes_per_token(self) -> float:
        return self.nbytes() / (len(self.refcount) * self.block_size)

    # -- allocation ---------------------------------------------------------------------------
    def _alloc(self) -> int:
        if not self.free_blocks:
            raise OutOfBlocks("KV cache pool exhausted")
        b = self.free_blocks.pop()
        self.refcount[b] = 1
        return b

    def _release(self, b: int) -> None:
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            self.free_blocks.append(b)

    def new_sequence(self) -> int:
        sid = self._next_id
        self._next_id += 1
        self.tables[sid], self.lengths[sid] = [], 0
        return sid

    def free(self, sid: int) -> None:
        for b in self.tables.pop(sid):
            self._release(b)
        del self.lengths[sid]

    def fork(self, sid: int) -> int:
        """New sequence sharing every block of ``sid`` (no data copied)."""
        child = self.new_sequence()
        self.tables[child] = list(self.tables[sid])
        self.lengths[child] = self.lengths[sid]
        for b in self.tables[child]:
            self.refcount[b] += 1
        return child

    # -- writes -------------------------------------------------------------------------------
    def append(self, sid: int, k, v) -> None:
        """Append ``T`` tokens; k, v have shape (T, n_kv_heads, head_dim)."""
        k, v = np.asarray(k), np.asarray(v)
        needed_new = -(-(self.lengths[sid] + len(k)) // self.block_size) - len(self.tables[sid])
        shared_tail = (self.lengths[sid] % self.block_size != 0
                       and self.refcount[self.tables[sid][-1]] > 1)
        if needed_new + int(shared_tail) > len(self.free_blocks):
            raise OutOfBlocks("KV cache pool exhausted")
        for t in range(len(k)):
            n = self.lengths[sid]
            slot = n % self.block_size
            if slot == 0:
                self.tables[sid].append(self._alloc())
            else:
                tail = self.tables[sid][-1]
                if self.refcount[tail] > 1:                      # copy-on-write
                    fresh = self._alloc()
                    self._copy_block(tail, fresh)
                    self._release(tail)
                    self.tables[sid][-1] = fresh
                    self.copies += 1
            b = self.tables[sid][-1]
            self._store(b, slot, k[t], v[t])
            self.lengths[sid] = n + 1

    # -- reads --------------------------------------------------------------------------------
    def gather(self, sid: int):
        """Contiguous (K, V), each (n_kv_heads, length, head_dim) - for tests and debugging."""
        n = self.lengths[sid]
        if n == 0:
            e = np.zeros((self.n_kv_heads, 0, self.head_dim), dtype=self.compute_dtype)
            return e, e.copy()
        bs = self.block_size
        loaded = [self._load(b, min(bs, n - i * bs)) for i, b in enumerate(self.tables[sid])]
        ks = np.concatenate([a for a, _ in loaded])
        vs = np.concatenate([c for _, c in loaded])
        return ks.transpose(1, 0, 2), vs.transpose(1, 0, 2)

    def attend(self, sid: int, q, *, causal=True, window=None, alibi_slopes=None, scale=None):
        """Attention of ``q`` (H, nq, d) - the *last* nq tokens of the sequence - over the paged
        cache, one physical block per key tile, never gathering the cache. Returns (out, lse)."""
        q = np.asarray(q)
        n, bs = self.lengths[sid], self.block_size
        off = n - q.shape[1]
        parts = []
        for bi, b in enumerate(self.tables[sid]):
            valid = min(bs, n - bi * bs)
            kb, vb = (a.transpose(1, 0, 2) for a in self._load(b, valid))
            parts.append(attend_partial(q, kb, vb, causal=causal, window=window,
                                        alibi_slopes=alibi_slopes, q_offset=off,
                                        k_start=bi * bs, block_q=max(q.shape[1], 1),
                                        block_k=bs, scale=scale, dtype=self.compute_dtype))
        if not parts:
            raise ValueError("empty sequence")
        return finalize(*merge_partials(parts))

    # -- accounting ---------------------------------------------------------------------------
    def blocks_in_use(self) -> int:
        return int((self.refcount > 0).sum())

    def utilisation(self) -> dict:
        """Token slots actually holding data vs allocated, counting a shared block once."""
        used = self.blocks_in_use()
        return {"blocks_in_use": used, "logical_tokens": sum(self.lengths.values()),
                "physical_slots": used * self.block_size,
                "internal_waste_slots": used * self.block_size - self._physical_tokens()}

    def _physical_tokens(self) -> int:
        """Distinct (block, slot) pairs holding live data."""
        seen = {}
        for s, tab in self.tables.items():
            for i, b in enumerate(tab):
                seen[b] = max(seen.get(b, 0), min(self.block_size, self.lengths[s] - i * self.block_size))
        return sum(seen.values())


class QuantPagedKVCache(PagedKVCache):
    """A paged cache whose blocks hold bit-packed low-bit keys and values (see :mod:`quant`).

    Tokens are quantised once, when appended; attention dequantises one block at a time, so the
    cache is never expanded to full precision. ``k_bits`` and ``v_bits`` may differ. Copy-on-write,
    forking and refcounts are inherited unchanged - they move packed bytes and scales.
    """

    def __init__(self, num_blocks, block_size, n_kv_heads, head_dim, *, k_bits=8, v_bits=8,
                 group_size=None, scale_dtype=np.float16, dtype=np.float64):
        self.k_bits, self.v_bits, self.group_size = k_bits, v_bits, group_size
        self.scale_dtype = scale_dtype
        quant.qmax(k_bits), quant.qmax(v_bits)
        super().__init__(num_blocks, block_size, n_kv_heads, head_dim, dtype=dtype)

    def _init_storage(self, num_blocks, dtype):
        g, ng = quant._groups(self.head_dim, self.group_size)
        sh = (num_blocks, self.block_size, self.n_kv_heads)
        for name, bits in (("k", self.k_bits), ("v", self.v_bits)):
            if self.head_dim % (8 // bits):
                raise ValueError(f"head_dim must be a multiple of {8 // bits} for {bits}-bit")
            setattr(self, name + "q", np.zeros(sh + (self.head_dim * bits // 8,), dtype=np.uint8))
            setattr(self, name + "s", np.ones(sh + (ng,), dtype=self.scale_dtype))

    def _copy_block(self, src, dst):
        for n in ("kq", "ks", "vq", "vs"):
            a = getattr(self, n)
            a[dst] = a[src]

    def _store(self, b, slot, k, v):
        for n, x, bits in (("k", k, self.k_bits), ("v", v, self.v_bits)):
            codes, sc = quant.quantize(x, bits, self.group_size, self.scale_dtype)
            getattr(self, n + "q")[b, slot] = quant.pack(codes, bits)
            getattr(self, n + "s")[b, slot] = sc

    def _load(self, b, valid):
        out = []
        for n, bits in (("k", self.k_bits), ("v", self.v_bits)):
            codes = quant.unpack(getattr(self, n + "q")[b, :valid], bits, self.head_dim)
            out.append(quant.dequantize(codes, getattr(self, n + "s")[b, :valid], self.group_size,
                                        self.compute_dtype))
        return out[0], out[1]

    def nbytes(self) -> int:
        return int(self.kq.nbytes + self.ks.nbytes + self.vq.nbytes + self.vs.nbytes)
