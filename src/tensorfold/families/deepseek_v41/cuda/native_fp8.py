"""DeepSeek-V4.1's own FP8 projections (e4m3, one e8m0 power of two per 32x32 block) on the MXFP8 kernels.

cebu-profiler packs keep every projection but the routed experts in the checkpoint's native FP8. ``Mx8Linear`` takes
e4m3 [N, K] with one e8m0 exponent per 32 inputs of each row; a block's exponent repeated over the block's 32 rows is
exactly that, so the conversion is bit-identical (nothing is requantized). The head and the embedding are bf16.

The adapters expose what the engine's single-launch paths read from an ``Exl3Linear``: ``k`` (inputs), ``n``
(outputs), ``words`` (the bytes an L2 prefetch range covers), ``suh = None`` (no input rotation: grouped paths must
not take these layers), and a call ``layer(x, out_dtype=...)`` for decode rows plus ``prefill(x)`` for prompt rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorfold.cuda.nvfp4.linear import Mx8Linear

BF16 = torch.bfloat16
BLOCK = 32


_PLAIN_PATHS_SET = False


def use_plain_paths() -> None:
    """A pack with native FP8 projections: the grouped launches and the fused rotations (rot_q, rot_attn, rot_wob,
    hc_rot) share one EXL3 input rotation across several Exl3Linears, which these layers do not take. Turn them off once
    so every projection goes through model.mm (milestone 1 of docs/cebu-native-fp8-port.md); all-EXL3 packs never
    reach this and keep every fusion."""

    global _PLAIN_PATHS_SET
    if _PLAIN_PATHS_SET:
        return
    from . import kernels as K
    from . import model as M

    M.GROUPED = False
    for name in ("rot_q", "rot_attn", "rot_wob", "hc_rot"):
        if name in K.SMALL_SWITCHES:
            K.set_switch(name, False)
    _PLAIN_PATHS_SET = True
    print("[tensorfold] native FP8 backbone: grouped launches and fused rotations off (plain mm paths)", flush=True)


def is_native(sh, prefix: str) -> bool:
    """A projection stored as native FP8 (``<prefix>.weight`` e4m3 + ``<prefix>.scale``) rather than an EXL3 trellis."""

    return f"{prefix}.trellis" not in sh and f"{prefix}.weight" in sh and f"{prefix}.scale" in sh


def _slice(w: torch.Tensor, s: torch.Tensor, cols, rows) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's part: ``cols`` = output columns [lo, hi) (weight rows), ``rows`` = input rows [lo, hi) (weight
    columns), both on 32-aligned boundaries so the block scales slice with them."""

    if cols is not None:
        lo, hi = cols
        assert lo % BLOCK == 0 and hi % BLOCK == 0, f"output slice {cols} not 32-aligned"
        w, s = w[lo:hi], s[lo // BLOCK:hi // BLOCK]
    if rows is not None:
        lo, hi = rows
        assert lo % BLOCK == 0 and hi % BLOCK == 0, f"input slice {rows} not 32-aligned"
        w, s = w[:, lo:hi], s[:, lo // BLOCK:hi // BLOCK]
    return w.contiguous(), s.contiguous()


import os

# cebu 2026-10-07: K slices for the decode GEMV by shape (still a function of the shape only, so a row's bits never
# depend on the row count). Measured on GB10 with 48 distinct copies a pass (no L2 reuse), this rank's v5 shapes:
# qmm.split_k's 8-way cluster split was the slowest choice for every small shape (wq_a 38.9 us vs 31-32, wkv 19.0 vs
# 15.1, wo_a slice 27.2 vs 20.7, shared w1 36.3 vs 28.7); 16 slices through the partial buffer won; outputs of >= 4096
# columns ran best unsplit (wq_b 93.9 -> 92.8, wo_b 98.3 -> 94.3, shared w2 29.6 -> 27.8). TF_DS_MX8_SK=0: qmm.split_k.
MX8_SK = os.environ.get("TF_DS_MX8_SK", "1") != "0"


def _split(n: int, k: int) -> int:
    groups = k // 64
    if -(-n // 64) >= 64:
        return 1
    for s in (16, 4, 2):
        if groups % s == 0 and groups // s >= 2:
            return s
    return 1


def _gemv(mx: Mx8Linear, x: torch.Tensor, f32: bool = False) -> torch.Tensor:
    from tensorfold.cuda.kernels import qmm
    from tensorfold.cuda.nvfp4 import linear as nl

    if not MX8_SK and not f32:
        return mx(x)
    m = x.shape[0]
    sk = _split(mx.n, mx.k)
    y = torch.empty((m, mx.n), dtype=torch.float32 if f32 else BF16, device=x.device)
    part = torch.empty((sk, m, mx.n), dtype=torch.float32, device=x.device) if sk > 8 else None
    nl._ext().qmmf(x, mx.w8, mx.bs, 1.0, y, part, nl.MXFP8, mx.n, sk, mx.npad, qmm.bucket(m), f32)
    return y


@dataclass
class NativeFp8Linear:
    mx: Mx8Linear
    suh = None                     # no EXL3 input rotation
    f32_out = False                # fp32 asked for: written by the kernel in fp32 (only the drafter's head copy)

    @classmethod
    def load(cls, sh, prefix: str, cols=None, rows=None) -> "NativeFp8Linear":
        use_plain_paths()
        w = sh.get(f"{prefix}.weight")
        s = sh.get(f"{prefix}.scale")
        w, s = _slice(w, s, cols, rows)
        n, k = w.shape
        assert s.shape == (n // BLOCK, k // BLOCK), f"{prefix}: scale {tuple(s.shape)} for weight {tuple(w.shape)}"
        s_rows = s.view(torch.uint8).repeat_interleave(BLOCK, dim=0)          # [N, K/32]: MXFP8's per-row exponents
        return cls(Mx8Linear.from_checkpoint(w.cuda().view(torch.float8_e4m3fn), s_rows.cuda()))

    @property
    def k(self) -> int:
        return self.mx.k

    @property
    def n(self) -> int:
        return self.mx.n

    @property
    def words(self) -> torch.Tensor:
        return self.mx.w8

    def nbytes(self) -> int:
        return self.mx.nbytes()

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None, out_dtype=BF16) -> torch.Tensor:
        y = _gemv(self.mx, x.contiguous().to(BF16), f32=self.f32_out and out_dtype == torch.float32 and out is None)
        if out is not None:
            out.copy_(y)
            return out
        return y if out_dtype == BF16 else y.to(out_dtype)

    def prefill(self, x: torch.Tensor, out_dtype=BF16) -> torch.Tensor:
        y = self.mx.prefill(x.contiguous().to(BF16))
        return y if out_dtype == BF16 else y.to(out_dtype)


@dataclass
class Bf16Linear:
    """A bf16 projection (the head): y = x @ W.T; ``w`` [N, K] on the GPU."""

    w: torch.Tensor
    suh = None

    @classmethod
    def load(cls, sh, prefix: str, cols=None, rows=None) -> "Bf16Linear":
        w = sh.get(f"{prefix}.weight")
        if cols is not None:
            w = w[cols[0]:cols[1]]
        if rows is not None:
            w = w[:, rows[0]:rows[1]]
        return cls(w.contiguous().to(BF16).cuda())

    @property
    def k(self) -> int:
        return self.w.shape[1]

    @property
    def n(self) -> int:
        return self.w.shape[0]

    @property
    def words(self) -> torch.Tensor:
        return self.w

    @property
    def device(self):
        return self.w.device

    def nbytes(self) -> int:
        return self.w.numel() * 2

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None, out_dtype=BF16) -> torch.Tensor:
        x = x.to(BF16)
        y = rowinv_linear(x, self.w) if ROWINV else torch.nn.functional.linear(x, self.w)
        if out is not None:
            out.copy_(y)
            return out
        return y if out_dtype == BF16 else y.to(out_dtype)

    def prefill(self, x: torch.Tensor, out_dtype=BF16) -> torch.Tensor:
        y = torch.nn.functional.linear(x.to(BF16), self.w)                # prompt rows: cuBLAS
        return y if out_dtype == BF16 else y.to(out_dtype)


# cebu 2026-10-08: decode rows of the bf16 projections (the head, the compressor's wkv / wgate, the indexer's wk) on a
# row-invariant GEMM. cuBLAS picks its kernel by the row count (gemv at 1 row, a CUTLASS GEMM at 2-16), so a row's
# logits changed with the verify window's size, and the greedy reply with the drafts: the feature-edit prompt's reply
# took 943 tokens serial and with DSpark, 974 with 16-row suffix windows. Here every row is 16-row tiles x the same K
# order (tl.dot, fp32 accumulation, a bf16 store as cuBLAS's), whatever the row count. TF_DS_ROWINV_BF16=0: cuBLAS.
ROWINV = os.environ.get("TF_DS_ROWINV_BF16", "1") != "0"

import triton
import triton.language as tl


@triton.jit
def _rowinv_mm(X, W, Y, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pn = tl.program_id(0)
    pm = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + rm[:, None] * K + (k0 + rk)[None, :], mask=rm[:, None] < M, other=0.0)
        w = tl.load(W + rn[:, None] * K + (k0 + rk)[None, :], mask=rn[:, None] < N, other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(Y + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def rowinv_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [M, K] bf16 @ w [N, K].T -> bf16 [M, N], each row's bits independent of M and of the other rows."""

    m, k = x.shape
    n = w.shape[0]
    assert k % 64 == 0, f"row-invariant GEMM wants K in 64s, got {k}"
    x = x.contiguous()
    y = torch.empty((m, n), dtype=BF16, device=x.device)
    # 2026-10-08 sweep on the head's half (64640 x 5120, L2 flushed): BN 64 x BK 256, 3 stages, 4 warps 2.87 / 2.88 ms
    # at 1 / 6 rows vs cuBLAS 2.84 / 2.96; BK is fixed by K alone, so the row's K order is too
    BM, BN, BK = 16, 64, (256 if k % 256 == 0 else 64)
    _rowinv_mm[(triton.cdiv(n, BN), triton.cdiv(m, BM))](x, w, y, m, n, k, BM=BM, BN=BN, BK=BK, num_warps=4,
                                                         num_stages=3)
    return y


def mxfp8_of(w: torch.Tensor) -> NativeFp8Linear:
    """A bf16 [N, K] matrix as MXFP8: e4m3 codes with one e8m0 power of two per 32 inputs of each row (the smallest
    2^e with amax / 2^e <= 448, codes rounded to nearest). Lossy: only for the drafter's head (``draft_head``)."""

    n, k = w.shape
    rows = 4096                                           # 128-row multiple: a slice's tiles are whole tiles
    assert n % 128 == 0, f"MXFP8 copy wants whole 128-row tiles, got {n} rows"
    w8 = torch.empty((n * k,), dtype=torch.uint8, device=w.device)
    bs = torch.empty((n // 64, k // 64, 64, 2), dtype=torch.uint8, device=w.device)
    # slice by slice: the fragment reorder gathers through int64 indices the size of its input (~5 GB for the whole
    # head), and its layout leads with the 64-row tile, so slices land in order
    for lo in range(0, n, rows):
        hi = min(n, lo + rows)
        wb = w[lo:hi].float().view(hi - lo, k // BLOCK, BLOCK)
        e = torch.ceil(torch.log2(wb.abs().amax(-1).clamp(min=2.0 ** -126) / 448.0)).clamp(-127, 127)
        codes = (wb / torch.exp2(e)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(hi - lo, k)
        part = Mx8Linear.from_checkpoint(codes, (e + 127).to(torch.uint8))
        w8[lo * k:hi * k] = part.w8
        bs[lo // 64:hi // 64] = part.bs
        del wb, e, codes, part
    return NativeFp8Linear(Mx8Linear(w8, bs, n, k, n))


# cebu 2026-10-08: TF_DS_DRAFT_HEAD8=1: the DSpark drafter scores its vocabulary half with an MXFP8 copy of the head
# (half the bytes of the bf16 head it reads every round, 331 MB a rank more). Only the drafts can change (a near-tie in
# the drafter's argmax); the target's head stays bf16 and verification is unchanged, so every emitted token is the
# target's as before.
DRAFT_HEAD8 = os.environ.get("TF_DS_DRAFT_HEAD8", "0") == "1"


def draft_head(w):
    """The head the drafter reads: ``w.head``, or its MXFP8 copy under TF_DS_DRAFT_HEAD8 (made on first use, outside
    any graph capture: the drafter runs its body once before capturing)."""

    if not DRAFT_HEAD8 or not isinstance(w.head, Bf16Linear):
        return w.head
    h8 = getattr(w, "_draft_head8", None)
    if h8 is None:
        h8 = w._draft_head8 = mxfp8_of(w.head.w)
        h8.f32_out = True
        torch.cuda.empty_cache()
        print(f"[tensorfold] drafter head: MXFP8 copy {h8.nbytes() / 2**20:.0f} MiB (target head stays bf16)", flush=True)
    return h8


def native_mm(layer, x: torch.Tensor, out_dtype=BF16):
    """``model.mm`` for native layers: decode rows on the GEMV, prompt rows on the prompt GEMM; None for EXL3."""

    if not isinstance(layer, (NativeFp8Linear, Bf16Linear)):
        return None
    return layer(x, out_dtype=out_dtype) if x.shape[0] <= 128 else layer.prefill(x, out_dtype=out_dtype)


@dataclass
class NativeSharedMLP:
    """The shared expert in native FP8 (cebu packs), run beside the EXL3 routed experts: this rank's columns of w1/w3
    (gate/up) and input rows of w2 (down), the same split as the routed experts, so its output is a partial sum the
    MoE's all-reduce completes. Activation as the reference (model._moe_exact): gate and up rounded to bf16, up clamped
    to +-limit, gate to <= limit, silu(gate) * up, then down."""

    w13: NativeFp8Linear          # gate (w1) then up (w3) rows: one launch reads both (cebu 2026-10-07)
    w2: NativeFp8Linear
    inter: int                    # this rank's columns of each

    @classmethod
    def load(cls, sh, prefix: str, cols) -> "NativeSharedMLP":
        use_plain_paths()
        parts = [_slice(sh.get(f"{prefix}.{m}.weight"), sh.get(f"{prefix}.{m}.scale"), cols, None) for m in ("w1", "w3")]
        w = torch.cat([p[0] for p in parts])
        s = torch.cat([p[1] for p in parts]).view(torch.uint8).repeat_interleave(BLOCK, dim=0)
        w13 = NativeFp8Linear(Mx8Linear.from_checkpoint(w.cuda().view(torch.float8_e4m3fn), s.cuda()))
        return cls(w13, NativeFp8Linear.load(sh, f"{prefix}.w2", rows=cols), parts[0][0].shape[0])

    @property
    def words(self) -> list:
        return [self.w13.words, self.w2.words]

    def __call__(self, x: torch.Tensor, limit: float) -> torch.Tensor:
        gu = native_mm(self.w13, x)
        g = gu[:, :self.inter].float().clamp(max=limit)
        u = gu[:, self.inter:].float().clamp(-limit, limit)
        h = (torch.nn.functional.silu(g) * u).to(BF16)
        return native_mm(self.w2, h, torch.float32)


def fuse_inputs(parts: list) -> "NativeFp8Linear | None":
    """One MXFP8 matrix for projections of the same input (cebu 2026-10-07: wq_a and wkv, one decode launch). The
    parts become zero-copy views of its 64-row tiles (the fragment order keeps a 64-row tile's bytes together), so each
    still runs alone where a path calls it alone, and nothing is held twice. None unless every part is native and
    64-row aligned."""

    if not parts or not all(isinstance(p, NativeFp8Linear) and p.mx.n % 64 == 0 for p in parts):
        return None
    k = parts[0].mx.k
    if any(p.mx.k != k for p in parts):
        return None
    fused = NativeFp8Linear(Mx8Linear.stack([p.mx for p in parts]))
    at = 0
    for p in parts:
        n = p.mx.n
        lo, hi = at // 64, (at + n) // 64                  # bs: [npad/64, K/64, 64, 2]; w8: flat, 64 x K bytes a tile
        p.mx = Mx8Linear(fused.mx.w8[at * k:(at + n) * k], fused.mx.bs[lo:hi], n, k, n)
        at += n
    torch.cuda.empty_cache()
    return fused
