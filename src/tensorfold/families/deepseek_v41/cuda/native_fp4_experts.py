# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kristian Aaron (Krishando). Added in this fork; see NOTICE, "Further changes in this fork".
"""DSpark draft experts in the checkpoint's own MXFP4 on the NVFP4 expert kernels (cebu milestone 1b, 2026-10-07).

DeepSeek-V4.1 ships the routed experts of its MTP / DSpark draft blocks as MXFP4: E2M1 codes two a byte (low nibble
first) and one E8M0 power of two per 32 inputs. ``tensorfold.cuda.nvfp4.experts`` runs NVFP4: the same E2M1 codes, one
E4M3 scale per 16 inputs and one fp32 scale per (expert, matrix). A block's 2^p written as two per-16 scales
2^(p - G) with the matrix scale 2^G is exact when every 2^(p - G) is an E4M3 value, i.e. p - G in [-9, 8] for every
block of the matrix; G = max p - 8 makes the largest exact and leaves a 17-octave span below it. A matrix whose scales
span more is refused (no silent rounding): the draft heads stay the checkpoint's bits or the load fails.
"""

from __future__ import annotations

import torch

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.nvfp4 import experts as nvx

BLOCK = 32
E4M3_LO, E4M3_HI = -9, 8          # exact powers of two in E4M3 (subnormal 2^-9 .. 2^8)


def mx_to_nv(words: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MXFP4 words [E, N, K/2] uint8 and E8M0 bytes [E, N, K/32] -> (words, E4M3 scales [E, N, K/16], fp32 [E])."""

    p = scale.view(torch.uint8).to(torch.int32) - 127                       # [E, N, K/32] exponents
    g = p.amax(dim=(1, 2)) - E4M3_HI                                       # [E] matrix exponents
    rel = p - g[:, None, None]
    lo = int(rel.amin())
    if lo < E4M3_LO:
        raise ValueError(f"MXFP4 scales span {E4M3_HI - lo} octaves in some matrix: not exact in NVFP4 (max 17)")
    s16 = torch.ldexp(torch.ones_like(rel, dtype=torch.float32), rel).repeat_interleave(2, dim=-1)
    return words.view(torch.uint8).contiguous(), s16.to(torch.float8_e4m3fn).contiguous(), \
        torch.ldexp(torch.ones_like(g, dtype=torch.float32), g)


def _stack(sh, prefixes: list, which: str, cols=None, rows=None) -> tuple:
    """This rank's slice of one matrix kind over the experts: cols = output rows [lo, hi), rows = input columns."""

    ws, ss = [], []
    for pre in prefixes:
        w = sh.get(f"{pre}.{which}.weight")                                 # [N, K/2] packed
        s = sh.get(f"{pre}.{which}.scale")                                  # [N, K/32]
        if cols is not None:
            w, s = w[cols[0]:cols[1]], s[cols[0]:cols[1]]
        if rows is not None:
            assert rows[0] % BLOCK == 0 and rows[1] % BLOCK == 0
            w, s = w[:, rows[0] // 2:rows[1] // 2], s[:, rows[0] // BLOCK:rows[1] // BLOCK]
        ws.append(w.contiguous())
        ss.append(s.contiguous())
    return mx_to_nv(torch.stack(ws).cuda(), torch.stack(ss).cuda())


def is_mxfp4(sh, prefix: str) -> bool:
    return f"{prefix}.w1.trellis" not in sh and f"{prefix}.w1.weight" in sh and f"{prefix}.w1.scale" in sh


def load(sh, p: str, n_experts: int, cols: tuple, limit: float) -> nvx.Experts4:
    """The routed experts of draft block ``p`` (``<p>.ffn.experts.<e>``), this rank's half of the intermediate."""

    pre = [f"{p}.ffn.experts.{e}" for e in range(n_experts)]
    return nvx.make(_stack(sh, pre, "w1", cols=cols), _stack(sh, pre, "w3", cols=cols),
                    _stack(sh, pre, "w2", rows=cols), limit=limit)


_PLANS: dict = {}


def routed(x: torch.Tensor, pick: torch.Tensor, wts: torch.Tensor, ex: nvx.Experts4) -> torch.Tensor:
    """fp32 [n, D]: sum over a row's routed slots of weight * expert(x). Slots whose expert id is >= ex.count (the
    shared slot, computed elsewhere) are skipped."""

    n, slots = pick.shape
    prefill = n > 64
    key = (str(x.device), slots, ex.count, prefill)
    plan = _PLANS.get(key)
    if plan is None or plan.rows < n:
        plan = _PLANS[key] = grouped.Plan(max(n, 64), slots, ex.count + 1, x.device, prefill=prefill)
    picks = pick.contiguous().to(torch.int32)
    grouped.route(picks, plan, nvx.PREFILL_TILE)
    act = torch.empty((n * slots, ex.width), dtype=torch.bfloat16, device=x.device)
    nvx.gate_up(x.contiguous().to(torch.bfloat16), ex, plan, act, n, skip=ex.count)
    y = torch.zeros((n * slots, ex.dims), dtype=torch.float32, device=x.device)
    nvx.down(act, ex, plan, y, n, skip=ex.count)
    w = torch.where(picks < ex.count, wts, torch.zeros_like(wts)).to(torch.float32)
    return (y.view(n, slots, ex.dims) * w[..., None]).sum(dim=1)
