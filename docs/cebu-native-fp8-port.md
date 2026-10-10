# DeepSeek-V4.1 TP2 with the native FP8 backbone (cebu port, 2026-10-07)

_Written by Kristian Aaron (Krishando), 2026. Apache-2.0, part of this fork's changes (see `NOTICE`)._

Goal: serve packs whose routed experts are EXL3 (any width 1-8, mixed per layer, `mcg` or `mul1`) while every other
projection stays in DeepSeek's own FP8 (cebu-profiler v4.1/v5 packs), on this engine at TP=2 over two GB10 Sparks.
Upstream today builds every linear as `Exl3Linear` (Mia-style all-EXL3 packs only).

## The checkpoint format

| Tensor | dtype | shape (layer 3) |
| --- | --- | --- |
| `attn.wq_a.weight` / `.scale` | F8_E4M3 / F8_E8M0 | [1280, 5120] / [40, 160] |
| `attn.wq_b` | same | [32768, 1280] / [1024, 40] |
| `attn.wkv` | same | [512, 5120] / [16, 160] |
| `attn.wo_a` | same | [8192, 4096] / [256, 128] |
| `attn.wo_b` | same | [5120, 8192] / [160, 256] |
| `ffn.shared_experts.w1/w3` | same | [2304, 5120] / [72, 160] |
| `ffn.shared_experts.w2` | same | [5120, 2304] / [160, 72] |
| `head.weight`, `embed.weight` | BF16 | [129280, 5120] |

Scales are one E8M0 power of two per 32x32 block. `Mx8Linear` (cuda/nvfp4/linear.py) takes e4m3 [N, K] with one
E8M0 per 32 inputs per row: repeating each block scale over its 32 rows gives exactly that, so the conversion is
bit-identical (no requantization). Every K here (5120, 1280, 4096, 8192, 2304) is a multiple of 64 as Mx8 requires.

## Where the engine is EXL3-specific

- `weights.linear()` -> `Exl3Linear` only; `load_block` packs the shared expert into the routed-expert kernel as the
  always-selected last expert (`exl3_experts.prepare`, `ex.keep[e - 1]`).
- `model.mm()` dispatches on EXL3 sizes (glinear up to 128 rows, `exl3_prefill.matmul` beyond).
- Grouped paths (`attn_in`, `q_proj`, `wo_a_out`, `wo_ab`, `*_rot`) share one Hadamard input rotation across several
  `Exl3Linear`s (`Exl3Group`) and fold rotations into RMSNorm / sparse-attn epilogues.
- L2 prefetch lists read `.words` of each linear.
- dspark.py groups `wkv` of the draft blocks; markov head uses `self.head.device`.

## Plan

1. `native_fp8.NativeFp8Linear` (done in this branch: adapter with `k`, `n`, `words`, `suh=None`, decode and prefill
   calls over `Mx8Linear`), `Bf16Linear` for the head.
2. `weights.linear()`: native when `<prefix>.trellis` is absent and `<prefix>.weight` is e4m3; TP slices by
   output columns (`cols`) or input rows (`rows`) on 32-aligned boundaries (scales sliced with them).
3. `model.mm()`: native layers call their own decode / prefill path; F32 outputs cast.
4. Milestone 1 runs with `TF_DS_GROUPED_LINEAR=0` (every projection through `mm`), `hc_rot` and `rot_wob` off.
5. Shared expert: load it as a native FP8 MLP beside `exl3_experts.prepare(...)` of the routed experts only; `moe()`
   adds silu(x w1) * (x w3) @ w2 (TP: w1/w3 by output column, w2 by input row, then the existing all-reduce).
6. Milestone 2: grouped paths take native layers per layer (no shared rotation for them), so routed experts keep
   their grouped launches and the backbone gets Mx8 decode GEMVs inside the same CUDA graph.
7. Validation: greedy output on the cebu fixed prompt set equal to vLLM's on the same pack; then the cebu bench
   (code / structured / prose separately), vision, DSpark, 262k context, memory floor.
