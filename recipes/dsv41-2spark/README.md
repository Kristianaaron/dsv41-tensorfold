# DeepSeek-V4.1-Flash · EXL3 · 2.86 bpw · Hessian-traced

The two-DGX-Spark serving recipe.

Weights: [Krishando/DeepSeek-V4.1-Flash-EXL3-2.86bpw](https://huggingface.co/Krishando/DeepSeek-V4.1-Flash-EXL3-2.86bpw).
It runs on TensorFold, tensor parallel 2 over the Sparks' QSFP link, with 262k context, vision on, and the native
DSpark drafter plus suffix drafts.

## Credits and provenance

The quant is my own work. The engine that serves it is mostly other people's, and they're credited here in full.

| Part | Who | Source | Licence |
|---|---|---|---|
| The quant: calibration, per-expert 2–5-bit allocation, encode, evaluation | Kristian Aaron (Krishando), original work | [Krishando/DeepSeek-V4.1-Flash-EXL3-2.86bpw](https://huggingface.co/Krishando/DeepSeek-V4.1-Flash-EXL3-2.86bpw) | MIT |
| Base model, its FP8 weights and the Engram tables | DeepSeek ([@deepseek_ai](https://x.com/deepseek_ai)) | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | MIT |
| TensorFold, the inference engine | ashhart ([@ashxhart](https://x.com/ashxhart)) and the TensorFold contributors | [ashhart/TensorFold](https://github.com/ashhart/TensorFold) | Apache-2.0 (MIT before 0.6.0) |
| The `deepseek_v41` family: DeepSeek-V4.1 tensor-parallel over two GB10s, the RDMA all-gather, DSpark drafting, vision | BertholomusAI, Albert Lee ([@bertholomusai](https://x.com/bertholomusai)) | [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold) at `808eb4a` | Apache-2.0 |
| The EXL3 format and the encoder the quant was made with | turboderp | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | MIT |
| Native FP8 backbone and mixed-width EXL3 experts, MXFP4 draft experts, suffix drafts, memory watch, this recipe | Kristian Aaron (Krishando) | this repository | Apache-2.0 |
| Run-time image | NVIDIA | `nvcr.io/nvidia/pytorch:26.07-py3` (not included) | NVIDIA's container licence |

How the provenance is kept:

- This repository carries bertholomus/TensorFold's full history, which carries ashhart/TensorFold's. Nothing is squashed,
  and every upstream author keeps their commits. Mine sit on top: `git log 808eb4a..HEAD` lists exactly what I changed.
- `NOTICE` keeps TensorFold's and BertholomusAI's notices unchanged and lists every file I added or modified
  (Apache-2.0, section 4(b)). Each of those files also says so in its first line.
- The sources behind the `deepseek_v41` family and their licences are in BertholomusAI's
  [`tools/dsv41/ATTRIBUTION.md`](../../tools/dsv41/ATTRIBUTION.md).
- "DeepSeek" and "DeepSeek-V4.1-Flash" belong to DeepSeek; "DGX Spark" and "GB10" belong to NVIDIA. This project is not
  affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors or BertholomusAI.

## What you need

- Two DGX Sparks connected by a QSFP cable, with RDMA working (the defaults assume interface `enp1s0f1np1`, RDMA device
  `rocep1s0f1`, GID index 3), passwordless SSH from the head node to the worker, and Docker with the NVIDIA runtime.
- At least 110 GiB free on each Spark before you start. The engine refuses to allocate below a 3 GiB floor
  (`MEM_FLOOR_GIB`) and logs every new memory low-water mark.
- About 214 GB for the quant and 196 GB for the two Engram shards, on both nodes (or on shared storage).

## Steps

```bash
# 1. Weights (both nodes, or shared storage)
hf download Krishando/DeepSeek-V4.1-Flash-EXL3-2.86bpw --local-dir ~/models/dsv41-2.86
hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --include "model-00047-of-00048.safetensors" --include "model-00048-of-00048.safetensors" \
  --local-dir ~/models/dsv41-engram

# 2. Image (build on the head node, then copy it to the worker)
docker build -f recipes/dsv41-2spark/Dockerfile -t tensorfold-dsv41:2.86 .
docker save tensorfold-dsv41:2.86 | ssh spark-2 docker load

# 3. Serve
PACK=~/models/dsv41-2.86 ENGRAM=~/models/dsv41-engram SRC=$PWD WORKER=spark-2 HEAD_IP=10.0.0.1 \
  recipes/dsv41-2spark/launch.sh start

# 4. Stop (graceful, head node first)
PACK=~/models/dsv41-2.86 ENGRAM=~/models/dsv41-engram SRC=$PWD WORKER=spark-2 HEAD_IP=10.0.0.1 \
  recipes/dsv41-2spark/launch.sh stop
```

The server speaks the OpenAI chat API on port 8000 (`PORT`). The first start compiles CUDA extensions and Triton
kernels into `~/.cache/tensorfold-dsv41`; later starts reuse them and copy them to the worker before weights load.

## Settings used for the published numbers

`PARALLEL=4`, `DRAFTS=5` (native DSpark), `SUFFIX=1`, prefill chunk 2048, exact fp32 accumulation (TF32 off),
`expandable_segments:False`. Measured on the release build (2026-10-08), one stream unless noted:

| | tok/s |
|---|---|
| New code | 69–76 |
| Structured output | 85–105 |
| Prose | 41–42 |
| Code edits | 137–138 |
| Prefill, 8k / 32k / 131k / 245k | 1,680 / 1,582 / 1,726 / 1,706 (warm runs 2,044–2,072 at 8k–32k) |
| 4 streams, total | 104 |

A 6-hour soak (3,822 replies, mixed traffic, 4 streams) had 0 failures; the lowest free memory was 4.7 GiB on the head
node and 6.1 GiB on the worker.

## Safety notes for GB10

GPU memory on a Spark is system memory. Keep the memory floor on, start with the 110 GiB margin above, and stop with
`launch.sh stop` (it sends SIGTERM and waits) rather than killing containers. An external watchdog can be added with
`MEMGUARD=/path/to/script`; none is bundled.
