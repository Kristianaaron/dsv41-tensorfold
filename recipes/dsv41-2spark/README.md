# DeepSeek-V4.1-Flash 2.86 bpw on two DGX Sparks

The serving recipe for [Krishando/DeepSeek-V4.1-Flash-EXL3-2.86bpw](https://huggingface.co/Krishando/DeepSeek-V4.1-Flash-EXL3-2.86bpw):
TensorFold, tensor parallel 2 over the Sparks' QSFP link, 262k context, vision on, native DSpark drafter plus suffix drafts.

## Credits and provenance

| Part | Source | Licence |
|---|---|---|
| Model | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | MIT |
| Inference engine | [ashhart/TensorFold](https://github.com/ashhart/TensorFold) | Apache-2.0 (MIT before 0.6.0) |
| DeepSeek-V4.1 family for TensorFold | [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold) @ `808eb4a` | Apache-2.0 |
| EXL3 format and encoder | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | MIT |
| Native FP8 backbone, mixed-width experts, suffix drafts, memory watch, this recipe | this fork (see `NOTICE`) | Apache-2.0 |

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
