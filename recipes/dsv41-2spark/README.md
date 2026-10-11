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

### Everything the quant was made and measured with

The same list is on the [model card](https://huggingface.co/Krishando/DeepSeek-V4.1-Flash-EXL3-2.86bpw#credits-and-provenance).

#### Model, format and code

| What | Who | Used for | Licence |
|---|---|---|---|
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), commit `df42c10` | DeepSeek ([@deepseek_ai](https://x.com/deepseek_ai)) | The weights: FP8 backbone, routed experts (the part quantized here), DSpark drafter, vision tower, Engram tables. Its reference inference code was used to check our full-model reference | MIT |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | turboderp | The EXL3 format, the encoder that made this quant, and the proxy-error measure behind the bit allocation | MIT |
| [QTIP](https://arxiv.org/abs/2406.11235) | Albert Tseng, Qingyao Sun, David Hou, Christopher De Sa | The trellis quantization method EXL3 builds on | Paper |
| [DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark) and [sfxnz/DeepSeek-V4.1-Flash-EXL3](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-EXL3) | Sufyan Shah ([sfxnz](https://huggingface.co/sfxnz)) | An earlier 2-Spark EXL3 pack and recipe for this model. This quant's encoder started from its `quantize_experts_exl3.py` and `pack_meta.py` (stream the official shards, EXL3 for the routed experts, every other tensor kept as stored, Engram tables read from disk); the Hessian-traced calibration and per-expert widths were added on top | MIT |
| [TensorFold](https://github.com/ashhart/TensorFold) | ashhart ([@ashxhart](https://x.com/ashxhart)) and the [TensorFold contributors](https://github.com/ashhart/TensorFold/graphs/contributors) | The inference engine. Every speed on this card is TensorFold's | Apache-2.0 (MIT before 0.6.0) |
| [`deepseek_v41` family for TensorFold](https://github.com/bertholomus/TensorFold) at `808eb4a` | BertholomusAI, Albert Lee ([@bertholomusai](https://x.com/bertholomusai)) | DeepSeek-V4.1 tensor-parallel over two GB10s; the runtime builds on it | Apache-2.0 |
| [Transformers](https://github.com/huggingface/transformers) | Hugging Face | The DeepSeek-V4.1 model code behind the layer-by-layer full-model reference and the Hessian capture | Apache-2.0 |
| [vLLM](https://github.com/vllm-project/vllm) | The vLLM project | Serving for the KLD measurements | Apache-2.0 |
| vLLM EXL3 overlay from the DeepSeek-V4.1 serving kit | MiaAI-Lab | Running EXL3 packs inside vLLM for the KLD measurements | AGPL-3.0 |
| PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3` | NVIDIA | The run-time image (not included) | NVIDIA's container licence |

#### Calibration data (1.03M tokens)

55% is the author's own agent sessions (private; rendered in the V4.1 chat format with thinking and tool calls). The rest:

| Dataset | Who | Domain | Licence |
|---|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample-10BT) | Hugging Face | Prose | ODC-By 1.0 |
| [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) (train) | Salesforce (Merity et al.) | Prose | CC BY-SA 3.0 |
| [codeparrot-clean-valid](https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid) | CodeParrot | Code | See its dataset card |
| [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | ise-uiuc (Magicoder authors) | Code | MIT |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | Paster et al. | Maths | ODC-By 1.0 |
| [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | NVIDIA | Maths | CC BY 4.0 |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (train) | OpenAI | Maths | MIT |
| [Hermes Function Calling v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | Nous Research | Tool calls | Apache-2.0 |
| [PG-19](https://huggingface.co/datasets/emozilla/pg19) | Google DeepMind (Rae et al.) | Long documents | Apache-2.0 |
| [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) 20231101 (ar, de, es, fr, hi, ja, ru, zh) | Wikimedia | Multilingual | CC BY-SA 3.0 |

#### Evaluation

| Dataset or tool | Who | Used for | Licence |
|---|---|---|---|
| [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) (test) | Salesforce (Merity et al.) | General holdout: prose and long documents | CC BY-SA 3.0 |
| [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) | OpenAI | General holdout: code | MIT |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (test) | OpenAI | General holdout: maths; the GSM8K score | MIT |
| [Belebele](https://huggingface.co/datasets/facebook/belebele) (8 languages) | Meta AI | General holdout: multilingual | CC BY-SA 4.0 |
| [MMLU](https://huggingface.co/datasets/cais/mmlu) | Hendrycks et al. | The MMLU score | MIT |
| [three.js](https://github.com/mrdoob/three.js) | The three.js authors | The long-generation task (a 3D game) | MIT |
| [Playwright](https://github.com/microsoft/playwright) with Chromium | Microsoft; the Chromium project | Headless play test of the generated games | Apache-2.0; BSD-3-Clause |

#### Reference builds measured during development

Measured on the same harness while choosing this build:
[Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) and
[0xSero/DeepSeek-V4.1-Flash-Spark](https://huggingface.co/0xSero/DeepSeek-V4.1-Flash-Spark) (served with its launcher,
[0xSero/DeepSeek-V4.1-Flash-Two-Sparks](https://github.com/0xSero/DeepSeek-V4.1-Flash-Two-Sparks) at `ee68818`). Their weights
are not part of this quant.

#### Names

"DeepSeek" and "DeepSeek-V4.1-Flash" belong to DeepSeek; "DGX Spark" and "GB10" belong to NVIDIA. This project is not
affiliated with or endorsed by DeepSeek, NVIDIA, or any of the people and projects above.

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

## Decision points in agent sessions

The headline KLD (0.168 on agent sessions) averages every token. An agent run is usually derailed by a few of them:
whether to call a tool or answer, which tool, whether to make another call, and when to stop. This scores only those
tokens, from the same served dumps and the same full-model reference as the headline number:

| Decision | Positions | This quant: KLD, same top-1 | Mia-AiLab 2.9 bpw: KLD, same top-1 | KLD ratio (95% CI) |
|---|---|---|---|---|
| Call a tool, or answer (the tokens after `</think>`) | 105 | 0.059, 96.2% | 0.091, 94.3% | 0.65 (0.38–0.87) |
| Which tool (every token of the name) | 96 | 0.032, 93.8% | 0.107, 90.6% | 0.30 (0.12–0.70) |
| Which tool (first token of the name) | 42 | 0.052, 88.1% | 0.197, 85.7% | 0.26 (0.09–0.70) |
| Another call, or done | 41 | 0.003, 95.1% | 0.010, 92.7% | 0.30 (0.22–0.40) |
| Stop (end of the turn) | 48 | 0.004, 100% | 0.005, 100% | 0.75 (0.32–1.71) |
| **All decision points** | **285** | **0.034, 95.8%** | **0.071, 93.7%** | **0.47 (0.21–0.73)** |
| Every position (the headline) | 70,570 | 0.168, 90.7% | 0.248, 88.4% | |

"Same top-1" is how often the quant's most likely token is the full model's, which is the choice it makes greedily.
The ratio is this quant's mean KLD over the reference's on the same positions; its interval resamples whole sessions.

What this does and doesn't show:

- At decision points this quant stays about twice as close to the full model as the reference build. The same-top-1
  gaps are small (95.8% vs 93.7%); most of the difference is in how sure each quant is, not in which token wins.
- Both quants are closer to the full model at decision points than on average. Stopping is near-identical for both.
- It is small: 18 sessions, 42 tool calls. Teacher-forced: each decision is scored on the full model's own history, so it
  measures single decisions, not whole agent runs. A tool-calling benchmark is still to do.
- Served through vLLM, like the headline KLD. TensorFold cannot return prompt log-probabilities yet.

To reproduce on your own holdout (`tools/dsv41/decision_kld.py`; needs `torch` and `tokenizers`):

```bash
python3 tools/dsv41/decision_kld.py --manifest holdout-manifest.json --tokenizer tokenizer.json \
  --teacher teacher_ref_b000-009.pt teacher_ref_b009-018.pt --student this-quant.pt other-quant.pt --out decisions.json
```

The "every position" row reproduces the headline KLD exactly, which checks that the decision rows come from the same data.

## Safety notes for GB10

GPU memory on a Spark is system memory. Keep the memory floor on, start with the 110 GiB margin above, and stop with
`launch.sh stop` (it sends SIGTERM and waits) rather than killing containers. An external watchdog can be added with
`MEMGUARD=/path/to/script`; none is bundled.
