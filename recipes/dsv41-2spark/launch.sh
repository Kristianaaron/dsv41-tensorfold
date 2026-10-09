#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Serve Krishando/DeepSeek-V4.1-Flash-EXL3-2.86bpw on two DGX Sparks with TensorFold, tensor parallel 2.
# Rank 1 starts on the worker first, then rank 0 on this (head) node. Stop is graceful, head first.
#
#   launch.sh start | stop
#
# Required environment (no defaults on purpose):
#   PACK      path to the downloaded model folder (same path on both nodes, or set WORKER_PACK)
#   ENGRAM    folder holding the base model's model-00047/00048-of-00048.safetensors (same path on both nodes)
#   SRC       path to this repository checkout (synced to the worker on start)
#   WORKER    ssh host of the second Spark, e.g. spark-2
#   HEAD_IP   head node address on the QSFP link, e.g. 10.0.0.1
# Optional:
#   NET_IF (enp1s0f1np1)  RDMA_DEV (rocep1s0f1)  GID (3)  PORT (8000)  PARALLEL (4)  DRAFTS (5)  SUFFIX (1)
#   IMG (tensorfold-dsv41:2.86)  CACHE (~/.cache/tensorfold-dsv41)  MEM_FLOOR_GIB (3)
#   MEMGUARD  path to an external memory-guard script (called as: MEMGUARD <container> <GiB> <log> <pidfile>)
set -u
for v in PACK ENGRAM SRC WORKER HEAD_IP; do [ -n "${!v:-}" ] || { echo "set $v"; exit 2; }; done
WPACK=${WORKER_PACK:-$PACK}; PORT=${PORT:-8000}; IMG=${IMG:-tensorfold-dsv41:2.86}
NET_IF=${NET_IF:-enp1s0f1np1}; RDMA_DEV=${RDMA_DEV:-rocep1s0f1}; GID=${GID:-3}
CACHE=${CACHE:-$HOME/.cache/tensorfold-dsv41}; LOGS=${LOGS:-$HOME/.cache/tensorfold-dsv41/logs}
ENVS="-e TF_DS_RANK_CACHE=/cache -e TF_DS_RANK_CACHE_READERS=32 -e TF_DS_ENGRAM=/engram -e TF_DS_PREFILL_CHUNK=2048 \
 -e TF_DS_SUFFIX_DRAFT=${SUFFIX:-1} -e NCCL_IB_HCA=$RDMA_DEV -e NCCL_IB_GID_INDEX=$GID -e NCCL_SOCKET_IFNAME=$NET_IF \
 -e GLOO_SOCKET_IFNAME=$NET_IF -e TF_RDMA_DEVICES=$RDMA_DEV -e TF_DS_MEMWATCH=1 -e TF_DS_MEM_FLOOR_GIB=${MEM_FLOOR_GIB:-3} \
 -e TORCH_EXTENSIONS_DIR=/cache/torch-ext -e TRITON_CACHE_DIR=/cache/triton -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False ${EXTRA_ENVS:-}"
RUN="docker run -d --init --gpus all --network host --ipc host --shm-size 32g --device /dev/infiniband --cap-add IPC_LOCK \
 --ulimit memlock=-1 --ulimit stack=67108864 -v $CACHE:/cache -v $SRC:/opt/tensorfold:ro -v $ENGRAM:/engram:ro"
ARGS="--tp 2 --master $HEAD_IP --host 0.0.0.0 --port $PORT --context 262144 --vision --parallel ${PARALLEL:-4} --mtp-drafts ${DRAFTS:-5} --temperature 0"
case "${1:-}" in
start)
  if docker ps -q --filter name=^tf-dsv41-r0$ | grep -q . || ssh $WORKER "docker ps -q --filter name=^tf-dsv41-r1$" < /dev/null | grep -q .; then
    echo "a TensorFold container is still running: run '$0 stop' first"; exit 1; fi
  mkdir -p $CACHE/torch-ext $CACHE/triton $LOGS; ssh $WORKER "mkdir -p $CACHE/torch-ext $CACHE/triton" < /dev/null
  rsync -a --delete $SRC/ $WORKER:$SRC/ || exit 1
  # Ship the head's compiled CUDA extensions and Triton kernels to the worker before weights load, so the worker
  # does not compile them at the end of its load, when memory is tightest. The files are root-owned, so a root
  # container replaces them.
  (cd $CACHE && tar -cf - torch-ext triton) | ssh $WORKER "docker run --rm -i -v $CACHE:/c ubuntu:24.04 sh -c 'rm -rf /c/torch-ext /c/triton && tar -xf - -C /c'" || exit 1
  ssh $WORKER "docker rm -f tf-dsv41-r1 >/dev/null 2>&1; $RUN -v $WPACK:/model:ro --name tf-dsv41-r1 $ENVS $IMG tensorfold serve /model --rank 1 $ARGS" < /dev/null || exit 1
  docker rm -f tf-dsv41-r0 >/dev/null 2>&1
  $RUN -v $PACK:/model:ro --name tf-dsv41-r0 $ENVS $IMG tensorfold serve /model --rank 0 $ARGS || exit 1
  if [ -n "${MEMGUARD:-}" ]; then
    nohup setsid bash $MEMGUARD tf-dsv41-r0 1.5 $LOGS/memguard-r0.log $LOGS/memguard-r0.pid > /dev/null 2>&1 < /dev/null &
    scp -q $MEMGUARD $WORKER:/tmp/tf-memguard.sh && ssh $WORKER "nohup setsid bash /tmp/tf-memguard.sh tf-dsv41-r1 1.5 /tmp/tf-memguard-r1.log /tmp/tf-memguard-r1.pid > /dev/null 2>&1 < /dev/null &" < /dev/null
  fi
  for i in $(seq 1 270); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health)" = 200 ] && { echo "ready after $((i * 10)) s on port $PORT"; exit 0; }
    docker ps --format '{{.Names}}' | grep -qx tf-dsv41-r0 || { echo "rank 0 exited, logs in $LOGS"; docker logs tf-dsv41-r0 > $LOGS/r0.log 2>&1; ssh $WORKER "docker logs tf-dsv41-r1" > $LOGS/r1.log 2>&1 < /dev/null; exit 1; }
    sleep 10
  done
  echo "not ready after 45 min"; exit 1 ;;
stop)
  mkdir -p $LOGS; docker logs tf-dsv41-r0 > $LOGS/r0.log 2>&1; ssh $WORKER "docker logs tf-dsv41-r1" > $LOGS/r1.log 2>&1 < /dev/null
  # graceful, head first: SIGTERM with 180 s each; remove containers only once both have stopped
  docker stop -t 180 tf-dsv41-r0 >/dev/null 2>&1
  timeout 240 ssh $WORKER "docker stop -t 180 tf-dsv41-r1 >/dev/null 2>&1" < /dev/null
  docker rm tf-dsv41-r0 >/dev/null 2>&1; ssh $WORKER "docker rm tf-dsv41-r1 >/dev/null 2>&1" < /dev/null
  [ -f $LOGS/memguard-r0.pid ] && kill $(cat $LOGS/memguard-r0.pid) 2>/dev/null
  ssh $WORKER "[ -f /tmp/tf-memguard-r1.pid ] && kill \$(cat /tmp/tf-memguard-r1.pid) 2>/dev/null" < /dev/null; exit 0 ;;
*) echo "usage: $0 start|stop"; exit 2 ;;
esac
