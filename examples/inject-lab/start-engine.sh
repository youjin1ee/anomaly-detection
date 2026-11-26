#!/bin/bash
# Starts a vLLM server with a hard cap on GPU memory (unified memory machines can be
# exhausted otherwise). ENGINE_PYTHON_BIN is the bin directory of an environment with vLLM.
#   ENGINE_BIN=/path/to/venv/bin MODEL=Qwen/Qwen3-4B ./start-engine.sh
set -euo pipefail
: "${ENGINE_BIN:?bin directory of an environment with vllm}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
GPU_UTIL="${GPU_UTIL:-0.2}"   # fraction of GPU memory, keep it small on shared machines
PORT="${PORT:-8000}"
exec "$ENGINE_BIN/vllm" serve "$MODEL" --dtype bfloat16 --max-model-len 2048 \
  --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs 32 --no-enable-prefix-caching \
  --enforce-eager --host 127.0.0.1 --port "$PORT" --served-model-name lab-model
