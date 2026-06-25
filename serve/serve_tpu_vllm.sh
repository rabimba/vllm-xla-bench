#!/usr/bin/env bash
# serve_tpu_vllm.sh — launch vLLM-TPU (the tpu-inference / JAX-native backend) on a
# Cloud TPU host (v5e-8 or v6e-8).
#
# Usage:
#   bash serve_tpu_vllm.sh <hf_model_id> <tensor_parallel> <max_model_len> [port]
# Example (Gemma 4 31B on a full v6e-8 host):
#   bash serve_tpu_vllm.sh google/gemma-4-31b-it 8 32768 8000
#
# Background: since Oct-2025 vLLM-TPU is powered by `tpu-inference`, a JAX-native
# plugin that lowers models (even PyTorch defs, via Torchax) through JAX -> XLA. The
# JetStream standalone engine was archived (Feb-1-2026) and its functionality folded
# into tpu-inference. So THIS path is the canonical "vLLM frontend on the JAX/XLA
# backend." Attention uses Ragged Paged Attention (RPA) for chunked prefill + prefix
# caching on TPU.
#
# IMPORTANT (XLA cold start): the FIRST launch compiles XLA graphs (~20-30 min cold);
# subsequent launches reuse the on-disk cache (~5 min). We point the cache at a stable
# path so cold-vs-warm is reproducible and measurable with measure_startup.py.
set -euo pipefail

MODEL="${1:?hf model id}"
TP="${2:?tensor parallel = number of TPU chips in the tp group}"
MAXLEN="${3:?max model len}"
PORT="${4:-8000}"

export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-$HOME/.cache/vllm/xla_cache}"
mkdir -p "$VLLM_XLA_CACHE_PATH"

echo "[serve_tpu_vllm] model=$MODEL tp=$TP max_len=$MAXLEN port=$PORT"
echo "[serve_tpu_vllm] XLA cache: $VLLM_XLA_CACHE_PATH (delete it to force a COLD run)"
echo "[serve_tpu_vllm] tpu-inference: $(python -c 'import tpu_inference; print(getattr(tpu_inference,\"__version__\",\"present\"))' 2>/dev/null || echo 'check install')"

# Two ways to run, pick one:
#
# (A) Containerized (recommended; matches the published Gemma 4 TPU recipe):
#     docker run -itd --name vllm-tpu --privileged --network host --shm-size 16G \
#       -v /dev/shm:/dev/shm -v "$VLLM_XLA_CACHE_PATH":/root/.cache/vllm/xla_cache \
#       -e HF_TOKEN="$HF_TOKEN" vllm/vllm-tpu:latest \
#       --model "$MODEL" --tensor-parallel-size "$TP" --max-model-len "$MAXLEN" \
#       --host 0.0.0.0 --port "$PORT"
#     # For Gemma 4 specifically the pinned image tag is vllm/vllm-tpu:gemma4 .
#
# (B) Native (vllm installed with the TPU plugin in the current venv):
exec vllm serve "$MODEL" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAXLEN" \
  --enable-chunked-prefill \
  --host 0.0.0.0 --port "$PORT" \
  --disable-log-requests
