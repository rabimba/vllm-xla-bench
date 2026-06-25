#!/usr/bin/env bash
# serve_gpu_vllm.sh — launch vLLM (CUDA backend) on a GPU host (H200 / RTX PRO 6000).
#
# Usage:
#   bash serve_gpu_vllm.sh <hf_model_id> <tensor_parallel> <max_model_len> [precision] [port]
# Example:
#   bash serve_gpu_vllm.sh google/gemma-4-31b-it 1 32768 bf16 8000
#
# This is the MATURE reference path. Keep its config minimal and standard so it is a
# fair, well-understood baseline (no exotic flags the TPU paths can't match).
#
# Notes:
#  - RTX PRO 6000 Blackwell has NO NVLink; multi-GPU TP runs over PCIe Gen5. Expect an
#    interconnect penalty at TP>1 vs H200's NVLink. Record this; it is a real finding.
#  - bf16 is the cross-platform apples-to-apples baseline. Use fp8 only in a SEPARATE
#    cell (fp8 is native on H200/Blackwell; precision is a confound, not a free win).
set -euo pipefail

MODEL="${1:?hf model id}"
TP="${2:?tensor parallel}"
MAXLEN="${3:?max model len}"
PRECISION="${4:-bf16}"
PORT="${5:-8000}"

DTYPE_FLAG="--dtype bfloat16"
QUANT_FLAG=""
case "$PRECISION" in
  bf16) ;;
  fp8)  QUANT_FLAG="--quantization fp8" ;;   # native on H200 + Blackwell
  fp4)  QUANT_FLAG="--quantization modelopt_fp4" ;;  # Blackwell-only; verify build
  *) echo "unknown precision $PRECISION"; exit 1 ;;
esac

echo "[serve_gpu_vllm] model=$MODEL tp=$TP max_len=$MAXLEN precision=$PRECISION port=$PORT"
echo "[serve_gpu_vllm] vllm version: $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"

# --enable-chunked-prefill is on by default in recent vLLM; keep it explicit so the
# TPU paths (which also chunk prefill via Ragged Paged Attention) are matched.
exec vllm serve "$MODEL" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  $DTYPE_FLAG $QUANT_FLAG \
  --host 0.0.0.0 --port "$PORT" \
  --disable-log-requests
