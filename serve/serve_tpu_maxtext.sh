#!/usr/bin/env bash
# serve_tpu_maxtext.sh — run the MaxText model implementation on TPU, served through
# vLLM's out-of-tree MaxText plugin. This is the "JAX-native MODEL code" path (MaxText
# defines the model; vLLM provides the scheduler/API), distinct from serve_tpu_vllm.sh
# where tpu-inference provides vLLM's own model implementations.
#
# Comparing these two isolates "whose model implementation" while holding the frontend
# (vLLM) and the compiler (XLA) constant — a clean controlled variable.
#
# Usage:
#   bash serve_tpu_maxtext.sh <model_name> <hf_tokenizer_id> <converted_ckpt> <tp> [port]
# Example (dense):
#   bash serve_tpu_maxtext.sh gemma4-31b google/gemma-4-31b-it $CKPT 4 8000
# Example (MoE — note prefuse flag handled below):
#   bash serve_tpu_maxtext.sh gemma4-26b google/gemma-4-26b-a4b-it $CKPT 4 8000
#
# Prereqs (verify against the current MaxText inference tutorial — versions move fast):
#   - MaxText installed with the `tpu-post-train` extra (provides the vLLM adapter
#     plugin + pinned tpu-inference / vllm versions).
#   - An UNSCANNED Orbax checkpoint (the vLLM path requires scan_layers=False).
#   - Convert HF -> MaxText/Orbax with the scripts under
#     maxtext/tests/end_to_end/tpu/gemma4 first.
set -euo pipefail

MODEL_NAME="${1:?maxtext model_name, e.g. gemma4-31b}"
TOKENIZER="${2:?hf tokenizer id}"
CKPT="${3:?path to converted unscanned Orbax checkpoint}"
TP="${4:?ici_tensor_parallelism = TPU chip count}"
PORT="${5:-8000}"

# MoE models need the experts pre-fused into the per-shard layout for TP>1.
PREFUSE=""
case "$MODEL_NAME" in
  *26b*|*moe*) PREFUSE="prefuse_moe_weights=True" ;;
esac

export NEW_MODEL_DESIGN=1   # required by the MaxText->vLLM adapter for direct serve
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

echo "[serve_tpu_maxtext] model=$MODEL_NAME tok=$TOKENIZER tp=$TP prefuse='${PREFUSE:-none}' port=$PORT"

# ---- ONLINE serving (preferred for this benchmark) ----
# vLLM serves the MaxText model via the architecture override. Confirm the exact flag
# surface in your MaxText/vLLM versions; the adapter applies configs/inference/vllm.yml
# internally.
exec vllm serve "$CKPT" \
  --tokenizer "$TOKENIZER" \
  --tensor-parallel-size "$TP" \
  --hf-overrides '{"architectures": ["MaxTextForCausalLM"]}' \
  --host 0.0.0.0 --port "$PORT" \
  --disable-log-requests

# ---- OFFLINE decode alternative (documented MaxText path) ----
# If online serving via the plugin is not yet wired in your version, use offline decode
# for an Offline-scenario throughput number (MLPerf-style), then compare to the GPU
# Offline number rather than to online latency:
#
# python3 -m maxtext.inference.vllm_decode src/maxtext/configs/base.yml \
#   model_name="$MODEL_NAME" tokenizer_path="$TOKENIZER" \
#   load_parameters_path="$CKPT" \
#   vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
#   ici_tensor_parallelism="$TP" scan_layers=False $PREFUSE \
#   use_chat_template=True
