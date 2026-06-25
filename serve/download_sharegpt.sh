#!/usr/bin/env bash
# download_sharegpt.sh — fetch the standard ShareGPT serving-benchmark dataset.
# This is the canonical real-conversation workload used by vLLM's bench harness.
set -euo pipefail
OUT="${1:-ShareGPT_V3_unfiltered_cleaned_split.json}"
URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
if [ -f "$OUT" ]; then
  echo "[sharegpt] already present: $OUT"
  exit 0
fi
echo "[sharegpt] downloading -> $OUT"
curl -L -o "$OUT" "$URL"
echo "[sharegpt] done. Records: $(python -c "import json;print(len(json.load(open('$OUT'))))" 2>/dev/null || echo '?')"
