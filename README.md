# vLLM-on-XLA Benchmark Suite

Reproducible, backend-agnostic LLM-serving benchmarks comparing **vLLM-CUDA (GPU)**,
**vLLM-TPU / `tpu-inference` (JAX-native, TPU)**, and the **MaxText model
implementation served through vLLM (TPU)**, across dense and MoE models.

The design principle: **one identical client drives every backend.** All three stacks
expose an OpenAI-compatible API, so the load generator and metric computation are held
constant and only the server varies. That makes the client a non-confound.

```
vllm-xla-bench/
├── serve/                      # per-backend server launchers (run ON the machine)
│   ├── serve_gpu_vllm.sh       #   vLLM-CUDA  (H200 / RTX PRO 6000)
│   ├── serve_tpu_vllm.sh       #   vLLM-TPU native (tpu-inference)
│   ├── serve_tpu_maxtext.sh    #   MaxText model via vLLM plugin (TPU)
│   └── download_sharegpt.sh
├── bench/
│   ├── bench_serving.py        # the async client + metrics (TTFT/TPOT/ITL/goodput)
│   ├── run_matrix.py           # QPS sweep across workloads against a live server
│   ├── measure_startup.py      # cold-start + per-shape XLA compilation timing
│   └── collect_env.py          # reproducibility env capture (run after each sweep)
├── analyze/analyze.py          # tidy CSV + all talk figures from the result JSONs
├── config/
│   ├── models.yaml             # model registry + per-hardware tp/max-len
│   ├── workloads.yaml          # ShareGPT + fixed-shape synthetic buckets
│   ├── slos.yaml               # goodput SLO targets by model class
│   └── pricing.yaml            # $/hr per hardware (FILL IN) for perf-per-dollar
└── results/                    # JSONs land here; this is what you send back
```

## What runs where

| Cell | Machine | Server script | Backend label |
|---|---|---|---|
| vLLM-CUDA | H200 | `serve_gpu_vllm.sh` | `vllm-cuda` / `h200` |
| vLLM-CUDA | RTX PRO 6000 | `serve_gpu_vllm.sh` | `vllm-cuda` / `rtx-pro-6000` |
| vLLM-TPU native | v5e-8 | `serve_tpu_vllm.sh` | `vllm-tpu` / `tpu-v5e-8` |
| vLLM-TPU native | v6e-8 | `serve_tpu_vllm.sh` | `vllm-tpu` / `tpu-v6e-8` |
| MaxText via vLLM | v5e-8 / v6e-8 | `serve_tpu_maxtext.sh` | `maxtext-vllm` / `tpu-…` |

## Procedure (per cell)

The client and the server can live on the same host. Run the client from a venv with
just `requirements.txt`; the server uses its own per-machine install.

**1. One-time client setup (any box that can reach the server):**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash serve/download_sharegpt.sh        # fetches ShareGPT json
```

**2. Validate the harness on a cheap dense model FIRST** (do not start with 70B):
```bash
# terminal A — server:
bash serve/serve_gpu_vllm.sh google/gemma-4-e4b-it 1 32768 bf16 8000
# terminal B — one smoke point:
python bench/bench_serving.py --base-url http://127.0.0.1:8000 \
  --model google/gemma-4-e4b-it --dataset synthetic --input-len 1024 --output-len 128 \
  --num-prompts 64 --request-rate 4 --ignore-eos \
  --ttft-slo-ms 800 --tpot-slo-ms 40 --result-file results/smoke.json
```
Advance only when repeated runs are stable (TTFT/TPOT within ~10%).

**3. Full sweep for the cell** (repeat per backend/hardware/model):
```bash
# server (example: vLLM-CUDA, Gemma-4 31B on H200):
bash serve/serve_gpu_vllm.sh google/gemma-4-31b-it 1 32768 bf16 8000

# sweep (client):
python bench/run_matrix.py \
  --base-url http://127.0.0.1:8000 \
  --model google/gemma-4-31b-it --model-tag gemma-4-31b \
  --backend-label vllm-cuda --hardware-label h200 \
  --workloads config/workloads.yaml --slos config/slos.yaml \
  --rates 1,2,4,8,16,32,inf --repeats 3 \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
```
For TPU cells use `serve_tpu_vllm.sh` / `serve_tpu_maxtext.sh` and the matching
`--backend-label` / `--hardware-label`. For models needing `--trust-remote-code`
(Qwen3-*) pass it to `run_matrix.py` too.

**4. Cold-start / compilation regime** (the XLA-specific result; especially on TPU):
```bash
# Delete the XLA cache first to force a TRUE cold run, then measure:
rm -rf "$HOME/.cache/vllm/xla_cache"
python bench/measure_startup.py \
  --launch-cmd "bash serve/serve_tpu_vllm.sh google/gemma-4-31b-it 8 32768 8000" \
  --base-url http://127.0.0.1:8000 --model google/gemma-4-31b-it \
  --shapes 128:128,1024:128,1024:1024,4096:256 \
  --ready-timeout 2400 \
  --result-file results/startup_vllm-tpu_tpu-v6e-8_gemma-4-31b.json
# Run again WITHOUT deleting the cache to capture the warm-start number too.
```

**5. Capture the environment for the cell** (right after the sweep, on that machine):
```bash
python bench/collect_env.py results/env_<backend>_<hardware>.json
```

## What to send back

Zip and return the **entire `results/` tree** (all `*.json`, the `_manifest.json`
files, the `startup_*.json` files, and every `env_*.json`). That is everything needed
to regenerate the figures and the analysis:

```bash
tar czf results_$(hostname)_$(date +%Y%m%d).tgz results/
```

I will run `analyze/analyze.py` on the combined tree, drop the figures into the report
and the deck, and write the results narrative.

## Generating figures locally
```bash
# fill config/pricing.yaml first if you want the perf-per-dollar chart
python analyze/analyze.py --results results --pricing config/pricing.yaml --out analysis_out
# -> analysis_out/summary.csv and analysis_out/figures/*.png
```

## Methodology guardrails (why this is defensible)

- **bf16 everywhere** for the apples-to-apples cell; native precisions (fp8/fp4) only
  in clearly-labeled separate cells. Precision is a confound, not a free win.
- **Hold the SLO fixed** per model class across all backends — goodput is only fair as
  a single yardstick (DistServe, OSDI'24).
- **Report cold AND warm** regimes; never silently discard XLA compilation cost.
- **Record everything** with `collect_env.py`: versions, the XLA/libtpu build, flags,
  seeds, hardware SKUs (OSDI/SOSP/MLSys artifact norms).
- **Disclose engineering-effort asymmetry**: vLLM-CUDA is more mature than the TPU
  paths today. State it next to every cross-platform cost number.

License: Apache-2.0 (matches the OpenXLA/JAX community norm).
