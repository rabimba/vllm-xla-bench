# Experiment Protocol — what to run, in what order, and how to know it's valid

This is the experimental design (the science). The suite README is the tool mechanics
(how to launch a server, sweep, hand back results). Read them together. The governing
idea: **change exactly one thing per comparison.** Every cell fixes everything except
the variable under test, so a difference in the numbers maps to a cause.

---

## 0. The cell matrix

A **cell** = (backend × hardware × model × precision). Within a cell you sweep
(workload × QPS × repeats). The full matrix:

- **Backends (3):** `vllm-cuda`, `vllm-tpu` (native tpu-inference), `maxtext-vllm`.
- **Hardware (4):** `h200`, `rtx-pro-6000`, `tpu-v5e-8`, `tpu-v6e-8`.
- **Models (8):** llama-3.1-8b/70b · gemma-4-31b · gemma-4-26b · gemma-4-e4b ·
  nemotron-nano-30b · qwen3-next-80b · (qwen3-coder-480b, optional).
- **Precision:** `bf16` for the apples-to-apples comparison. Native (`fp8`/`fp4`/`int8`)
  only as **separate, explicitly-labeled** cells — never mixed into the bf16 numbers.

The full Cartesian product is large and mostly pointless. **Do not run all of it.**
Run the prioritized subset below; it answers every question in the talk.

---

## 1. The validity contract (applies to every run)

1. **bf16 everywhere** for the comparison. Precision is a confound, not a free win.
2. **One identical client** (the suite) drives all backends — already enforced.
3. **Fixed SLO per model class** (`config/slos.yaml`), identical across backends.
4. **Both regimes:** warm steady-state sweep **and** cold-start/compilation.
5. **Capture the environment** after every sweep (`collect_env.py`) and keep the JSON.
6. **≥3 repeats** per point; report mean ± std. If std > ~10% of mean, add repeats or
   investigate (thermal throttling, noisy neighbor, background compaction).
7. **Disclose effort asymmetry** when comparing cost — vLLM-CUDA is more mature today.

If any of these is violated, the number is not comparable. Fix it before moving on.

---

## 2. Phased plan (run in this order; each phase gates the next)

### Phase 0 — Plumbing (½ day)
**Goal:** prove the harness + each backend talk to each other before spending real time.
- On each machine: launch the server for **gemma-4-e4b** (cheapest), then run ONE
  smoke point (`bench_serving.py … --num-prompts 64 --request-rate 4`).
- **Gate:** a smoke run completes with 0 failures and non-zero throughput on every
  (backend × hardware) you intend to use. Confirm `collect_env.py` records versions.
- **Why e4b:** it loads fast and compiles fast, so you find wiring/auth/quota problems
  in minutes, not after a 30-minute TPU compile.

### Phase 1 — Harness validation against known-good (½ day)
**Goal:** prove the numbers are *sane*, not just non-zero.
- Run a full QPS sweep for **llama-3.1-8b** on **vllm-cuda / h200** (the most
  characterized cell in the world).
- **Gate:** your throughput/TTFT/TPOT curves are in the same ballpark as published
  vLLM Llama-3.1-8B numbers on H200. If they're 3× off, your client, SLOs, or
  `max-model-len` are wrong — debug here, where ground truth exists, not later on TPU.
- Then repeat the same sweep on **vllm-tpu / v6e-8** and **vllm-cuda / rtx-pro-6000**
  to shake out each backend's launch path on a known model.

### Phase 2 — The dense reference cells (1 day)
**Goal:** the dense baseline across the full matrix.
- Sweep **gemma-4-31b** and **llama-3.1-70b** on all (backend × hardware) you can fit.
  Use `config/models.yaml` tp values; 70B won't fit a single RTX without TP=4 (PCIe).
- This is the first cross-platform comparison: vLLM-CUDA vs vLLM-TPU vs MaxText-vLLM
  on the *same dense models*. It fills the throughput / latency / goodput figures.

### Phase 3 — Both regimes for every Phase-2 cell (½ day, mostly waiting)
**Goal:** the cold-start/compilation data — the talk's novel measurement.
- For each TPU cell, run `measure_startup.py` **twice**:
  1. `rm -rf $VLLM_XLA_CACHE_PATH` first → **cold** (captures first XLA compile).
  2. again **without** deleting → **warm** (cache hit).
- Do the same on a GPU cell for the eager baseline (near-zero compile).
- **Gate:** you have a `startup_*.json` per cell with `time_to_ready_s` (cold & warm)
  and per-shape `compile_overhead_s`. These drive `fig_coldstart`.

### Phase 4 — The named experiment: dense vs MoE (1 day)
**Goal:** the controlled centerpiece.
- Sweep **gemma-4-31b (dense)** and **gemma-4-26b (MoE)** on the **same hardware**,
  **same SLO**, **same workloads** — ideally on both a GPU cell (h200) and a TPU cell
  (v6e-8). On the MaxText path the MoE needs `prefuse_moe_weights=True` for TP>1.
- **The comparison:** 31B-dense vs 26B-A4B-MoE peak throughput and goodput, per
  hardware. **The money question:** does the dense-vs-MoE verdict *flip* between GPU
  and TPU? (Hypothesis: MoE's routing taxes the systolic array more than GPU SMs.)
- This is the only place architecture is isolated with training held constant — guard
  it carefully (identical workloads, identical SLO, identical precision).

### Phase 5 — MoE / hybrid stress (1–2 days; expect friction)
**Goal:** push the TPU path where it's weakest.
- Sweep **nemotron-nano-30b** and **qwen3-next-80b** across backends. Add
  **qwen3-coder-480b** only if you have multi-host TPU / 8×H200 — otherwise skip.
- **Expect feature gaps on vLLM-TPU** (irregular MoE routing, exotic hybrid layers).
  Record what *doesn't* run as a finding — "supported on CUDA, not yet on TPU-native"
  is a legitimate, useful result for this audience. Don't force it; note it.

### Phase 6 (optional) — Native-precision cells
**Goal:** show the precision headroom without polluting the bf16 comparison.
- Re-sweep the Gemma-4 pair at **fp8** on h200/v6e and **fp4** on rtx-pro-6000, in
  **separately labeled** cells. Present as "native precision" deltas, never merged
  into the bf16 ranking.

---

## 3. What each figure needs (so you don't under-collect)

| Figure | Needs |
|---|---|
| `fig_throughput_vs_qps` | Phase 2 sweeps, all cells, the QPS ladder |
| `fig_p99_ttft/tpot_vs_qps` | same sweeps (TTFT separates on prefill-heavy shapes, TPOT on decode-heavy) |
| `fig_goodput_vs_qps` | same sweeps + SLOs set in `slos.yaml` (the headline) |
| `fig_latency_throughput` | same sweeps (Pareto view) |
| `fig_dense_vs_moe` | Phase 4 (Gemma-4 31B vs 26B on matched hardware) |
| `fig_coldstart` | Phase 3 (`startup_*.json`, cold + warm) |
| `fig_perf_per_dollar` | any goodput data + `pricing.yaml` filled in (same basis for all) |

If a figure is empty after your run, you skipped its phase. Cross-check before sending.

---

## 4. Dials and defaults

- **QPS ladder:** `--rates 1,2,4,8,16,32,inf`. `inf` = saturation (all-at-once). If a
  small model saturates above 32, extend the ladder; if a 70B saturates below 8,
  shorten it. The goal is to bracket the knee of each curve.
- **Open vs closed loop:** default open-loop (request-rate). Add a closed-loop pass
  (`--max-concurrency`) only if you want a fixed-concurrency throughput ceiling.
- **Workloads:** the full ladder in `workloads.yaml` (ShareGPT + 5 synthetic shapes).
  At minimum run `sharegpt_real`, `synth_1024_128` (prefill story), and
  `synth_256_2048` (decode story) — they carry most of the narrative.
- **Repeats:** 3 (default). Bump to 5 for the headline goodput cells.
- **Decode length:** `ignore_eos` on synthetic so every backend emits exactly the same
  number of tokens — otherwise TPOT comparisons are apples-to-oranges.

---

## 5. Per-backend pitfalls (the ones that will actually bite)

- **vLLM-TPU first launch is slow** (XLA compile, ~20–30 min cold). That's expected and
  is itself data — measure it, don't kill it. Use `VLLM_XLA_CACHE_PATH` so warm
  restarts are fast and reproducible.
- **MaxText needs an unscanned Orbax checkpoint** (`scan_layers=False`) and, for MoE,
  `prefuse_moe_weights=True`. Convert HF→MaxText first (scripts under the gemma4 e2e
  tests). If online serving via the plugin isn't wired in your version, fall back to
  the offline `vllm_decode` path and compare it to the GPU **Offline** scenario, not to
  online latency.
- **RTX PRO 6000 has no NVLink** → don't expect H200-like TP scaling; TP>1 runs over
  PCIe. Capture TP=1 and TP=2 separately so the interconnect penalty is visible.
- **Locked-down install (VPC-SC)**: the prebuilt-binary fetch can fail *silently at
  install, loudly at runtime*. Verify the plugin imports cleanly in Phase 0.
- **Gemma 4 is multimodal by default** — run **text-only** so it's comparable to the
  rest of the suite. MTP spec-decode needs nightly vLLM; pin the wheel if you use it.
- **Qwen3-* need `--trust-remote-code`** — pass it to both the server and `run_matrix.py`.
- **Confirm HF model ids** in `models.yaml` (Nemotron/Qwen tags marked to verify) before
  a long run, and **fill `pricing.yaml`** with one consistent pricing basis before the
  cost figure means anything.

---

## 6. Definition of done

You're done collecting when, for every cell you committed to:
- a full QPS sweep exists (3+ reps) with `_manifest.json`,
- a cold and a warm `startup_*.json` exist (TPU cells),
- an `env_*.json` exists,
- and `analyze.py` over the combined tree produces all 7 figures with no empty panels.

Then: `tar czf results_$(hostname)_$(date +%Y%m%d).tgz results/` per machine and send
the tarballs back. I run `analyze.py` on the union, write the results narrative, and
drop the real figures into the report and the deck.

---

## 7. The minimum viable run (if you're short on time / quota)

If you can only do a slice, do this — it still tells a complete story:
1. **gemma-4-31b** and **gemma-4-26b** on **h200** and **tpu-v6e-8**, `vllm-cuda` vs
   `vllm-tpu`, bf16, the 3 core workloads, 3 reps. → throughput, latency, goodput,
   dense-vs-MoE.
2. **cold + warm `measure_startup.py`** for the two TPU cells. → the compilation figure.
3. `collect_env.py` on both machines; `pricing.yaml` filled. → cost figure.

That's 4 sweeps + 2 startup runs and it populates every figure in the deck. Everything
else (RTX, v5e, MaxText path, 70B, the hybrids) deepens the story but isn't required to
have a coherent, defensible talk.
