#!/usr/bin/env python3
"""
bench_serving.py — a standalone, backend-agnostic LLM serving benchmark client.

Why this exists
---------------
We compare three serving stacks (vLLM-CUDA on GPU, vLLM-TPU/tpu-inference on TPU,
and the MaxText model implementation served through vLLM on TPU). All three expose
an OpenAI-compatible HTTP API. By driving every backend with *one identical client*
that generates load and computes metrics the same way, the client is never a source
of cross-backend variance — only the server differs. This mirrors the methodology of
`vllm bench serve` (Poisson arrivals, burstiness, percentile reporting) but has no
vLLM dependency, so it runs anywhere with just aiohttp + numpy + transformers.

Metrics (all standard in the serving literature):
  - TTFT  : time-to-first-token (s)              [first streamed content chunk]
  - TPOT  : time-per-output-token (s/token)      [(E2E - TTFT) / (out_tokens - 1)]
  - ITL   : inter-token latency list (s)         [gaps between successive chunks]
  - E2E   : end-to-end request latency (s)
  - Throughput: output tokens/s and requests/s (completed / wall-clock)
  - Goodput  : completed requests/s that satisfy BOTH a TTFT SLO and a TPOT SLO
               (Zhong et al., DistServe, OSDI'24).

References for definitions:
  Kwon et al., PagedAttention, SOSP'23; Yu et al., Orca, OSDI'22;
  Agrawal et al., Sarathi-Serve, OSDI'24; Zhong et al., DistServe, OSDI'24.

Usage (single point):
  python bench_serving.py \
      --base-url http://127.0.0.1:8000 \
      --model google/gemma-4-31b-it \
      --dataset sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-prompts 500 --request-rate 8 --burstiness 1.0 \
      --ttft-slo-ms 2000 --tpot-slo-ms 100 \
      --result-file results/run.json

Outputs a JSON file with full metadata, aggregates, and per-request records.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import aiohttp
except ImportError:
    aiohttp = None  # only needed for the networked path; pure functions work without it


def _require_aiohttp():
    if aiohttp is None:
        sys.exit("aiohttp is required for the networked benchmark: pip install aiohttp")


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class RequestInput:
    prompt: str
    prompt_len: int          # measured input tokens (tokenizer)
    output_len: int          # requested max_tokens for this request


@dataclass
class RequestOutput:
    success: bool = False
    prompt_len: int = 0
    requested_output_len: int = 0
    generated_tokens: int = 0          # from server `usage`, else chunk count
    ttft: float = 0.0                  # s
    e2e_latency: float = 0.0           # s
    itl: List[float] = field(default_factory=list)  # s, per decode chunk gap
    tpot: float = 0.0                  # s/token
    error: str = ""
    start_wall: float = 0.0            # absolute perf_counter at dispatch
    end_wall: float = 0.0


# --------------------------------------------------------------------------- #
# Workload construction
# --------------------------------------------------------------------------- #
def _get_tokenizer(model: str, trust_remote_code: bool):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)


def build_sharegpt_requests(
    path: str,
    tokenizer,
    num_prompts: int,
    fixed_output_len: Optional[int],
    min_prompt_len: int = 4,
    max_prompt_len: int = 8192,
    seed: int = 0,
) -> List[RequestInput]:
    """Sample (prompt, completion) pairs from ShareGPT, tokenize, and filter.

    If fixed_output_len is given, every request asks for exactly that many output
    tokens (clean control over decode length). Otherwise we use the length of the
    dataset's reference completion (realistic distribution).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Keep conversations with at least one human turn followed by a gpt turn.
    pairs: List[Tuple[str, str]] = []
    for entry in raw:
        conv = entry.get("conversations", [])
        if len(conv) < 2:
            continue
        if conv[0].get("from") not in ("human", "user"):
            continue
        prompt = conv[0]["value"]
        completion = conv[1]["value"]
        pairs.append((prompt, completion))

    rng = random.Random(seed)
    rng.shuffle(pairs)

    out: List[RequestInput] = []
    for prompt, completion in pairs:
        if len(out) >= num_prompts:
            break
        p_ids = tokenizer(prompt).input_ids
        if not (min_prompt_len <= len(p_ids) <= max_prompt_len):
            continue
        if fixed_output_len is not None:
            o_len = fixed_output_len
        else:
            c_ids = tokenizer(completion).input_ids
            o_len = max(1, len(c_ids))
        out.append(RequestInput(prompt=prompt, prompt_len=len(p_ids), output_len=o_len))

    if len(out) < num_prompts:
        print(f"[warn] only {len(out)} prompts passed the length filter "
              f"(requested {num_prompts}). Proceeding with {len(out)}.", file=sys.stderr)
    return out


def build_synthetic_requests(
    tokenizer,
    num_prompts: int,
    input_len: int,
    output_len: int,
    seed: int = 0,
) -> List[RequestInput]:
    """Fixed-shape synthetic workload: every prompt is ~input_len tokens, every
    request asks for output_len tokens. Useful for isolating a single (in, out)
    bucket and for clean cross-backend / XLA-bucketing comparisons.

    We build prompts from random vocabulary token ids so we hit the target length
    precisely, then decode back to text for the API call.
    """
    rng = np.random.default_rng(seed)
    vocab = tokenizer.vocab_size
    # Avoid special tokens at the low end of the vocab where possible.
    lo = min(1000, max(1, vocab // 100))
    reqs: List[RequestInput] = []
    for _ in range(num_prompts):
        ids = rng.integers(low=lo, high=vocab, size=input_len).tolist()
        text = tokenizer.decode(ids)
        # Re-tokenize to get the true length after decode/encode round-trip.
        true_len = len(tokenizer(text).input_ids)
        reqs.append(RequestInput(prompt=text, prompt_len=true_len, output_len=output_len))
    return reqs


# --------------------------------------------------------------------------- #
# Arrival process
# --------------------------------------------------------------------------- #
def interarrival_delays(
    n: int, request_rate: float, burstiness: float, seed: int
) -> List[float]:
    """Return n inter-arrival gaps (s).

    request_rate == inf  -> all zeros (send everything at t=0; saturation test).
    Otherwise gaps ~ Gamma(shape=burstiness, scale=1/(rate*burstiness)), whose mean
    is 1/rate. burstiness=1.0 reduces to Exponential (a Poisson process), <1 is
    burstier, >1 is more uniform — identical semantics to `vllm bench serve`.
    """
    if request_rate == float("inf"):
        return [0.0] * n
    rng = np.random.default_rng(seed)
    shape = burstiness
    scale = 1.0 / (request_rate * burstiness)
    return rng.gamma(shape=shape, scale=scale, size=n).tolist()


# --------------------------------------------------------------------------- #
# Single request
# --------------------------------------------------------------------------- #
async def one_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    req: RequestInput,
    use_chat: bool,
    temperature: float,
    ignore_eos: bool,
    request_timeout: float,
    api_key: Optional[str],
) -> RequestOutput:
    """Issue one streaming request and time every chunk."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if use_chat:
        url = f"{base_url}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": req.prompt}],
            "max_tokens": req.output_len,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    else:
        url = f"{base_url}/v1/completions"
        payload = {
            "model": model,
            "prompt": req.prompt,
            "max_tokens": req.output_len,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    # ignore_eos forces the server to emit exactly max_tokens, giving clean,
    # comparable decode lengths across backends. vLLM honors this extension.
    if ignore_eos:
        payload["ignore_eos"] = True

    out = RequestOutput(prompt_len=req.prompt_len, requested_output_len=req.output_len)
    chunk_times: List[float] = []
    usage_completion: Optional[int] = None
    content_chunks = 0

    start = time.perf_counter()
    out.start_wall = start
    try:
        async with session.post(url, json=payload, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=request_timeout)) as resp:
            if resp.status != 200:
                body = await resp.text()
                out.error = f"HTTP {resp.status}: {body[:300]}"
                out.end_wall = time.perf_counter()
                return out
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                now = time.perf_counter()
                # usage chunk (final, when include_usage=True)
                if obj.get("usage"):
                    usage_completion = obj["usage"].get("completion_tokens")
                choices = obj.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                if use_chat:
                    delta = ch.get("delta", {})
                    piece = delta.get("content")
                else:
                    piece = ch.get("text")
                if piece:  # non-empty content => a decode step we can time
                    chunk_times.append(now)
                    content_chunks += 1
    except asyncio.TimeoutError:
        out.error = "timeout"
        out.end_wall = time.perf_counter()
        return out
    except Exception as e:  # noqa: BLE001 — surface any client/transport error
        out.error = f"{type(e).__name__}: {e}"
        out.end_wall = time.perf_counter()
        return out

    end = time.perf_counter()
    out.end_wall = end
    out.e2e_latency = end - start

    if not chunk_times:
        out.error = out.error or "no content received"
        return out

    out.ttft = chunk_times[0] - start
    # inter-token latencies = gaps between successive content chunks
    out.itl = [chunk_times[i] - chunk_times[i - 1] for i in range(1, len(chunk_times))]
    out.generated_tokens = usage_completion if usage_completion else content_chunks
    denom = max(1, out.generated_tokens - 1)
    out.tpot = (out.e2e_latency - out.ttft) / denom
    out.success = True
    return out


# --------------------------------------------------------------------------- #
# Benchmark driver (one workload point)
# --------------------------------------------------------------------------- #
async def _run(
    base_url: str,
    model: str,
    requests: List[RequestInput],
    request_rate: float,
    burstiness: float,
    max_concurrency: Optional[int],
    use_chat: bool,
    temperature: float,
    ignore_eos: bool,
    request_timeout: float,
    api_key: Optional[str],
    seed: int,
) -> Tuple[List[RequestOutput], float]:
    _require_aiohttp()
    delays = interarrival_delays(len(requests), request_rate, burstiness, seed)
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    results: List[RequestOutput] = []

    async with aiohttp.ClientSession(connector=connector) as session:
        async def guarded(req: RequestInput) -> RequestOutput:
            if sem:
                async with sem:
                    return await one_request(session, base_url, model, req, use_chat,
                                             temperature, ignore_eos, request_timeout, api_key)
            return await one_request(session, base_url, model, req, use_chat,
                                     temperature, ignore_eos, request_timeout, api_key)

        tasks: List[asyncio.Task] = []
        bench_start = time.perf_counter()
        for req, gap in zip(requests, delays):
            if gap > 0:
                await asyncio.sleep(gap)
            tasks.append(asyncio.create_task(guarded(req)))
        results = await asyncio.gather(*tasks)
        bench_end = time.perf_counter()

    return results, bench_end - bench_start


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _pct(xs: List[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else 0.0


def summarize(
    results: List[RequestOutput],
    duration: float,
    ttft_slo_ms: Optional[float],
    tpot_slo_ms: Optional[float],
) -> Dict[str, Any]:
    ok = [r for r in results if r.success]
    n_ok, n_total = len(ok), len(results)
    ttfts = [r.ttft * 1000 for r in ok]                       # ms
    tpots = [r.tpot * 1000 for r in ok]                       # ms/token
    e2es = [r.e2e_latency * 1000 for r in ok]                 # ms
    itls = [x * 1000 for r in ok for x in r.itl]              # ms
    total_out = sum(r.generated_tokens for r in ok)
    total_in = sum(r.prompt_len for r in ok)

    def block(name: str, xs: List[float]) -> Dict[str, float]:
        return {
            f"{name}_mean": float(np.mean(xs)) if xs else 0.0,
            f"{name}_median": _pct(xs, 50),
            f"{name}_p90": _pct(xs, 90),
            f"{name}_p95": _pct(xs, 95),
            f"{name}_p99": _pct(xs, 99),
            f"{name}_max": float(np.max(xs)) if xs else 0.0,
        }

    good = 0
    if ttft_slo_ms is not None and tpot_slo_ms is not None:
        for r in ok:
            if (r.ttft * 1000 <= ttft_slo_ms) and (r.tpot * 1000 <= tpot_slo_ms):
                good += 1

    summary: Dict[str, Any] = {
        "completed": n_ok,
        "total_requested": n_total,
        "failures": n_total - n_ok,
        "duration_s": duration,
        "request_throughput_rps": n_ok / duration if duration > 0 else 0.0,
        "output_throughput_tok_s": total_out / duration if duration > 0 else 0.0,
        "total_token_throughput_tok_s": (total_in + total_out) / duration if duration > 0 else 0.0,
        "mean_input_tokens": total_in / n_ok if n_ok else 0.0,
        "mean_output_tokens": total_out / n_ok if n_ok else 0.0,
    }
    summary.update(block("ttft_ms", ttfts))
    summary.update(block("tpot_ms", tpots))
    summary.update(block("itl_ms", itls))
    summary.update(block("e2e_ms", e2es))
    if ttft_slo_ms is not None and tpot_slo_ms is not None:
        summary["ttft_slo_ms"] = ttft_slo_ms
        summary["tpot_slo_ms"] = tpot_slo_ms
        summary["goodput_rps"] = good / duration if duration > 0 else 0.0
        summary["slo_attainment"] = good / n_ok if n_ok else 0.0
    return summary


# --------------------------------------------------------------------------- #
# Warmup
# --------------------------------------------------------------------------- #
async def warmup(base_url, model, sample: RequestInput, n: int, use_chat,
                 request_timeout, api_key) -> List[float]:
    """Send n serial warmup requests; return their TTFTs (s).

    On XLA backends the FIRST request of each new shape triggers compilation, so
    warmup TTFTs expose compilation/bucketing cost. We return them so the caller
    can record the cold-vs-warm gap rather than silently discarding it.
    """
    ttfts = []
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        for _ in range(n):
            r = await one_request(session, base_url, model, sample, use_chat,
                                  0.0, True, request_timeout, api_key)
            ttfts.append(r.ttft)
    return ttfts


# --------------------------------------------------------------------------- #
# Public callable (used by run_matrix.py) + CLI
# --------------------------------------------------------------------------- #
def run_benchmark(args) -> Dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)

    tok = _get_tokenizer(args.model, args.trust_remote_code)

    if args.dataset == "sharegpt":
        if not args.dataset_path or not os.path.exists(args.dataset_path):
            sys.exit(f"--dataset-path not found: {args.dataset_path}")
        reqs = build_sharegpt_requests(
            args.dataset_path, tok, args.num_prompts,
            fixed_output_len=args.fixed_output_len,
            max_prompt_len=args.max_prompt_len, seed=args.seed)
    elif args.dataset == "synthetic":
        reqs = build_synthetic_requests(
            tok, args.num_prompts, args.input_len, args.output_len, seed=args.seed)
    else:
        sys.exit(f"unknown dataset {args.dataset}")

    if not reqs:
        sys.exit("no requests built; check dataset / filters")

    # Warmup (separate, serial). Records cold-start / compilation behavior.
    warm_ttfts: List[float] = []
    if args.warmup > 0:
        warm_ttfts = asyncio.run(warmup(
            args.base_url, args.model, reqs[0], args.warmup,
            args.use_chat, args.request_timeout, args.api_key))

    results, duration = asyncio.run(_run(
        base_url=args.base_url, model=args.model, requests=reqs,
        request_rate=args.request_rate, burstiness=args.burstiness,
        max_concurrency=args.max_concurrency, use_chat=args.use_chat,
        temperature=args.temperature, ignore_eos=args.ignore_eos,
        request_timeout=args.request_timeout, api_key=args.api_key, seed=args.seed))

    summary = summarize(results, duration, args.ttft_slo_ms, args.tpot_slo_ms)

    record = {
        "metadata": {
            "base_url": args.base_url,
            "model": args.model,
            "backend_label": args.backend_label,
            "hardware_label": args.hardware_label,
            "dataset": args.dataset,
            "dataset_path": args.dataset_path,
            "num_prompts": args.num_prompts,
            "request_rate": (None if args.request_rate == float("inf") else args.request_rate),
            "burstiness": args.burstiness,
            "max_concurrency": args.max_concurrency,
            "use_chat": args.use_chat,
            "ignore_eos": args.ignore_eos,
            "input_len": args.input_len if args.dataset == "synthetic" else None,
            "output_len": args.output_len if args.dataset == "synthetic" else None,
            "fixed_output_len": args.fixed_output_len,
            "seed": args.seed,
            "warmup": args.warmup,
            "warmup_ttft_s": warm_ttfts,
            "cold_start_ttft_s": (warm_ttfts[0] if warm_ttfts else None),
            "warm_ttft_s": (warm_ttfts[-1] if len(warm_ttfts) > 1 else None),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "summary": summary,
    }
    if args.dump_per_request:
        record["per_request"] = [asdict(r) for r in results]

    # Pretty console summary
    print(json.dumps(summary, indent=2))
    if warm_ttfts:
        print(f"[cold-start] first warmup TTFT = {warm_ttfts[0]*1000:.1f} ms; "
              f"last = {warm_ttfts[-1]*1000:.1f} ms "
              f"(gap exposes XLA compilation/bucketing on first shape)")

    if args.result_file:
        os.makedirs(os.path.dirname(args.result_file) or ".", exist_ok=True)
        with open(args.result_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"[saved] {args.result_file}")
    return record


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backend-agnostic LLM serving benchmark client")
    p.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000")
    p.add_argument("--model", required=True, help="served model id (must match server)")
    p.add_argument("--backend-label", default="unknown",
                   help="vllm-cuda | vllm-tpu | maxtext-vllm (for bookkeeping)")
    p.add_argument("--hardware-label", default="unknown",
                   help="h200 | rtx-pro-6000 | tpu-v5e-8 | tpu-v6e-8 ...")

    p.add_argument("--dataset", choices=["sharegpt", "synthetic"], default="sharegpt")
    p.add_argument("--dataset-path", default=None, help="ShareGPT json path")
    p.add_argument("--num-prompts", type=int, default=500)
    p.add_argument("--max-prompt-len", type=int, default=8192)
    p.add_argument("--fixed-output-len", type=int, default=None,
                   help="sharegpt: force this many output tokens for every request")
    p.add_argument("--input-len", type=int, default=1024, help="synthetic input tokens")
    p.add_argument("--output-len", type=int, default=128, help="synthetic output tokens")

    p.add_argument("--request-rate", type=float, default=float("inf"),
                   help="requests/s; 'inf' sends all at t=0 (saturation)")
    p.add_argument("--burstiness", type=float, default=1.0,
                   help="gamma shape; 1.0=Poisson, <1 burstier, >1 more uniform")
    p.add_argument("--max-concurrency", type=int, default=None,
                   help="cap in-flight requests (closed-loop); default open-loop")

    p.add_argument("--use-chat", action="store_true", help="hit /v1/chat/completions")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--ignore-eos", action="store_true",
                   help="force exactly max_tokens output (clean decode lengths)")
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))

    p.add_argument("--ttft-slo-ms", type=float, default=None)
    p.add_argument("--tpot-slo-ms", type=float, default=None)

    p.add_argument("--warmup", type=int, default=3,
                   help="serial warmup requests (also measures cold-start)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--result-file", default=None)
    p.add_argument("--dump-per-request", action="store_true")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_benchmark(args)
