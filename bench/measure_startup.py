#!/usr/bin/env python3
"""
measure_startup.py — quantify the XLA compilation / cold-start regime.

Steady-state throughput hides the cost that matters for autoscaling and bursty
traffic: ahead-of-time XLA compilation. This harness measures it in two parts.

  (1) TIME-TO-READY: launch the server as a subprocess and poll /health until it
      answers. On TPU this interval includes weight load + the first XLA graph
      compilation; on GPU/eager it is mostly weight load. We log it.

  (2) PER-SHAPE FIRST-REQUEST LATENCY: once ready, send ONE request at each (input,
      output) shape in a sweep. On XLA backends a previously-unseen shape triggers
      a recompile (unless it falls into an existing bucket), so the first request at
      a new shape spikes; the second at the same shape is warm. We send each shape
      twice and report cold vs warm, which directly exposes bucketing granularity.

This isolates the phenomenon the talk frames as a first-class result, instead of
discarding warmup as most harnesses do.

Example:
  python measure_startup.py \
    --launch-cmd "bash ../serve/serve_tpu_vllm.sh google/gemma-4-31b-it 8 32768" \
    --base-url http://127.0.0.1:8000 --model google/gemma-4-31b-it \
    --shapes 128:128,1024:128,1024:1024,4096:256 \
    --ready-timeout 2400 --result-file results/startup_tpu_gemma4-31b.json
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error


def http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def post_completion(base_url, model, prompt, max_tokens, timeout=600):
    """Non-streaming POST; return (ttft_proxy_e2e_s, ok, err). For a single-request
    cold-start probe we use end-to-end latency as the compile-inclusive signal."""
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "ignore_eos": True, "stream": False,
    }).encode()
    req = urllib.request.Request(f"{base_url}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        return time.perf_counter() - t0, True, ""
    except Exception as e:  # noqa: BLE001
        return time.perf_counter() - t0, False, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch-cmd", required=True,
                    help="shell command that starts the server in the foreground")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--health-path", default="/health")
    ap.add_argument("--shapes", default="128:128,1024:128,1024:1024,4096:256",
                    help="comma list of input_chars:output_tokens probes")
    ap.add_argument("--ready-timeout", type=float, default=2400.0,
                    help="max seconds to wait for /health (TPU first compile is slow)")
    ap.add_argument("--poll-interval", type=float, default=2.0)
    ap.add_argument("--keep-alive", action="store_true",
                    help="leave the server running after measuring")
    ap.add_argument("--result-file", default=None)
    args = ap.parse_args()

    health_url = args.base_url.rstrip("/") + args.health_path
    record = {"launch_cmd": args.launch_cmd, "model": args.model,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    print(f"[launch] {args.launch_cmd}")
    proc = subprocess.Popen(args.launch_cmd, shell=True, preexec_fn=os.setsid)

    # (1) time-to-ready
    t_launch = time.perf_counter()
    ready = False
    while time.perf_counter() - t_launch < args.ready_timeout:
        if http_ok(health_url):
            ready = True
            break
        if proc.poll() is not None:
            print("[fatal] server process exited before becoming ready", file=sys.stderr)
            break
        time.sleep(args.poll_interval)
    ttr = time.perf_counter() - t_launch
    record["time_to_ready_s"] = ttr
    record["became_ready"] = ready
    print(f"[ready] {'yes' if ready else 'NO'} after {ttr:.1f}s "
          f"(includes weight load + first XLA compile on TPU)")

    # (2) per-shape cold vs warm
    shape_results = []
    if ready:
        for spec in args.shapes.split(","):
            in_chars, out_toks = spec.split(":")
            in_chars, out_toks = int(in_chars), int(out_toks)
            prompt = "word " * (in_chars // 5 + 1)  # rough; exact len not critical here
            cold, ok1, e1 = post_completion(args.base_url, args.model, prompt, out_toks)
            warm, ok2, e2 = post_completion(args.base_url, args.model, prompt, out_toks)
            shape_results.append({
                "shape": spec, "input_chars": in_chars, "output_tokens": out_toks,
                "cold_e2e_s": cold, "warm_e2e_s": warm,
                "compile_overhead_s": max(0.0, cold - warm),
                "cold_ok": ok1, "warm_ok": ok2, "err": (e1 or e2),
            })
            print(f"[shape {spec}] cold={cold:.2f}s warm={warm:.2f}s "
                  f"=> compile overhead ~{max(0.0, cold-warm):.2f}s")
    record["shapes"] = shape_results

    if args.result_file:
        os.makedirs(os.path.dirname(args.result_file) or ".", exist_ok=True)
        with open(args.result_file, "w") as f:
            json.dump(record, f, indent=2)
        print(f"[saved] {args.result_file}")

    if not args.keep_alive:
        print("[teardown] stopping server")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        print(f"[keep-alive] server still running (pid {proc.pid})")


if __name__ == "__main__":
    main()
