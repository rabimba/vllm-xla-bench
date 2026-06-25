#!/usr/bin/env python3
"""
run_matrix.py — drive a full QPS sweep across workloads against ONE running server.

You launch a server (see ../serve/*.sh), then point this at it. It runs every
(workload x request-rate) point, repeats each `--repeats` times for variance, and
writes one JSON per point under results/<backend>/<hardware>/<model_tag>/.

Because the server is fixed per invocation, the only things that vary within a run
are the workload and the offered load — exactly the controlled-variable discipline
the analysis assumes. Re-run this script once per (backend, hardware, model) cell of
the matrix.

Example:
  python run_matrix.py \
     --base-url http://127.0.0.1:8000 \
     --model google/gemma-4-31b-it --model-tag gemma4-31b \
     --backend-label vllm-cuda --hardware-label h200 \
     --workloads config/workloads.yaml --slos config/slos.yaml \
     --rates 1,2,4,8,16,32,inf --repeats 3 \
     --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
"""
import argparse
import os
import sys
import time
import json
from types import SimpleNamespace

import yaml

# Import the client in-process (no subprocess => no interpreter-startup noise).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_serving import run_benchmark  # noqa: E402


def parse_rates(s: str):
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        out.append(float("inf") if tok in ("inf", "infinity") else float(tok))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-tag", required=True, help="short tag for result paths")
    ap.add_argument("--backend-label", required=True)
    ap.add_argument("--hardware-label", required=True)
    ap.add_argument("--workloads", required=True, help="YAML list of workloads")
    ap.add_argument("--slos", required=True, help="YAML mapping of SLOs by model class")
    ap.add_argument("--slo-class", default=None,
                    help="key into slos.yaml; defaults to model-tag, then 'default'")
    ap.add_argument("--rates", default="1,2,4,8,16,inf")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--num-prompts", type=int, default=500)
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--use-chat", action="store_true")
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out-root", default="results")
    args = ap.parse_args()

    with open(args.workloads) as f:
        workloads = yaml.safe_load(f)["workloads"]
    with open(args.slos) as f:
        slos = yaml.safe_load(f)
    slo_key = args.slo_class or (args.model_tag if args.model_tag in slos else "default")
    slo = slos.get(slo_key, slos["default"])
    print(f"[slo] using class '{slo_key}': {slo}")

    rates = parse_rates(args.rates)
    out_dir = os.path.join(args.out_root, args.backend_label,
                           args.hardware_label, args.model_tag)
    os.makedirs(out_dir, exist_ok=True)

    manifest = []
    for wl in workloads:
        for rate in rates:
            for rep in range(args.repeats):
                rate_tag = "inf" if rate == float("inf") else f"{rate:g}"
                fname = f"{wl['name']}__rate{rate_tag}__rep{rep}.json"
                fpath = os.path.join(out_dir, fname)
                a = SimpleNamespace(
                    base_url=args.base_url, model=args.model,
                    backend_label=args.backend_label, hardware_label=args.hardware_label,
                    dataset=wl.get("dataset", "synthetic"),
                    dataset_path=args.dataset_path,
                    num_prompts=wl.get("num_prompts", args.num_prompts),
                    max_prompt_len=wl.get("max_prompt_len", 8192),
                    fixed_output_len=wl.get("fixed_output_len"),
                    input_len=wl.get("input_len", 1024),
                    output_len=wl.get("output_len", 128),
                    request_rate=rate, burstiness=wl.get("burstiness", 1.0),
                    max_concurrency=wl.get("max_concurrency"),
                    use_chat=args.use_chat, temperature=0.0,
                    ignore_eos=args.ignore_eos or wl.get("ignore_eos", False),
                    request_timeout=wl.get("request_timeout", 600.0),
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    ttft_slo_ms=slo["ttft_slo_ms"], tpot_slo_ms=slo["tpot_slo_ms"],
                    warmup=(args.warmup if rep == 0 else 0),  # warm once per workload
                    seed=rep, trust_remote_code=args.trust_remote_code,
                    result_file=fpath, dump_per_request=False,
                )
                print(f"\n=== {wl['name']} | rate={rate_tag} | rep={rep} ===")
                try:
                    run_benchmark(a)
                    manifest.append({"workload": wl["name"], "rate": rate_tag,
                                     "rep": rep, "file": fpath, "status": "ok"})
                except SystemExit as e:
                    print(f"[skip] {e}")
                    manifest.append({"workload": wl["name"], "rate": rate_tag,
                                     "rep": rep, "file": fpath, "status": f"skip:{e}"})
                except Exception as e:  # noqa: BLE001
                    print(f"[error] {type(e).__name__}: {e}")
                    manifest.append({"workload": wl["name"], "rate": rate_tag,
                                     "rep": rep, "file": fpath, "status": f"error:{e}"})
                time.sleep(2)  # let the server drain between points

    mpath = os.path.join(out_dir, "_manifest.json")
    with open(mpath, "w") as f:
        json.dump({"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "cell": {"backend": args.backend_label,
                            "hardware": args.hardware_label,
                            "model": args.model_tag},
                   "runs": manifest}, f, indent=2)
    print(f"\n[done] manifest -> {mpath}")
    print("Remember to run ../bench/collect_env.py on THIS machine and keep its JSON "
          "next to these results.")


if __name__ == "__main__":
    main()
