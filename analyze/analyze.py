#!/usr/bin/env python3
"""
analyze.py — turn the result JSONs into a tidy table + the talk's figures.

Reads every results/**/<workload>__rate*__rep*.json (plus startup_*.json), builds a
long-format CSV (one row per run), aggregates repeats (mean + std), and renders the
core figures:

  fig_throughput_vs_qps   output tok/s vs offered QPS, one line per (backend,hw)
  fig_p99_ttft_vs_qps     P99 TTFT vs offered QPS (the latency-under-load story)
  fig_p99_tpot_vs_qps     P99 TPOT vs offered QPS
  fig_goodput_vs_qps      SLO-constrained goodput vs offered QPS (the fair metric)
  fig_latency_throughput  P99 latency vs achieved throughput (the Pareto frontier)
  fig_perf_per_dollar     peak goodput per $/hr, bar chart (uses config/pricing.yaml)
  fig_coldstart           time-to-ready + per-shape compile overhead bars
  fig_dense_vs_moe        Gemma-4 31B dense vs 26B-A4B MoE, training held constant

All figures are written to <out>/figures/*.png and the table to <out>/summary.csv.
Hand me the JSONs (or the CSV) and I regenerate these for the deck.

Usage:
  python analyze.py --results results --pricing config/pricing.yaml --out analysis_out
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

# Headless plotting.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import yaml
except ImportError:
    yaml = None


def load_runs(results_dir):
    rows = []
    for path in glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True):
        base = os.path.basename(path)
        if base.startswith("_manifest") or base.startswith("env_"):
            continue
        with open(path) as f:
            try:
                rec = json.load(f)
            except json.JSONDecodeError:
                continue
        if base.startswith("startup_"):
            rec["_kind"] = "startup"
            rec["_path"] = path
            rows.append(rec)
            continue
        if "summary" not in rec:
            continue
        m, s = rec["metadata"], rec["summary"]
        rows.append({
            "_kind": "serving",
            "backend": m.get("backend_label"),
            "hardware": m.get("hardware_label"),
            "model": m.get("model"),
            "workload": _wl_from_path(base),
            "request_rate": (m.get("request_rate") if m.get("request_rate") is not None
                             else float("inf")),
            "rep": _rep_from_path(base),
            "completed": s.get("completed"),
            "failures": s.get("failures"),
            "rps": s.get("request_throughput_rps"),
            "out_tok_s": s.get("output_throughput_tok_s"),
            "ttft_p50": s.get("ttft_ms_median"),
            "ttft_p99": s.get("ttft_ms_p99"),
            "tpot_p50": s.get("tpot_ms_median"),
            "tpot_p99": s.get("tpot_ms_p99"),
            "e2e_p99": s.get("e2e_ms_p99"),
            "goodput_rps": s.get("goodput_rps"),
            "slo_attainment": s.get("slo_attainment"),
            "cold_start_ttft_s": m.get("cold_start_ttft_s"),
            "warm_ttft_s": m.get("warm_ttft_s"),
        })
    return rows


def _wl_from_path(base):
    return base.split("__rate")[0] if "__rate" in base else base.replace(".json", "")


def _rep_from_path(base):
    if "__rep" in base:
        try:
            return int(base.split("__rep")[1].split(".")[0])
        except ValueError:
            return 0
    return 0


def write_csv(rows, out):
    serving = [r for r in rows if r.get("_kind") == "serving"]
    if not serving:
        print("[warn] no serving rows found")
        return
    cols = ["backend", "hardware", "model", "workload", "request_rate", "rep",
            "completed", "failures", "rps", "out_tok_s",
            "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99", "e2e_p99",
            "goodput_rps", "slo_attainment", "cold_start_ttft_s", "warm_ttft_s"]
    path = os.path.join(out, "summary.csv")
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in serving:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"[saved] {path} ({len(serving)} rows)")


def _agg(serving, metric):
    """Return {(backend,hardware,workload): {rate: (mean,std)}} for a metric."""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in serving:
        if r.get(metric) is None:
            continue
        key = (r["backend"], r["hardware"], r["workload"])
        buckets[key][r["request_rate"]].append(r[metric])
    agg = {}
    for key, by_rate in buckets.items():
        agg[key] = {rate: (float(np.mean(v)), float(np.std(v)))
                    for rate, v in by_rate.items()}
    return agg


def _finite_rates(d):
    return sorted([x for x in d if np.isfinite(x)])


def line_plot(serving, metric, ylabel, title, fname, out, workload_filter=None):
    agg = _agg(serving, metric)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for (backend, hw, wl), by_rate in sorted(agg.items()):
        if workload_filter and wl != workload_filter:
            continue
        rates = _finite_rates(by_rate)
        if not rates:
            continue
        means = [by_rate[x][0] for x in rates]
        stds = [by_rate[x][1] for x in rates]
        ax.errorbar(rates, means, yerr=stds, marker="o", capsize=3,
                    label=f"{backend} / {hw}" + (f" [{wl}]" if not workload_filter else ""))
        plotted = True
    if not plotted:
        plt.close(fig)
        print(f"[skip] {fname}: no data")
        return
    ax.set_xlabel("Offered load (requests/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out, "figures", fname)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[saved] {p}")


def latency_throughput(serving, fname, out, workload_filter=None):
    """P99 E2E latency (y) vs achieved output tok/s (x): the Pareto frontier."""
    by_series = defaultdict(list)
    for r in serving:
        if workload_filter and r["workload"] != workload_filter:
            continue
        if r.get("out_tok_s") is None or r.get("e2e_p99") is None:
            continue
        by_series[(r["backend"], r["hardware"])].append((r["out_tok_s"], r["e2e_p99"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for (backend, hw), pts in sorted(by_series.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=f"{backend} / {hw}")
        plotted = True
    if not plotted:
        plt.close(fig)
        print(f"[skip] {fname}: no data")
        return
    ax.set_xlabel("Achieved output throughput (tok/s)")
    ax.set_ylabel("P99 end-to-end latency (ms)")
    ax.set_title("Latency–throughput frontier" +
                 (f" [{workload_filter}]" if workload_filter else ""))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out, "figures", fname)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[saved] {p}")


def perf_per_dollar(serving, pricing, fname, out):
    if not pricing:
        print("[skip] perf_per_dollar: no pricing config")
        return
    # peak goodput per (backend, hardware) across all rates/workloads
    peak = defaultdict(float)
    for r in serving:
        g = r.get("goodput_rps")
        if g is None:
            continue
        peak[(r["backend"], r["hardware"])] = max(
            peak[(r["backend"], r["hardware"])], g)
    labels, values = [], []
    for (backend, hw), g in sorted(peak.items()):
        price = pricing.get(hw)
        if not price:
            continue
        labels.append(f"{backend}\n{hw}")
        values.append(g / price * 3600.0)  # good-requests per dollar (per hour)
    if not values:
        print("[skip] perf_per_dollar: no matching pricing")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color="#0D9488")
    ax.set_ylabel("Peak goodput per dollar (good-req / $)")
    ax.set_title("Cost efficiency at SLO (uses public on-demand pricing)")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p = os.path.join(out, "figures", fname)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[saved] {p}  (NOTE: report engineering-effort asymmetry alongside this)")


def coldstart(rows, fname, out):
    starts = [r for r in rows if r.get("_kind") == "startup"]
    if not starts:
        print("[skip] coldstart: no startup_*.json")
        return
    labels, ttr, compile_max = [], [], []
    for r in starts:
        lbl = os.path.basename(r["_path"]).replace("startup_", "").replace(".json", "")
        labels.append(lbl)
        ttr.append(r.get("time_to_ready_s", 0.0))
        shapes = r.get("shapes", [])
        compile_max.append(max([s.get("compile_overhead_s", 0.0) for s in shapes],
                               default=0.0))
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, ttr, w, label="time-to-ready (incl. first compile)",
           color="#1E2761")
    ax.bar(x + w / 2, compile_max, w, label="max per-shape compile overhead",
           color="#F96167")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("seconds")
    ax.set_title("Cold-start & compilation regime")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out, "figures", fname)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[saved] {p}")


def dense_vs_moe(serving, fname, out, dense_tag="gemma-4-31b", moe_tag="gemma-4-26b"):
    """Peak output throughput: matched dense vs MoE from one family."""
    def peak_for(substr):
        by_hw = defaultdict(float)
        for r in serving:
            if substr in (r.get("model") or ""):
                if r.get("out_tok_s"):
                    by_hw[r["hardware"]] = max(by_hw[r["hardware"]], r["out_tok_s"])
        return by_hw
    dense, moe = peak_for(dense_tag), peak_for(moe_tag)
    hws = sorted(set(dense) | set(moe))
    if not hws:
        print("[skip] dense_vs_moe: no Gemma-4 rows")
        return
    x = np.arange(len(hws))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, [dense.get(h, 0) for h in hws], w,
           label=f"{dense_tag} (dense)", color="#028090")
    ax.bar(x + w / 2, [moe.get(h, 0) for h in hws], w,
           label=f"{moe_tag} (MoE)", color="#02C39A")
    ax.set_xticks(x)
    ax.set_xticklabels(hws, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Peak output throughput (tok/s)")
    ax.set_title("Dense vs MoE, training held constant (Gemma 4)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out, "figures", fname)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[saved] {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--pricing", default="config/pricing.yaml")
    ap.add_argument("--out", default="analysis_out")
    ap.add_argument("--workload", default=None,
                    help="optional: restrict line plots to one workload name")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    rows = load_runs(args.results)
    serving = [r for r in rows if r.get("_kind") == "serving"]
    print(f"[load] {len(serving)} serving runs, "
          f"{len([r for r in rows if r.get('_kind')=='startup'])} startup records")

    write_csv(rows, args.out)

    pricing = None
    if yaml and os.path.exists(args.pricing):
        with open(args.pricing) as f:
            pricing = yaml.safe_load(f)

    line_plot(serving, "out_tok_s", "Output throughput (tok/s)",
              "Throughput vs offered load", "fig_throughput_vs_qps.png",
              args.out, args.workload)
    line_plot(serving, "ttft_p99", "P99 TTFT (ms)",
              "P99 TTFT vs offered load", "fig_p99_ttft_vs_qps.png",
              args.out, args.workload)
    line_plot(serving, "tpot_p99", "P99 TPOT (ms/token)",
              "P99 TPOT vs offered load", "fig_p99_tpot_vs_qps.png",
              args.out, args.workload)
    line_plot(serving, "goodput_rps", "Goodput (good req/s)",
              "SLO-constrained goodput vs offered load", "fig_goodput_vs_qps.png",
              args.out, args.workload)
    latency_throughput(serving, "fig_latency_throughput.png", args.out, args.workload)
    perf_per_dollar(serving, pricing, "fig_perf_per_dollar.png", args.out)
    coldstart(rows, "fig_coldstart.png", args.out)
    dense_vs_moe(serving, "fig_dense_vs_moe.png", args.out)

    print(f"\n[done] figures in {args.out}/figures, table in {args.out}/summary.csv")


if __name__ == "__main__":
    main()
