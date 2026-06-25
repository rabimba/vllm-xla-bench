#!/usr/bin/env python3
"""
collect_env.py — capture everything needed to reproduce a result cell.

Run this ON EACH MACHINE right after a sweep and keep its JSON next to the results.
Artifact-evaluation norms (OSDI/SOSP/MLSys) require pinned versions, hardware SKUs,
and flags; this records them automatically so the talk's numbers are defensible.
"""
import json
import os
import platform
import subprocess
import sys
import time


def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT,
                                       text=True, timeout=60).strip()
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>"


def pkg_versions(names):
    out = {}
    for n in names:
        try:
            mod = __import__(n)
            out[n] = getattr(mod, "__version__", "<no __version__>")
        except Exception:
            out[n] = "<not installed>"
    return out


def main():
    env = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "os": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "cpu_count": os.cpu_count(),
    }

    # Python packages relevant to the serving stacks.
    env["packages"] = pkg_versions([
        "vllm", "torch", "jax", "jaxlib", "transformers", "numpy",
        "aiohttp", "flax", "optax", "orbax", "tpu_inference",
    ])

    # CLI version probes (best-effort; fields are '<unavailable>' if absent).
    env["pip_freeze_grep"] = run(
        "pip freeze | grep -iE '^(vllm|torch|jax|jaxlib|libtpu|transformers|"
        "tpu-inference|flax|optax|orbax|maxtext)' || true")
    env["nvidia_smi"] = run("nvidia-smi --query-gpu=name,memory.total,driver_version,"
                            "compute_cap --format=csv,noheader || true")
    env["cuda"] = run("nvcc --version 2>/dev/null | tail -1 || true")
    # TPU detection (GCE metadata + libtpu).
    env["tpu_metadata"] = run(
        "curl -s -H 'Metadata-Flavor: Google' "
        "http://metadata.google.internal/computeMetadata/v1/instance/"
        "machine-type 2>/dev/null || true")
    env["tpu_accelerator"] = run(
        "curl -s -H 'Metadata-Flavor: Google' "
        "http://metadata.google.internal/computeMetadata/v1/instance/"
        "attributes/accelerator-type 2>/dev/null || true")
    env["libtpu"] = run("python -c \"import libtpu, os; "
                        "print(os.path.dirname(libtpu.__file__))\" 2>/dev/null || true")

    # XLA / vLLM relevant environment variables actually set in this shell.
    keys = [k for k in os.environ
            if k.startswith(("XLA_", "VLLM_", "JAX_", "LIBTPU_", "PJRT_",
                             "NEW_MODEL_DESIGN", "TPU_", "PT_XLA"))]
    env["xla_vllm_env"] = {k: os.environ[k] for k in sorted(keys)}

    # Repo commit, if launched from a checkout.
    env["bench_git"] = run("git rev-parse HEAD 2>/dev/null || true")

    out = "env_" + platform.node().replace(".", "_") + ".json"
    if len(sys.argv) > 1:
        out = sys.argv[1]
    with open(out, "w") as f:
        json.dump(env, f, indent=2)
    print(json.dumps(env, indent=2))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
