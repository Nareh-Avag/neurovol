"""
Benchmark the three IV backends on the deterministic synthetic chain.

  (a) python  original per-row solver (scipy.stats.norm per call, Newton then
              scipy brentq), the code compute_iv_batch has always used
  (b) numpy   vectorized Newton with vectorized bisection fallback
  (c) cpp     C++ batch solver through pybind11, GIL released

All three use the same Brenner-Subrahmanyam initial guess, Newton update,
clipping, and tolerances, so the comparison is about implementation cost, not
about one side having a smarter algorithm.

What is timed: implied_vol_array(price, S, K, T, r, is_call, backend=...) on
NumPy arrays that already exist. Building the chain and converting the
DataFrame are outside the timer. Each timed run solves the whole chain once.

Methodology: warm-up runs (not recorded), then N timed runs with
time.perf_counter. Reports median, p95, min, and throughput
(contracts / median seconds). Single process, single thread.

Usage:
    python bench/benchmark.py                          # defaults used for results.md
    python bench/benchmark.py --sizes 1000,10000 --repeats 20 --python-repeats 20,20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import sysconfig
import time
import timeit
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import norm  # noqa: E402

from vol_surface_engine import iv_solver  # noqa: E402
from vol_surface_engine.synthetic_chain import generate_chain  # noqa: E402

RATE = 0.03
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def cpu_model() -> str:
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=30).stdout.strip()
            if out:
                return out
        elif sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        else:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith(("model name", "hardware")):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001 - best effort only
        pass
    return platform.processor() or "unknown"


def environment() -> dict:
    py_plat = sysconfig.get_platform()
    host = platform.machine()
    emulated = sys.platform == "win32" and host.upper() == "ARM64" and py_plat == "win-amd64"
    env = {
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "os": platform.platform(),
        "host_arch": host,
        "python": f"{platform.python_version()} ({py_plat})",
        "x64_emulation": emulated,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cpp_available": iv_solver.CPP_AVAILABLE,
    }
    if iv_solver.CPP_AVAILABLE:
        core = iv_solver._iv_core
        env["cpp_compiler"] = getattr(core, "COMPILER", "unknown")
        env["cpp_build_type"] = getattr(core, "BUILD_TYPE", "unknown")
        env["cpp_flags"] = " ".join(getattr(core, "BUILD_FLAGS", "unknown").split())
    return env


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_runs(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def summarize(times: list[float], n: int) -> dict:
    a = np.asarray(times)
    med = float(np.median(a))
    return {
        "repeats": len(a),
        "median_s": med,
        "p95_s": float(np.percentile(a, 95)),
        "min_s": float(a.min()),
        "throughput": n / med,
    }


def fmt_time(s: float) -> str:
    if s >= 1.0:
        return f"{s:.2f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.2f} ms"
    return f"{s * 1e6:.1f} us"


def fmt_rate(r: float) -> str:
    if r >= 1e6:
        return f"{r / 1e6:.2f} M/s"
    if r >= 1e3:
        return f"{r / 1e3:.1f} k/s"
    return f"{r:.0f} /s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1000,10000,100000")
    ap.add_argument("--repeats", type=int, default=20,
                    help="timed runs for numpy and cpp")
    ap.add_argument("--python-repeats", default="20,20,3",
                    help="timed runs for the per-row python backend, one per size")
    ap.add_argument("--out", default=str(HERE / "results.md"))
    ap.add_argument("--render-only", action="store_true",
                    help="rebuild results.md from an existing results.json without timing")
    args = ap.parse_args()

    if args.render_only:
        raw = json.loads(Path(args.out).with_suffix(".json").read_text())
        res = {int(k): v for k, v in raw["results"].items()}
        write_markdown(Path(args.out), raw["environment"], res,
                       raw["micro_seconds_per_call"], raw["brent_fallback_share"], args)
        print(f"re-rendered {args.out} from results.json")
        return

    sizes = [int(s) for s in args.sizes.split(",")]
    py_repeats = [int(s) for s in args.python_repeats.split(",")]
    if len(py_repeats) != len(sizes):
        sys.exit("--python-repeats needs one value per size")

    env = environment()
    print(json.dumps(env, indent=2), flush=True)
    if not iv_solver.CPP_AVAILABLE:
        print("WARNING: C++ extension not built; cpp rows will be skipped", flush=True)

    backends = ["python", "numpy"] + (["cpp"] if iv_solver.CPP_AVAILABLE else [])
    results: dict = {}
    for n, py_rep in zip(sizes, py_repeats):
        df = generate_chain(n, rate=RATE)
        arrays = (np.ascontiguousarray(df["mid"].to_numpy()),
                  np.ascontiguousarray(df["underlying_price"].to_numpy()),
                  np.ascontiguousarray(df["strike"].to_numpy()),
                  np.ascontiguousarray(df["time_to_expiry"].to_numpy()),
                  RATE,
                  (df["option_type"] == "C").to_numpy())
        results[n] = {}
        for b in backends:
            rep = py_rep if b == "python" else args.repeats
            warm = 1 if b == "python" else 3
            if rep <= 0:
                continue
            times = time_runs(lambda b=b: iv_solver.implied_vol_array(*arrays, backend=b),
                              warm, rep)
            results[n][b] = summarize(times, n)
            r = results[n][b]
            print(f"n={n:>7,} {b:>6}: median {fmt_time(r['median_s'])}  "
                  f"p95 {fmt_time(r['p95_s'])}  {fmt_rate(r['throughput'])}  "
                  f"({rep} runs)", flush=True)

    # Per-call overhead microbenchmarks: where the per-row time goes.
    micro = {
        "scipy.stats.norm.cdf(float)": timeit.timeit(lambda: norm.cdf(0.3), number=2_000) / 2_000,
        "math.erfc based cdf": timeit.timeit(
            lambda: 0.5 * math.erfc(-0.3 / math.sqrt(2.0)), number=200_000) / 200_000,
    }
    if iv_solver.CPP_AVAILABLE:
        core = iv_solver._iv_core
        micro["C++ norm_cdf via pybind11 (one call)"] = timeit.timeit(
            lambda: core.norm_cdf(0.3), number=200_000) / 200_000
        p = iv_solver.black_scholes(100, 100, 0.5, RATE, 0.2, "C")
        micro["C++ implied_vol via pybind11 (one ATM contract)"] = timeit.timeit(
            lambda: core.implied_vol(p, 100.0, 100.0, 0.5, RATE, True), number=50_000) / 50_000

    # Fallback share in the chain (same for every size since each underlying
    # has the same layout).
    fallback_share = None
    if iv_solver.CPP_AVAILABLE:
        n0 = sizes[-1]
        df = generate_chain(n0, rate=RATE)
        status = np.empty(n0, dtype=np.int8)
        iv_solver._iv_core.implied_vol_batch_status(
            df["underlying_price"].to_numpy(), df["strike"].to_numpy(),
            df["time_to_expiry"].to_numpy(), df["mid"].to_numpy(),
            (df["option_type"] == "C").to_numpy(), RATE, np.empty(n0), status)
        fallback_share = float(np.mean(status == 1))

    raw = {"environment": env, "results": {str(k): v for k, v in results.items()},
           "micro_seconds_per_call": micro, "brent_fallback_share": fallback_share}
    Path(args.out).with_suffix(".json").write_text(json.dumps(raw, indent=2))
    write_markdown(Path(args.out), env, results, micro, fallback_share, args)
    print(f"wrote {args.out}")


def write_markdown(path: Path, env: dict, results: dict, micro: dict,
                   fallback_share, args) -> None:
    L = []
    L.append("# IV solver benchmark results")
    L.append("")
    L.append("Generated by `python bench/benchmark.py "
             f"--sizes {args.sizes} --repeats {args.repeats} "
             f"--python-repeats {args.python_repeats}`. Raw numbers are in `results.json`.")
    L.append("")
    L.append("## Environment")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    for k, v in env.items():
        L.append(f"| {k} | `{v}` |")
    L.append("")
    if env.get("x64_emulation"):
        L.append("**Note:** this Python is an x64 build running under Windows x64 emulation "
                 "on an ARM64 CPU, so the C++ extension is also x64 and every number below "
                 "(Python, NumPy, and C++) was measured under emulation. The ratios between "
                 "backends are the meaningful part; absolute times on native hardware will "
                 "differ.")
        L.append("")
    L.append("## Results")
    L.append("")
    L.append("Time to solve IV for the whole chain once. Lower is better. "
             "Speedup is relative to the per-row python backend (median vs median).")
    L.append("")
    L.append("| Contracts | Backend | Runs | Median | p95 | Throughput | Speedup vs python "
             "| Speedup vs numpy |")
    L.append("|---|---|---|---|---|---|---|---|")
    for n, by_b in results.items():
        py = by_b.get("python", {}).get("median_s")
        npm = by_b.get("numpy", {}).get("median_s")
        for b, r in by_b.items():
            sp = f"{py / r['median_s']:,.0f}x" if py else "n/a"
            sn = f"{npm / r['median_s']:.3g}x" if npm else "n/a"
            L.append(f"| {n:,} | {b} | {r['repeats']} | {fmt_time(r['median_s'])} | "
                     f"{fmt_time(r['p95_s'])} | {fmt_rate(r['throughput'])} | {sp} | {sn} |")
    L.append("")
    if fallback_share is not None:
        L.append(f"Share of contracts where Newton gave up and the bracketing fallback ran: "
                 f"{fallback_share:.1%} (same in every backend, same algorithm).")
        py_runs = [r["repeats"] for by_b in results.values() for b, r in by_b.items() if b == "python"]
    if py_runs and min(py_runs) < 20:
        L.append("The per-row python backend at the largest size has fewer timed runs (each "
                 "run takes several minutes), so its p95 is close to its maximum and is less "
                 "stable than the other rows.")
        L.append("")
    L.append("## Per-call overhead")
    L.append("")
    L.append("| Call | Time per call |")
    L.append("|---|---|")
    for k, v in micro.items():
        L.append(f"| {k} | {fmt_time(v)} |")
    L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
