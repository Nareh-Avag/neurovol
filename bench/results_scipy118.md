# IV solver benchmark results

Generated in a fresh virtualenv (NumPy 2.5.5, SciPy 1.18.1) by `python bench/benchmark.py --sizes 1000,10000,100000 --repeats 20 --python-repeats 0,0,0`. Raw numbers are in `results_scipy118.json`.

## Environment

| | |
|---|---|
| cpu | `Snapdragon(R) X Plus - X1P42100 - Qualcomm(R) Oryon(TM) CPU` |
| logical_cpus | `8` |
| os | `Windows-11-10.0.26300-SP0` |
| host_arch | `ARM64` |
| python | `3.14.2 (win-amd64)` |
| x64_emulation | `True` |
| numpy | `2.5.5` |
| scipy | `1.18.1` |
| cpp_available | `True` |
| cpp_compiler | `clang 23.1.6 (https://github.com/llvm/llvm-project.git 5b10d2d74929eae78912771222c9855af97496f4)` |
| cpp_build_type | `Release` |
| cpp_flags | `-O2 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic -Werror` |

**Note:** this Python is an x64 build running under Windows x64 emulation on an ARM64 CPU, so the C++ extension is also x64 and every number below (Python, NumPy, and C++) was measured under emulation. The ratios between backends are the meaningful part; absolute times on native hardware will differ.

## Results

Time to solve IV for the whole chain once. Lower is better. Speedup is relative to the per-row python backend (median vs median).

| Contracts | Backend | Runs | Median | p95 | Throughput | Speedup vs python | Speedup vs numpy |
|---|---|---|---|---|---|---|---|
| 1,000 | numpy | 20 | 5.91 ms | 6.73 ms | 169.3 k/s | n/a | 1x |
| 1,000 | cpp | 20 | 3.52 ms | 4.16 ms | 284.1 k/s | n/a | 1.68x |
| 10,000 | numpy | 20 | 31.39 ms | 33.45 ms | 318.6 k/s | n/a | 1x |
| 10,000 | cpp | 20 | 35.81 ms | 38.10 ms | 279.3 k/s | n/a | 0.877x |
| 100,000 | numpy | 20 | 315.69 ms | 406.09 ms | 316.8 k/s | n/a | 1x |
| 100,000 | cpp | 20 | 372.44 ms | 480.56 ms | 268.5 k/s | n/a | 0.848x |

Share of contracts where Newton gave up and the bracketing fallback ran: 39.0% (same in every backend, same algorithm).
## Per-call overhead

| Call | Time per call |
|---|---|
| scipy.stats.norm.cdf(float) | 120.0 us |
| math.erfc based cdf | 0.7 us |
| C++ norm_cdf via pybind11 (one call) | 0.3 us |
| C++ implied_vol via pybind11 (one ATM contract) | 1.6 us |

