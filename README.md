# NeuroVol

Options analytics in Python with a C++ core for implied volatility.

What is in this repository today:

* **Implied volatility engine** with three interchangeable backends that run the same
  algorithm (Brenner-Subrahmanyam initial guess, Newton-Raphson, bracketing fallback):
  * `python`: the original per-row solver (scipy `norm.cdf` and `brentq`)
  * `numpy`: vectorized Newton with a vectorized bisection fallback
  * `cpp`: a C++17 batch solver exposed through pybind11 (zero-copy NumPy input,
    GIL released, no heap allocation in the hot loop)
* **SVI calibration** of implied-variance slices with a butterfly-arbitrage check
* **Volatility features**: realized volatility estimators (close-to-close, Parkinson,
  Yang-Zhang, GARCH(1,1)) and IV term-structure features
* **Walk-forward validation** fold splitting with an embargo gap
* **Backtesting toolkit**: signal generation, vol-carry P&L simulation, Greek P&L
  attribution, transaction cost model
* **Tests** in Python (pytest) and C++ (a standalone test executable), and a
  **benchmark** with recorded results

See [`docs/architecture.md`](docs/architecture.md) for how the pieces fit and what state
each one is in.

## Benchmark summary

Solving implied volatility for a 100,000-contract synthetic chain, single thread
(median of 20 runs, except 3 runs for the per-row Python solver):

| Backend | Median time | Throughput | vs per-row Python | vs NumPy |
|---|---|---|---|---|
| `python` (original per-row) | 533.12 s | 188 contracts/s | 1x | |
| `numpy` (vectorized) | 1.69 s | 59.2k contracts/s | 316x | 1x |
| `cpp` (this repo's C++ core) | 354.08 ms | 282.4k contracts/s | 1,506x | 4.77x |

Measured on a Snapdragon X Plus laptop with x64 Python under Windows emulation (so the
C++ extension ran emulated too); the ratios are the meaningful part. Across the 100k
chain the C++ and Python IVs differ by at most 1.58e-12.

The large speedup over the per-row solver is mostly the cost of calling
`scipy.stats.norm.cdf` on one float (108.2 us per call here, versus 0.7 us for a
`math.erfc` version), not Python arithmetic itself. The fair comparison is against
vectorized NumPy: 4.8x to 5.9x depending on chain size.

Full methodology, environment, and per-call overhead numbers:
[`bench/results.md`](bench/results.md). Cross-backend accuracy:
[`bench/agreement.md`](bench/agreement.md).

## Repository layout

```
cpp/
  include/neurovol/iv_core.hpp   C++ API: pricing, vega, IV solver, batch, status codes
  src/iv_core.cpp                Implementation (noexcept, no heap, bounded loops)
  src/bindings.cpp               pybind11 module vol_surface_engine._iv_core
  tests/test_iv_core.cpp         C++ unit tests (no framework needed)
  CMakeLists.txt
vol_surface_engine/
  iv_solver.py                   Python reference, NumPy backend, backend dispatch
  svi_calibrator.py              SVI slice fitting
  synthetic_chain.py             Seeded synthetic options chain with a known smile
  schemas.py                     Dataclasses
features/                        Realized-vol estimators and IV features
backtesting/                     Signals, backtest loop, P&L attribution, costs
training/                        Walk-forward splitter, Lightning module (see status)
research/analysis/               Surface PCA, regime classification
bench/                           benchmark.py, compare_backends.py, recorded results
tests/                           pytest suite
docs/                            architecture.md, data_dictionary.md, INTERVIEW_PREP.md
notebooks/                       Exploration notebooks
airflow/dags/, kafka/            Scaffolding only (see Status below)
```

## Quick start (Python only)

Requires Python 3.11 or newer.

```bash
git clone https://github.com/Nareh-Avag/neurovol.git
cd neurovol
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Without the C++ extension the `cpp` tests are skipped and `backend="cpp"` falls back to
the NumPy backend with a warning.

```python
from vol_surface_engine.iv_solver import compute_iv_batch
from vol_surface_engine.synthetic_chain import generate_chain

chain = generate_chain(10_000)                       # deterministic, seed=12345
out = compute_iv_batch(chain, rate=0.03, backend="cpp")
print(out[["strike", "time_to_expiry", "option_type", "implied_vol", "true_iv"]].head())
```

## Building the C++ extension

The build needs CMake 3.18+, a C++17 compiler, and pybind11. All but the compiler come
from pip:

```bash
pip install -e ".[dev,cpp]"        # pybind11, cmake, ninja
```

### Linux / macOS (gcc or clang)

```bash
cmake -S cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNEUROVOL_WERROR=ON
cmake --build build
ctest --test-dir build --output-on-failure     # C++ unit tests
pytest                                         # now includes the cpp backend
```

### Windows with llvm-mingw (what the recorded results used)

Download an llvm-mingw release from https://github.com/mstorsjo/llvm-mingw/releases
(`ucrt-x86_64` for an Intel/AMD PC, `ucrt-aarch64` for a Windows on ARM PC), unzip it,
and put its `bin` folder on `PATH`. The compiler must target the same architecture as
your Python (`python -c "import sysconfig; print(sysconfig.get_platform())"`); for a
`win-amd64` Python use `x86_64-w64-mingw32-clang++`:

```powershell
cmake -S cpp -B build -G Ninja -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-clang++ -DNEUROVOL_WERROR=ON
cmake --build build
ctest --test-dir build --output-on-failure
pytest
```

The build writes `vol_surface_engine/_iv_core.<python-tag>.pyd` (or `.so`), next to
`iv_solver.py`. The C++ runtime is linked statically on MinGW so the module has no extra
DLL dependencies.

An MSVC branch exists in `cpp/CMakeLists.txt` (`/W4 /permissive- /O2`) but has not been
built or tested.

Compiler flags: `-O2 -std=c++17 -Wall -Wextra -Wpedantic`, plus `-Werror` with
`-DNEUROVOL_WERROR=ON`. The build is warning-free with clang 23 (llvm-mingw).

## Running the benchmark and accuracy check

```bash
python bench/benchmark.py            # writes bench/results.md and bench/results.json
python bench/compare_backends.py     # writes bench/agreement.md (100k contracts)
```

`benchmark.py` defaults to 20 timed runs per size for `numpy` and `cpp`. The per-row
`python` backend is slow enough that the default is 20 runs at 1k and 10k and 3 runs at
100k; change this with `--python-repeats`. The full default run took 53 minutes on
the machine in `bench/results.md`, almost all of it in the per-row Python backend.

## Status and known limitations

* **Initial guess.** Brenner-Subrahmanyam is an at-the-money approximation. For
  out-of-the-money quotes it lands near zero, vega there underflows, and Newton hands
  the contract to the bracketing fallback. On the synthetic chain that is
  39.0% of all contracts. The behavior is the same in all backends.
* **Backtester P&L is a simulation.** `BacktestEngine.run` uses the forecast's
  `rv_realized` if provided and otherwise assumes realized volatility is 92% of ATM
  implied, which builds a positive vol-carry return into the default. Use it with real
  realized-vol inputs.
* **Training** (`training/trainer.py`, `training/walk_forward.py`) imports
  `models.build_model`, `models.losses.CombinedVolLoss`, and `training.callbacks`, which
  do not exist yet. `WalkForwardValidator.get_fold_splits` works on its own, but the
  module imports `pytorch_lightning` at the top, so it needs the `training` extra.
* **airflow/dags/** are DAG skeletons. Every task body is a placeholder that only passes
  file paths between tasks; no data is fetched or written. **kafka/create_topics.sh**
  creates topics that no code produces to or consumes from.
* `features/realized_vol.py` warns at import time if the optional `arch` package is
  missing.

## Roadmap (not built)

None of the following exists in the repository yet:

* Neural surface forecasting models (ConvLSTM, transformer) and their loss functions
* Live market data ingestion and streaming (Kafka producers and consumers)
* REST API and dashboard
* Working Airflow pipelines and container deployment
* A better initial guess in the IV solver to reduce fallback calls, and multithreading
  across contracts in the C++ batch

## License

MIT. Copyright (c) 2025 Nareh Avagyan.
