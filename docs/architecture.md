# Architecture

This describes the code that exists in the repository today. Planned pieces
are listed in the README under "Roadmap" and are not described here as if
they were built.

## Implied volatility engine

```
pandas DataFrame (one row per option quote)
  underlying_price, strike, time_to_expiry, option_type, mid
        |
        v
compute_iv_batch(df, rate, backend=...)          vol_surface_engine/iv_solver.py
        |
        |  backend="python"  per-row loop: implied_vol_newton, then implied_vol_brent
        |  backend="numpy"   implied_vol_vectorized: array Newton + array bisection
        |  backend="cpp"     implied_vol_array -> _iv_core.implied_vol_batch
        |                     (falls back to "numpy" if the extension is not built)
        v
DataFrame + implied_vol, delta, gamma, vega, theta, log_moneyness, iv_valid
```

### C++ path in detail

```
Python                                   C++ (cpp/)
------                                   ----------
NumPy arrays (float64, C-contiguous)
  price, S, K, T, is_call (bool -> uint8 view, no copy)
out = np.empty(n)  (caller allocates)
        |
        |  pybind11 (cpp/src/bindings.cpp)
        |   - inputs: py::array_t<double, c_style | forcecast>  -> view, no copy
        |   - out: c_style + noconvert -> must be the caller's buffer
        |   - shape checks, then raw pointers, then release the GIL
        v
neurovol::implied_vol_batch(S, K, T, price, is_call, n, r, cfg, out, status)
        |   for i in 0..n-1:      (cpp/src/iv_core.cpp, noexcept, no heap)
        |     implied_vol(...)
        |       1. validate inputs (finite, > 0)
        |       2. no-arbitrage bounds (intrinsic <= price < upper)
        |       3. Newton from the Brenner-Subrahmanyam guess, <= 100 iterations
        |       4. Brent on [1e-6, 20] if Newton gives up, <= 200 iterations
        |     out[i] = sigma or NaN; status[i] = int8 code
        v
out filled in place, number of valid IVs returned
```

The same algorithm (guess, Newton update, clipping, tolerances, bracket) is
used by all three backends, so they agree to within the tolerances measured
in `bench/agreement.md`.

## Other modules

| Module | What it does | Status |
|---|---|---|
| `vol_surface_engine/svi_calibrator.py` | Fits SVI total-variance slices per expiry with L-BFGS-B and random restarts; butterfly-arbitrage check | Works on synthetic input |
| `vol_surface_engine/schemas.py` | Dataclasses for quotes, SVI params, surfaces | Used by SVI |
| `vol_surface_engine/synthetic_chain.py` | Seeded synthetic chain with a known smile | Used by tests and benchmarks |
| `features/realized_vol.py` | Close-to-close, Parkinson, Yang-Zhang, GARCH(1,1) (needs `arch`) | Works |
| `features/iv_features.py` | ATM IV, IV rank/percentile, VRP, term-structure features | Works |
| `backtesting/` | Signal generator, vol-carry P&L simulation, Greek P&L attribution, cost model | Runs on dict inputs; see README caveats |
| `training/walk_forward.py` | Walk-forward fold splitting with an embargo gap | Fold splitting works; training a fold needs model code that does not exist yet |
| `training/trainer.py`, `training/datamodule.py` | Lightning module and datamodule | Depend on `models.build_model`, not implemented |
| `research/analysis/` | Surface PCA, regime classification | Need scikit-learn and plotly |
| `airflow/dags/` | DAG definitions | Skeletons only: task bodies are placeholders |
| `kafka/create_topics.sh` | Creates Kafka topics | Nothing produces to or consumes from them |
