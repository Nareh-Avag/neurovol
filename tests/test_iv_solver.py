"""
Correctness tests for the implied volatility solvers.

Every test that solves for IV runs against each available backend:
  python  original per-row solver (Newton, then scipy Brent)
  numpy   vectorized Newton with vectorized bisection fallback
  cpp     C++ extension (skipped if it has not been built)

A note on tolerances. The Newton solver stops when the *price* error is below
1e-8, so the volatility error is roughly 1e-8 / vega. Round-trip tests that
demand 1e-6 accuracy in sigma therefore only use contracts with vega >= 0.01
(a one vol point move changes the price by at least 1e-4). Below that, IV is
numerically ill-conditioned for any solver, and the tests only check that the
solver either returns NaN or returns a sigma that reproduces the price.
"""

import itertools
import logging
import math
import time

import numpy as np
import pandas as pd
import pytest

from vol_surface_engine import iv_solver as ivs
from vol_surface_engine.iv_solver import (
    black_scholes,
    bs_vega,
    compute_iv_batch,
    implied_vol_array,
)

R = 0.03
BACKENDS = ["python", "numpy",
            pytest.param("cpp", marks=pytest.mark.skipif(
                not ivs.CPP_AVAILABLE, reason="C++ extension not built"))]


def solve(backend, price, S, K, T, is_call, r=R):
    """Solve one or more quotes with the given backend; returns an array."""
    arrs = [np.atleast_1d(np.asarray(a, dtype=float)) for a in (price, S, K, T)]
    n = max(a.size for a in arrs)
    price, S, K, T = (np.broadcast_to(a, (n,)).copy() for a in arrs)
    flags = np.broadcast_to(np.atleast_1d(np.asarray(is_call, dtype=bool)), (n,)).copy()
    return implied_vol_array(price, S, K, T, r, flags, backend=backend)


# ---------------------------------------------------------------------------
# Pricing identities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S,K,T,sigma", itertools.product(
    [50.0, 100.0], [40.0, 80.0, 100.0, 125.0, 200.0], [0.01, 0.25, 2.0], [0.05, 0.2, 0.8]))
def test_put_call_parity(S, K, T, sigma):
    c = black_scholes(S, K, T, R, sigma, "C")
    p = black_scholes(S, K, T, R, sigma, "P")
    assert c - p == pytest.approx(S - K * math.exp(-R * T), abs=1e-10 * S)


def test_vectorized_pricer_matches_scalar():
    rng = np.random.default_rng(0)
    S = rng.uniform(50, 150, 200)
    K = rng.uniform(50, 150, 200)
    T = rng.uniform(0.01, 2, 200)
    sig = rng.uniform(0.05, 1.0, 200)
    call = rng.random(200) < 0.5
    vec = ivs.black_scholes_vec(S, K, T, R, sig, call)
    ref = [black_scholes(s, k, t, R, v, "C" if c else "P")
           for s, k, t, v, c in zip(S, K, T, sig, call)]
    np.testing.assert_allclose(vec, ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Round trip: price at a known sigma, recover sigma
# ---------------------------------------------------------------------------

def _round_trip_cases():
    cases = []
    for sigma, m, T, opt in itertools.product(
            [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            [0.7, 0.9, 1.0, 1.1, 1.3],
            [0.02, 0.25, 1.0, 2.0],
            ["C", "P"]):
        S, K = 100.0, 100.0 * m
        if bs_vega(S, K, T, R, sigma) >= 0.01:
            cases.append((sigma, S, K, T, opt))
    return cases


ROUND_TRIP = _round_trip_cases()


@pytest.mark.parametrize("backend", BACKENDS)
def test_round_trip_grid(backend):
    assert len(ROUND_TRIP) > 150  # make sure the vega filter did not empty the grid
    sig = np.array([c[0] for c in ROUND_TRIP])
    S = np.array([c[1] for c in ROUND_TRIP])
    K = np.array([c[2] for c in ROUND_TRIP])
    T = np.array([c[3] for c in ROUND_TRIP])
    call = np.array([c[4] == "C" for c in ROUND_TRIP])
    price = ivs.black_scholes_vec(S, K, T, R, sig, call)
    iv = solve(backend, price, S, K, T, call)
    assert np.all(np.isfinite(iv))
    np.testing.assert_allclose(iv, sig, rtol=0, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("S,K,T,sigma,opt", [
    (100.0, 50.0, 0.5, 0.3, "C"),    # deep ITM call
    (100.0, 200.0, 0.5, 0.3, "C"),   # deep OTM call
    (100.0, 200.0, 0.5, 0.3, "P"),   # deep ITM put
    (100.0, 50.0, 0.5, 0.3, "P"),    # deep OTM put
    (100.0, 60.0, 1.0, 0.25, "P"),   # OTM put, the case Newton's ATM guess handles badly
])
def test_deep_itm_otm_round_trip(backend, S, K, T, sigma, opt):
    assert bs_vega(S, K, T, R, sigma) >= 0.01
    price = black_scholes(S, K, T, R, sigma, opt)
    iv = solve(backend, price, S, K, T, opt == "C")[0]
    assert iv == pytest.approx(sigma, abs=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("S,K,T,sigma,opt", [
    (100.0, 20.0, 0.1, 0.2, "C"),    # time value underflows: price == intrinsic in float64
    (100.0, 500.0, 0.05, 0.2, "C"),  # price underflows to ~0
    (100.0, 70.0, 0.01, 0.1, "P"),   # tiny time value
])
def test_ill_conditioned_is_nan_or_reprices(backend, S, K, T, sigma, opt):
    price = black_scholes(S, K, T, R, sigma, opt)
    iv = solve(backend, price, S, K, T, opt == "C")[0]
    if not math.isnan(iv):
        assert 1e-6 <= iv <= 20.0
        assert black_scholes(S, K, T, R, iv, opt) == pytest.approx(price, abs=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("T", [1.0 / 252.0, 1.0 / (252.0 * 6.5), 1e-5])  # 1 day, 1 hour, ~5 min
@pytest.mark.parametrize("opt", ["C", "P"])
def test_very_short_expiry_atm(backend, T, opt):
    S, K, sigma = 100.0, 100.0, 0.25
    price = black_scholes(S, K, T, R, sigma, opt)
    iv = solve(backend, price, S, K, T, opt == "C")[0]
    assert iv == pytest.approx(sigma, abs=1e-6)


# ---------------------------------------------------------------------------
# Invalid prices and inputs: NaN, quickly, never an exception
# ---------------------------------------------------------------------------

INVALID = [
    # price, S, K, T, is_call, description
    (15.0, 100.0, 80.0, 0.5, True, "call below intrinsic"),
    (15.0, 80.0, 100.0, 0.5, False, "put below intrinsic"),
    (100.5, 100.0, 80.0, 0.5, True, "call above upper bound S"),
    (120.0, 100.0, 110.0, 0.5, False, "put above upper bound K*exp(-rT)"),
    (0.0, 100.0, 100.0, 0.5, True, "zero price"),
    (-1.0, 100.0, 100.0, 0.5, True, "negative price"),
    (5.0, 0.0, 100.0, 0.5, True, "zero spot"),
    (5.0, 100.0, -1.0, 0.5, True, "negative strike"),
    (5.0, 100.0, 100.0, 0.0, True, "expired"),
    (5.0, 100.0, 100.0, -0.1, False, "negative time"),
    (float("nan"), 100.0, 100.0, 0.5, True, "NaN price"),
    (5.0, float("nan"), 100.0, 0.5, True, "NaN spot"),
    (float("inf"), 100.0, 100.0, 0.5, True, "inf price"),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("price,S,K,T,is_call,desc", INVALID, ids=[c[-1] for c in INVALID])
def test_invalid_inputs_return_nan(backend, price, S, K, T, is_call, desc):
    t0 = time.perf_counter()
    iv = solve(backend, price, S, K, T, is_call)[0]
    elapsed = time.perf_counter() - t0
    assert math.isnan(iv), desc
    assert elapsed < 2.0, f"{desc}: took {elapsed:.2f}s"


@pytest.mark.parametrize("backend", BACKENDS)
def test_invalid_rows_do_not_affect_valid_rows(backend):
    price = np.array([c[0] for c in INVALID] + [black_scholes(100, 100, 0.5, R, 0.2, "C")])
    S = np.array([c[1] for c in INVALID] + [100.0])
    K = np.array([c[2] for c in INVALID] + [100.0])
    T = np.array([c[3] for c in INVALID] + [0.5])
    call = np.array([c[4] for c in INVALID] + [True])
    iv = solve(backend, price, S, K, T, call)
    assert np.all(np.isnan(iv[:-1]))
    assert iv[-1] == pytest.approx(0.2, abs=1e-6)


def test_python_scalar_solvers_return_nan_not_raise():
    for price, S, K, T, is_call, _ in INVALID:
        opt = "C" if is_call else "P"
        assert math.isnan(ivs.implied_vol_newton(price, S, K, T, R, opt))
        assert math.isnan(ivs.implied_vol_brent(price, S, K, T, R, opt))


# ---------------------------------------------------------------------------
# compute_iv_batch (DataFrame API)
# ---------------------------------------------------------------------------

def _quotes():
    rows = []
    for K, opt in itertools.product([80.0, 100.0, 120.0], ["C", "P"]):
        rows.append(dict(underlying_price=100.0, strike=K, time_to_expiry=0.25,
                         option_type=opt, mid=black_scholes(100.0, K, 0.25, R, 0.3, opt)))
    rows.append(dict(underlying_price=100.0, strike=80.0, time_to_expiry=0.25,
                     option_type="C", mid=1.0))  # below intrinsic
    return pd.DataFrame(rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_compute_iv_batch_columns_and_flags(backend):
    out = compute_iv_batch(_quotes(), R, backend=backend)
    for col in ["implied_vol", "delta", "gamma", "vega", "theta", "log_moneyness", "iv_valid"]:
        assert col in out.columns
    assert out["iv_valid"].tolist() == [True] * 6 + [False]
    np.testing.assert_allclose(out["implied_vol"].iloc[:6], 0.3, atol=1e-6)
    assert out.iloc[6][["implied_vol", "delta", "gamma", "vega", "theta"]].isna().all()


def test_compute_iv_batch_greeks_match_across_backends():
    ref = compute_iv_batch(_quotes(), R, backend="python")
    vec = compute_iv_batch(_quotes(), R, backend="numpy")
    for col in ["implied_vol", "delta", "gamma", "vega", "theta", "log_moneyness"]:
        np.testing.assert_allclose(vec[col], ref[col], rtol=1e-7, atol=1e-9, equal_nan=True)


def test_compute_iv_batch_missing_column():
    with pytest.raises(ValueError):
        compute_iv_batch(_quotes().drop(columns=["mid"]), R)


def test_unknown_backend():
    with pytest.raises(ValueError):
        compute_iv_batch(_quotes(), R, backend="fortran")


def test_falls_back_to_numpy_when_extension_missing(monkeypatch, caplog):
    from vol_surface_engine.synthetic_chain import generate_chain
    df = generate_chain(1_000)
    args = (df["mid"].to_numpy(), df["underlying_price"].to_numpy(), df["strike"].to_numpy(),
            df["time_to_expiry"].to_numpy(), R, (df["option_type"] == "C").to_numpy())
    monkeypatch.setattr(ivs, "CPP_AVAILABLE", False)
    with caplog.at_level(logging.WARNING, logger=ivs.logger.name):
        out = ivs.implied_vol_array(*args, backend="cpp")
    assert "falling back" in caplog.text
    np.testing.assert_array_equal(out, ivs.implied_vol_array(*args, backend="numpy"))
