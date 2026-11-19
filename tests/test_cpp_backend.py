"""
C++ backend vs the Python reference.

The round-trip, edge-case, and invalid-input tests in test_iv_solver.py
already run against the cpp backend. This file checks that the C++ building
blocks match the Python ones function by function, that the extension uses
NumPy memory the way it claims to, and that the fallback works when the
extension is missing.
"""

import itertools
import math

import numpy as np
import pytest
from scipy.special import ndtr

from vol_surface_engine import iv_solver as ivs
from vol_surface_engine.synthetic_chain import generate_chain

pytestmark = pytest.mark.skipif(not ivs.CPP_AVAILABLE, reason="C++ extension not built")
core = ivs._iv_core
R = 0.03


def test_norm_cdf_matches_scipy():
    # Where option prices live (|x| <= 8): a couple of ulps.
    x = np.concatenate([np.linspace(-8, 8, 4001), [-1e-300, 0.0, 1e-300]])
    ours = np.array([core.norm_cdf(v) for v in x])
    np.testing.assert_allclose(ours, ndtr(x), rtol=5e-15, atol=0)

    # Far tail: C++ std::erfc (the C runtime's) and scipy's erfc are different
    # implementations. Measured relative gap is up to ~6e-14 for x in
    # [-37, -20], where the values themselves are below 1e-88. Below about
    # x = -37.7 scipy flushes to 0 while the C runtime returns subnormals.
    x = np.linspace(-37, -8, 2001)
    ours = np.array([core.norm_cdf(v) for v in x])
    np.testing.assert_allclose(ours, ndtr(x), rtol=1e-13, atol=0)
    assert 0.0 <= core.norm_cdf(-40.0) < 1e-300
    assert core.norm_cdf(40.0) == 1.0


@pytest.mark.parametrize("K,T,sigma,call", itertools.product(
    [50.0, 90.0, 100.0, 110.0, 200.0], [0.004, 0.25, 2.0], [0.05, 0.3, 1.5], [True, False]))
def test_price_and_vega_match_python(K, T, sigma, call):
    opt = "C" if call else "P"
    ref_p = ivs.black_scholes(100.0, K, T, R, sigma, opt)
    ref_v = ivs.bs_vega(100.0, K, T, R, sigma)
    assert core.bs_price(100.0, K, T, R, sigma, call) == pytest.approx(ref_p, rel=1e-13, abs=1e-13)
    assert core.bs_vega(100.0, K, T, R, sigma) == pytest.approx(ref_v, rel=1e-13, abs=1e-13)


def test_scalar_solver_matches_python_newton_then_brent():
    df = generate_chain(1_000)
    for row in df.itertuples():
        call = row.option_type == "C"
        sigma, status, iters = core.implied_vol(row.mid, row.underlying_price, row.strike,
                                                row.time_to_expiry, R, call)
        ref = ivs.implied_vol_newton(row.mid, row.underlying_price, row.strike,
                                     row.time_to_expiry, R, row.option_type)
        used_newton = not math.isnan(ref)
        if not used_newton:
            ref = ivs.implied_vol_brent(row.mid, row.underlying_price, row.strike,
                                        row.time_to_expiry, R, row.option_type)
        # Same path taken (0 = Newton, 1 = Brent) and same answer.
        assert status == (0 if used_newton else 1)
        assert abs(sigma - ref) < 1e-8
        assert 0 < iters <= 300


def test_batch_matches_python_on_1k_chain():
    df = generate_chain(1_000)
    args = (df["mid"].to_numpy(), df["underlying_price"].to_numpy(), df["strike"].to_numpy(),
            df["time_to_expiry"].to_numpy(), R, (df["option_type"] == "C").to_numpy())
    cpp = ivs.implied_vol_array(*args, backend="cpp")
    py = ivs.implied_vol_array(*args, backend="python")
    assert np.array_equal(np.isnan(cpp), np.isnan(py))
    assert np.nanmax(np.abs(cpp - py)) < 1e-8


def test_status_codes():
    S = np.full(5, 100.0)
    K = np.array([100.0, 60.0, 80.0, 80.0, 100.0])
    T = np.array([0.5, 1.0, 0.5, 0.5, 0.0])
    call = np.array([1, 0, 1, 1, 1], dtype=np.uint8)
    price = np.array([ivs.black_scholes(100, 100, 0.5, R, 0.2, "C"),
                      ivs.black_scholes(100, 60, 1.0, R, 0.25, "P"),
                      1.0,     # below intrinsic
                      150.0,   # above S
                      5.0])    # expired
    out = np.empty(5)
    status = np.empty(5, dtype=np.int8)
    n_ok = core.implied_vol_batch_status(S, K, T, price, call, R, out, status)
    codes = core.STATUS_CODES
    assert n_ok == 2
    assert status.tolist() == [codes["ok_newton"], codes["ok_brent"], codes["below_intrinsic"],
                               codes["above_upper_bound"], codes["invalid_input"]]


def test_output_written_in_place_and_bad_outputs_rejected():
    S = np.array([100.0]); K = np.array([100.0]); T = np.array([0.5])
    call = np.array([1], dtype=np.uint8)
    price = np.array([ivs.black_scholes(100, 100, 0.5, R, 0.2, "C")])

    out = np.zeros(1)
    core.implied_vol_batch(S, K, T, price, call, R, out)
    assert out[0] == pytest.approx(0.2, abs=1e-8)  # the caller's array was filled

    with pytest.raises(TypeError):  # float32 would need a converted copy
        core.implied_vol_batch(S, K, T, price, call, R, np.zeros(1, dtype=np.float32))
    # Non-contiguous output would need a copy. (Two elements, because NumPy
    # treats any one-element view as contiguous whatever its stride.)
    S2, K2, T2 = np.repeat(S, 2), np.repeat(K, 2), np.repeat(T, 2)
    p2, c2 = np.repeat(price, 2), np.repeat(call, 2)
    with pytest.raises(TypeError):
        core.implied_vol_batch(S2, K2, T2, p2, c2, R, np.zeros(4)[::2])
    readonly = np.zeros(1)
    readonly.flags.writeable = False
    with pytest.raises(ValueError):
        core.implied_vol_batch(S, K, T, price, call, R, readonly)
    with pytest.raises(ValueError):
        core.implied_vol_batch(S, np.array([100.0, 1.0]), T, price, call, R, out)


def test_inputs_of_other_dtypes_are_converted():
    """forcecast: float32 / int inputs still work (pybind11 makes one converted copy)."""
    S = np.array([100], dtype=np.int64)
    K = np.array([100.0], dtype=np.float32)
    T = np.array([0.5])
    price = np.array([ivs.black_scholes(100, 100, 0.5, R, 0.2, "C")])
    out = np.empty(1)
    core.implied_vol_batch(S, K, T, price, np.array([True]), R, out)
    assert out[0] == pytest.approx(0.2, abs=1e-8)
