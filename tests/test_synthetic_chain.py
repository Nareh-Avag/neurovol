import numpy as np
import pandas as pd
import pytest

from vol_surface_engine.iv_solver import implied_vol_array
from vol_surface_engine.synthetic_chain import generate_chain


@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
def test_sizes_and_sanity(n):
    df = generate_chain(n)
    assert len(df) == n
    assert set(df["option_type"]) == {"C", "P"}
    assert (df["option_type"] == "C").sum() == n // 2
    assert np.all(np.isfinite(df[["underlying_price", "strike", "time_to_expiry", "mid"]]))
    assert (df["mid"] > 0).all()
    assert df["time_to_expiry"].nunique() == 20
    assert df["true_iv"].between(0.05, 1.5).all()


def test_deterministic():
    pd.testing.assert_frame_equal(generate_chain(10_000), generate_chain(10_000))
    assert not generate_chain(10_000, seed=1)["mid"].equals(generate_chain(10_000, seed=2)["mid"])


def test_has_a_smile():
    """Within one expiry, wing IVs are higher than the ATM IV."""
    df = generate_chain(1_000)
    one = df[(df.time_to_expiry == df.time_to_expiry.unique()[8]) & (df.option_type == "C")]
    iv = one["true_iv"].to_numpy()
    assert iv[0] > iv[len(iv) // 2]  # low strike (put wing) above ATM


def test_rejects_bad_sizes():
    for n in (0, 999, 1500):
        with pytest.raises(ValueError):
            generate_chain(n)


def test_round_trip_1k_chain_python_backend():
    df = generate_chain(1_000)
    iv = implied_vol_array(df["mid"].to_numpy(), df["underlying_price"].to_numpy(),
                           df["strike"].to_numpy(), df["time_to_expiry"].to_numpy(),
                           0.03, (df["option_type"] == "C").to_numpy(), backend="python")
    assert np.all(np.isfinite(iv))
    np.testing.assert_allclose(iv, df["true_iv"], atol=1e-6, rtol=0)
