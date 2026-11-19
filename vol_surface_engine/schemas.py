from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np


@dataclass
class OptionQuote:
    underlying: str
    expiry: date
    strike: float
    option_type: str  # 'C' or 'P'
    bid: float
    ask: float
    volume: int
    open_interest: int
    underlying_price: float
    timestamp: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def moneyness(self) -> float:
        return self.strike / self.underlying_price

    @property
    def days_to_expiry(self) -> int:
        return max((self.expiry - self.timestamp.date()).days, 0)

    @property
    def time_to_expiry(self) -> float:
        return self.days_to_expiry / 252.0


@dataclass
class IVQuote:
    quote: OptionQuote
    implied_vol: float
    delta: float
    gamma: float
    vega: float
    theta: float
    log_moneyness: float
    is_valid: bool = True
    confidence: float = 1.0


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    expiry_days: int

    def total_variance(self, k: float) -> float:
        """Evaluate SVI total variance w(k) at log-moneyness k."""
        return self.a + self.b * (
            self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma ** 2)
        )

    def implied_vol(self, k: float, T: float) -> float:
        """Implied volatility from SVI at log-moneyness k and time T (years)."""
        w = self.total_variance(k)
        return np.sqrt(max(w / T, 0.0001))


@dataclass
class VolSurface:
    underlying: str
    timestamp: datetime
    spot_price: float
    forward_prices: np.ndarray   # shape (W,) one per expiry
    strike_grid: np.ndarray      # shape (H,) as moneyness K/S
    expiry_grid: np.ndarray      # shape (W,) in calendar days
    iv_surface: np.ndarray       # shape (H, W)
    delta_surface: np.ndarray    # shape (H, W)
    vega_surface: np.ndarray     # shape (H, W)
    local_vol_surface: Optional[np.ndarray] = None   # shape (H, W)
    bid_ask_surface: Optional[np.ndarray] = None     # shape (H, W)
    confidence_mask: Optional[np.ndarray] = None     # shape (H, W)
    svi_params: Optional[dict] = None               # {expiry_days: SVIParams}

    @property
    def atm_iv_by_expiry(self) -> np.ndarray:
        atm_idx = np.argmin(np.abs(self.strike_grid - 1.0))
        return self.iv_surface[atm_idx, :]

    @property
    def skew_25d_approx(self) -> np.ndarray:
        idx_90 = np.argmin(np.abs(self.strike_grid - 0.90))
        idx_110 = np.argmin(np.abs(self.strike_grid - 1.10))
        return self.iv_surface[idx_90, :] - self.iv_surface[idx_110, :]

    @property
    def term_structure_slope(self) -> float:
        atm_ivs = self.atm_iv_by_expiry
        log_expiries = np.log(self.expiry_grid.astype(float))
        valid = atm_ivs > 0
        if valid.sum() < 2:
            return 0.0
        coeffs = np.polyfit(log_expiries[valid], atm_ivs[valid], 1)
        return float(coeffs[0])

    def to_tensor(self) -> np.ndarray:
        """Stack surface channels into (H, W, 6) tensor."""
        channels = []
        channels.append(self.iv_surface)
        channels.append(
            self.local_vol_surface if self.local_vol_surface is not None
            else self.iv_surface
        )
        channels.append(self.delta_surface)
        channels.append(self.vega_surface)
        channels.append(
            self.confidence_mask if self.confidence_mask is not None
            else np.ones_like(self.iv_surface)
        )
        channels.append(
            self.bid_ask_surface if self.bid_ask_surface is not None
            else np.zeros_like(self.iv_surface)
        )
        return np.stack(channels, axis=-1)  # (H, W, 6)


@dataclass
class SurfaceForecast:
    underlying: str
    forecast_time: datetime
    horizon_days: list
    iv_surfaces: dict            # {horizon_days: np.ndarray (H, W)}
    inversion_probs: dict        # {horizon_days: float}
    skew_forecast: dict          # {horizon_days: np.ndarray}
    uncertainty_lower: Optional[dict] = None
    uncertainty_upper: Optional[dict] = None
    model_version: str = "unknown"
    inference_latency_ms: float = 0.0
