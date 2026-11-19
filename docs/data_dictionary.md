# Data Dictionary — Neural Volatility Surface Forecasting System

This document defines all data structures used across the pipeline: raw
options quotes, the IV surface tensor, the full feature vector, and SVI
model parameters.

---

## 1. `options_quotes` Table

Stored in PostgreSQL (`vol_surface` schema). One row per option contract snapshot.

| Column | Type | Unit | Description |
|---|---|---|---|
| `id` | BIGSERIAL | — | Auto-increment primary key |
| `snapshot_ts` | TIMESTAMPTZ | — | UTC timestamp of the market data snapshot |
| `trade_date` | DATE | — | Business date of the snapshot (ET) |
| `underlying` | VARCHAR(10) | — | Ticker of the underlying index/stock (e.g. `SPX`) |
| `expiry_date` | DATE | — | Option expiry / expiration date |
| `dte` | SMALLINT | calendar days | Days to expiry at snapshot time |
| `strike` | NUMERIC(10,2) | USD | Option strike price |
| `option_type` | CHAR(1) | — | `C` = call, `P` = put |
| `bid` | NUMERIC(10,4) | USD | Best bid price |
| `ask` | NUMERIC(10,4) | USD | Best offer price |
| `mid` | NUMERIC(10,4) | USD | Mid-market price `(bid + ask) / 2` |
| `last` | NUMERIC(10,4) | USD | Last trade price |
| `volume` | INTEGER | contracts | Daily traded volume |
| `open_interest` | INTEGER | contracts | Open interest at snapshot |
| `underlying_price` | NUMERIC(12,4) | USD | Underlying spot price at snapshot |
| `forward_price` | NUMERIC(12,4) | USD | Dividend-adjusted forward price |
| `risk_free_rate` | NUMERIC(8,6) | decimal | Continuously compounded risk-free rate |
| `dividend_yield` | NUMERIC(8,6) | decimal | Continuously compounded dividend yield |
| `implied_vol` | NUMERIC(8,6) | decimal | Black-Scholes implied volatility (Newton solver) |
| `iv_bid` | NUMERIC(8,6) | decimal | IV computed from bid price |
| `iv_ask` | NUMERIC(8,6) | decimal | IV computed from ask price |
| `delta` | NUMERIC(8,6) | — | Black-Scholes delta ∂V/∂S |
| `gamma` | NUMERIC(10,8) | 1/USD | Black-Scholes gamma ∂²V/∂S² |
| `vega` | NUMERIC(10,6) | USD/vol-pt | Black-Scholes vega ∂V/∂σ (per 1% move in IV) |
| `theta` | NUMERIC(10,6) | USD/day | Black-Scholes theta ∂V/∂t (per calendar day) |
| `rho` | NUMERIC(8,6) | USD/% | Black-Scholes rho ∂V/∂r |
| `vanna` | NUMERIC(10,8) | — | ∂²V/∂S∂σ (delta sensitivity to IV) |
| `volga` | NUMERIC(10,8) | — | ∂²V/∂σ² (vega convexity) |
| `bid_ask_spread` | NUMERIC(8,4) | USD | `ask - bid` |
| `moneyness_ln` | NUMERIC(8,6) | — | `ln(K/F)` where F is forward price |
| `iv_filter_flag` | BOOLEAN | — | TRUE if contract excluded from surface fitting |
| `iv_filter_reason` | VARCHAR(64) | — | Reason for exclusion (e.g. `zero_bid`, `crossed_market`) |
| `data_source` | VARCHAR(32) | — | Data vendor identifier |

**Indexes**: `(trade_date, underlying)`, `(underlying, expiry_date, strike, option_type)`

---

## 2. IV Surface Tensor Channels

The surface tensor has shape `(C, H, W)` where:
- `C` = number of channels (see below)
- `H` = 15 expiry nodes (fixed DTE grid: 7, 14, 21, 30, 45, 60, 90, 120, 150, 180, 210, 252, 300, 360, 504 days)
- `W` = 25 moneyness nodes (fixed log-moneyness grid: −0.30 to +0.30 in 0.025 steps)

| Channel Index | Name | Description |
|---|---|---|
| 0 | `iv_mid` | Mid-market Black-Scholes implied volatility |
| 1 | `iv_bid` | Bid-side implied volatility |
| 2 | `iv_ask` | Ask-side implied volatility |
| 3 | `iv_spread` | `iv_ask - iv_bid` — bid-ask spread in vol units |
| 4 | `delta` | Black-Scholes delta at each grid node |
| 5 | `vega_density` | Vega-weighted open interest density (normalised) |
| 6 | `svi_fit` | SVI model-fitted IV at each grid node |
| 7 | `svi_residual` | `iv_mid - svi_fit` — fitting residual |

Surface tensors are stored as `.npz` files with arrays cast to `float32`.

---

## 3. Feature Vector Fields

The feature vector contains **62 scalar features** computed daily from the surface and its
30-day rolling history.  All realised-vol quantities use 252-day annualisation.

### 3a. Level Features (indices 0–9)

| Index | Name | Unit | Description |
|---|---|---|---|
| 0 | `atm_iv_1m` | decimal | ATM IV interpolated at 30-DTE |
| 1 | `atm_iv_3m` | decimal | ATM IV interpolated at 91-DTE |
| 2 | `atm_iv_6m` | decimal | ATM IV interpolated at 182-DTE |
| 3 | `atm_iv_1y` | decimal | ATM IV interpolated at 365-DTE |
| 4 | `rv_5d` | decimal | 5-day close-to-close realised volatility |
| 5 | `rv_21d` | decimal | 21-day close-to-close realised volatility |
| 6 | `rv_63d` | decimal | 63-day close-to-close realised volatility |
| 7 | `vix_spot` | decimal | VIX index level (if available) |
| 8 | `iv_rv_ratio_1m` | — | `atm_iv_1m / rv_21d` — vol risk premium proxy |
| 9 | `vol_of_vol_21d` | decimal | 21-day rolling std dev of daily IV changes |

### 3b. Term-Structure Features (indices 10–19)

| Index | Name | Unit | Description |
|---|---|---|---|
| 10 | `ts_slope_1m3m` | decimal | `atm_iv_3m - atm_iv_1m` — front slope |
| 11 | `ts_slope_3m6m` | decimal | `atm_iv_6m - atm_iv_3m` — back slope |
| 12 | `ts_slope_6m1y` | decimal | `atm_iv_1y - atm_iv_6m` — long-end slope |
| 13 | `ts_curvature` | decimal | `2×atm_iv_3m - atm_iv_1m - atm_iv_6m` |
| 14 | `ts_inversion_flag` | binary | 1 if `atm_iv_1m > atm_iv_3m` |
| 15 | `ts_percentile_1m` | — | Percentile rank of `atm_iv_1m` over trailing 252 days |
| 16 | `ts_percentile_3m` | — | Percentile rank of `atm_iv_3m` over trailing 252 days |
| 17 | `contango_depth_1m` | decimal | `atm_iv_3m - atm_iv_1m` scaled by `atm_iv_1m` |
| 18 | `ts_z_score_1m` | — | Z-score of `atm_iv_1m` vs 63-day rolling mean/std |
| 19 | `ts_z_score_3m` | — | Z-score of `atm_iv_3m` vs 63-day rolling mean/std |

### 3c. Skew Features (indices 20–31)

| Index | Name | Unit | Description |
|---|---|---|---|
| 20 | `skew_25d_1m` | decimal | 25Δ put IV − 25Δ call IV at 30-DTE |
| 21 | `skew_25d_3m` | decimal | 25Δ put IV − 25Δ call IV at 91-DTE |
| 22 | `skew_10d_1m` | decimal | 10Δ put IV − 10Δ call IV at 30-DTE |
| 23 | `rr_25d_1m` | decimal | 25Δ risk-reversal at 30-DTE |
| 24 | `rr_25d_3m` | decimal | 25Δ risk-reversal at 91-DTE |
| 25 | `bf_25d_1m` | decimal | 25Δ butterfly at 30-DTE |
| 26 | `bf_25d_3m` | decimal | 25Δ butterfly at 91-DTE |
| 27 | `skew_slope_ratio` | — | `skew_25d_3m / skew_25d_1m` |
| 28 | `skew_z_score_1m` | — | Z-score of `skew_25d_1m` vs 63-day rolling stats |
| 29 | `skew_z_score_3m` | — | Z-score of `skew_25d_3m` vs 63-day rolling stats |
| 30 | `skew_percentile_1m` | — | Percentile rank of `skew_25d_1m` over 252 days |
| 31 | `put_call_skew_diff` | decimal | Change in `skew_25d_1m` over past 5 days |

### 3d. SVI Parameter Features (indices 32–43)

| Index | Name | Unit | Description |
|---|---|---|---|
| 32 | `svi_a_1m` | var | SVI level parameter `a` for 30-DTE slice |
| 33 | `svi_b_1m` | var | SVI slope parameter `b` for 30-DTE slice |
| 34 | `svi_rho_1m` | — | SVI correlation parameter `ρ` for 30-DTE slice |
| 35 | `svi_m_1m` | — | SVI ATM offset `m` for 30-DTE slice |
| 36 | `svi_sigma_1m` | — | SVI smile width `σ` for 30-DTE slice |
| 37 | `svi_a_3m` | var | SVI level parameter `a` for 91-DTE slice |
| 38 | `svi_b_3m` | var | SVI slope parameter `b` for 91-DTE slice |
| 39 | `svi_rho_3m` | — | SVI correlation parameter `ρ` for 91-DTE slice |
| 40 | `svi_m_3m` | — | SVI ATM offset `m` for 91-DTE slice |
| 41 | `svi_sigma_3m` | — | SVI smile width `σ` for 91-DTE slice |
| 42 | `svi_fit_rmse_1m` | vol-pt | RMSE of SVI fit vs market quotes at 30-DTE |
| 43 | `svi_fit_rmse_3m` | vol-pt | RMSE of SVI fit vs market quotes at 91-DTE |

### 3e. PCA Score Features (indices 44–53)

| Index | Name | Unit | Description |
|---|---|---|---|
| 44 | `pc1_score` | — | PC1 projection (level factor) |
| 45 | `pc2_score` | — | PC2 projection (skew factor) |
| 46 | `pc3_score` | — | PC3 projection (curvature factor) |
| 47 | `pc4_score` | — | PC4 projection |
| 48 | `pc5_score` | — | PC5 projection |
| 49 | `pc1_zscore` | — | Z-score of `pc1_score` vs 63-day rolling stats |
| 50 | `pc2_zscore` | — | Z-score of `pc2_score` vs 63-day rolling stats |
| 51 | `pc3_zscore` | — | Z-score of `pc3_score` vs 63-day rolling stats |
| 52 | `pc1_delta_1d` | — | 1-day change in `pc1_score` |
| 53 | `pc2_delta_1d` | — | 1-day change in `pc2_score` |

### 3f. Regime & Market Microstructure Features (indices 54–61)

| Index | Name | Unit | Description |
|---|---|---|---|
| 54 | `regime_label` | int | Current market regime (0=low_vol, 1=normal, 2=elevated, 3=crisis) |
| 55 | `regime_prob_0` | — | HMM posterior probability of regime 0 |
| 56 | `regime_prob_1` | — | HMM posterior probability of regime 1 |
| 57 | `regime_prob_2` | — | HMM posterior probability of regime 2 |
| 58 | `regime_prob_3` | — | HMM posterior probability of regime 3 |
| 59 | `avg_bid_ask_spread` | vol-pt | Average bid-ask spread across ATM options |
| 60 | `total_open_interest` | contracts | Total OI summed across all options in chain |
| 61 | `put_call_oi_ratio` | — | Total put OI / total call OI |

---

## 4. SVI Parameter Definitions

The Stochastic Volatility Inspired (SVI) model (Gatheral 2004) parameterises the
total implied variance `w(k) = σ²(k) × T` as a function of log-moneyness `k = ln(K/F)`:

```
w(k; a, b, ρ, m, σ) = a + b × (ρ × (k − m) + sqrt((k − m)² + σ²))
```

| Parameter | Symbol | Domain | Description |
|---|---|---|---|
| Level | `a` | `a ≥ 0` | Overall level of implied variance; vertical shift of the smile. As `a` increases the entire smile shifts up uniformly. |
| Slope | `b` | `b ≥ 0` | Controls the rate at which the wings of the smile grow with log-moneyness. A higher `b` produces more pronounced OTM wings. |
| Correlation | `ρ` | `−1 < ρ < 1` | Skewness parameter. Negative `ρ` (typical for equity indices) tilts the smile so that puts are more expensive than calls. |
| ATM offset | `m` | `m ∈ ℝ` | Horizontal shift of the minimum-variance point relative to `k = 0`. Usually close to 0 for liquid underlyings. |
| Smile width | `σ` | `σ > 0` | Controls the curvature (convexity) of the smile near the money. Smaller `σ` gives a sharper smile. |

**Arbitrage-free constraints (Lee 2004 / Gatheral–Jacquier SSVI)**:
- `b × (1 + |ρ|) < 4 / T`  (no calendar arbitrage when combined with SSVI)
- `4σ² ≥ b²(1 − ρ²)`  (no butterfly arbitrage within a single slice)

**Conversion to implied vol**: `σ_implied(k) = sqrt(w(k) / T)` where `T` is time to
expiry in years.

**Fitting**: Parameters are estimated by minimising the weighted sum of squared
differences between SVI-implied vols and mid-market quotes, with weights proportional
to open interest and inversely proportional to bid-ask spread.
