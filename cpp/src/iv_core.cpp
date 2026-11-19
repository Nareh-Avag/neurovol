// iv_core.cpp
//
// Implementation notes for each design rule are inline. The short version:
// everything here works on a handful of doubles that live in registers or
// on the stack, nothing allocates, nothing throws, and every loop is capped.

#include "neurovol/iv_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace neurovol {
namespace {

// constexpr constants are read-only and computed at compile time. They are
// the only "globals" in this file, so there is no shared mutable state and
// no initialization order problem.
constexpr double kSqrt1_2 = 0.70710678118654752440;   // 1 / sqrt(2)
constexpr double kSqrt2Pi = 2.50662827463100050242;   // sqrt(2 * pi)
constexpr double kTwoPi = 6.28318530717958647693;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
// scipy.optimize.brentq's default relative tolerance: 4 * machine epsilon.
constexpr double kBrentRtol = 4.0 * std::numeric_limits<double>::epsilon();

// std::clamp would be fine too, but it is undefined if lo > hi and it says
// nothing about NaN. This version passes NaN through (both comparisons are
// false), which matches numpy.clip and lets the caller detect it.
inline double clip(double x, double lo, double hi) noexcept {
    return std::min(std::max(x, lo), hi);
}

inline IvResult fail(IvStatus status, int iterations) noexcept {
    return IvResult{kNaN, status, iterations};
}

// The pieces of Black-Scholes that pricing and vega share.
struct D1D2 {
    double d1;
    double d2;
    double sqrt_T;
};

inline D1D2 compute_d1_d2(double S, double K, double T, double r, double sigma) noexcept {
    const double sqrt_T = std::sqrt(T);
    // Same operation order as the Python reference so the two implementations
    // round the same way as often as possible.
    const double d1 = (std::log(S / K) + (r + 0.5 * (sigma * sigma)) * T) / (sigma * sqrt_T);
    return D1D2{d1, d1 - sigma * sqrt_T, sqrt_T};
}

// Brent's method, written to follow the same steps as scipy's brentq.c so the
// fallback lands on the same root as the Python reference. f(sigma) is the
// pricing error BS(sigma) - price. Everything lives in local variables; the
// function object is a small struct passed by const reference.
struct PriceError {
    double S, K, T, r, price;
    bool is_call;
    double operator()(double sigma) const noexcept {
        return bs_price(S, K, T, r, sigma, is_call) - price;
    }
};

IvResult brent(const PriceError& f, double xa, double xb, double xtol, int max_iter,
               int iterations_so_far) noexcept {
    double xpre = xa;
    double xcur = xb;
    double xblk = 0.0;
    double fpre = f(xpre);
    double fcur = f(xcur);
    double fblk = 0.0;
    double spre = 0.0;
    double scur = 0.0;

    // Same pre-check as the Python reference: no sign change means the
    // market price is not reachable with any sigma in the bracket.
    if (!std::isfinite(fpre) || !std::isfinite(fcur) || fpre * fcur > 0.0) {
        return fail(IvStatus::kNoBracket, iterations_so_far);
    }
    if (fpre == 0.0) return IvResult{xpre, IvStatus::kOkBrent, iterations_so_far};
    if (fcur == 0.0) return IvResult{xcur, IvStatus::kOkBrent, iterations_so_far};

    for (int i = 0; i < max_iter; ++i) {
        const int iters = iterations_so_far + i + 1;
        if (fpre != 0.0 && fcur != 0.0 && (std::signbit(fpre) != std::signbit(fcur))) {
            xblk = xpre;
            fblk = fpre;
            spre = scur = xcur - xpre;
        }
        if (std::fabs(fblk) < std::fabs(fcur)) {
            xpre = xcur; xcur = xblk; xblk = xpre;
            fpre = fcur; fcur = fblk; fblk = fpre;
        }

        const double delta = (xtol + kBrentRtol * std::fabs(xcur)) / 2.0;
        const double sbis = (xblk - xcur) / 2.0;
        if (fcur == 0.0 || std::fabs(sbis) < delta) {
            return IvResult{xcur, IvStatus::kOkBrent, iters};
        }

        if (std::fabs(spre) > delta && std::fabs(fcur) < std::fabs(fpre)) {
            double stry;
            if (xpre == xblk) {
                // secant (linear interpolation)
                stry = -fcur * (xcur - xpre) / (fcur - fpre);
            } else {
                // inverse quadratic interpolation
                const double dpre = (fpre - fcur) / (xpre - xcur);
                const double dblk = (fblk - fcur) / (xblk - xcur);
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
            }
            if (2.0 * std::fabs(stry) < std::min(std::fabs(spre), 3.0 * std::fabs(sbis) - delta)) {
                spre = scur;  // interpolation step is good, take it
                scur = stry;
            } else {
                spre = sbis;  // interpolation step is too big, bisect
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        xcur += (std::fabs(scur) > delta) ? scur : (sbis > 0.0 ? delta : -delta);
        fcur = f(xcur);
    }
    return fail(IvStatus::kNoConvergence, iterations_so_far + max_iter);
}

}  // namespace

// ---------------------------------------------------------------------------
// Normal distribution
// ---------------------------------------------------------------------------

// Same branch structure as scipy.special.ndtr (which scipy.stats.norm.cdf
// calls). Near zero, erf is accurate. In the tails, erfc avoids the
// cancellation you would get from computing 1 - (something close to 1).
double norm_cdf(double x) noexcept {
    const double z = x * kSqrt1_2;
    const double az = std::fabs(z);
    if (az < kSqrt1_2) {
        return 0.5 + 0.5 * std::erf(z);
    }
    const double y = 0.5 * std::erfc(az);
    return z > 0.0 ? 1.0 - y : y;
}

double norm_pdf(double x) noexcept {
    return std::exp(-(x * x) / 2.0) / kSqrt2Pi;
}

// ---------------------------------------------------------------------------
// Black-Scholes
// ---------------------------------------------------------------------------

double bs_price(double S, double K, double T, double r, double sigma, bool is_call) noexcept {
    const D1D2 d = compute_d1_d2(S, K, T, r, sigma);
    const double disc_K = K * std::exp(-r * T);
    if (is_call) {
        return S * norm_cdf(d.d1) - disc_K * norm_cdf(d.d2);
    }
    return disc_K * norm_cdf(-d.d2) - S * norm_cdf(-d.d1);
}

double bs_vega(double S, double K, double T, double r, double sigma) noexcept {
    const D1D2 d = compute_d1_d2(S, K, T, r, sigma);
    return S * norm_pdf(d.d1) * d.sqrt_T;
}

double brenner_subrahmanyam_guess(double price, double S, double T,
                                  const SolverConfig& cfg) noexcept {
    // sigma ~= price / (S * sqrt(T / (2 pi))). Exact only at the money; in
    // the wings it can be far off, which is why the Brent fallback exists.
    const double guess = price / (S * std::sqrt(T / kTwoPi));
    return clip(guess, cfg.guess_lo, cfg.guess_hi);
}

// ---------------------------------------------------------------------------
// Implied volatility
// ---------------------------------------------------------------------------

IvResult implied_vol(double price, double S, double K, double T, double r, bool is_call,
                     const SolverConfig& cfg) noexcept {
    // 1. Reject bad input up front. Note the NaN handling: a comparison with
    //    NaN is always false, so "price <= 0" alone would let NaN through.
    if (!std::isfinite(price) || !std::isfinite(S) || !std::isfinite(K) ||
        !std::isfinite(T) || !std::isfinite(r)) {
        return fail(IvStatus::kInvalidInput, 0);
    }
    if (price <= 0.0 || S <= 0.0 || K <= 0.0 || T <= 0.0) {
        return fail(IvStatus::kInvalidInput, 0);
    }

    // 2. No-arbitrage bounds. A call is worth at least S - K e^{-rT} and at
    //    most S. A put is worth at least K e^{-rT} - S and at most K e^{-rT}.
    //    Outside those bounds no sigma can produce the price.
    const double disc_K = K * std::exp(-r * T);
    const double intrinsic = is_call ? std::max(S - disc_K, 0.0) : std::max(disc_K - S, 0.0);
    if (price < intrinsic - cfg.intrinsic_slack) {
        return fail(IvStatus::kBelowIntrinsic, 0);
    }
    const double upper = is_call ? S : disc_K;
    if (price >= upper) {
        return fail(IvStatus::kAboveUpperBound, 0);
    }

    // 3. Newton-Raphson: sigma <- sigma - (BS(sigma) - price) / vega.
    //    Converges quadratically near the root, but can stall when vega is
    //    tiny (deep in the wings, or when the guess is far too low) and can
    //    overshoot when the price function is very curved. The clip keeps
    //    sigma inside the bracket, the vega floor avoids dividing by ~0, and
    //    the iteration cap bounds the worst case.
    double sigma = brenner_subrahmanyam_guess(price, S, T, cfg);
    int it = 0;
    for (; it < cfg.max_newton_iter; ++it) {
        const double diff = bs_price(S, K, T, r, sigma, is_call) - price;
        const double vega = bs_vega(S, K, T, r, sigma);
        if (std::fabs(diff) < cfg.price_tol) {
            return IvResult{sigma, IvStatus::kOkNewton, it + 1};
        }
        if (vega < cfg.vega_floor) {
            break;  // Newton step would be huge or inf; hand over to Brent
        }
        sigma = clip(sigma - diff / vega, cfg.sigma_lo, cfg.sigma_hi);
        if (!std::isfinite(sigma)) {
            break;  // NaN can only come from NaN arithmetic; stop wasting iterations
        }
    }

    // 4. Fallback: Brent on [sigma_lo, sigma_hi]. Price is monotonic in sigma,
    //    so if the bracket has a sign change there is exactly one root and
    //    Brent is guaranteed to find it (it never leaves the bracket).
    const PriceError f{S, K, T, r, price, is_call};
    return brent(f, cfg.sigma_lo, cfg.sigma_hi, cfg.brent_xtol, cfg.max_brent_iter, it);
}

std::size_t implied_vol_batch(const double* S, const double* K, const double* T,
                              const double* price, const std::uint8_t* is_call,
                              std::size_t n, double r, const SolverConfig& cfg,
                              double* out_iv, std::int8_t* out_status) noexcept {
    if (n == 0) return 0;
    if (S == nullptr || K == nullptr || T == nullptr || price == nullptr ||
        is_call == nullptr || out_iv == nullptr) {
        return 0;
    }

    // Hot loop. Things to notice:
    //  * Struct of arrays: each input is its own contiguous array, read front
    //    to back exactly once. The hardware prefetcher handles that pattern
    //    well, and no bytes are loaded that are not used.
    //  * Zero heap allocation: implied_vol works only on stack locals, and
    //    the output buffers were allocated by the caller before the loop.
    //  * Contracts are independent, so there is no state carried between
    //    iterations. The same input always gives the same output.
    std::size_t n_ok = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const IvResult res = implied_vol(price[i], S[i], K[i], T[i], r, is_call[i] != 0, cfg);
        out_iv[i] = res.sigma;
        if (out_status != nullptr) {
            out_status[i] = static_cast<std::int8_t>(res.status);
        }
        if (static_cast<std::int8_t>(res.status) >= 0) {
            ++n_ok;
        }
    }
    return n_ok;
}

}  // namespace neurovol
