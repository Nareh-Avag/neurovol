// iv_core.hpp
//
// Black-Scholes pricing and implied volatility (IV) solver.
//
// This is a C++17 port of the Python reference in
// vol_surface_engine/iv_solver.py. It uses the same initial guess, the same
// Newton update and clipping, the same tolerances, and a Brent fallback that
// follows the same algorithm as scipy.optimize.brentq. Keeping the algorithm
// identical is deliberate: it lets the tests compare the two implementations
// directly, and it means the benchmark measures implementation cost, not a
// better or worse algorithm.
//
// Design rules (the reasons are spelled out next to the code):
//   * No heap allocation anywhere in this library.
//   * No exceptions. Every function is noexcept. Failures are reported with
//     NaN plus an explicit IvStatus code.
//   * No global mutable state. The only globals are constexpr constants, so
//     the functions are reentrant and safe to call from several threads.
//   * Every loop has a hard iteration cap, so the worst-case run time per
//     contract is bounded.

#ifndef NEUROVOL_IV_CORE_HPP
#define NEUROVOL_IV_CORE_HPP

#include <cstddef>
#include <cstdint>

namespace neurovol {

// Result codes. int8_t keeps a status array at one byte per contract.
// Values >= 0 mean success, values < 0 mean the output is NaN.
enum class IvStatus : std::int8_t {
    kOkNewton = 0,           // Newton-Raphson converged
    kOkBrent = 1,            // Newton gave up, Brent fallback converged
    kInvalidInput = -1,      // NaN/inf input, or price/S/K/T <= 0
    kBelowIntrinsic = -2,    // price below the no-arbitrage lower bound
    kAboveUpperBound = -3,   // price at or above the no-arbitrage upper bound
    kNoBracket = -4,         // no sigma in [sigma_lo, sigma_hi] matches the price
    kNoConvergence = -5,     // hit the iteration cap
};

// Solver settings. Passed by const reference so the struct is never copied.
// The defaults match vol_surface_engine/iv_solver.py.
struct SolverConfig {
    double price_tol = 1e-8;       // Newton stops when |model - market| < price_tol
    double brent_xtol = 1e-8;      // Brent stops when the bracket is this narrow
    int max_newton_iter = 100;
    int max_brent_iter = 200;
    double sigma_lo = 1e-6;        // search bracket and Newton clipping range
    double sigma_hi = 20.0;
    double guess_lo = 0.01;        // clipping range for the initial guess
    double guess_hi = 10.0;
    double vega_floor = 1e-10;     // below this, a Newton step is meaningless
    double intrinsic_slack = 1e-6; // allowance for quotes rounded below intrinsic
};

struct IvResult {
    double sigma;      // NaN on failure
    IvStatus status;
    int iterations;    // Newton iterations plus Brent iterations used
};

// Standard normal CDF and PDF.
double norm_cdf(double x) noexcept;
double norm_pdf(double x) noexcept;

// Black-Scholes price of a European option. Requires T > 0 and sigma > 0
// (the caller checks this; the solver never calls it otherwise).
double bs_price(double S, double K, double T, double r, double sigma, bool is_call) noexcept;

// dPrice/dSigma. Requires T > 0 and sigma > 0.
double bs_vega(double S, double K, double T, double r, double sigma) noexcept;

// Brenner and Subrahmanyam (1988) ATM approximation, clipped to
// [cfg.guess_lo, cfg.guess_hi].
double brenner_subrahmanyam_guess(double price, double S, double T,
                                  const SolverConfig& cfg) noexcept;

// Implied volatility for one contract.
IvResult implied_vol(double price, double S, double K, double T, double r, bool is_call,
                     const SolverConfig& cfg) noexcept;

// Batch solver over a struct of arrays (SoA).
//
// All input arrays have length n. is_call[i] is 0 (put) or nonzero (call).
// Results go into out_iv, which the caller allocates. out_status may be
// nullptr if the caller does not need status codes.
//
// out_status is int8_t*, not IvStatus*, on purpose. From Python that buffer
// is a NumPy int8 array. Casting its pointer to IvStatus* and writing through
// it would break C++'s strict aliasing rule (only char, unsigned char, and
// std::byte may alias other types), which is undefined behavior. Writing
// static_cast<int8_t>(status) into an int8_t buffer is always well defined.
//
// Returns the number of contracts that produced a valid IV. If n > 0 and any
// required pointer is null, nothing is written and 0 is returned.
std::size_t implied_vol_batch(const double* S, const double* K, const double* T,
                              const double* price, const std::uint8_t* is_call,
                              std::size_t n, double r, const SolverConfig& cfg,
                              double* out_iv, std::int8_t* out_status) noexcept;

}  // namespace neurovol

#endif  // NEUROVOL_IV_CORE_HPP
