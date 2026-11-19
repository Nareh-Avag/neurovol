// test_iv_core.cpp
//
// Standalone unit tests for iv_core, no test framework needed.
//
// Why not assert()? Release builds define NDEBUG, which turns assert() into
// nothing, so the tests would silently stop testing. CHECK always runs.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <new>

#include "neurovol/iv_core.hpp"

using neurovol::IvResult;
using neurovol::IvStatus;
using neurovol::SolverConfig;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failures;                                                        \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) <= (tol))

}  // namespace

// ---------------------------------------------------------------------------
// Heap allocation counter.
//
// Replacing the global operator new is allowed by the C++ standard. Every
// `new` in the program (including inside the standard library) now goes
// through these functions, so if the solver allocated anything, the counter
// would move.
// ---------------------------------------------------------------------------
static std::size_t g_allocations = 0;

void* operator new(std::size_t size) {
    ++g_allocations;
    if (void* p = std::malloc(size == 0 ? 1 : size)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) {
    ++g_allocations;
    if (void* p = std::malloc(size == 0 ? 1 : size)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

// ---------------------------------------------------------------------------

static void test_norm_cdf() {
    CHECK(neurovol::norm_cdf(0.0) == 0.5);
    // Standard table values.
    CHECK_NEAR(neurovol::norm_cdf(1.96), 0.9750021048517795, 1e-15);
    CHECK_NEAR(neurovol::norm_cdf(-1.0), 0.15865525393145707, 1e-15);
    // Symmetry, including the tails where erfc matters.
    for (double x = -8.0; x <= 8.0; x += 0.25) {
        CHECK_NEAR(neurovol::norm_cdf(x) + neurovol::norm_cdf(-x), 1.0, 2e-16);
    }
    // Extreme tails stay in [0, 1] and do not produce NaN.
    CHECK(neurovol::norm_cdf(-40.0) >= 0.0 && neurovol::norm_cdf(-40.0) < 1e-300);
    CHECK(neurovol::norm_cdf(40.0) == 1.0);
    // Far tail keeps relative accuracy: N(-10) = 7.6198530241605e-24.
    // (1 - N(10) computed naively would be exactly 0 here.)
    CHECK_NEAR(neurovol::norm_cdf(-10.0) / 7.6198530241605e-24, 1.0, 1e-10);
}

static void test_bs_price_textbook() {
    // Worked example from Hull, "Options, Futures, and Other Derivatives"
    // (Black-Scholes-Merton chapter): S=42, K=40, r=10%, sigma=20%, T=0.5
    // gives c = 4.76 and p = 0.81 (rounded to cents in the book).
    const double c = neurovol::bs_price(42.0, 40.0, 0.5, 0.10, 0.20, true);
    const double p = neurovol::bs_price(42.0, 40.0, 0.5, 0.10, 0.20, false);
    CHECK_NEAR(c, 4.76, 0.005);
    CHECK_NEAR(p, 0.81, 0.005);
}

static void test_put_call_parity() {
    const double r = 0.03;
    for (double K : {50.0, 90.0, 100.0, 110.0, 200.0}) {
        for (double T : {0.01, 0.5, 2.0}) {
            for (double sigma : {0.05, 0.2, 1.0}) {
                const double c = neurovol::bs_price(100.0, K, T, r, sigma, true);
                const double p = neurovol::bs_price(100.0, K, T, r, sigma, false);
                CHECK_NEAR(c - p, 100.0 - K * std::exp(-r * T), 1e-10 * 100.0);
            }
        }
    }
}

static void test_vega_matches_finite_difference() {
    const double h = 1e-5;
    for (double K : {80.0, 100.0, 120.0}) {
        for (double sigma : {0.1, 0.3, 0.8}) {
            const double up = neurovol::bs_price(100.0, K, 0.5, 0.03, sigma + h, true);
            const double dn = neurovol::bs_price(100.0, K, 0.5, 0.03, sigma - h, true);
            const double fd = (up - dn) / (2.0 * h);
            const double v = neurovol::bs_vega(100.0, K, 0.5, 0.03, sigma);
            CHECK_NEAR(v, fd, 1e-6 * v + 1e-8);
        }
    }
}

static void test_round_trip() {
    const SolverConfig cfg{};
    const double r = 0.03;
    int tested = 0;
    for (double sigma : {0.05, 0.1, 0.2, 0.5, 1.0, 2.0}) {
        for (double m : {0.7, 0.9, 1.0, 1.1, 1.3}) {
            for (double T : {0.02, 0.25, 1.0, 2.0}) {
                for (bool call : {true, false}) {
                    const double K = 100.0 * m;
                    // IV error is about price_tol / vega, so only demand 1e-6
                    // where vega >= 0.01 (same rule as the Python tests).
                    if (neurovol::bs_vega(100.0, K, T, r, sigma) < 0.01) continue;
                    const double price = neurovol::bs_price(100.0, K, T, r, sigma, call);
                    const IvResult res = neurovol::implied_vol(price, 100.0, K, T, r, call, cfg);
                    CHECK(static_cast<int>(res.status) >= 0);
                    CHECK_NEAR(res.sigma, sigma, 1e-6);
                    CHECK(res.iterations <= cfg.max_newton_iter + cfg.max_brent_iter);
                    ++tested;
                }
            }
        }
    }
    CHECK(tested > 150);
}

static void test_otm_uses_brent_fallback() {
    // OTM put: the price is small, so the ATM guess is clipped to 0.01, vega
    // there is ~0, Newton gives up immediately and Brent must take over.
    const SolverConfig cfg{};
    const double price = neurovol::bs_price(100.0, 60.0, 1.0, 0.03, 0.25, false);
    const IvResult res = neurovol::implied_vol(price, 100.0, 60.0, 1.0, 0.03, false, cfg);
    CHECK(res.status == IvStatus::kOkBrent);
    CHECK_NEAR(res.sigma, 0.25, 1e-6);
}

static void test_invalid_inputs() {
    const SolverConfig cfg{};
    const double nan = std::nan("");
    const double inf = HUGE_VAL;
    struct Case { double price, S, K, T; bool call; IvStatus expect; };
    const Case cases[] = {
        {15.0, 100.0, 80.0, 0.5, true, IvStatus::kBelowIntrinsic},
        {15.0, 80.0, 100.0, 0.5, false, IvStatus::kBelowIntrinsic},
        {100.5, 100.0, 80.0, 0.5, true, IvStatus::kAboveUpperBound},
        {120.0, 100.0, 110.0, 0.5, false, IvStatus::kAboveUpperBound},
        {0.0, 100.0, 100.0, 0.5, true, IvStatus::kInvalidInput},
        {-1.0, 100.0, 100.0, 0.5, true, IvStatus::kInvalidInput},
        {5.0, 0.0, 100.0, 0.5, true, IvStatus::kInvalidInput},
        {5.0, 100.0, -1.0, 0.5, true, IvStatus::kInvalidInput},
        {5.0, 100.0, 100.0, 0.0, true, IvStatus::kInvalidInput},
        {nan, 100.0, 100.0, 0.5, true, IvStatus::kInvalidInput},
        {5.0, nan, 100.0, 0.5, true, IvStatus::kInvalidInput},
        {inf, 100.0, 100.0, 0.5, true, IvStatus::kInvalidInput},
    };
    for (const Case& c : cases) {
        const IvResult res = neurovol::implied_vol(c.price, c.S, c.K, c.T, 0.03, c.call, cfg);
        CHECK(std::isnan(res.sigma));
        CHECK(res.status == c.expect);
    }
}

static void test_batch_and_no_heap_allocation() {
    // Mixed batch: valid contracts with invalid ones in between.
    constexpr std::size_t n = 6;
    const double r = 0.03;
    const double S[n] = {100.0, 100.0, 100.0, 100.0, 100.0, 100.0};
    const double K[n] = {100.0, 80.0, 120.0, 60.0, 100.0, 90.0};
    const double T[n] = {0.5, 0.5, 0.5, 1.0, 0.0, 0.25};
    const std::uint8_t call[n] = {1, 1, 0, 0, 1, 1};
    double price[n];
    for (std::size_t i = 0; i < n; ++i) {
        price[i] = (T[i] > 0.0) ? neurovol::bs_price(S[i], K[i], T[i], r, 0.3, call[i] != 0) : 5.0;
    }
    price[1] = 1.0;  // below intrinsic for a call with S=100, K=80

    double iv[n];
    std::int8_t status[n];
    const SolverConfig cfg{};

    const std::size_t before = g_allocations;
    const std::size_t n_ok =
        neurovol::implied_vol_batch(S, K, T, price, call, n, r, cfg, iv, status);
    const std::size_t after = g_allocations;

    CHECK(after == before);  // zero heap allocations inside the batch
    CHECK(n_ok == 4);
    CHECK(status[1] == static_cast<std::int8_t>(IvStatus::kBelowIntrinsic));
    CHECK(status[4] == static_cast<std::int8_t>(IvStatus::kInvalidInput));
    CHECK(std::isnan(iv[1]) && std::isnan(iv[4]));
    for (std::size_t i : {std::size_t{0}, std::size_t{2}, std::size_t{3}, std::size_t{5}}) {
        CHECK_NEAR(iv[i], 0.3, 1e-6);
    }

    // out_status is optional.
    double iv2[n];
    CHECK(neurovol::implied_vol_batch(S, K, T, price, call, n, r, cfg, iv2, nullptr) == 4);

    // Determinism: identical bits on a second run.
    CHECK(std::memcmp(iv, iv2, sizeof(iv)) == 0);

    // Edge cases: empty batch, null pointers.
    CHECK(neurovol::implied_vol_batch(S, K, T, price, call, 0, r, cfg, iv, status) == 0);
    CHECK(neurovol::implied_vol_batch(nullptr, K, T, price, call, n, r, cfg, iv, status) == 0);
    CHECK(neurovol::implied_vol_batch(S, K, T, price, call, n, r, cfg, nullptr, status) == 0);
}

int main() {
    test_norm_cdf();
    test_bs_price_textbook();
    test_put_call_parity();
    test_vega_matches_finite_difference();
    test_round_trip();
    test_otm_uses_brent_fallback();
    test_invalid_inputs();
    test_batch_and_no_heap_allocation();

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
