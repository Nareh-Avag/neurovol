// bench_kernels.cpp
//
// Times the scalar math the C++ solver spends its time in, on 1M doubles,
// for comparison with bench/kernels.py (the NumPy/SciPy array versions).
//
// Build: cmake --build build --target bench_kernels
// Run:   ./build/bench_kernels

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "neurovol/iv_core.hpp"

namespace {

constexpr std::size_t kN = 1000000;
constexpr int kRepeats = 15;

// Writing every result into an output array (and summing it at the end)
// stops the compiler from deleting the loop as dead code.
template <typename F>
double median_ms(const std::vector<double>& in, std::vector<double>& out, F f) {
    std::vector<double> times;
    for (int rep = 0; rep < kRepeats; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < kN; ++i) out[i] = f(in[i]);
        const auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

}  // namespace

int main() {
    std::mt19937_64 rng(0);
    std::uniform_real_distribution<double> wide(-4.0, 4.0);
    std::uniform_real_distribution<double> positive(0.5, 2.0);
    std::vector<double> x(kN), pos(kN), out(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        x[i] = wide(rng);
        pos[i] = positive(rng);
    }

    double checksum = 0.0;
    std::printf("C++ scalar loops, 1M float64, median of %d\n", kRepeats);
    const double t_cdf = median_ms(x, out, [](double v) { return neurovol::norm_cdf(v); });
    for (double v : out) checksum += v;
    std::printf("  norm_cdf  %7.2f ms\n", t_cdf);
    const double t_exp = median_ms(x, out, [](double v) { return std::exp(v); });
    for (double v : out) checksum += v;
    std::printf("  std::exp  %7.2f ms\n", t_exp);
    const double t_log = median_ms(pos, out, [](double v) { return std::log(v); });
    for (double v : out) checksum += v;
    std::printf("  std::log  %7.2f ms\n", t_log);
    std::printf("(checksum %.6e)\n", checksum);
    return 0;
}
