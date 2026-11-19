// bindings.cpp
//
// pybind11 glue that exposes the C++ solver to Python as
// vol_surface_engine._iv_core.
//
// The rule here is: all checking and anything that can throw happens before
// the hot loop. Once we have raw pointers, the GIL is released and the
// noexcept core runs.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <string>
#include <tuple>

#include "neurovol/iv_core.hpp"

namespace py = pybind11;

namespace {

// Input arrays: c_style | forcecast.
//  * If the caller passes a C-contiguous float64 array (the normal case),
//    pybind11 hands us a view of the same memory. No copy.
//  * If the caller passes something else (float32, a strided slice, a list),
//    forcecast lets pybind11 make one converted contiguous copy. That is
//    correct, just slower, and only affects inputs.
using InArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using FlagArray = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

// Output arrays: c_style only, and the argument is marked noconvert() below.
// If the output were allowed to be converted, pybind11 would silently write
// into a temporary copy and the caller's array would never change. Refusing
// conversion turns that silent bug into a TypeError.
using OutArray = py::array_t<double, py::array::c_style>;
using StatusArray = py::array_t<std::int8_t, py::array::c_style>;

void require_1d(const py::array& a, py::ssize_t n, const char* name) {
    if (a.ndim() != 1 || a.shape(0) != n) {
        throw py::value_error(std::string(name) + " must be 1-D with the same length as S");
    }
}

std::size_t batch_impl(const InArray& S, const InArray& K, const InArray& T,
                       const InArray& price, const FlagArray& is_call, double r,
                       OutArray& out, StatusArray* status) {
    if (S.ndim() != 1) throw py::value_error("S must be 1-D");
    const py::ssize_t n = S.shape(0);
    require_1d(K, n, "K");
    require_1d(T, n, "T");
    require_1d(price, n, "price");
    require_1d(is_call, n, "is_call");
    require_1d(out, n, "out");
    if (status != nullptr) require_1d(*status, n, "status");

    // Grab raw pointers while we still hold the GIL. mutable_data() throws
    // if the array is read-only, which is what we want.
    const double* s = S.data();
    const double* k = K.data();
    const double* t = T.data();
    const double* p = price.data();
    const std::uint8_t* c = is_call.data();
    double* o = out.mutable_data();
    std::int8_t* st = (status != nullptr) ? status->mutable_data() : nullptr;

    const neurovol::SolverConfig cfg{};  // defaults match the Python solver
    std::size_t n_ok = 0;
    {
        // Release the GIL for the pure C++ part. Other Python threads can
        // run meanwhile. This is safe because the core never touches a
        // Python object, and the array objects above stay alive (we hold
        // references) until this function returns.
        py::gil_scoped_release release;
        n_ok = neurovol::implied_vol_batch(s, k, t, p, c, static_cast<std::size_t>(n), r, cfg,
                                           o, st);
    }
    return n_ok;
}

}  // namespace

PYBIND11_MODULE(_iv_core, m) {
    m.doc() = "C++ Black-Scholes implied volatility solver (see cpp/ in the repo)";

    m.def(
        "implied_vol_batch",
        [](const InArray& S, const InArray& K, const InArray& T, const InArray& price,
           const FlagArray& is_call, double r, OutArray out) {
            return batch_impl(S, K, T, price, is_call, r, out, nullptr);
        },
        py::arg("S"), py::arg("K"), py::arg("T"), py::arg("price"), py::arg("is_call"),
        py::arg("r"), py::arg("out").noconvert(),
        "Solve IV for every contract, writing into the preallocated float64 array `out`.\n"
        "Returns the number of valid IVs.");

    m.def(
        "implied_vol_batch_status",
        [](const InArray& S, const InArray& K, const InArray& T, const InArray& price,
           const FlagArray& is_call, double r, OutArray out, StatusArray status) {
            return batch_impl(S, K, T, price, is_call, r, out, &status);
        },
        py::arg("S"), py::arg("K"), py::arg("T"), py::arg("price"), py::arg("is_call"),
        py::arg("r"), py::arg("out").noconvert(), py::arg("status").noconvert(),
        "Like implied_vol_batch, and also writes an int8 status code per contract.");

    // Scalar functions, mainly for tests.
    m.def("norm_cdf", &neurovol::norm_cdf, py::arg("x"));
    m.def("bs_price", &neurovol::bs_price, py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("sigma"), py::arg("is_call"));
    m.def("bs_vega", &neurovol::bs_vega, py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("sigma"));
    m.def(
        "implied_vol",
        [](double price, double S, double K, double T, double r, bool is_call) {
            const neurovol::SolverConfig cfg{};
            const neurovol::IvResult res = neurovol::implied_vol(price, S, K, T, r, is_call, cfg);
            return std::make_tuple(res.sigma, static_cast<int>(res.status), res.iterations);
        },
        py::arg("price"), py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
        py::arg("is_call"), "Returns (sigma, status_code, iterations).");

#if defined(__clang__)
    m.attr("COMPILER") = "clang " __clang_version__;
#elif defined(__GNUC__)
    m.attr("COMPILER") = "gcc " __VERSION__;
#elif defined(_MSC_FULL_VER)
    m.attr("COMPILER") = "msvc " + std::to_string(_MSC_FULL_VER);
#else
    m.attr("COMPILER") = "unknown";
#endif
#ifdef NEUROVOL_BUILD_FLAGS
    m.attr("BUILD_FLAGS") = NEUROVOL_BUILD_FLAGS;
    m.attr("BUILD_TYPE") = NEUROVOL_BUILD_TYPE;
#endif

    m.attr("STATUS_CODES") = py::dict(
        py::arg("ok_newton") = 0, py::arg("ok_brent") = 1, py::arg("invalid_input") = -1,
        py::arg("below_intrinsic") = -2, py::arg("above_upper_bound") = -3,
        py::arg("no_bracket") = -4, py::arg("no_convergence") = -5);
}
