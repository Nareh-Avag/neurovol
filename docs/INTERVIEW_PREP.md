# Interview prep: the C++ implied volatility core

This is written for you, not for the repo's users. It walks through the C++ code as if
you know Python well and C++ basics, ties every performance claim to a number we
actually measured, and ends with likely interview questions and an honest resume bullet.

All numbers come from `bench/results.md`, `bench/results_scipy118.md`,
`bench/agreement.md`, `bench/kernels.py`, and `cpp/bench/bench_kernels.cpp`, measured on
a Snapdragon X Plus (ARM64) Windows laptop. Python and the C++ extension were both x64 builds running under
Windows' x64 emulation on that ARM CPU, so treat the absolute times as
"this laptop, emulated" and the ratios as the real result.

**The headline, before anything else:** with the SciPy version a fresh install gets today
(1.18.1), the C++ core is *not* faster than the vectorized NumPy backend on chains of
10k contracts or more. NumPy is 1.14x to 1.18x faster there. With SciPy 1.17.1, C++
was 4.8x to 5.9x faster. Sections 3 and 4 show the measurements and the cause. Know
this cold; it is the most important thing in this document.

---

## 1. The problem in one paragraph

An option's market price is known. Black-Scholes gives price as a function of
volatility, `price = BS(S, K, T, r, sigma)`. Implied volatility (IV) is the `sigma` that
makes `BS(...)` equal the market price. There is no closed-form inverse, so you solve
`f(sigma) = BS(sigma) - market_price = 0` numerically for every contract. A chain has
thousands of contracts, every quote update needs a new IV, and each solve calls `exp`,
`log`, and `erfc` dozens of times. That is why it is worth making fast.

## 2. Walkthrough, file by file

### `cpp/include/neurovol/iv_core.hpp` (the public API)

A header declares what exists; the `.cpp` defines how it works. Think of the header as
a Python module's function signatures and docstrings with the bodies removed.

* **Include guard** (`#ifndef NEUROVOL_IV_CORE_HPP ...`). C++ pastes headers in
  textually; the guard stops the same declarations from being pasted twice.
* **`namespace neurovol`**. Like a Python module name: `neurovol::implied_vol` cannot
  clash with someone else's `implied_vol`.
* **`enum class IvStatus : std::int8_t`**. A set of named error codes. `enum class`
  (unlike a plain `enum`) does not convert to `int` silently, so you cannot accidentally
  compare a status with a random integer. `: std::int8_t` fixes the size at one byte,
  so an array of 100k statuses is 100 KB.
  Negative means failure, so "did it work" is `status >= 0`.
* **`struct SolverConfig`** with default values (`double price_tol = 1e-8;`). Like a
  Python dataclass with defaults. It is passed as `const SolverConfig&`: a reference
  (no copy, like passing a Python object) that the function promises not to modify
  (`const`).
* **`struct IvResult { double sigma; IvStatus status; int iterations; }`**. Returning a
  small struct by value is cheap (16 bytes, returned in registers or a stack slot). It
  is the C++ version of returning a tuple.
* **Function declarations marked `noexcept`.** A promise that the function never throws.
  If something inside did throw, the program would call `std::terminate` instead of
  unwinding, so the promise has to be true. Here it is: the code only does arithmetic
  and calls `<cmath>` functions, which report problems with NaN/inf, not exceptions.
* **`implied_vol_batch(const double* S, ..., std::size_t n, ..., double* out_iv,
  std::int8_t* out_status)`**. This is the "pointer plus length" style. Instead of
  taking a `std::vector` (which owns memory and would have to be filled by copying),
  the function borrows memory that somebody else owns: a NumPy array, a
  `std::vector`'s `.data()`, a static buffer on a microcontroller. `const double*`
  means "I will only read through this pointer". The output pointers are not `const`,
  because we write results there.
* **Why `std::int8_t*` and not `IvStatus*` for statuses.** From Python, that buffer is
  a NumPy `int8` array. If C++ cast the pointer to `IvStatus*` and wrote through it,
  that would break the strict aliasing rule: the compiler may assume pointers to
  unrelated types never point at the same memory, and only `char`, `unsigned char`,
  and `std::byte` are allowed to break that assumption. Violating it is undefined
  behavior. Writing `static_cast<std::int8_t>(status)` into an `int8_t` buffer is
  always defined.

### `cpp/src/iv_core.cpp` (the implementation)

* **Anonymous namespace** (`namespace { ... }`). Everything inside is private to this
  file (internal linkage), like a leading underscore in Python but enforced by the
  linker.
* **`constexpr` constants** (`kSqrt2Pi`, `kNaN`, ...). Computed at compile time and
  read-only. These are the only file-level variables, and none of them can change, so
  there is no global mutable state. Two threads can call `implied_vol` at the same time
  without locks.
* **`clip(x, lo, hi)`** is `min(max(x, lo), hi)`. Comparisons with NaN are always false,
  so a NaN passes straight through, same as `numpy.clip`. (`std::clamp` would also work
  but is undefined if `lo > hi`.)
* **`compute_d1_d2`** computes the shared Black-Scholes terms once:
  `d1 = (log(S/K) + (r + sigma^2/2) T) / (sigma sqrt(T))`, `d2 = d1 - sigma sqrt(T)`.
  The expression is written in the same order as the Python code. Floating point
  addition is not associative, so `(a + b) + c` can differ from `a + (b + c)` in the last
  bit. Matching the order keeps the two implementations as close as possible.
* **`norm_cdf`** (standard normal CDF, N(x)). Uses `erf` near zero and `erfc` in the
  tails, the same branch structure as scipy. The reason: for x = -10,
  N(x) = 7.6e-24, but computing it as `1 - N(10)` gives exactly 0 because N(10) rounds
  to 1.0 in a double. `erfc` computes the small tail directly. The C++ test checks this.
* **`bs_price`, `bs_vega`**. The textbook formulas. Vega is dPrice/dSigma, the slope
  Newton needs.
* **`brenner_subrahmanyam_guess`**. `sigma ~ price / (S sqrt(T / 2pi))`. Exact-ish at
  the money. Far from the money it is badly wrong (see question 6 in section 5).
* **`PriceError`**. A tiny struct with `operator()`, so an instance can be called like a
  function: `f(sigma)` returns `BS(sigma) - price`. It is the C++ equivalent of a Python
  closure (`lambda sigma: bs(...) - price`) with no heap allocation: the captured values
  are plain fields on the stack.
* **`brent`**. Brent's root finder, following scipy's `brentq.c` step for step so both
  backends land on the same root. It keeps a bracket `[a, b]` where `f` changes sign,
  tries a fast interpolation step, and falls back to bisection if the fast step is not
  safe. It never leaves the bracket, so it always converges if a root exists.
* **`implied_vol`**, in four steps:
  1. **Validate.** Reject NaN/inf and non-positive inputs. Note the explicit
     `isfinite` checks: `price <= 0` is false for NaN, so it would let NaN through.
  2. **No-arbitrage bounds.** A call is worth at least `S - K e^(-rT)` and at most `S`.
     A put is worth at least `K e^(-rT) - S` and at most `K e^(-rT)`. Outside that
     range no volatility can produce the price, so we return NaN plus a specific
     status (`kBelowIntrinsic`, `kAboveUpperBound`) instead of searching.
  3. **Newton-Raphson.** `sigma -= (BS(sigma) - price) / vega`, at most 100 times. Stop
     when the price error is under 1e-8. Give up (break) if vega is below 1e-10,
     because dividing by almost zero sends sigma somewhere meaningless.
  4. **Brent fallback** on `[1e-6, 20]`, at most 200 iterations.
* **`implied_vol_batch`**. Checks for null pointers, then a plain `for` loop over `i`,
  calling `implied_vol` and writing `out_iv[i]`. No allocation, no shared state between
  iterations, and the inputs are read front to back exactly once.

### `cpp/src/bindings.cpp` (Python glue with pybind11)

* **`py::array_t<double, py::array::c_style | py::array::forcecast>`** for inputs. If you
  pass a C-contiguous float64 NumPy array, pybind11 gives C++ a view of the same memory:
  no copy. If you pass something else (float32, a strided slice, a list), `forcecast`
  allows pybind11 to make one converted copy. Correct, just slower.
* **Outputs use `c_style` without `forcecast`, and the argument is `.noconvert()`.** This
  is the subtle one. If pybind11 were allowed to convert the output array, it would
  make a temporary copy, C++ would fill the copy, and the copy would be thrown away.
  Your array would stay empty and nothing would error. `noconvert` turns that silent
  bug into a `TypeError`. There is a Python test for exactly this.
* **Shape checks throw `py::value_error`** (becomes Python `ValueError`). All checking
  happens before the hot loop, so the loop itself never needs to throw.
* **`py::gil_scoped_release`**. The Global Interpreter Lock (GIL) lets only one thread
  run Python bytecode at a time. While C++ crunches numbers it does not need Python,
  so it releases the GIL and other Python threads can run. This is only safe because
  (a) we grabbed the raw pointers before releasing it and (b) the core never touches a
  Python object. The lock is reacquired automatically when `release` goes out of scope
  (RAII: resource acquisition is initialization, the C++ habit of tying cleanup to a
  variable's lifetime, like a Python `with` block).
* **`m.attr("COMPILER")`, `m.attr("BUILD_FLAGS")`.** Baked in at compile time so the
  benchmark reports what the loaded binary was really built with.

### `cpp/tests/test_iv_core.cpp`

* **`CHECK(cond)` instead of `assert`.** Release builds define `NDEBUG`, which deletes
  every `assert`. Tests built that way would silently test nothing. The macro is
  wrapped in `do { ... } while (0)` so it behaves like a single statement inside an
  `if` without braces.
* **Replacing the global `operator new`.** The C++ standard lets a program provide its
  own `operator new`. Ours counts calls. The batch test records the count, runs the
  batch, and checks the count did not move: a test that proves the hot loop does not
  allocate. (Limit: it catches `new`, and so every standard container; it would not
  catch a direct `malloc` call. The code has none.)
* Also tested: textbook prices from Hull, put-call parity, vega against a finite
  difference, IV round trip, that an OTM put really takes the Brent path, every invalid
  input with its expected status code, null pointers, an empty batch, and determinism
  (two runs produce identical bytes, checked with `memcmp`).

### `cpp/CMakeLists.txt`

* Three targets: `iv_core` (static library, the solver), `test_iv_core` (test
  executable), `_iv_core` (the Python module). The library has no Python dependency,
  so it could be linked into firmware or a C++ service unchanged.
* Flags: `-O2 -std=c++17 -Wall -Wextra -Wpedantic`, optional `-Werror`. CMake's default
  Release flag is `-O3`; it is overridden to `-O2` on purpose. No `-ffast-math`, because
  it lets the compiler assume NaN never occurs, and this code uses NaN as its error
  value (`isnan` checks could be optimized away).
* `-static` on MinGW so the `.pyd` and the test do not need `libc++.dll` at run time.

### `vol_surface_engine/iv_solver.py` (the Python side)

* `implied_vol_array(..., backend="python" | "numpy" | "cpp")` dispatches. For `"cpp"`,
  it makes sure inputs are contiguous float64 (`np.ascontiguousarray` returns the same
  array if it already is), turns the bool call/put array into `uint8` with `.view()`
  (same bytes, no copy), allocates `out` once, and calls C++.
* If the extension was not built, `import _iv_core` fails at module load,
  `CPP_AVAILABLE` is `False`, and `"cpp"` logs a warning and runs the NumPy backend.

---

## 3. Measured results and why each choice matters

### The numbers

Single thread, same synthetic chain, same algorithm in all three backends. Median of 20
timed runs after warm-up (3 runs for the per-row Python backend at 100k, because each
run takes about 9 minutes).

| Contracts | Backend | Median | p95 | Throughput | vs per-row Python | vs NumPy |
|---|---|---|---|---|---|---|
| 1,000 | python | 3.88 s | 5.09 s | 258/s | 1x | |
| 1,000 | numpy | 24.73 ms | 57.71 ms | 40.4k/s | 157x | 1x |
| 1,000 | cpp | 4.35 ms | 5.30 ms | 229.8k/s | 892x | 5.68x |
| 10,000 | python | 46.88 s | 48.39 s | 213/s | 1x | |
| 10,000 | numpy | 208.07 ms | 328.77 ms | 48.1k/s | 225x | 1x |
| 10,000 | cpp | 35.11 ms | 42.00 ms | 284.9k/s | 1,335x | 5.93x |
| 100,000 | python | 533.12 s | 701.77 s | 188/s | 1x | |
| 100,000 | numpy | 1.69 s | 1.81 s | 59.2k/s | 316x | 1x |
| 100,000 | cpp | 354.08 ms | 361.03 ms | 282.4k/s | 1,506x | 4.77x |

The same NumPy and C++ benchmark in a fresh virtualenv, which installed NumPy 2.5.5 and
SciPy 1.18.1 (`bench/results_scipy118.md`, 20 runs each):

| Contracts | Backend | Median | p95 | Throughput | C++ vs NumPy |
|---|---|---|---|---|---|
| 1,000 | numpy | 5.91 ms | 6.73 ms | 169.3k/s | |
| 1,000 | cpp | 3.52 ms | 4.16 ms | 284.1k/s | 1.68x faster |
| 10,000 | numpy | 31.39 ms | 33.45 ms | 318.6k/s | |
| 10,000 | cpp | 35.81 ms | 38.10 ms | 279.3k/s | 1.14x slower |
| 100,000 | numpy | 315.69 ms | 406.09 ms | 316.8k/s | |
| 100,000 | cpp | 372.44 ms | 480.56 ms | 268.5k/s | 1.18x slower |

The C++ code is identical in both runs; its 100k median moved from 354.08 ms to
372.44 ms, about 5%, which is ordinary run-to-run variation on this laptop. NumPy moved
from 1.69 s to 315.69 ms.

The kernels behind that change, one million float64 values, median of 15:

| Kernel | SciPy 1.17.1 / NumPy 2.4.4 | SciPy 1.18.1 / NumPy 2.5.5 | C++ scalar loop |
|---|---|---|---|
| normal CDF (`ndtr` / `norm_cdf`) | 296.64 ms | 40.72 ms | 144.81 ms |
| `exp` | 15.19 ms | 14.74 ms | 12.02 ms |
| `log` | 17.77 ms | 17.29 ms | 14.66 ms |

Per-call costs measured in the first run:

| Call | Time |
|---|---|
| `scipy.stats.norm.cdf(0.3)` on one Python float | 108.2 us |
| A `math.erfc` based normal CDF in pure Python | 0.7 us |
| C++ `norm_cdf` called once through pybind11 | 0.4 us |
| C++ `implied_vol` for one ATM contract through pybind11 | 1.8 us |

Accuracy (`bench/agreement.md`, 100k chain): C++ vs Python max absolute IV difference
1.58e-12; NumPy vs Python 4.81e-9; no contract where one backend returned NaN and
another did not. 61.0% of contracts were solved by Newton and 39.0% by the Brent
fallback.

### How each design choice connects to those numbers

* **Batching across the language boundary.** One C++ `implied_vol` call through pybind11
  costs 1.8 us for a single easy contract, and calling the trivial `norm_cdf` costs
  0.4 us, most of which is crossing from Python into C++ and back. The batch averages
  about 3.5 us per contract *including* the 39% hard contracts that need Brent, and
  crosses the boundary once per chain. Crossing it once per contract would throw away
  much of the gain. That is why the API takes whole arrays.
* **Zero-copy inputs, caller-owned output.** These keep the per-call cost independent
  of chain size. C++ throughput is about flat from 10k to 100k (284.9k/s vs 282.4k/s):
  there is no per-call setup left to amortize. It is a bit lower at 1k (229.8k/s),
  where the fixed cost of the call and argument checks is a larger share of 4.35 ms.
* **No heap allocation in the loop.** Not directly visible as one number, but it fits
  the flat time per contract and a p95 close to the median (361.03 ms vs 354.08 ms at
  100k). Compare NumPy, which allocates a new temporary array for every operation: its
  p95 at 10k is 58% above its median. (I did not isolate allocation as the cause of
  NumPy's variance; it is one plausible contributor.)
* **Bounded iterations, no exceptions.** These are about predictability, not speed.
  They put a hard ceiling on the work per contract (at most 100 Newton plus 200 Brent
  iterations) and make failures cheap: an invalid quote is rejected in a few
  comparisons.
* **Contiguous struct of arrays.** Honest answer: this matters for zero copy, not for
  cache here. At 100k the batch reads and writes about 41 bytes per contract (four
  doubles and a flag in, one double out), about 4.1 MB in 354 ms, roughly 12 MB/s. That
  is a tiny fraction of memory bandwidth. The work is compute-bound: `exp`, `log`, and
  `erfc` inside every iteration.
* **`-O2`, no `-ffast-math`.** I did not benchmark `-O3` or fast-math, so I make no
  claim either way about speed. Fast-math is excluded for correctness: the code relies
  on NaN, and the 1.58e-12 agreement depends on not reordering arithmetic.
* **Releasing the GIL.** Not measured. The benchmark is single-threaded, so it gains
  nothing there. It only helps a program that runs other Python threads, or that splits
  a chain across threads, neither of which I benchmarked.

---

## 4. Where C++ won, where NumPy matched or beat it, and why

### Against the original per-row Python: 892x to 1,506x

This is a big number, so be precise about where it comes from. The per-row solver calls
`scipy.stats.norm.cdf` and `norm.pdf` on single floats. Each Newton iteration needs at
least two CDF calls for the price and one PDF call for vega, and each scalar `norm.cdf`
call cost 108.2 us here, versus 0.7 us for the same math written with `math.erfc`.
scipy's distribution objects are built for arrays; on a single float they spend almost
all their time on argument checking and array conversion. So most of this speedup is
"stop paying scipy's per-call overhead", which a pure-Python rewrite using `math.erfc`
would also partly get. I did not benchmark that rewrite, so I don't quote a number for
it. **Do not say "C++ is 1,500x faster than Python"** in an interview. Say "1,500x faster
than the original per-row implementation, mostly from removing per-call overhead; and
about even with a fair vectorized NumPy version on current SciPy: faster at 1k
contracts, slightly slower at 10k and 100k."

The ratio also grows with size (892x at 1k, 1,506x at 100k) because the per-row solver
got slower per contract at larger sizes (258/s to 188/s) while C++ stayed flat. I have
not investigated why. The 100k Python figure is also only 3 runs and its p95
(701.77 s) is 32% above its median, so it is the noisiest number in the table. In the
separate agreement run the same 100k Python solve took 404 s, which shows how much that
figure moves between runs.

### Against vectorized NumPy: it depends on SciPy, and today NumPy wins at scale

With SciPy 1.17.1, C++ was 4.8x to 5.9x faster. With SciPy 1.18.1, NumPy was 1.14x
faster at 10k and 1.18x faster at 100k, and C++ was only ahead at 1k (1.68x).

**What changed.** Only one kernel: SciPy's normal CDF, `scipy.special.ndtr`, went from
296.64 ms to 40.72 ms per million values (7.3x faster). NumPy's `exp` and `log` did not
change. The NumPy backend calls `ndtr` four times per active row per iteration, so its
total time fell from 1.69 s to 315.69 ms while the C++ time stayed put. I do not know
what SciPy changed internally, so I don't claim a reason, only the measurement.

**Why that is enough to flip the result.** The normal CDF is the expensive part of both
backends. My C++ `norm_cdf` calls the C runtime's `erf`/`erfc` once per value and costs
144.81 ms per million: twice as fast as the old SciPy, but 3.6x slower than the new one.
`exp` and `log` cost about the same everywhere. So once SciPy had a much faster CDF, the
backend that used it won, even though everything around the CDF is still less efficient
in NumPy.

**What is still true about NumPy's overhead** (and is why C++ still wins at 1k):

1. **Every operation is a separate pass with a temporary array.** `d1 = (log(S/K) + ...)
   / (...)` in NumPy is about eight array operations, each writing a new array. C++
   evaluates the whole formula for one contract in registers.
2. **Both branches are computed.** `np.where(is_call, call_price, put_price)` computes
   both prices for every row and throws one away. C++ computes one.
3. **Per-iteration Python overhead.** Each Newton iteration issues dozens of NumPy calls
   and fancy-indexes the active rows. With small arrays this fixed cost dominates, which
   is why NumPy's throughput rises with size (169.3k/s at 1k to 316.8k/s at 100k on
   SciPy 1.18.1) while C++ is flat.
4. **A weaker fallback.** The NumPy backend uses 32 fixed bisection steps where Brent
   usually needs fewer. Despite that handicap NumPy now wins at scale, which shows how
   much the CDF dominates.

**The lesson to say in an interview:** "Rewriting in C++ is not automatically faster. A
well-vectorized NumPy pipeline runs compiled, often SIMD-optimized library code, and
once the library improved its normal CDF it beat my scalar C++ loop. The profile, not the
language, decides. In my case the profile says the C++ core needs a faster normal CDF."

### What the GIL and cache did *not* contribute

Neither explains any of the measured gaps. The runs are single-threaded, so releasing the
GIL changed nothing in these numbers. Memory traffic is about 12 MB/s, so cache behavior
is not the bottleneck. These results are about compute (mostly the normal CDF) and
per-call overhead.

### The emulation caveat

Python, NumPy, and the C++ extension were all x64 code running under emulation on an ARM
CPU. All three paid that cost, so the ratios are comparable, but absolute times would
differ on native hardware, and emulation could favor one backend over another in ways I
did not measure. The C++ unit tests also build and pass natively on ARM64, but I did not
benchmark a native build.

---

## 5. Fifteen interview questions with answers you can say out loud

**1. Where is memory allocated in the batch path, and who owns it?**
"The caller owns all of it. From Python, the inputs are NumPy arrays and the output is
a NumPy array allocated once before the call. C++ only borrows pointers. Inside the
solver everything is local doubles on the stack or in registers. I have a unit test
that replaces global `operator new` with a counting version and checks the count does
not change across a batch."

**2. Stack versus heap: what lives where in `implied_vol`?**
"All stack. A handful of doubles, a small `PriceError` struct for the Brent callback,
and an `IvResult` returned by value, which is 16 bytes. Stack allocation is just moving
the stack pointer, it is freed automatically on return, and its size is known at
compile time. The heap needs an allocator call, can fragment, and can fail, which is
why embedded code avoids it in loops."

**3. Why pointer plus length instead of `std::vector`?**
"A `std::vector` owns its memory, so accepting one means the caller has to copy their
data into it. Pointer plus length lets me read memory owned by anyone: NumPy, a
vector's `.data()`, a static buffer, a DMA region. It is also a plain C-style interface.
The cost is that the compiler can't check the length for me, so the binding layer
checks that all arrays have the same length before calling in. In C++20 I would use
`std::span`, which is the same thing with the length attached."

**4. What undefined behavior did you have to think about?**
"Three things. Strict aliasing: the status output is a NumPy int8 buffer, so I write
`int8_t` values into an `int8_t*` instead of casting the pointer to my enum type. Null
pointers: the batch checks them and returns 0 rather than dereferencing. And
`-ffast-math`: not UB itself, but it tells the compiler NaN can't happen, and my error
value is NaN, so I don't use it. I also avoided `std::clamp`, which is UB if low is
greater than high."

**5. Why `noexcept`, and why no exceptions in the hot path?**
"Exceptions have unpredictable cost when thrown and add unwinding tables. Many embedded
toolchains turn them off. The core reports errors as NaN plus an explicit status code,
which the caller can check per contract. `noexcept` documents that and lets the
compiler skip unwinding code. If something did throw inside a `noexcept` function the
program would terminate, so the promise has to be true. Here only arithmetic and
`<cmath>` are called, and those signal problems with NaN, not exceptions."

**6. Why can Newton's method diverge or stall here?**
"Newton divides by the derivative, vega. Vega goes to zero deep in or out of the money,
and at very low volatility. If the initial guess is far off, the first step can shoot
out of range. That is exactly what happens with the Brenner-Subrahmanyam guess on
out-of-the-money options: the price is tiny, so the guess clips to 0.01, vega there is
about zero, and Newton gives up. In our synthetic chain that was 39.0% of
contracts. I guard with clipping to [1e-6, 20], a vega floor, a 100-iteration cap, and
a Brent fallback."

**7. Why is the fallback guaranteed to work when a solution exists?**
"Black-Scholes price is strictly increasing in volatility. So if the market price is
between the price at sigma = 1e-6 and at sigma = 20, there is exactly one root in that
bracket, and Brent never leaves the bracket. If the price is outside, there is no
answer and I return NaN with a status code instead of searching."

**8. Why are deep in- or out-of-the-money IVs unreliable no matter how good the solver is?**
"The sigma error is roughly the price error divided by vega. With a price tolerance of
1e-8 and vega of 1e-4, sigma is only good to about 1e-4. With real quotes rounded to a
cent, it is much worse. So the tests only require 1e-6 accuracy where vega is at least
0.01, and elsewhere they check that the returned sigma reproduces the price or is NaN."

**9. Is the output deterministic?**
"Same binary, same input: yes, bit for bit. There are no threads, no uninitialized
reads, no fast-math, and no state carried between contracts; the C++ test runs the batch
twice and compares bytes with `memcmp`. Across compilers or platforms: not guaranteed.
Different C libraries implement `erfc` differently. I measured up to about 6e-14
relative difference against scipy's in the far tail. Compilers may also fuse a
multiply and add into one FMA instruction on ARM, which rounds once instead of twice.
For cross-platform bit reproducibility I'd add `-ffp-contract=off` and ship my own
`erfc`."

**10. What is the worst-case execution time per contract?**
"Bounded by construction: at most 100 Newton iterations and 200 Brent iterations, each
a fixed number of `exp`, `log`, and `erfc` calls, so there is a hard upper bound. The
typical case is a few Newton iterations. Measured average across the 100k chain was
about 3.5 microseconds per contract (354.08 ms / 100,000) under emulation, including the fallback contracts."

**11. Why release the GIL, and what could go wrong?**
"So other Python threads can run while C++ works. It's safe because I take the raw
pointers first and the core never touches Python objects. The risk is another Python
thread writing to the same arrays during the call; that's a data race and the results
would be garbage, though not a crash, because the arrays are kept alive. Releasing the
GIL also makes it possible to split one batch across threads later."

**12. Why struct of arrays instead of an array of structs?**
"Two reasons. First, zero copy: pandas and NumPy store each column as its own contiguous
array, so SoA maps straight onto that, while an array of structs would need a copy to
interleave. Second, sequential access to each array is ideal for the prefetcher. But to
be honest, this workload is compute-bound: each contract reads 33 bytes and then does
dozens of transcendental function calls. Memory layout mattered here mostly for zero
copy, not for cache misses."

**13. How did you verify the C++ matches the Python?**
"Same algorithm, so I can compare directly. Function by function: `norm_cdf` against
scipy, price and vega to 1e-13 relative. Path by path: for every contract in the 1k chain
the C++ status code says whether Newton or Brent solved it, and I check the Python
reference took the same path and got the same answer within 1e-8. On the full 100k chain
the max absolute IV difference was 1.58e-12. Plus round trips, parity, invalid
inputs, and the C++-only tests."

**14. What would change on a microcontroller with no heap?**
"Very little in the core, because it already doesn't allocate. I'd drop pybind11, compile
with `-fno-exceptions -fno-rtti`, and feed it fixed-size buffers declared `static` or on
the stack, processing the chain in chunks of, say, 64 contracts. The pointer-plus-length
API works unchanged. I'd also check stack usage with `-fstack-usage`, because on a small
MCU the stack is a few KB."

**15. And with no FPU?**
"Double-precision in software is very slow, so I'd change the math. If the chip has a
single-precision FPU, like a Cortex-M4F, I'd move to `float`: epsilon is about 1e-7, so
the 1e-8 price tolerance becomes a relative tolerance around 1e-5, which is still well
inside a bid-ask spread. With no FPU at all I'd use fixed-point (for example Q16.16 or
Q32.32), replace `exp`, `log`, and `erfc` with polynomial or table approximations whose
error I've bounded, and watch for overflow in `S/K` and `sigma^2 T`. Bisection is
attractive there because it needs only comparisons and the pricing function, no
division by vega, and has a fixed iteration count."

---

## 6. Three things I would improve next

1. **A faster normal CDF in C++.** This is the measured bottleneck. The C runtime's
   `erf`/`erfc` path costs 144.81 ms per million values, against 40.72 ms for SciPy
   1.18.1's `ndtr`. Options: a branch-light rational or polynomial approximation with a
   bounded, tested error (so the compiler can vectorize the loop), or restructuring the
   batch so each Newton iteration evaluates the CDF for many contracts at once instead of
   one at a time. Success criterion: `bench_kernels` CDF time well below 40.72 ms per
   million, the 1.58e-12 agreement held (or any loosening documented), and then a re-run
   of the full benchmark.
2. **A better initial guess.** 39.0% of contracts skip Newton almost immediately and go
   to the slower bracketing fallback, because Brenner-Subrahmanyam is only accurate at
   the money. A guess that handles the wings (for example the Corrado-Miller formula, or
   Peter Jaeckel's "Let's Be Rational" method) would move most of those contracts to the
   fast path in every backend. I would measure the fallback share and time per contract
   before and after.
3. **Use the cores, on native hardware.** The batch runs on one of 8 logical CPUs.
   Contracts are independent and the GIL is already released, so splitting the index
   range across `std::thread`s (each writing its own slice of the output: no locks, no
   shared writes) is straightforward. I would measure scaling from 1 to 8 threads, and do
   it with a native ARM64 build instead of under x64 emulation. (CI already builds and
   tests the core on gcc, clang, Apple clang, and MSVC with warnings as errors, so a
   threading change would be checked on all four.)

---

## 7. Resume bullet (measured results only)

> Built a C++17 implied-volatility engine (Newton-Raphson with Brent fallback) for
> options-chain processing, exposed to Python through pybind11 with zero-copy NumPy
> buffers and no heap allocation in the hot loop, processing 282k contracts/s single
> threaded on a 100k-contract chain, 1,506x faster than the original per-row Python
> solver, with IVs matching the Python reference to within 1.6e-12.

Every figure in that bullet is in `bench/results.md` or `bench/agreement.md`.

**Do not claim it beats NumPy.** It did with SciPy 1.17.1 (4.8x) but not with SciPy
1.18.1 at 10k+ contracts. If you want to mention NumPy, make it the story, because it is
a good one:

> Benchmarked the C++ core against a vectorized NumPy baseline and found the advantage
> (4.8x with SciPy 1.17) disappeared on SciPy 1.18; profiled kernels and traced it to the
> normal CDF (SciPy's new implementation is 3.6x faster than the C runtime's), which set
> the next optimization target.

About the old claim, "C++ backend for low-latency options-chain processing": the C++
backend now exists and processes options chains. **"Low-latency" is the weak word.** I
measured batch throughput, not latency per quote update under load, and the numbers were
taken under emulation. If an interviewer asks "what latency?", the honest answer is: "I
measured throughput, about 3.5 microseconds per contract averaged over a 100k batch on a
laptop, single thread. I haven't measured tick-to-result latency." Prefer the bullets
above, which say exactly what was measured.

If asked about the 1,506x figure, say right away that it is mostly removing scipy's
per-call overhead (section 4), and that against well-vectorized NumPy on current SciPy
the C++ core is about even. Volunteering both is much better than having them pulled
out of you.
