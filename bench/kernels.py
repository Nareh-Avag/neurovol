"""Time the array kernels the NumPy IV backend spends its time in (1M float64).

Run under different NumPy/SciPy versions to see which kernel changed. Usage:
    python bench/kernels.py
"""
import timeit

import numpy as np
import scipy
from scipy.special import ndtr

rng = np.random.default_rng(0)
x = rng.uniform(-4, 4, 1_000_000)
pos = rng.uniform(0.5, 2.0, 1_000_000)
print(f"numpy {np.__version__} scipy {scipy.__version__}  (1M float64, median of 15)")
for name, fn in [("ndtr", lambda: ndtr(x)), ("np.exp", lambda: np.exp(x)),
                 ("np.log", lambda: np.log(pos)), ("np.sqrt", lambda: np.sqrt(pos)),
                 ("multiply", lambda: x * pos)]:
    t = timeit.repeat(fn, number=1, repeat=15)
    print(f"  {name:9s} {np.median(t) * 1e3:7.2f} ms")
