import os
import sys

# Add MSYS2/MinGW DLL paths so the .pyd can find its runtime dependencies
_mingw_bin = r"C:\msys64\ucrt64\bin"
_python_dll_dir = r"C:\Users\sadeq\AppData\Roaming\uv\python\cpython-3.10-windows-x86_64-none"
for _d in (_mingw_bin, _python_dll_dir):
    if os.path.isdir(_d):
        os.add_dll_directory(_d)

import numpy as np
from numba import njit, prange
import time

# Import C++ pybind11 bindings
from euclid_distance_cpp import (
    simd_sqeuclidean,
    multithread_simd_sqeuclidean,
    eigen_sqeuclidean,
    eigen_omp_sqeuclidean,
)

# create two large arrays of double with the size of 1 million
size = 1_000_000
a = np.random.rand(size).astype(np.float64)
b = np.random.rand(size).astype(np.float64)


@njit(parallel=True, fastmath=True)
def numba_euclidean_distance(a, b):
    n = a.size
    sum = 0.0
    for i in prange(n):
        d = a[i] - b[i]
        sum += d * d
    return np.sqrt(sum)


def np_euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def euclidean_distance(a, b):
    sum = 0.0
    for i in range(a.size):
        d = a[i] - b[i]
        sum += d * d
    return np.sqrt(sum)


# --- C++ wrappers (return squared distance, so we sqrt to match) ---
def cpp_simd_euclidean(a, b):
    return np.sqrt(simd_sqeuclidean(a, b))


def cpp_multithread_simd_euclidean(a, b):
    return np.sqrt(multithread_simd_sqeuclidean(a, b))


def cpp_eigen_euclidean(a, b):
    return np.sqrt(eigen_sqeuclidean(a, b))


def cpp_eigen_omp_euclidean(a, b):
    return np.sqrt(eigen_omp_sqeuclidean(a, b))


def benchmark(label, func, a, b, iterations=10):
    """Time a function over multiple iterations, return avg seconds."""
    # Warm-up
    _ = func(a, b)

    start = time.perf_counter()
    for _ in range(iterations):
        _ = func(a, b)
    elapsed = time.perf_counter() - start

    avg = elapsed / iterations
    print(f"{label:>30s} | avg: {avg*1e6:>8.2f} us | result: {func(a, b):.10f}")
    return avg


if __name__ == "__main__":
    print(f"Array size: {size:,} elements")
    print(f"Iterations: 10")
    print("-" * 75)

    # warm-up / accuracy
    ref = euclidean_distance(a, b)

    funcs = [
        ("Standard (pure Python)", euclidean_distance),
        ("NumPy (linalg.norm)", np_euclidean_distance),
        ("Numba (parallel)", numba_euclidean_distance),
        ("C++ SIMD (AVX2)", cpp_simd_euclidean),
        ("C++ OpenMP+SIMD", cpp_multithread_simd_euclidean),
        ("C++ Eigen", cpp_eigen_euclidean),
        ("C++ Eigen+OMP", cpp_eigen_omp_euclidean),
    ]

    times = {}
    for label, func in funcs:
        t = benchmark(label, func, a, b, iterations=10)
        times[label] = t

    print("-" * 75)

    # Speedups vs Standard
    base = times["Standard (pure Python)"]
    print(f"\nSpeedup vs Standard (pure Python):")
    for label, t in times.items():
        if t > 0:
            print(f"  {label:>30s}: {base / t:>6.2f}x")

    # Accuracy check
    print("\nAccuracy check (vs Standard):")
    for label, func in funcs:
        diff = abs(ref - func(a, b))
        ok = "OK" if diff < 1e-6 else "MISMATCH"
        print(f"  {label:>30s}: diff={diff:.2e}  [{ok}]")
