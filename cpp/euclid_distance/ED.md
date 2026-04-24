# Squared Euclidean Distance — SIMD (AVX2) & Eigen Benchmark

## How to Build & Run

```bash
cd cpp/euclid_distance

# Compile with AVX2, FMA, OpenMP, Eigen (make sure to adjust the include path for Eigen)
g++ -std=c++17 -mavx2 -mfma -fopenmp -O2 -I C:/msys64/ucrt64/include/eigen3 -o bench.exe main.cpp simd_ops.cpp simd_omp.cpp eigen_ops.cpp

# Run
./bench.exe
```

## Files

| File | Role |
|------|------|
| [`main.cpp`](main.cpp) | Baseline scalar implementation + benchmark harness |
| [`simd_ops.h`](simd_ops.h) | Header declaring all distance functions |
| [`simd_ops.cpp`](simd_ops.cpp) | Hand-written AVX2 implementation (4 doubles/iter via `_mm256_fmadd_pd`) |
| [`simd_omp.cpp`](simd_omp.cpp) | OpenMP + AVX2 parallel implementation |
| [`eigen_ops.cpp`](eigen_ops.cpp) | Eigen-based implementation using lazy `(va - vb).squaredNorm()` |

## Results (1,000,000 elements, 100 iterations, `-O2`)

| Variant | Avg time (μs) | Speedup vs Scalar |
|---------|--------------|-------------------|
| Scalar | 1262.8 | 1.00× |
| SIMD (AVX2) | 1045.7 | 1.21× |
| OpenMP+SIMD | 827.9 | 1.53× |
| **Eigen** | **745.9** | **1.69×** |

## Explanation

- **SIMD (AVX2):** Uses `_mm256_loadu_pd`, `_mm256_sub_pd`, and `_mm256_fmadd_pd` to process **4 doubles per instruction**. Guarantees vectorization regardless of compiler flags.
- **OpenMP+SIMD:** Splits the work across threads (`#pragma omp for`), each thread using its own AVX2 accumulator. Adds thread-level parallelism on top of vectorization.
- **Eigen:** Uses `Eigen::Map` to wrap raw arrays without copying, then `(va - vb).squaredNorm()` — Eigen's lazy evaluation engine fuses the subtraction and norm into a single optimized SIMD pass, outperforming the hand-written version.
For maximum performance, use multithreading with Eigen's SIMD-optimized operations. Eigen's implementation is additionaly more memory efficient due to expression templates, which avoid unnecessary temporary vectors.
- The tiny differences between results (~2–9e-3) are due to floating-point associativity: different summation orderings change rounding.
