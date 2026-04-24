# SIMD + pybind11 + Multithreading Benchmark Showcase

This project demonstrates and benchmarks **SIMD (AVX2) vectorization**, **OpenMP multithreading**, and **Eigen** optimizations for the squared Euclidean distance computation — all exposed to Python via **pybind11**.

It compares **7 implementations** side-by-side: pure Python, NumPy, Numba (parallel), hand-written C++ AVX2 intrinsics, C++ OpenMP+SIMD, Eigen, and Eigen+OpenMP.

## Implementations

| Variant | Description |
|---------|-------------|
| **Standard (pure Python)** | Naive loop in Python — baseline reference |
| **NumPy (`linalg.norm`)** | Vectorized BLAS-backed NumPy call |
| **Numba (parallel)** | JIT-compiled with `@njit(parallel=True)` using `prange` |
| **C++ SIMD (AVX2)** | Hand-written AVX2 intrinsics — processes 4 doubles per instruction via `_mm256_fmadd_pd` |
| **C++ OpenMP+SIMD** | OpenMP thread-parallel loop, each thread using its own AVX2 accumulator |
| **C++ Eigen** | Eigen's lazy-evaluated `(va - vb).squaredNorm()` via `Eigen::Map` |
| **C++ Eigen+OMP** | Manual array partitioning across OpenMP threads, each computing Eigen's `squaredNorm()` independently |

## Benchmark Results

**Array size:** 1,000,000 elements · **Iterations:** 10 · **CPU:** (AVX2-capable)

```
Array size: 1,000,000 elements
Iterations: 10
---------------------------------------------------------------------------
           Standard (pure Python) | avg: 58073.65 us | result: 666627.0179018675
             NumPy (linalg.norm) | avg:  1312.70 us | result: 666627.0179018675
              Numba (parallel) | avg:   133.82 us | result: 666627.0179018675
               C++ SIMD (AVX2) | avg:   441.53 us | result: 666627.0179018675
            C++ OpenMP+SIMD | avg:   108.72 us | result: 666627.0179018675
                  C++ Eigen | avg:   434.39 us | result: 666627.0179018675
               C++ Eigen+OMP | avg:   107.70 us | result: 666627.0179018675
---------------------------------------------------------------------------

Speedup vs Standard (pure Python):
           Standard (pure Python):   1.00x
             NumPy (linalg.norm):  44.24x
              Numba (parallel): 433.96x
               C++ SIMD (AVX2): 131.53x
            C++ OpenMP+SIMD: 534.17x
                  C++ Eigen: 133.69x
               C++ Eigen+OMP: 539.22x
```

### Key Takeaways

- **C++ Eigen+OMP** is the fastest overall at **~539× speedup** over pure Python.
- **C++ OpenMP+SIMD** is nearly identical (~534×), showing that for this memory-bound operation, threading is the dominant factor.
- **Numba (parallel)** is remarkably competitive at ~434×, outperforming the single-threaded C++ variants — a testament to Numba's JIT + auto-parallelization.
*Important Note: Although Numba implementation might seem fast and simple (which it is!), this performance is because of the C++ implementation loading with pybind11, meaning in more complex and practical scenarios, it will fall behind the optimized C++ versions.*
- **NumPy** provides a solid 44× speedup with zero code effort.
- The single-threaded **C++ SIMD (AVX2)** and **Eigen** variants both land around ~132×, confirming that Eigen's lazy evaluation produces near-hand-tuned SIMD code.
*Key takeaway is that some implementations have performance advantages but their performance compared to each otehr and the choice to use them depends on the specific use case. My suggestion is always using a dedicated C++ kernel using Eigen+OMP for the most intense and large computations, while smaller sized ones can benefit  from the simplicity of the Numba approach while achieving good performance. It's also very benifical to be aware and use the proper numpy practices since it comes with highly optimized objects and simple operations.*

## Project Structure

```
├── cpp/euclid_distance/
│   ├── bindings.cpp          # pybind11 module definition
│   ├── simd_ops.h            # Header declaring all distance functions
│   ├── simd_ops.cpp          # Hand-written AVX2 intrinsics
│   ├── simd_omp.cpp          # OpenMP + AVX2 parallel implementation
│   ├── eigen_ops.cpp         # Eigen-based implementation
│   ├── eigen_omp.cpp         # Eigen + OpenMP parallel implementation
│   ├── main.cpp              # C++ standalone benchmark harness
│   ├── build_pyd.bat         # Script to build the .pyd extension
│   └── ED.md                 # C++-only benchmark details
├── py/euclid_distance/
│   ├── main.py               # Python benchmark runner
│   └── euclid_distance_cpp.*.pyd  # Built extension module
├── pyproject.toml
├── uv.lock
└── README.md
```

## How to Run

### Prerequisites

- Python ≥ 3.10
- A C++ compiler with AVX2, FMA, and OpenMP support (e.g., MinGW-w64 / MSYS2)
- [Eigen](https://eigen.tuxfamily.org/) (header-only, no installation needed beyond include path)

### Using `uv` (recommended)

```bash
# Install dependencies
uv sync

# Run the Python benchmark
uv run python py/euclid_distance/main.py
```

> **Note:** The `.pyd` extension module must be built first (see below) before running the Python benchmark. The pre-built `.pyd` is not distributed — you build it for your own system.

### Standalone C++ Benchmark

If you only want the C++ results without Python:

```bash
cd cpp/euclid_distance
g++ -std=c++17 -mavx2 -mfma -fopenmp -O2 -I <eigen-include-dir> -o bench.exe main.cpp simd_ops.cpp simd_omp.cpp eigen_ops.cpp eigen_omp.cpp
./bench.exe
```

## License

MIT
