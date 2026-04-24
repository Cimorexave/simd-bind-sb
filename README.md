### SIMD instructions

This project acts as a playground for SIMD instructions. It is benchmarked against conventional implementations of the same algorithms and OpenMP parallelized implementations. The SIMD instructions are implemented using intrinsics, which are functions that map directly to specific CPU instructions.

`pybind11` is used to create Python bindings for the C++ code, allowing us to call the SIMD implementations from Python.