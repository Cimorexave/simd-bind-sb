#ifndef SIMD_OPS_H
#define SIMD_OPS_H

#include <immintrin.h>  // AVX2
#include <cstddef>

// compute squared Euclidean distance using AVX2 (256-bit SIMD)
// both arrays must be aligned to 32 bytes for best performance.
double simd_squared_euclidean_distance(const double* a, const double* b, std::size_t n);

#endif // SIMD_OPS_H
