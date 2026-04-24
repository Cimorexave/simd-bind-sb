#ifndef SIMD_OPS_H
#define SIMD_OPS_H

#include <immintrin.h>  // AVX2
#include <cstddef>

// compute squared Euclidean distance using AVX2 (256-bit SIMD)
// both arrays should be aligned to 32 bytes for best performance.
double simd_squared_euclidean_distance(const double* a, const double* b, std::size_t n);

// same computation but parallelized with OpenMP (SIMD inside each thread)
double multithread_simd_squared_euclidean_distance(const double* a, const double* b, std::size_t n);

// compute squared Euclidean distance using Eigen library
double eigen_squared_euclidean_distance(const double* a, const double* b, std::size_t n);

// compute squared Euclidean distance using Eigen + OpenMP (manual chunking)
double eigen_omp_squared_euclidean_distance(const double* a, const double* b, std::size_t n);

#endif // SIMD_OPS_H
