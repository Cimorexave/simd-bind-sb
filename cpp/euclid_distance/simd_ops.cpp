#include "simd_ops.h"
#include <cstring>

double simd_squared_euclidean_distance(const double* a, const double* b, std::size_t n) {
    __m256d sum_vec = _mm256_setzero_pd();

    std::size_t i = 0;

    // process 4 doubles at a time (256-bit AVX2)
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d diff = _mm256_sub_pd(va, vb); 
        sum_vec = _mm256_fmadd_pd(diff, diff, sum_vec);  // sum += diff * diff
    }

    // horizontal sum of the 4 lanes
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, sum_vec);
    double result = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    // handle remaining elements (0–3)
    for (; i < n; ++i) {
        double d = a[i] - b[i];
        result += d * d;
    }

    return result;
}
