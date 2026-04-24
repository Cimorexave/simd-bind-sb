#include "simd_ops.h"
#include <omp.h>
#include <cstddef>

double multithread_simd_squared_euclidean_distance(const double* a, const double* b, std::size_t n) {
    double result = 0.0;

    // Number of full 4-element chunks
    std::ptrdiff_t chunk_count = static_cast<std::ptrdiff_t>(n / 4);

    // Each thread gets its own partial sum to avoid false sharing
    #pragma omp parallel
    {
        __m256d local_sum = _mm256_setzero_pd();

        #pragma omp for nowait
        for (std::ptrdiff_t c = 0; c < chunk_count; ++c) {
            std::size_t i = static_cast<std::size_t>(c) * 4;
            __m256d va = _mm256_loadu_pd(a + i);
            __m256d vb = _mm256_loadu_pd(b + i);
            __m256d diff = _mm256_sub_pd(va, vb);
            local_sum = _mm256_fmadd_pd(diff, diff, local_sum);
        }

        // Horizontal sum of this thread's 4 lanes
        alignas(32) double tmp[4];
        _mm256_store_pd(tmp, local_sum);
        double thread_partial = tmp[0] + tmp[1] + tmp[2] + tmp[3];

        // Reduce into global result (atomic to avoid race)
        #pragma omp atomic
        result += thread_partial;
    }

    // Handle remaining elements (0–3) — single-threaded
    std::size_t remainder_start = chunk_count * 4;
    for (std::size_t i = remainder_start; i < n; ++i) {
        double d = a[i] - b[i];
        result += d * d;
    }

    return result;
}
