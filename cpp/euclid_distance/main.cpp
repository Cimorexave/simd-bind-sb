#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>
#include <omp.h>

#include "simd_ops.h"

// -------------------------------------------------------------------
// baseline: plain scalar squared Euclidean distance
// -------------------------------------------------------------------
double scalar_squared_euclidean_distance(const double* a, const double* b, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// -------------------------------------------------------------------
// benchmark helper
// -------------------------------------------------------------------
template <typename Func>
double benchmark(const char* label, Func f,
                 const double* a, const double* b, std::size_t n,
                 int iterations) {

    // Warm-up
    volatile double warm = f(a, b, n);
    (void)warm;

    auto start = std::chrono::high_resolution_clock::now();
    double result = 0.0;
    for (int iter = 0; iter < iterations; ++iter) {
        result += f(a, b, n);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    double avg_us = (elapsed.count() * 1'000'000.0) / iterations;

    std::cout << std::left << std::setw(20) << label
              << " | avg time: " << std::setw(10) << avg_us << " us"
              << " | result (partial): " << result / iterations
              << std::endl;

    return avg_us;
}

// -------------------------------------------------------------------
// main
// -------------------------------------------------------------------
int main() {
    constexpr std::size_t N = 1'000'000;
    constexpr int ITERATIONS = 100;

    // Allocate aligned memory for potential SIMD alignment benefit
    double* a = static_cast<double*>(_mm_malloc(N * sizeof(double), 32));
    double* b = static_cast<double*>(_mm_malloc(N * sizeof(double), 32));

    if (!a || !b) {
        std::cerr << "Allocation failed.\n";
        return 1;
    }

    // Fill with random data
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    std::cout << "Squared Euclidean Distance Benchmark\n";
    std::cout << "Array size: " << N << " elements\n";
    std::cout << "Iterations: " << ITERATIONS << "\n";
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
    std::cout << std::string(75, '-') << "\n";

    // Verify correctness
    double expected = scalar_squared_euclidean_distance(a, b, N);
    double simd     = simd_squared_euclidean_distance(a, b, N);
    double omp_simd = multithread_simd_squared_euclidean_distance(a, b, N);
    double eigen    = eigen_squared_euclidean_distance(a, b, N);

    std::cout << "\nCorrectness check:\n";
    std::cout << "  Scalar result:           " << expected << "\n";
    std::cout << "  SIMD result:             " << simd     << "\n";
    std::cout << "  OpenMP+SIMD result:      " << omp_simd << "\n";
    std::cout << "  Eigen result:            " << eigen    << "\n";
    std::cout << "  Scalar vs SIMD diff:     " << std::abs(expected - simd)      << "\n";
    std::cout << "  Scalar vs OMP+SIMD diff: " << std::abs(expected - omp_simd)  << "\n";
    std::cout << "  Scalar vs Eigen diff:    " << std::abs(expected - eigen)     << "\n\n";

    // Benchmark
    double scalar_us = benchmark("Scalar", scalar_squared_euclidean_distance, a, b, N, ITERATIONS);
    double simd_us   = benchmark("SIMD (AVX2)", simd_squared_euclidean_distance, a, b, N, ITERATIONS);
    double omp_us    = benchmark("OpenMP+SIMD", multithread_simd_squared_euclidean_distance, a, b, N, ITERATIONS);
    double eigen_us  = benchmark("Eigen", eigen_squared_euclidean_distance, a, b, N, ITERATIONS);

    std::cout << std::string(75, '-') << "\n";
    if (simd_us > 0.0) {
        std::cout << "Speedup (SIMD vs Scalar):        " << std::fixed << std::setprecision(2)
                  << (scalar_us / simd_us) << "x\n";
    }
    if (omp_us > 0.0) {
        std::cout << "Speedup (OpenMP+SIMD vs Scalar):  " << std::fixed << std::setprecision(2)
                  << (scalar_us / omp_us) << "x\n";
        std::cout << "Speedup (OpenMP+SIMD vs SIMD):    " << std::fixed << std::setprecision(2)
                  << (simd_us / omp_us) << "x\n";
    }
    if (eigen_us > 0.0) {
        std::cout << "Speedup (Eigen vs Scalar):        " << std::fixed << std::setprecision(2)
                  << (scalar_us / eigen_us) << "x\n";
        std::cout << "Speedup (Eigen vs SIMD):          " << std::fixed << std::setprecision(2)
                  << (simd_us / eigen_us) << "x\n";
    }

    _mm_free(a);
    _mm_free(b);
    return 0;
}
