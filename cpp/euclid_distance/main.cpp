#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>

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
    std::cout << std::string(65, '-') << "\n";

    // Verify correctness
    double expected = scalar_squared_euclidean_distance(a, b, N);
    double actual   = simd_squared_euclidean_distance(a, b, N);
    double diff     = std::abs(expected - actual);

    std::cout << "\nCorrectness check:\n";
    std::cout << "  Scalar result: " << expected << "\n";
    std::cout << "  SIMD   result: " << actual   << "\n";
    std::cout << "  Difference:    " << diff     << "\n\n";

    // Benchmark
    double scalar_us = benchmark("Scalar", scalar_squared_euclidean_distance, a, b, N, ITERATIONS);
    double simd_us   = benchmark("SIMD (AVX2)", simd_squared_euclidean_distance, a, b, N, ITERATIONS);

    std::cout << std::string(65, '-') << "\n";
    if (simd_us > 0.0) {
        std::cout << "Speedup: " << std::fixed << std::setprecision(2)
                  << (scalar_us / simd_us) << "x\n";
    }

    _mm_free(a);
    _mm_free(b);
    return 0;
}
