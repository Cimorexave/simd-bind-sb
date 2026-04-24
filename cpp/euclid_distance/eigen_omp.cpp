#include "simd_ops.h"
#include <Eigen/Dense>
#include <omp.h>

double eigen_omp_squared_euclidean_distance(const double* a, const double* b, std::size_t n) {
    double result = 0.0;

    #pragma omp parallel
    {
        std::size_t tid   = static_cast<std::size_t>(omp_get_thread_num());
        std::size_t nthreads = static_cast<std::size_t>(omp_get_num_threads());

        std::size_t chunk = (n + nthreads - 1) / nthreads;
        std::size_t start = tid * chunk;
        std::size_t end   = std::min(start + chunk, n);
        std::size_t len   = end - start;

        double partial = 0.0;
        if (len > 0) {
            // Map segment — no copy
            Eigen::Map<const Eigen::VectorXd> va(a + start, static_cast<Eigen::Index>(len));
            Eigen::Map<const Eigen::VectorXd> vb(b + start, static_cast<Eigen::Index>(len));

            // Lazy expression: Eigen fuses (va - vb).squaredNorm() into SIMD
            partial = (va - vb).squaredNorm();
        }

        #pragma omp atomic
        result += partial;
    }

    return result;
}
