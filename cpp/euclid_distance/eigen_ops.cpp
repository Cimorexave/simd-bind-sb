#include "simd_ops.h"
#include <Eigen/Dense>

double eigen_squared_euclidean_distance(const double* a, const double* b, std::size_t n) {
    // map raw arrays to Eigen vectors (no copy)
    Eigen::Map<const Eigen::VectorXd> va(a, static_cast<Eigen::Index>(n));
    Eigen::Map<const Eigen::VectorXd> vb(b, static_cast<Eigen::Index>(n));

    // lazy expression: no temporary vector allocated.
    // Eigen fuses (va - vb).squaredNorm() into a single SIMD-optimized pass.
    return (va - vb).squaredNorm();
}
