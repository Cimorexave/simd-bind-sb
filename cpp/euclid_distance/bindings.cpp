#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "simd_ops.h"

namespace py = pybind11;

// Wrappers that accept NumPy arrays
double py_simd_sqeuclidean(py::array_t<double> a, py::array_t<double> b) {
    py::buffer_info ai = a.request(), bi = b.request();
    if (ai.size != bi.size)
        throw std::runtime_error("Arrays must have the same size");
    return simd_squared_euclidean_distance(
        static_cast<const double*>(ai.ptr),
        static_cast<const double*>(bi.ptr),
        static_cast<std::size_t>(ai.size));
}

double py_multithread_simd_sqeuclidean(py::array_t<double> a, py::array_t<double> b) {
    py::buffer_info ai = a.request(), bi = b.request();
    if (ai.size != bi.size)
        throw std::runtime_error("Arrays must have the same size");
    return multithread_simd_squared_euclidean_distance(
        static_cast<const double*>(ai.ptr),
        static_cast<const double*>(bi.ptr),
        static_cast<std::size_t>(ai.size));
}

double py_eigen_sqeuclidean(py::array_t<double> a, py::array_t<double> b) {
    py::buffer_info ai = a.request(), bi = b.request();
    if (ai.size != bi.size)
        throw std::runtime_error("Arrays must have the same size");
    return eigen_squared_euclidean_distance(
        static_cast<const double*>(ai.ptr),
        static_cast<const double*>(bi.ptr),
        static_cast<std::size_t>(ai.size));
}

double py_eigen_omp_sqeuclidean(py::array_t<double> a, py::array_t<double> b) {
    py::buffer_info ai = a.request(), bi = b.request();
    if (ai.size != bi.size)
        throw std::runtime_error("Arrays must have the same size");
    return eigen_omp_squared_euclidean_distance(
        static_cast<const double*>(ai.ptr),
        static_cast<const double*>(bi.ptr),
        static_cast<std::size_t>(ai.size));
}

PYBIND11_MODULE(euclid_distance_cpp, m) {
    m.doc() = "C++ SIMD/Eigen squared Euclidean distance bindings";

    m.def("simd_sqeuclidean", &py_simd_sqeuclidean,
          "Squared Euclidean distance via hand-written AVX2");
    m.def("multithread_simd_sqeuclidean", &py_multithread_simd_sqeuclidean,
          "Squared Euclidean distance via OpenMP + AVX2");
    m.def("eigen_sqeuclidean", &py_eigen_sqeuclidean,
          "Squared Euclidean distance via Eigen");
    m.def("eigen_omp_sqeuclidean", &py_eigen_omp_sqeuclidean,
          "Squared Euclidean distance via Eigen + OpenMP");
}
