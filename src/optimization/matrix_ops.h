#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>

namespace py = pybind11;

// Function declarations
py::array_t<std::complex> fast_matrix_multiply(
    py::array_t<std::complex> a,
    py::array_t<std::complex> b);

py::array_t fast_eigenvalues(
    py::array_t<std::complex> matrix);

py::array_t<std::complex> tensor_product(
    py::array_t<std::complex> a,
    py::array_t<std::complex> b);

#endif // MATRIX_OPS_H