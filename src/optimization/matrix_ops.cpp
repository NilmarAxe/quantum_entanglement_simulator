#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include 
#include 

namespace py = pybind11;
using namespace Eigen;

// Fast complex matrix multiplication using Eigen
py::array_t<std::complex> fast_matrix_multiply(
    py::array_t<std::complex> a,
    py::array_t<std::complex> b) {
    
    py::buffer_info buf_a = a.request();
    py::buffer_info buf_b = b.request();
    
    if (buf_a.ndim != 2 || buf_b.ndim != 2)
        throw std::runtime_error("Input arrays must be 2-dimensional");
    
    size_t rows_a = buf_a.shape[0];
    size_t cols_a = buf_a.shape[1];
    size_t rows_b = buf_b.shape[0];
    size_t cols_b = buf_b.shape[1];
    
    if (cols_a != rows_b)
        throw std::runtime_error("Matrix dimensions incompatible for multiplication");
    
    // Map numpy arrays to Eigen matrices
    Map<Matrix<std::complex, Dynamic, Dynamic, RowMajor>> mat_a(
        static_cast<std::complex*>(buf_a.ptr), rows_a, cols_a);
    Map<Matrix<std::complex, Dynamic, Dynamic, RowMajor>> mat_b(
        static_cast<std::complex*>(buf_b.ptr), rows_b, cols_b);
    
    // Perform multiplication
    MatrixXcd result = mat_a * mat_b;
    
    // Create output array
    py::array_t<std::complex> output({rows_a, cols_b});
    py::buffer_info buf_out = output.request();
    
    std::complex* ptr_out = static_cast<std::complex*>(buf_out.ptr);
    
    for (size_t i = 0; i < rows_a; ++i) {
        for (size_t j = 0; j < cols_b; ++j) {
            ptr_out[i * cols_b + j] = result(i, j);
        }
    }
    
    return output;
}

// Fast eigenvalue computation
py::array_t fast_eigenvalues(py::array_t<std::complex> matrix) {
    py::buffer_info buf = matrix.request();
    
    if (buf.ndim != 2)
        throw std::runtime_error("Input must be 2-dimensional");
    
    size_t n = buf.shape[0];
    
    if (n != static_cast(buf.shape[1]))
        throw std::runtime_error("Matrix must be square");
    
    Map<Matrix<std::complex, Dynamic, Dynamic, RowMajor>> mat(
        static_cast<std::complex*>(buf.ptr), n, n);
    
    // Compute eigenvalues using Eigen's solver
    SelfAdjointEigenSolver solver(mat);
    VectorXd eigenvalues = solver.eigenvalues();
    
    // Create output array
    py::array_t output(n);
    py::buffer_info buf_out = output.request();
    double* ptr_out = static_cast(buf_out.ptr);
    
    for (size_t i = 0; i < n; ++i) {
        ptr_out[i] = eigenvalues(i);
    }
    
    return output;
}

// Tensor product for quantum states
py::array_t<std::complex> tensor_product(
    py::array_t<std::complex> a,
    py::array_t<std::complex> b) {
    
    py::buffer_info buf_a = a.request();
    py::buffer_info buf_b = b.request();
    
    size_t rows_a = buf_a.shape[0];
    size_t cols_a = buf_a.shape[1];
    size_t rows_b = buf_b.shape[0];
    size_t cols_b = buf_b.shape[1];
    
    size_t rows_out = rows_a * rows_b;
    size_t cols_out = cols_a * cols_b;
    
    py::array_t<std::complex> output({rows_out, cols_out});
    py::buffer_info buf_out = output.request();
    
    std::complex* ptr_a = static_cast<std::complex*>(buf_a.ptr);
    std::complex* ptr_b = static_cast<std::complex*>(buf_b.ptr);
    std::complex* ptr_out = static_cast<std::complex*>(buf_out.ptr);
    
    for (size_t i = 0; i < rows_a; ++i) {
        for (size_t j = 0; j < cols_a; ++j) {
            for (size_t k = 0; k < rows_b; ++k) {
                for (size_t l = 0; l < cols_b; ++l) {
                    size_t out_row = i * rows_b + k;
                    size_t out_col = j * cols_b + l;
                    ptr_out[out_row * cols_out + out_col] = 
                        ptr_a[i * cols_a + j] * ptr_b[k * cols_b + l];
                }
            }
        }
    }
    
    return output;
}

PYBIND11_MODULE(matrix_ops_cpp, m) {
    m.doc() = "C++ accelerated matrix operations for quantum simulation";
    
    m.def("fast_matrix_multiply", &fast_matrix_multiply,
          "Fast complex matrix multiplication using Eigen",
          py::arg("a"), py::arg("b"));
    
    m.def("fast_eigenvalues", &fast_eigenvalues,
          "Fast eigenvalue computation for Hermitian matrices",
          py::arg("matrix"));
    
    m.def("tensor_product", &tensor_product,
          "Compute tensor product of two matrices",
          py::arg("a"), py::arg("b"));
}