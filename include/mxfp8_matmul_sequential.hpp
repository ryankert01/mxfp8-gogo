/**
 * MXFP8 Matrix Multiplication - Sequential Implementation
 * 
 * Vanilla sequential matrix multiplication using MXFP8 format.
 * A (M x K) * B (K x N) = C (M x N)
 */

#ifndef MXFP8_MATMUL_SEQUENTIAL_HPP
#define MXFP8_MATMUL_SEQUENTIAL_HPP

#include "mxfp8.hpp"
#include <vector>
#include <stdexcept>

namespace mxfp8 {

/**
 * Reference float matrix multiplication
 * Uses cache-friendly i,k,j loop order to avoid striding through B matrix
 */
inline std::vector<float> matmul_reference(const std::vector<float>& A, 
                                           const std::vector<float>& B,
                                           size_t M, size_t K, size_t N) {
    std::vector<float> C(M * N, 0.0f);
    
    // i,k,j loop order for better cache utilization
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            const float a_val = A[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
    
    return C;
}

/**
 * Sequential matrix multiplication returning float result
 * (for verification purposes)
 * 
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @return Result as float vector (M x N)
 */
inline std::vector<float> matmul_sequential_float(const MxfpMatrix& A, const MxfpMatrix& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    // Convert to float first for efficiency
    std::vector<float> A_float = A.to_float();
    std::vector<float> B_float = B.to_float();
    
    return matmul_reference(A_float, B_float, A.rows, A.cols, B.cols);
}

/**
 * Sequential matrix multiplication for MXFP8 matrices
 * 
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @return Result matrix C (M x N)
 */
inline MxfpMatrix matmul_sequential(const MxfpMatrix& A, const MxfpMatrix& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    // Perform sequential multiplication on float representations
    std::vector<float> result_float = matmul_sequential_float(A, B);
    
    // Convert result to MXFP8
    return MxfpMatrix::from_float(result_float, A.rows, B.cols);
}

} // namespace mxfp8

#endif // MXFP8_MATMUL_SEQUENTIAL_HPP
