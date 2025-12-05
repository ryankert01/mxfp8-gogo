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

namespace mxfp8 {

/**
 * Sequential matrix multiplication for MXFP8 matrices
 * 
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @return Result matrix C (M x N)
 */
inline MxfpMatrix matmul_sequential(const MxfpMatrix& A, const MxfpMatrix& B) {
    size_t M = A.rows;
    size_t K = A.cols;
    size_t N = B.cols;
    
    // First compute result in float, then convert to MXFP8
    std::vector<float> result_float(M * N, 0.0f);
    
    // Classic O(M*N*K) matrix multiplication
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                // Get values with scaling applied
                float a_val = A.get(i, k);
                float b_val = B.get(k, j);
                sum += a_val * b_val;
            }
            result_float[i * N + j] = sum;
        }
    }
    
    // Convert result to MXFP8
    return MxfpMatrix::from_float(result_float, M, N);
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
    size_t M = A.rows;
    size_t K = A.cols;
    size_t N = B.cols;
    
    std::vector<float> result(M * N, 0.0f);
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float a_val = A.get(i, k);
                float b_val = B.get(k, j);
                sum += a_val * b_val;
            }
            result[i * N + j] = sum;
        }
    }
    
    return result;
}

/**
 * Reference float matrix multiplication for verification
 */
inline std::vector<float> matmul_reference(const std::vector<float>& A, 
                                           const std::vector<float>& B,
                                           size_t M, size_t K, size_t N) {
    std::vector<float> C(M * N, 0.0f);
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    
    return C;
}

} // namespace mxfp8

#endif // MXFP8_MATMUL_SEQUENTIAL_HPP
