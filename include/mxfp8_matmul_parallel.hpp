/**
 * MXFP8 Matrix Multiplication - Parallel Implementation
 * 
 * Optimized parallel matrix multiplication using:
 * - pthread for multi-threading
 * - Cache blocking (tiling) for better cache utilization
 * - SIMD (AVX2) vectorization
 */

#ifndef MXFP8_MATMUL_PARALLEL_HPP
#define MXFP8_MATMUL_PARALLEL_HPP

#include "mxfp8.hpp"
#include <vector>
#include <pthread.h>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <stdexcept>

#ifdef __x86_64__
#include <immintrin.h>
#endif

namespace mxfp8 {

// Cache line size for alignment
constexpr size_t TILE_SIZE = 32; // Tile size for cache blocking
constexpr size_t DEFAULT_THREAD_COUNT = 4; // Default number of threads when hardware_concurrency fails

/**
 * Thread arguments for parallel matrix multiplication
 */
struct ThreadArgs {
    const float* A;
    const float* B;
    float* C;
    size_t M, K, N;
    size_t row_start;
    size_t row_end;
};

/**
 * Worker function for each thread
 * Uses cache blocking (tiling) for better cache utilization
 */
inline void* matmul_worker(void* arg) {
    ThreadArgs* args = static_cast<ThreadArgs*>(arg);
    
    const float* A = args->A;
    const float* B = args->B;
    float* C = args->C;
    size_t K = args->K;
    size_t N = args->N;
    
    // Process assigned rows with tiling for cache optimization
    for (size_t i = args->row_start; i < args->row_end; ++i) {
        // Tile-based computation for cache efficiency
        for (size_t kk = 0; kk < K; kk += TILE_SIZE) {
            size_t k_end = std::min(kk + TILE_SIZE, K);
            
            for (size_t jj = 0; jj < N; jj += TILE_SIZE) {
                size_t j_end = std::min(jj + TILE_SIZE, N);
                
                for (size_t k = kk; k < k_end; ++k) {
                    float a_val = A[i * K + k];
                    
#ifdef __x86_64__
                    // SIMD vectorization using AVX
                    size_t j = jj;
                    for (; j + 8 <= j_end; j += 8) {
                        __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                        __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                        __m256 a_vec = _mm256_set1_ps(a_val);
                        c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                        _mm256_storeu_ps(&C[i * N + j], c_vec);
                    }
                    // Handle remaining elements
                    for (; j < j_end; ++j) {
                        C[i * N + j] += a_val * B[k * N + j];
                    }
#else
                    // Scalar fallback
                    for (size_t j = jj; j < j_end; ++j) {
                        C[i * N + j] += a_val * B[k * N + j];
                    }
#endif
                }
            }
        }
    }
    
    return nullptr;
}

/**
 * Parallel matrix multiplication using pthreads with SIMD and cache optimization
 * 
 * @param A Input matrix A (M x K) in float format (pre-converted from MXFP8)
 * @param B Input matrix B (K x N) in float format
 * @param M Number of rows in A
 * @param K Common dimension (columns in A, rows in B)
 * @param N Number of columns in B
 * @param num_threads Number of threads to use (0 = auto-detect)
 * @return Result matrix C (M x N) as float vector
 */
inline std::vector<float> matmul_parallel_float(const std::vector<float>& A,
                                                const std::vector<float>& B,
                                                size_t M, size_t K, size_t N,
                                                size_t num_threads = 0) {
    // Validate input matrix sizes
    if (A.size() != M * K || B.size() != K * N) {
        throw std::invalid_argument("Matrix dimensions don't match provided sizes");
    }
    
    // Auto-detect number of threads
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = DEFAULT_THREAD_COUNT;
    }
    
    // Limit threads to number of rows
    num_threads = std::min(num_threads, M);
    
    // Allocate result matrix (cache-line aligned would be ideal but vector is simpler)
    std::vector<float> C(M * N, 0.0f);
    
    // Create thread pool
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadArgs> args(num_threads);
    
    // Distribute rows among threads
    size_t rows_per_thread = M / num_threads;
    size_t remainder = M % num_threads;
    
    size_t row_start = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t rows = rows_per_thread + (t < remainder ? 1 : 0);
        
        args[t].A = A.data();
        args[t].B = B.data();
        args[t].C = C.data();
        args[t].M = M;
        args[t].K = K;
        args[t].N = N;
        args[t].row_start = row_start;
        args[t].row_end = row_start + rows;
        
        int rc = pthread_create(&threads[t], nullptr, matmul_worker, &args[t]);
        if (rc != 0) {
            // Wait for already created threads and throw error
            for (size_t i = 0; i < t; ++i) {
                pthread_join(threads[i], nullptr);
            }
            throw std::runtime_error("Failed to create thread");
        }
        row_start += rows;
    }
    
    // Wait for all threads to complete
    for (size_t t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], nullptr);
    }
    
    return C;
}

/**
 * Parallel matrix multiplication for MXFP8 matrices
 * 
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param num_threads Number of threads to use (0 = auto-detect)
 * @return Result matrix C (M x N)
 */
inline MxfpMatrix matmul_parallel(const MxfpMatrix& A, const MxfpMatrix& B,
                                  size_t num_threads = 0) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    // Convert MXFP8 to float
    std::vector<float> A_float = A.to_float();
    std::vector<float> B_float = B.to_float();
    
    // Perform parallel multiplication
    std::vector<float> C_float = matmul_parallel_float(A_float, B_float,
                                                       A.rows, A.cols, B.cols,
                                                       num_threads);
    
    // Convert result back to MXFP8
    return MxfpMatrix::from_float(C_float, A.rows, B.cols);
}

/**
 * Parallel MXFP8 matrix multiplication returning float result
 * (for verification purposes)
 */
inline std::vector<float> matmul_parallel_float(const MxfpMatrix& A, 
                                                const MxfpMatrix& B,
                                                size_t num_threads = 0) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    std::vector<float> A_float = A.to_float();
    std::vector<float> B_float = B.to_float();
    
    return matmul_parallel_float(A_float, B_float,
                                 A.rows, A.cols, B.cols,
                                 num_threads);
}

} // namespace mxfp8

#endif // MXFP8_MATMUL_PARALLEL_HPP
