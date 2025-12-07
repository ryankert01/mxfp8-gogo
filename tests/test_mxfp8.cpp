/**
 * MXFP8 Matrix Multiplication Tests
 * 
 * Verifies correctness of both sequential and parallel implementations
 */

#include "../include/mxfp8.hpp"
#include "../include/mxfp8_matmul_sequential.hpp"
#include "../include/mxfp8_matmul_parallel.hpp"

#include <iostream>
#include <cmath>
#include <random>
#include <iomanip>
#include <cassert>

using namespace mxfp8;

// Tolerance for floating point comparison (MXFP8 has limited precision)
constexpr float FP8_RELATIVE_ERROR_TOLERANCE = 0.3f;  // 30% relative error for FP8 conversions
constexpr float MATMUL_RELATIVE_ERROR_TOLERANCE = 0.5f;  // 50% relative error for matmul due to quantization
constexpr float ABSOLUTE_ERROR_TOLERANCE = 0.5f;  // Absolute error tolerance for small values

/**
 * Test FP8_E4M3 conversion
 */
void test_fp8_conversion() {
    std::cout << "Testing FP8_E4M3 conversion..." << std::endl;
    
    // Test zero
    {
        FP8_E4M3 zero = FP8_E4M3::from_float(0.0f);
        assert(std::abs(zero.to_float()) < 1e-6f);
    }
    
    // Test positive numbers
    {
        float test_vals[] = {1.0f, 2.0f, 4.0f, 8.0f, 0.5f, 0.25f, 100.0f};
        for (float val : test_vals) {
            FP8_E4M3 fp8 = FP8_E4M3::from_float(val);
            float recovered = fp8.to_float();
            float rel_err = std::abs(recovered - val) / std::max(std::abs(val), 1e-6f);
            assert(rel_err < FP8_RELATIVE_ERROR_TOLERANCE);
        }
    }
    
    // Test negative numbers
    {
        FP8_E4M3 neg = FP8_E4M3::from_float(-3.14f);
        assert(neg.to_float() < 0);
    }
    
    std::cout << "  PASSED" << std::endl;
}

/**
 * Test MXFP8 matrix creation and conversion
 */
void test_mxfp8_matrix() {
    std::cout << "Testing MXFP8 matrix creation..." << std::endl;
    
    size_t rows = 4;
    size_t cols = 8;
    std::vector<float> data(rows * cols);
    
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i) - 16.0f;
    }
    
    MxfpMatrix mat = MxfpMatrix::from_float(data, rows, cols);
    std::vector<float> recovered = mat.to_float();
    
    // Check dimensions
    assert(mat.rows == rows);
    assert(mat.cols == cols);
    
    // Check values are close
    for (size_t i = 0; i < data.size(); ++i) {
        float err = std::abs(recovered[i] - data[i]);
        float rel_err = err / std::max(std::abs(data[i]), 1e-6f);
        assert(rel_err < FP8_RELATIVE_ERROR_TOLERANCE || err < ABSOLUTE_ERROR_TOLERANCE);
    }
    
    std::cout << "  PASSED" << std::endl;
}

/**
 * Test sequential matrix multiplication
 */
void test_sequential_matmul() {
    std::cout << "Testing sequential matrix multiplication..." << std::endl;
    
    size_t M = 4, K = 8, N = 4;
    
    // Create test matrices
    std::vector<float> A_data(M * K);
    std::vector<float> B_data(K * N);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    
    for (auto& v : A_data) v = dist(rng);
    for (auto& v : B_data) v = dist(rng);
    
    // Reference result
    std::vector<float> ref = matmul_reference(A_data, B_data, M, K, N);
    
    // MXFP8 result
    MxfpMatrix A = MxfpMatrix::from_float(A_data, M, K);
    MxfpMatrix B = MxfpMatrix::from_float(B_data, K, N);
    std::vector<float> result = matmul_sequential_float(A, B);
    
    // Compare (allow larger tolerance due to quantization)
    float max_err = 0;
    float max_ref = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_err = std::max(max_err, std::abs(result[i] - ref[i]));
        max_ref = std::max(max_ref, std::abs(ref[i]));
    }
    
    float rel_err = max_err / std::max(max_ref, 1e-6f);
    std::cout << "  Max relative error: " << rel_err << std::endl;
    assert(rel_err < MATMUL_RELATIVE_ERROR_TOLERANCE);
    
    std::cout << "  PASSED" << std::endl;
}

/**
 * Test parallel matrix multiplication
 */
void test_parallel_matmul() {
    std::cout << "Testing parallel matrix multiplication..." << std::endl;
    
    size_t M = 64, K = 128, N = 64;
    
    // Create test matrices
    std::vector<float> A_data(M * K);
    std::vector<float> B_data(K * N);
    
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    
    for (auto& v : A_data) v = dist(rng);
    for (auto& v : B_data) v = dist(rng);
    
    // Create MXFP8 matrices
    MxfpMatrix A = MxfpMatrix::from_float(A_data, M, K);
    MxfpMatrix B = MxfpMatrix::from_float(B_data, K, N);
    
    // Sequential result
    std::vector<float> seq_result = matmul_sequential_float(A, B);
    
    // Parallel result
    std::vector<float> par_result = matmul_parallel_float(A, B, 4);
    
    // Compare sequential and parallel results
    float max_err = 0;
    float max_val = 0;
    for (size_t i = 0; i < seq_result.size(); ++i) {
        max_err = std::max(max_err, std::abs(par_result[i] - seq_result[i]));
        max_val = std::max(max_val, std::abs(seq_result[i]));
    }
    
    float rel_err = max_err / std::max(max_val, 1e-6f);
    std::cout << "  Sequential vs Parallel max relative error: " << rel_err << std::endl;
    assert(rel_err < 1e-5f); // Should be nearly identical
    
    std::cout << "  PASSED" << std::endl;
}

/**
 * Test with larger matrices
 */
void test_large_matrices() {
    std::cout << "Testing large matrix multiplication..." << std::endl;
    
    size_t M = 256, K = 512, N = 256;
    
    std::vector<float> A_data(M * K);
    std::vector<float> B_data(K * N);
    
    std::mt19937 rng(456);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    
    for (auto& v : A_data) v = dist(rng);
    for (auto& v : B_data) v = dist(rng);
    
    MxfpMatrix A = MxfpMatrix::from_float(A_data, M, K);
    MxfpMatrix B = MxfpMatrix::from_float(B_data, K, N);
    
    // Just verify both complete without error
    std::vector<float> seq_result = matmul_sequential_float(A, B);
    std::vector<float> par_result = matmul_parallel_float(A, B);
    
    // Quick sanity check
    assert(seq_result.size() == M * N);
    assert(par_result.size() == M * N);
    
    float max_diff = 0;
    for (size_t i = 0; i < seq_result.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(seq_result[i] - par_result[i]));
    }
    std::cout << "  Max difference between seq and parallel: " << max_diff << std::endl;
    
    std::cout << "  PASSED" << std::endl;
}

int main() {
    std::cout << "=== MXFP8 Matrix Multiplication Tests ===" << std::endl << std::endl;
    
    test_fp8_conversion();
    test_mxfp8_matrix();
    test_sequential_matmul();
    test_parallel_matmul();
    test_large_matrices();
    
    std::cout << std::endl << "=== All tests PASSED ===" << std::endl;
    return 0;
}
