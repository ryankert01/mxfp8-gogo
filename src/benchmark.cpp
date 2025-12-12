/**
 * MXFP8 Matrix Multiplication Benchmark
 * 
 * Compares performance of vanilla, sequential (optimized), and parallel implementations
 */

#include "../include/mxfp8.hpp"
#include "../include/mxfp8_matmul_sequential.hpp"
#include "../include/mxfp8_matmul_parallel.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <thread>

using namespace mxfp8;

/**
 * Time a function and return duration in milliseconds
 */
template<typename Func>
double time_function(Func&& f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/**
 * Benchmark for a given matrix size
 */
void benchmark_size(size_t M, size_t K, size_t N, int iterations = 5) {
    std::cout << "\n--- Matrix Size: " << M << "x" << K << " * " << K << "x" << N << " ---" << std::endl;
    
    // Generate random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> A_data(M * K);
    std::vector<float> B_data(K * N);
    
    for (auto& v : A_data) v = dist(rng);
    for (auto& v : B_data) v = dist(rng);
    
    MxfpMatrix A = MxfpMatrix::from_float(A_data, M, K);
    MxfpMatrix B = MxfpMatrix::from_float(B_data, K, N);
    
    // Warmup
    matmul_vanilla_float(A, B);
    matmul_sequential_float(A, B);
    matmul_parallel_float(A, B);
    
    // Benchmark vanilla (naive i,j,k loop order)
    double vanilla_total = 0;
    for (int i = 0; i < iterations; ++i) {
        vanilla_total += time_function([&]() {
            volatile auto result = matmul_vanilla_float(A, B);
            (void)result;
        });
    }
    double vanilla_avg = vanilla_total / iterations;
    
    // Benchmark sequential (optimized i,k,j loop order)
    double seq_total = 0;
    for (int i = 0; i < iterations; ++i) {
        seq_total += time_function([&]() {
            volatile auto result = matmul_sequential_float(A, B);
            (void)result;
        });
    }
    double seq_avg = seq_total / iterations;
    
    // Benchmark parallel with different thread counts
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    double par_total = 0;
    for (int i = 0; i < iterations; ++i) {
        par_total += time_function([&]() {
            volatile auto result = matmul_parallel_float(A, B, num_threads);
            (void)result;
        });
    }
    double par_avg = par_total / iterations;
    
    // Calculate GFLOPS (2 * M * N * K operations for matmul)
    double gflops = (2.0 * M * N * K) / 1e9;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Vanilla:        " << vanilla_avg << " ms  (" 
              << (gflops / (vanilla_avg / 1000.0)) << " GFLOPS)" << std::endl;
    std::cout << "  Sequential:     " << seq_avg << " ms  (" 
              << (gflops / (seq_avg / 1000.0)) << " GFLOPS)" << std::endl;
    std::cout << "  Parallel (" << num_threads << "t): " << par_avg << " ms  (" 
              << (gflops / (par_avg / 1000.0)) << " GFLOPS)" << std::endl;
    std::cout << "  Speedup (vanilla→seq):      " << (vanilla_avg / seq_avg) << "x" << std::endl;
    std::cout << "  Speedup (vanilla→parallel): " << (vanilla_avg / par_avg) << "x" << std::endl;
    std::cout << "  Speedup (seq→parallel):     " << (seq_avg / par_avg) << "x" << std::endl;
}

int main() {
    std::cout << "=== MXFP8 Matrix Multiplication Benchmark ===" << std::endl;
    std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << std::endl;
    
    // Small matrices
    benchmark_size(128, 128, 128);
    
    // Medium matrices
    benchmark_size(256, 256, 256);
    benchmark_size(512, 512, 512);
    
    // Large matrices
    benchmark_size(1024, 1024, 1024);
    
    // Non-square matrices
    benchmark_size(512, 1024, 256);
    benchmark_size(1024, 512, 1024);
    
    std::cout << "\n=== Benchmark Complete ===" << std::endl;
    return 0;
}
