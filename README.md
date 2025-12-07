# mxfp8-gogo

MXFP8 (Microscaling Floating Point 8-bit) Matrix Multiplication implementation in C++.

## Overview

This project implements matrix multiplication using the MXFP8 format, which is commonly used in AI/ML workloads for memory-efficient computation. The implementation includes:

- **MXFP8 Format**: E4M3 format (1 sign + 4 exponent + 3 mantissa bits) with microscaling (32 elements share one scale factor)
- **Sequential Implementation**: Vanilla O(M×N×K) matrix multiplication
- **Parallel Implementation**: Optimized with pthreads, cache blocking (tiling), and SIMD (AVX2)

## Building

```bash
# Build all targets
make all

# Run tests
make test

# Run benchmarks
make benchmark

# Clean build artifacts
make clean
```

### Requirements

- C++17 compatible compiler (g++ recommended)
- pthread support
- x86_64 architecture with AVX2 and FMA support (for SIMD optimizations)

## File Structure

```
├── include/
│   ├── mxfp8.hpp                  # MXFP8 format and matrix class
│   ├── mxfp8_matmul_sequential.hpp # Sequential matmul implementation
│   └── mxfp8_matmul_parallel.hpp   # Parallel matmul implementation
├── src/
│   └── benchmark.cpp              # Performance benchmark program
├── tests/
│   └── test_mxfp8.cpp             # Unit tests
├── Makefile
└── README.md
```

## MXFP8 Format Details

MXFP8 uses microscaling to improve the dynamic range of 8-bit floating point:

- **Per-element format**: FP8 E4M3 (1 sign, 4 exponent, 3 mantissa bits)
- **Block size**: 32 elements share one 32-bit float scale factor
- **Range**: Approximately ±448 (before scaling)

## Implementation Details

### Sequential Version

Standard triple-nested loop matrix multiplication:

```cpp
for i in 0..M:
    for j in 0..N:
        for k in 0..K:
            C[i,j] += A[i,k] * B[k,j]
```

### Parallel Version

Optimizations applied:

1. **pthread parallelization**: Row-based work distribution across threads
2. **Cache blocking (tiling)**: 32×32 tiles for better L1/L2 cache utilization
3. **SIMD vectorization**: AVX2 intrinsics for 8-wide float operations (256-bit)
4. **FMA (Fused Multiply-Add)**: Single instruction for multiply-accumulate

## Benchmark Results

Tested on 4-core system:

| Matrix Size | Sequential | Parallel (4 threads) | Speedup |
|-------------|------------|---------------------|---------|
| 128×128     | 0.20 ms    | 0.35 ms            | 0.57× |
| 256×256     | 1.62 ms    | 1.69 ms            | 0.96× |
| 512×512     | 12.70 ms   | 11.75 ms           | 1.08× |
| 1024×1024   | 88.26 ms   | 98.47 ms           | 0.90× |

### Performance Analysis

- **Sequential**: ~20-24 GFLOPS (optimized with cache-friendly i,k,j loop order)
- **Parallel (4 threads)**: ~20-25 GFLOPS

After optimizing the sequential version with cache-friendly loop ordering (i,k,j instead of i,j,k), both versions achieve similar performance. The sequential version benefits from:
1. Better cache utilization from sequential memory access
2. No thread synchronization overhead

The parallel version still benefits from:
1. Thread parallelism (4×)
2. SIMD parallelism (8× for AVX2)
3. Cache blocking (tiling)

## API Usage

```cpp
#include "mxfp8.hpp"
#include "mxfp8_matmul_sequential.hpp"
#include "mxfp8_matmul_parallel.hpp"

using namespace mxfp8;

// Create matrices from float data
std::vector<float> A_data = {/* ... */};
std::vector<float> B_data = {/* ... */};

MxfpMatrix A = MxfpMatrix::from_float(A_data, M, K);
MxfpMatrix B = MxfpMatrix::from_float(B_data, K, N);

// Sequential multiplication
MxfpMatrix C_seq = matmul_sequential(A, B);

// Parallel multiplication (auto-detect thread count)
MxfpMatrix C_par = matmul_parallel(A, B);

// Convert result back to float
std::vector<float> result = C_par.to_float();
```

## License

MIT License