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

| Matrix Size | Vanilla    | Sequential (optimized) | Parallel (4t) | Vanilla→Seq | Vanilla→Parallel |
|-------------|------------|------------------------|---------------|-------------|------------------|
| 128×128     | 2.80 ms    | 0.19 ms                | 0.38 ms       | 14.68×      | 7.41×            |
| 256×256     | 23.03 ms   | 1.58 ms                | 1.80 ms       | 14.54×      | 12.76×           |
| 512×512     | 253.53 ms  | 13.46 ms               | 11.71 ms      | 18.83×      | 21.66×           |
| 1024×1024   | 6779.29 ms | 94.17 ms               | 100.86 ms     | 71.99×      | 67.22×           |

### Performance Analysis

- **Vanilla**: ~0.3-1.5 GFLOPS (naive i,j,k loop order, poor cache utilization)
- **Sequential (optimized)**: ~20-22 GFLOPS (cache-friendly i,k,j loop order)
- **Parallel (4 threads)**: ~18-24 GFLOPS (threading + SIMD + cache blocking)

The key optimization is the loop order change from i,j,k to i,k,j which dramatically improves cache utilization:
- **Vanilla→Sequential**: Up to 72× speedup just from loop reordering
- **Vanilla→Parallel**: Similar speedup as the optimized sequential version

The parallel version uses:
1. pthread threading (row distribution)
2. AVX2 SIMD (8-wide float operations)
3. 32×32 cache blocking (tiling)

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