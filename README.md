# mxfp8-gogo

MXFP8 (Microscaling Floating Point 8-bit) Matrix Multiplication implementation in C++.

## Overview

This project implements matrix multiplication using the MXFP8 format, which is commonly used in AI/ML workloads for memory-efficient computation. The implementation includes:

- **MXFP8 Format**: E4M3 format (1 sign + 4 exponent + 3 mantissa bits) with microscaling (32 elements share one scale factor)
- **Sequential Implementation**: Vanilla O(M×N×K) matrix multiplication
- **Parallel Implementation**: Optimized with pthreads, cache blocking, and SIMD (AVX)

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
| 128×128     | 6.70 ms    | 0.32 ms            | 20.75× |
| 256×256     | 53.84 ms   | 1.82 ms            | 29.62× |
| 512×512     | 433.66 ms  | 11.30 ms           | 38.39× |
| 1024×1024   | 3513.75 ms | 93.02 ms           | 37.77× |

### Performance Analysis

- **Sequential**: ~0.62 GFLOPS
- **Parallel (4 threads)**: ~23 GFLOPS
- **Superlinear speedup**: Observed due to improved cache utilization from tiling

The parallel version achieves >37× speedup on 4 cores because:
1. Thread parallelism (4×)
2. SIMD parallelism (8× for AVX)
3. Better cache efficiency from blocking

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