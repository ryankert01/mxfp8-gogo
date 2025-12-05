/**
 * MXFP8 (Microscaling Floating Point 8-bit) Format Implementation
 * 
 * MXFP8 is a format used in AI/ML workloads that uses 8 bits per element
 * with a shared scaling factor per block of elements (typically 32 elements).
 * 
 * Format: 1 sign bit, 4 exponent bits, 3 mantissa bits (E4M3)
 * Block size: 32 elements share one 8-bit scale factor
 */

#ifndef MXFP8_HPP
#define MXFP8_HPP

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>
#include <algorithm>

namespace mxfp8 {

// Block size for microscaling (32 elements share one scale)
constexpr size_t BLOCK_SIZE = 32;
constexpr size_t CACHE_LINE_SIZE = 64;

/**
 * E4M3 format: 1 sign + 4 exponent + 3 mantissa
 * Range: approximately ±448
 * Special values: NaN (S.1111.111)
 */
struct FP8_E4M3 {
    uint8_t data;
    
    FP8_E4M3() : data(0) {}
    explicit FP8_E4M3(uint8_t raw) : data(raw) {}
    
    // Convert from float to FP8_E4M3
    static FP8_E4M3 from_float(float f) {
        if (std::isnan(f)) {
            return FP8_E4M3(0x7F); // NaN representation
        }
        
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(float));
        
        uint8_t sign = (bits >> 31) & 1;
        int32_t exp = ((bits >> 23) & 0xFF) - 127; // unbias FP32 exponent
        uint32_t mant = bits & 0x7FFFFF;
        
        // Handle zero
        if (exp == -127 && mant == 0) {
            return FP8_E4M3(sign << 7);
        }
        
        // E4M3 bias is 7
        int32_t fp8_exp = exp + 7;
        
        // Clamp to representable range
        if (fp8_exp >= 15) {
            // Max representable value (avoid NaN encoding 0x7F/0xFF)
            return FP8_E4M3((sign << 7) | 0x7E);
        }
        if (fp8_exp <= 0) {
            return FP8_E4M3(sign << 7); // Subnormal -> zero
        }
        
        // Round mantissa from 23 bits to 3 bits
        uint8_t fp8_mant = (mant >> 20) & 0x7;
        
        return FP8_E4M3((sign << 7) | (fp8_exp << 3) | fp8_mant);
    }
    
    // Convert from FP8_E4M3 to float
    float to_float() const {
        uint8_t sign = (data >> 7) & 1;
        uint8_t exp = (data >> 3) & 0xF;
        uint8_t mant = data & 0x7;
        
        // Check for NaN
        if (exp == 15 && mant == 7) {
            return std::nanf("");
        }
        
        // Zero
        if (exp == 0 && mant == 0) {
            return sign ? -0.0f : 0.0f;
        }
        
        // Subnormal
        if (exp == 0) {
            float result = std::ldexp(static_cast<float>(mant), -9); // 2^(-7-3+1)
            return sign ? -result : result;
        }
        
        // Normal number
        int32_t fp32_exp = static_cast<int32_t>(exp) - 7 + 127;
        uint32_t fp32_mant = static_cast<uint32_t>(mant) << 20;
        
        uint32_t bits = (static_cast<uint32_t>(sign) << 31) | 
                        (static_cast<uint32_t>(fp32_exp) << 23) | 
                        fp32_mant;
        
        float result;
        std::memcpy(&result, &bits, sizeof(float));
        return result;
    }
};

/**
 * MXFP8 Matrix with microscaling
 * Each block of BLOCK_SIZE elements shares a scale factor
 */
class MxfpMatrix {
public:
    size_t rows;
    size_t cols;
    std::vector<FP8_E4M3> data;
    std::vector<float> scales; // One scale per block
    
    MxfpMatrix() : rows(0), cols(0) {}
    
    MxfpMatrix(size_t r, size_t c) : rows(r), cols(c) {
        size_t total_elements = r * c;
        size_t num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        data.resize(total_elements);
        scales.resize(num_blocks, 1.0f);
    }
    
    // Create from float matrix with microscaling
    static MxfpMatrix from_float(const std::vector<float>& input, size_t r, size_t c) {
        MxfpMatrix result(r, c);
        size_t total = r * c;
        
        for (size_t block = 0; block < result.scales.size(); ++block) {
            size_t start = block * BLOCK_SIZE;
            size_t end = std::min(start + BLOCK_SIZE, total);
            
            // Find max absolute value in block
            float max_abs = 0.0f;
            for (size_t i = start; i < end; ++i) {
                max_abs = std::max(max_abs, std::abs(input[i]));
            }
            
            // Compute scale factor (target max representable ~240)
            float scale = (max_abs > 0) ? (240.0f / max_abs) : 1.0f;
            result.scales[block] = 1.0f / scale;
            
            // Convert elements with scaling
            for (size_t i = start; i < end; ++i) {
                float scaled_val = input[i] * scale;
                result.data[i] = FP8_E4M3::from_float(scaled_val);
            }
        }
        
        return result;
    }
    
    // Convert back to float matrix
    std::vector<float> to_float() const {
        std::vector<float> result(rows * cols);
        size_t total = rows * cols;
        
        for (size_t block = 0; block < scales.size(); ++block) {
            size_t start = block * BLOCK_SIZE;
            size_t end = std::min(start + BLOCK_SIZE, total);
            float scale = scales[block];
            
            for (size_t i = start; i < end; ++i) {
                result[i] = data[i].to_float() * scale;
            }
        }
        
        return result;
    }
    
    // Access element (row-major order)
    float get(size_t i, size_t j) const {
        size_t idx = i * cols + j;
        size_t block = idx / BLOCK_SIZE;
        return data[idx].to_float() * scales[block];
    }
};

} // namespace mxfp8

#endif // MXFP8_HPP
