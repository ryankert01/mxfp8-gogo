# MXFP8 Matrix Multiplication Makefile

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -mavx2 -mfma
LDFLAGS = -pthread

INCLUDE_DIR = include
SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build

# Targets
.PHONY: all clean test benchmark

all: $(BUILD_DIR)/test_mxfp8 $(BUILD_DIR)/benchmark

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_mxfp8: $(TEST_DIR)/test_mxfp8.cpp $(INCLUDE_DIR)/*.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -I$(INCLUDE_DIR) $< -o $@

$(BUILD_DIR)/benchmark: $(SRC_DIR)/benchmark.cpp $(INCLUDE_DIR)/*.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -I$(INCLUDE_DIR) $< -o $@

test: $(BUILD_DIR)/test_mxfp8
	./$(BUILD_DIR)/test_mxfp8

benchmark: $(BUILD_DIR)/benchmark
	./$(BUILD_DIR)/benchmark

clean:
	rm -rf $(BUILD_DIR)
