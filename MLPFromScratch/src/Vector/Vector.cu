#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <stdexcept>

#include "../Matrix/Matrix.cuh"
#include "../config.h"
#include "Vector.cuh"

// Existing kernels
__global__ static void vector_add_parallel(double* a, double* b, double* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ static void vector_elem_mult_parallel(double* a, double* b, double* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void vector_sum_parallel(const double* input, double* results, size_t n) {
    extern __shared__ double partial_sums[];

    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    partial_sums[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) results[blockIdx.x] = partial_sums[0];
}

// New kernels for additional operations
__global__ static void vector_sub_parallel(double* a, double* b, double* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] - b[idx];
    }
}

__global__ static void vector_scalar_mult_parallel(double* a, double scalar, double* result,
                                                   size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * scalar;
    }
}

__global__ static void vector_square_parallel(double* a, double* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * a[idx];
    }
}

__global__ static void vector_outer_product_parallel(double* a, double* b, double* result,
                                                     size_t n1, size_t n2) {
    // Calculate global thread index
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n1 && col < n2) {
        // Perform the outer product
        result[row * n2 + col] = a[row] * b[col];
    }
}

// Existing implementations
Vector Vector::operator+(const Vector& other) const {
    assert(size_ == other.size() && "Vector dimensions must match.");

    // if small enough, do it on the host
    if (size_ < PARALLEL_THRESHOLD) {
        Vector result(size_);
        for (size_t i = 0; i < size_; ++i) {
            result[i] = values[i] + other.values[i];
        }
        return result;
    }

    // Allocate memory for arrays on the device
    double *d_a, *d_b, *d_result;
    size_t size = size_ * sizeof(double);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_result, size);

    cudaMemcpy(d_a, values, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, other.values, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (static_cast<int>(size_) + blockSize - 1) / blockSize;

    vector_add_parallel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, size_);

    // Copy the result back to the host
    Vector result(size_);
    cudaMemcpy(result.get_values().data(), d_result, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return result;
}

Vector Vector::elem_mult(const Vector& other) const {
    assert(size_ == other.size() && "Vector dimensions must match.");

    // if small enough, do it on the host
    if (size_ < PARALLEL_THRESHOLD) {
        Vector result(size_);
        for (size_t i = 0; i < size_; ++i) {
            result[i] = values[i] * other.values[i];
        }
        return result;
    }

    // Allocate memory for arrays on the device
    double *d_a, *d_b, *d_result;
    size_t size = size_ * sizeof(double);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_result, size);

    cudaMemcpy(d_a, values, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, other.values, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (static_cast<int>(size_) + blockSize - 1) / blockSize;

    vector_elem_mult_parallel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, size_);

    // Copy the result back to the host
    Vector result(size_);
    cudaMemcpy(result.get_values().data(), d_result, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return result;
}

double Vector::sum() const {
    if (size_ == 0) return 0.0;

    // if small enough, do it on the host
    if (size_ < PARALLEL_THRESHOLD) {
        double total_sum = 0.0;
        for (size_t i = 0; i < size_; ++i) {
            total_sum += values[i];
        }
        return total_sum;
    }

    // Allocate memory for arrays on the device
    double* d_a;
    double* d_result;
    size_t size = size_ * sizeof(double);

    int blockSize = 256;
    int numBlocks = (static_cast<int>(size_) + blockSize - 1) / blockSize;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_result, numBlocks * sizeof(double));

    cudaMemcpy(d_a, values, size, cudaMemcpyHostToDevice);

    vector_sum_parallel<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(d_a, d_result, size_);

    std::vector<double> partial_sums(numBlocks);
    cudaMemcpy(partial_sums.data(), d_result, numBlocks * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_result);

    // Sum the partial results on the host
    double total_sum = 0.0;
    for (size_t i = 0; i < partial_sums.size(); ++i) {
        total_sum += partial_sums[i];
    }

    return total_sum;
}

double Vector::dot(const Vector& other) const { return this->elem_mult(other).sum(); }

// New implementations
Vector Vector::operator-(const Vector& other) const {
    assert(size_ == other.size() && "Vector dimensions must match.");

    // if small enough, do it on the host
    if (size_ < PARALLEL_THRESHOLD) {
        Vector result(size_);
        for (size_t i = 0; i < size_; ++i) {
            result[i] = values[i] - other.values[i];
        }
        return result;
    }

    // Allocate memory for arrays on the device
    double *d_a, *d_b, *d_result;
    size_t size = size_ * sizeof(double);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_result, size);

    cudaMemcpy(d_a, values, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, other.values, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (static_cast<int>(size_) + blockSize - 1) / blockSize;

    vector_sub_parallel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, size_);

    // Copy the result back to the host
    Vector result(size_);
    cudaMemcpy(result.get_values().data(), d_result, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return result;
}

Vector Vector::operator*(double scalar) const {
    // if small enough, do it on the host
    if (size_ < PARALLEL_THRESHOLD) {
        Vector result(size_);
        for (size_t i = 0; i < size_; ++i) {
            result[i] = values[i] * scalar;
        }
        return result;
    }

    // Allocate memory for arrays on the device
    double *d_a, *d_result;
    size_t size = size_ * sizeof(double);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_result, size);

    cudaMemcpy(d_a, values, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (static_cast<int>(size_) + blockSize - 1) / blockSize;

    vector_scalar_mult_parallel<<<numBlocks, blockSize>>>(d_a, scalar, d_result, size_);

    // Copy the result back to the host
    Vector result(size_);
    cudaMemcpy(result.get_values().data(), d_result, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_result);

    return result;
}

Vector Vector::square() const {
    // if small enough, do it on the host
    if (size_ < PARALLEL_THRESHOLD) {
        Vector result(size_);
        for (size_t i = 0; i < size_; ++i) {
            result[i] = values[i] * values[i];
        }
        return result;
    }

    // Allocate memory for arrays on the device
    double *d_a, *d_result;
    size_t size = size_ * sizeof(double);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_result, size);

    cudaMemcpy(d_a, values, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (static_cast<int>(size_) + blockSize - 1) / blockSize;

    vector_square_parallel<<<numBlocks, blockSize>>>(d_a, d_result, size_);

    // Copy the result back to the host
    Vector result(size_);
    cudaMemcpy(result.get_values().data(), d_result, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_result);

    return result;
}

double Vector::norm() const { return std::sqrt(this->square().sum()); }

double Vector::mean() const {
    if (size_ == 0) {
        throw std::invalid_argument("Cannot compute mean of an empty vector.");
    }
    return sum() / static_cast<double>(size_);
}

Matrix Vector::outer_product(const Vector& other) const {
    // if small enough, do it on the host
    if (size_ < PARALLEL_THRESHOLD) {
        Matrix result(size_, other.size());
        for (size_t i = 0; i < size_; ++i) {
            for (size_t j = 0; j < other.size(); ++j) {
                result[i][j] = values[i] * other.values[j];
            }
        }
        return result;
    }

    // Allocate memory for arrays on the device
    double *d_a, *d_b, *d_result;
    size_t size_a = size_ * sizeof(double);
    size_t size_b = other.size() * sizeof(double);

    cudaMalloc((void**)&d_a, size_a);
    cudaMalloc((void**)&d_b, size_b);
    cudaMalloc((void**)&d_result, size_a * size_b);

    cudaMemcpy(d_a, values, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, other.values, size_b, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((static_cast<unsigned int>(other.size()) + blockSize.x - 1) / blockSize.x,
                  (size_ + blockSize.y - 1) / blockSize.y);

    vector_outer_product_parallel<<<gridSize, blockSize>>>(d_a, d_b, d_result, size_, other.size());

    double* h_result = new double[size_a * size_b];

    cudaMemcpy(h_result, d_result, size_a * size_b, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);

    // Copy the result back to the host
    Matrix result(size_, other.size(), h_result);

    delete[] h_result;

    return result;
}
