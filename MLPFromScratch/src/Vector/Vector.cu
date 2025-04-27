#include <cuda_runtime.h>

#include <stdexcept>

#include "Vector.cuh"

__global__ static void add_parallel(double* a, double* b, double* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

Vector Vector::operator+(const Vector& other) const {
    if (values.size() != other.size()) {
        throw std::invalid_argument("Vector dimensions must match.");
    }

    // Allocate memory for arrays on the device
    double *d_a, *d_b, *d_result;
    size_t size = values.size() * sizeof(double);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_result, size);

    cudaMemcpy(d_a, values.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, other.values.data(), size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (static_cast<int>(values.size()) + blockSize - 1) / blockSize;

    add_parallel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, values.size());

    // Copy the result back to the host
    Vector result(values.size());
    cudaMemcpy(result.get_values().data(), d_result, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return result;
}

__global__ static void elem_mult_parallel(double* a, double* b, double* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * b[idx];
    }
}

Vector Vector::elem_mult(const Vector& other) const {
    if (values.size() != other.size()) {
        throw std::invalid_argument("Vector dimensions must match.");
    }

    // Allocate memory for arrays on the device
    double *d_a, *d_b, *d_result;
    size_t size = values.size() * sizeof(double);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_result, size);

    cudaMemcpy(d_a, values.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, other.values.data(), size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (static_cast<int>(values.size()) + blockSize - 1) / blockSize;

    elem_mult_parallel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, values.size());

    // Copy the result back to the host
    Vector result(values.size());
    cudaMemcpy(result.get_values().data(), d_result, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return result;
}

__global__ void sum_parallel(const double* input, double* results, size_t n) {
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

double Vector::sum() const {
    if (values.empty()) return 0.0;

    // Allocate memory for arrays on the device
    double* d_a;
    double* d_result;
    size_t size = values.size() * sizeof(double);

    int blockSize = 256;
    int numBlocks = (static_cast<int>(values.size()) + blockSize - 1) / blockSize;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_result, numBlocks * sizeof(double));

    cudaMemcpy(d_a, values.data(), size, cudaMemcpyHostToDevice);

    sum_parallel<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(d_a, d_result,
                                                                       values.size());

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
