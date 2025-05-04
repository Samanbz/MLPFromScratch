#include <cuda_runtime.h>

#include <cassert>

#include "../config.h"
#include "Matrix.cuh"

// Addition kernel - already provided
__global__ static void matrix_add_parallel(double* a, double* b, double* result, size_t rows,
                                           size_t cols) {
    // Calculate global thread index
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within matrix bounds
    if (row < rows && col < cols) {
        // Calculate linear index from 2D coordinates
        size_t idx = row * cols + col;

        // Perform addition
        result[idx] = a[idx] + b[idx];
    }
}

// Subtraction kernel
__global__ static void matrix_subtract_parallel(double* a, double* b, double* result, size_t rows,
                                                size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        size_t idx = row * cols + col;
        result[idx] = a[idx] - b[idx];
    }
}

// Matrix-matrix multiplication kernel
__global__ static void matrix_multiply_parallel(double* a, double* b, double* result, size_t a_rows,
                                                size_t a_cols, size_t b_cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < a_rows && col < b_cols) {
        double sum = 0.0;
        for (size_t k = 0; k < a_cols; ++k) {
            sum += a[row * a_cols + k] * b[k * b_cols + col];
        }
        result[row * b_cols + col] = sum;
    }
}

// Matrix-vector multiplication kernel
__global__ static void matrix_vector_multiply_parallel(double* matrix, double* vector,
                                                       double* result, size_t rows, size_t cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        double sum = 0.0;
        for (size_t col = 0; col < cols; ++col) {
            sum += matrix[row * cols + col] * vector[col];
        }
        result[row] = sum;
    }
}

// Scalar multiplication kernel
__global__ static void matrix_scalar_multiply_parallel(double* matrix, double scalar,
                                                       double* result, size_t rows, size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        size_t idx = row * cols + col;
        result[idx] = matrix[idx] * scalar;
    }
}

// Matrix transpose kernel
__global__ static void matrix_transpose_parallel(double* input, double* output, size_t rows,
                                                 size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// Addition implementation - already provided
Matrix Matrix::operator+(const Matrix& other) const {
    assert(cols_ == other.cols() && rows_ == other.rows_ && "Dimensions must match.");

    if (cols_ * rows_ < PARALLEL_THRESHOLD) {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result.at(i, j) = this->at(i, j) + other.at(i, j);
            }
        }
        return result;
    }

    // Allocate memory for arrays on the device
    double *d_a, *d_b, *d_result;
    size_t size = rows_ * cols_ * sizeof(double);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_result, size);

    double* h_a = new double[rows_ * cols_];
    double* h_b = new double[rows_ * cols_];

    this->flatten(h_a);
    other.flatten(h_b);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    delete[] h_a;
    delete[] h_b;

    dim3 blockSize(16, 16);
    dim3 gridSize((static_cast<unsigned int>(cols_) + blockSize.x - 1) / blockSize.x,
                  (static_cast<unsigned int>(rows_) + blockSize.y - 1) / blockSize.y);

    matrix_add_parallel<<<gridSize, blockSize>>>(d_a, d_b, d_result, rows_, cols_);

    double* h_result = new double[rows_ * cols_];
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    Matrix result(rows_, cols_, h_result);

    delete[] h_result;

    return result;
}

// Subtraction implementation
Matrix Matrix::operator-(const Matrix& other) const {
    assert(cols_ == other.cols() && rows_ == other.rows_ && "Dimensions must match.");

    if (cols_ * rows_ < PARALLEL_THRESHOLD) {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result.at(i, j) = this->at(i, j) - other.at(i, j);
            }
        }
        return result;
    }

    // Allocate memory for arrays on the device
    double *d_a, *d_b, *d_result;
    size_t size = rows_ * cols_ * sizeof(double);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_result, size);

    double* h_a = new double[rows_ * cols_];
    double* h_b = new double[rows_ * cols_];

    this->flatten(h_a);
    other.flatten(h_b);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    delete[] h_a;
    delete[] h_b;

    dim3 blockSize(16, 16);
    dim3 gridSize((static_cast<unsigned int>(cols_) + blockSize.x - 1) / blockSize.x,
                  (static_cast<unsigned int>(rows_) + blockSize.y - 1) / blockSize.y);

    matrix_subtract_parallel<<<gridSize, blockSize>>>(d_a, d_b, d_result, rows_, cols_);

    double* h_result = new double[rows_ * cols_];
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    Matrix result(rows_, cols_, h_result);

    delete[] h_result;

    return result;
}

// Matrix-matrix multiplication implementation
Matrix Matrix::operator*(const Matrix& other) const {
    assert(cols_ == other.rows() && "Dimension mismatch for matrix multiplication.");

    if (rows_ * cols_ < PARALLEL_THRESHOLD) {
        // CPU implementation for small matrices
        Matrix result(rows_, other.cols());
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < other.cols(); ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < cols_; ++k) {
                    sum += this->at(i, k) * other.at(k, j);
                }
                result.at(i, j) = sum;
            }
        }
        return result;
    }

    // GPU implementation for large matrices
    double *d_a, *d_b, *d_result;
    size_t a_size = rows_ * cols_ * sizeof(double);
    size_t b_size = other.rows() * other.cols() * sizeof(double);
    size_t result_size = rows_ * other.cols() * sizeof(double);

    cudaMalloc((void**)&d_a, a_size);
    cudaMalloc((void**)&d_b, b_size);
    cudaMalloc((void**)&d_result, result_size);

    double* h_a = new double[rows_ * cols_];
    double* h_b = new double[other.rows() * other.cols()];

    this->flatten(h_a);
    other.flatten(h_b);

    cudaMemcpy(d_a, h_a, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, b_size, cudaMemcpyHostToDevice);

    delete[] h_a;
    delete[] h_b;

    dim3 blockSize(16, 16);
    dim3 gridSize((static_cast<unsigned int>(other.cols()) + blockSize.x - 1) / blockSize.x,
                  (static_cast<unsigned int>(rows_) + blockSize.y - 1) / blockSize.y);

    matrix_multiply_parallel<<<gridSize, blockSize>>>(d_a, d_b, d_result, rows_, cols_,
                                                      other.cols());

    double* h_result = new double[rows_ * other.cols()];
    cudaMemcpy(h_result, d_result, result_size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    Matrix result(rows_, other.cols(), h_result);

    delete[] h_result;

    return result;
}

// Matrix-vector multiplication implementation
Vector Matrix::operator*(const Vector& vector) const {
    assert(cols_ == vector.size() && "Dimension mismatch for matrix-vector multiplication.");

    if (rows_ * cols_ < PARALLEL_THRESHOLD) {
        // CPU implementation for small matrices
        Vector result(rows_);
        for (size_t i = 0; i < rows_; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < cols_; ++j) {
                sum += this->at(i, j) * vector[j];
            }
            result[i] = sum;
        }
        return result;
    }

    // GPU implementation for large matrices
    double *d_matrix, *d_vector, *d_result;
    size_t matrix_size = rows_ * cols_ * sizeof(double);
    size_t vector_size = cols_ * sizeof(double);
    size_t result_size = rows_ * sizeof(double);

    cudaMalloc((void**)&d_matrix, matrix_size);
    cudaMalloc((void**)&d_vector, vector_size);
    cudaMalloc((void**)&d_result, result_size);

    double* h_matrix = new double[rows_ * cols_];
    double* h_vector = new double[cols_];

    this->flatten(h_matrix);
    for (size_t i = 0; i < cols_; ++i) {
        h_vector[i] = vector[i];
    }

    cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, vector_size, cudaMemcpyHostToDevice);

    delete[] h_matrix;
    delete[] h_vector;

    // Configure kernel launch parameters
    int blockSize = 256;
    int gridSize = (static_cast<int>(rows_) + blockSize - 1) / blockSize;

    matrix_vector_multiply_parallel<<<gridSize, blockSize>>>(d_matrix, d_vector, d_result, rows_,
                                                             cols_);

    double* h_result = new double[rows_];
    cudaMemcpy(h_result, d_result, result_size, cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);

    Vector result(rows_);
    for (size_t i = 0; i < rows_; ++i) {
        result[i] = h_result[i];
    }

    delete[] h_result;

    return result;
}

// Scalar multiplication implementation
Matrix Matrix::operator*(double scalar) const {
    if (rows_ * cols_ < PARALLEL_THRESHOLD) {
        // CPU implementation for small matrices
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result.at(i, j) = this->at(i, j) * scalar;
            }
        }
        return result;
    }

    // GPU implementation for large matrices
    double *d_matrix, *d_result;
    size_t matrix_size = rows_ * cols_ * sizeof(double);

    cudaMalloc((void**)&d_matrix, matrix_size);
    cudaMalloc((void**)&d_result, matrix_size);

    double* h_matrix = new double[rows_ * cols_];
    this->flatten(h_matrix);

    cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice);
    delete[] h_matrix;

    dim3 blockSize(16, 16);
    dim3 gridSize((static_cast<unsigned int>(cols_) + blockSize.x - 1) / blockSize.x,
                  (static_cast<unsigned int>(rows_) + blockSize.y - 1) / blockSize.y);

    matrix_scalar_multiply_parallel<<<gridSize, blockSize>>>(d_matrix, scalar, d_result, rows_,
                                                             cols_);

    double* h_result = new double[rows_ * cols_];
    cudaMemcpy(h_result, d_result, matrix_size, cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);
    cudaFree(d_result);

    Matrix result(rows_, cols_, h_result);

    delete[] h_result;

    return result;
}

// Transpose implementation
Matrix Matrix::transpose() const {
    if (rows_ * cols_ < PARALLEL_THRESHOLD) {
        // CPU implementation for small matrices
        Matrix result(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result.at(j, i) = this->at(i, j);
            }
        }
        return result;
    }

    // GPU implementation for large matrices
    double *d_input, *d_output;
    size_t matrix_size = rows_ * cols_ * sizeof(double);

    cudaMalloc((void**)&d_input, matrix_size);
    cudaMalloc((void**)&d_output, matrix_size);

    double* h_input = new double[rows_ * cols_];
    this->flatten(h_input);

    cudaMemcpy(d_input, h_input, matrix_size, cudaMemcpyHostToDevice);
    delete[] h_input;

    dim3 blockSize(16, 16);
    dim3 gridSize((static_cast<unsigned int>(cols_) + blockSize.x - 1) / blockSize.x,
                  (static_cast<unsigned int>(rows_) + blockSize.y - 1) / blockSize.y);

    matrix_transpose_parallel<<<gridSize, blockSize>>>(d_input, d_output, rows_, cols_);

    double* h_result = new double[cols_ * rows_];
    cudaMemcpy(h_result, d_output, matrix_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    Matrix result(cols_, rows_, h_result);

    delete[] h_result;

    return result;
}