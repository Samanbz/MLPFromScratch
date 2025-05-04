#pragma once

#include <device_launch_parameters.h>

#include <cassert>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <string>

#include "../Vector/Vector.cuh"

/**
 * @brief Represents a row-major matrix of real numbers.
 */
class Matrix {
public:
    /**
     * @brief Default constructor that initializes an empty matrix.
     */
    Matrix();

    /**
     * @brief Creates a new matrix with the given row and column numbers.
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     */
    Matrix(size_t rows, size_t cols);

    /**
     * @brief Creates a new matrix with the given row and column numbers and initializes all
     * elements to the given value.
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param value The value to initialize all elements to.
     */
    Matrix(size_t rows, size_t cols, double value);

    /**
     * @brief Creates a Matrix object given the vector of vector of doubles.
     * @param values The vector of vector of doubles to initialize the matrix with.
     */
    Matrix(std::vector<std::vector<double>> values);

    /**
     * @brief Creates a Matrix object from an initializer list of initializer lists of doubles.
     * @param values The initializer list of initializer lists to initialize the matrix with.
     */
    Matrix(std::initializer_list<std::initializer_list<double>> values);

    /**
     * @brief Creates a Matrix object from a flattened array of doubles.
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param data Pointer to the flattened array of doubles (in row-major order).
     */
    Matrix(size_t rows, size_t cols, const double* data);

    /**
     * @brief Copy constructor
     * @param other The matrix to copy from
     */
    Matrix(const Matrix& other);

    /**
     * @brief Assignment operator
     * @param other The matrix to copy from
     * @return Reference to this matrix
     */
    Matrix& operator=(const Matrix& other);

    /**
     * @brief Destructor to free allocated memory
     */
    ~Matrix();

    /**
     * @brief Returns the number of rows in the matrix.
     * @returns The number of rows in the matrix.
     */
    __host__ __device__ size_t rows() const { return rows_; }

    /**
     * @brief Returns the number of columns in the matrix.
     * @returns The number of columns in the matrix.
     */
    __host__ __device__ size_t cols() const { return cols_; }

    /**
     * @brief Flattens the matrix into a 1D array.
     * @param output Pointer to the output array (must be pre-allocated with rows*cols elements)
     */
    void flatten(double* output) const;

    /**
     * @brief Accesses the specified row of the matrix.
     * @param row The index of the row to access.
     * @returns A reference to the vector representing the row.
     */
    Vector& operator[](size_t row);

    /**
     * @brief Accesses the specified row of the matrix (const version).
     * @param row The index of the row to access.
     * @returns A const reference to the vector representing the row.
     */
    const Vector& operator[](size_t row) const;

    /**
     * @brief Accesses the specified element of the matrix.
     * @param row The index of the row of the element.
     * @param col The index of the column of the element.
     * @returns A reference to the element at the specified position.
     */
    __host__ __device__ double& at(size_t row, size_t col) {
        assert(row < rows_ && col < cols_);
        return values[row].at(col);
    }

    /**
     * @brief Accesses the specified element of the matrix (const version).
     * @param row The index of the row of the element.
     * @param col The index of the column of the element.
     * @returns A const reference to the element at the specified position.
     */
    __host__ __device__ double at(size_t row, size_t col) const {
        assert(row < rows_ && col < cols_);
        return values[row].at(col);
    }

    /**
     * @brief Adds two matrices.
     * @param other The matrix to add.
     * @returns A new matrix that is the sum of this matrix and the other matrix.
     * @throws std::invalid_argument if the dimensions of the matrices do not match.
     */
    Matrix operator+(const Matrix& other) const;

    /**
     * @brief Subtracts one matrix from another.
     * @param other The matrix to subtract.
     * @returns A new matrix that is the difference of this matrix and the other matrix.
     * @throws std::invalid_argument if the dimensions of the matrices do not match.
     */
    Matrix operator-(const Matrix& other) const;

    /**
     * @brief Multiplies two matrices.
     * @param other The matrix to multiply with.
     * @returns A new matrix that is the product of this matrix and the other matrix.
     * @throws std::invalid_argument if the number of columns in this matrix does not match the
     * number of rows in the other matrix.
     */
    Matrix operator*(const Matrix& other) const;

    /**
     * @brief Multiplies the matrix by a vector.
     * @param vector The vector to multiply with.
     * @returns A new vector that is the product of this matrix and the vector.
     * @throws std::invalid_argument if the number of columns in the matrix does not match the
     * size of the vector.
     */
    Vector operator*(const Vector& vector) const;

    /**
     * @brief Multiplies the matrix by a scalar.
     * @param scalar The scalar to multiply with.
     * @returns A new matrix that is the product of this matrix and the scalar.
     */
    Matrix operator*(double scalar) const;

    /**
     * @brief Transposes the matrix.
     * @returns A new matrix that is the transpose of this matrix.
     */
    Matrix transpose() const;

    /**
     * @brief Converts the matrix to a string representation.
     * @returns A string representation of the matrix.
     */
    std::string to_string() const;

    /**
     * @brief Checks if two matrices are equal.
     * @param other The matrix to compare with.
     * @returns true if the matrices are equal, false otherwise.
     */
    bool operator==(const Matrix& other) const;

private:
    size_t rows_;    /// The number of rows in the matrix.
    size_t cols_;    /// The number of columns in the matrix.
    Vector* values;  /// The values of the matrix stored as an array of Vector objects
};