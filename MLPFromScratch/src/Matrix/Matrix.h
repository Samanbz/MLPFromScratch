#pragma once

#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

#include "../Vector/Vector.h"

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
     * @brief Returns the number of rows in the matrix.
     * @returns The number of rows in the matrix.
     */
    size_t rows() const;

    /**
     * @brief Returns the number of columns in the matrix.
     * @returns The number of columns in the matrix.
     */
    size_t cols() const;

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
    double& at(size_t row, size_t col);

    /**
     * @brief Accesses the specified element of the matrix (const version).
     * @param row The index of the row of the element.
     * @param col The index of the column of the element.
     * @returns A const reference to the element at the specified position.
     */
    const double& at(size_t row, size_t col) const;

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
     * @throws std::invalid_argument if the number of columns in the matrix does not match the size
     * of the vector.
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
    size_t rows_;                /// The number of rows in the matrix.
    size_t cols_;                /// The number of columns in the matrix.
    std::vector<Vector> values;  /// The values of the matrix stored as a vector of vectors.
};
