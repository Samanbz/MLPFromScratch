#include "Matrix.h"

Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), values(std::vector(rows, Vector(cols))){};

Matrix::Matrix(size_t rows, size_t cols, double value)
    : rows_(rows), cols_(cols), values(std::vector(rows, Vector(cols, value))){};

Matrix::Matrix(std::vector<std::vector<double>> values)
    : rows_(values.size()),
      cols_(values[0].size()),
      values(std::vector<Vector>(rows_, Vector(cols_, 0.0))) {
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
            this->values[i][j] = values[i][j];
        }
    }
}

size_t Matrix::rows() const { return rows_; }

size_t Matrix::cols() const { return cols_; }

Vector& Matrix::operator[](size_t row) { return values[row]; }

const Vector& Matrix::operator[](size_t row) const { return values[row]; }

double& Matrix::at(size_t row, size_t col) { return values[row][col]; }

const double& Matrix::at(size_t row, size_t col) const { return values[row][col]; }

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_ != other.rows() || cols_ != other.cols()) {
        throw std::invalid_argument("Matrix dimensions must match.");
    }

    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; i++) {
        result[i] = values[i] + other[i];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_ != other.rows() || cols_ != other.cols()) {
        throw std::invalid_argument("Matrix dimensions must match.");
    }

    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; i++) {
        result[i] = values[i] - other[i];
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows()) {
        throw std::invalid_argument("Matrix dimensions must match.");
    }

    Matrix result(rows_, other.cols());
    auto other_transpose = other.transpose();
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < other.cols(); j++) {
            result[i][j] = values[i].dot(other_transpose[j]);
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; i++) {
        result[i] = values[i] * scalar;
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
            result[j][i] = values[i][j];
        }
    }
    return result;
}

std::string Matrix::to_string() const {
    std::string result = "[\n";
    for (size_t i = 0; i < rows_; i++) {
        result += "\t" + values[i].to_string();
        if (i < rows_ - 1) {
            result += ",\n";
        }
    }
    result += "\n]";
    return result;
}
