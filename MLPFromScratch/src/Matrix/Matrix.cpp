#include "Matrix.h"

#include <cassert>
#include <cstring>

Matrix::Matrix() : rows_(0), cols_(0), values(nullptr) {}

Matrix::Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
    values = new Vector[rows];
    for (size_t i = 0; i < rows; ++i) {
        values[i] = Vector(cols);
    }
}

Matrix::Matrix(size_t rows, size_t cols, double value) : rows_(rows), cols_(cols) {
    values = new Vector[rows];
    for (size_t i = 0; i < rows; ++i) {
        values[i] = Vector(cols, value);
    }
}

Matrix::Matrix(std::vector<std::vector<double>> values)
    : rows_(values.size()), cols_(values[0].size()) {
    this->values = new Vector[rows_];
    for (size_t i = 0; i < rows_; i++) {
        this->values[i] = Vector(values[i]);
    }
}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> values)
    : rows_(values.size()), cols_(values.begin()->size()) {
    this->values = new Vector[rows_];
    size_t i = 0;
    for (const auto& row : values) {
        assert(row.size() == cols_ && "All rows must have the same number of columns");
        this->values[i] = Vector(row);
        i++;
    }
}

Matrix::Matrix(size_t rows, size_t cols, const double* data) : rows_(rows), cols_(cols) {
    values = new Vector[rows];
    for (size_t i = 0; i < rows; i++) {
        std::vector<double> row_values(cols);
        std::memcpy(row_values.data(), data + (i * cols), cols * sizeof(double));
        values[i] = Vector(row_values);
    }
}

Matrix::Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_) {
    values = new Vector[rows_];
    for (size_t i = 0; i < rows_; ++i) {
        values[i] = other.values[i];
    }
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        // Free existing resources
        if (values != nullptr) {
            delete[] values;
        }

        // Copy from other
        rows_ = other.rows_;
        cols_ = other.cols_;
        values = new Vector[rows_];
        for (size_t i = 0; i < rows_; ++i) {
            values[i] = other.values[i];
        }
    }
    return *this;
}

Matrix::~Matrix() {
    if (values != nullptr) {
        delete[] values;
        values = nullptr;
    }
}

void Matrix::flatten(double* output) const {
    for (size_t i = 0; i < rows_; i++) {
        std::memcpy(output + (i * cols_), values[i].get_values().data(), cols_ * sizeof(double));
    }
}

Vector& Matrix::operator[](size_t row) {
    assert(row < rows_);
    return values[row];
}

const Vector& Matrix::operator[](size_t row) const {
    assert(row < rows_);
    return values[row];
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_ != other.rows() || cols_ != other.cols()) {
        throw std::invalid_argument("Matrix dimensions must match for addition.");
    }

    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
            result[i][j] = values[i][j] + other[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_ != other.rows() || cols_ != other.cols()) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction.");
    }

    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
            result[i][j] = values[i][j] - other[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows()) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication.");
    }

    Matrix result(rows_, other.cols());
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < other.cols(); j++) {
            double sum = 0.0;
            for (size_t k = 0; k < cols_; k++) {
                sum += values[i][k] * other[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

Vector Matrix::operator*(const Vector& vector) const {
    if (cols_ != vector.size()) {
        throw std::invalid_argument(
            "Matrix and vector dimensions incompatible for multiplication.");
    }

    Vector result(rows_);
    for (size_t i = 0; i < rows_; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < cols_; j++) {
            sum += values[i][j] * vector[j];
        }
        result[i] = sum;
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
            result[i][j] = values[i][j] * scalar;
        }
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
        result += "\t" + values[i].to_string();  // Set precision to 2
        if (i < rows_ - 1) {
            result += ",\n";
        }
    }
    result += "\n]";
    return result;
}

bool Matrix::operator==(const Matrix& other) const {
    if (rows_ != other.rows() || cols_ != other.cols()) {
        return false;
    }
    for (size_t i = 0; i < rows_; i++) {
        if (values[i] != other[i]) {
            return false;
        }
    }
    return true;
}