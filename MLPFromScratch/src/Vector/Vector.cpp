#include "Vector.h"

#include <algorithm>
#include <cstring>
#include <format>

#include "../Matrix/Matrix.h"
#include "../config.h"

#ifndef __OMP_PARALLEL_FOR
#define __OMP_PARALLEL_FOR _Pragma("omp parallel for")
#endif

#define CONDITIONAL_PARALLEL(for_loop_code)       \
    if (size_ < PARALLEL_THRESHOLD) {             \
        /* Serial execution */                    \
        for_loop_code                             \
    } else {                                      \
        /* Parallel execution */                  \
        _Pragma("omp parallel for") for_loop_code \
    }

Vector::Vector() : values(nullptr), size_(0) {}

Vector::Vector(size_t size) : size_(size) {
    values = size_ > 0 ? new double[size_]() : nullptr;  // Initialize with zeros
}

Vector::Vector(size_t size, double value) : size_(size) {
    values = size_ > 0 ? new double[size_] : nullptr;
    std::fill_n(values, size_, value);
}

Vector::Vector(size_t size, const double* src_values) : size_(size) {
    values = size_ > 0 ? new double[size_] : nullptr;
    if (values && src_values) {
        std::memcpy(values, src_values, size_ * sizeof(double));
    }
}

Vector::Vector(std::initializer_list<double> init) : size_(init.size()) {
    values = size_ > 0 ? new double[size_] : nullptr;
    if (values) {
        std::copy(init.begin(), init.end(), values);
    }
}

Vector::Vector(std::vector<double> src_values) : size_(src_values.size()) {
    values = size_ > 0 ? new double[size_] : nullptr;
    if (values) {
        std::memcpy(values, src_values.data(), size_ * sizeof(double));
    }
}

// Copy constructor
Vector::Vector(const Vector& other) : size_(other.size_) {
    values = size_ > 0 ? new double[size_] : nullptr;
    if (values) {
        std::memcpy(values, other.values, size_ * sizeof(double));
    }
}

// Assignment operator
Vector& Vector::operator=(const Vector& other) {
    if (this != &other) {
        delete[] values;
        size_ = other.size_;
        values = size_ > 0 ? new double[size_] : nullptr;
        if (values) {
            std::memcpy(values, other.values, size_ * sizeof(double));
        }
    }
    return *this;
}

// Destructor
Vector::~Vector() { delete[] values; }

Vector Vector::random(double min, double max) {
    Vector result(1);
    result.values[0] =
        min + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (max - min)));
    return result;
}

double& Vector::operator[](size_t index) {
    if (index >= size_) {
        throw std::out_of_range("Vector index out of range");
    }
    return values[index];
}

const double& Vector::operator[](size_t index) const {
    if (index >= size_) {
        throw std::out_of_range("Vector index out of range");
    }
    return values[index];
}

std::vector<double> Vector::get_values() const {
    return std::vector<double>(values, values + size_);
}

double Vector::dot(const Vector& other) const {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vector dimensions must match.");
    }

    double result = 0;

    CONDITIONAL_PARALLEL(for (int i = 0; i < size_; i++) { result += values[i] * other.values[i]; })

    return result;
}

Vector Vector::operator+(const Vector& other) const {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vector dimensions must match.");
    }

    Vector result(size_);

    CONDITIONAL_PARALLEL(
        for (int i = 0; i < size_; i++) { result.values[i] = values[i] + other.values[i]; });
    return result;
}

Vector Vector::operator-(const Vector& other) const {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vector dimensions must match.");
    }

    Vector result(size_);

    CONDITIONAL_PARALLEL(
        for (int i = 0; i < size_; i++) { result.values[i] = values[i] - other.values[i]; });
    return result;
}

Vector Vector::operator*(double scalar) const {
    Vector result(size_);

    CONDITIONAL_PARALLEL(
        for (int i = 0; i < size_; i++) { result.values[i] = values[i] * scalar; });
    return result;
}

Vector Vector::elem_mult(const Vector& other) const {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vector dimensions must match.");
    }

    Vector result(size_);

    CONDITIONAL_PARALLEL(
        for (int i = 0; i < size_; i++) { result.values[i] = values[i] * other.values[i]; });
    return result;
}

Vector Vector::square() const {
    Vector result(size_);

    CONDITIONAL_PARALLEL(
        for (int i = 0; i < size_; i++) { result.values[i] = values[i] * values[i]; });
    return result;
}

double Vector::sum() const {
    double result = 0;

    for (int i = 0; i < size_; i++) {
        result += values[i];
    }
    return result;
}

double Vector::mean() const {
    if (size_ == 0) {
        throw std::invalid_argument("Cannot compute mean of an empty vector.");
    }
    return sum() / static_cast<double>(size_);
}

double Vector::norm() const { return std::sqrt(this->square().sum()); }

Matrix Vector::outer_product(const Vector& other) const {
    Matrix result(size_, other.size());

    CONDITIONAL_PARALLEL(for (int i = 0; i < size_; i++) {
        for (size_t j = 0; j < other.size(); j++) {
            result[i][j] = values[i] * other.values[j];
        }
    });
    return result;
}

Vector Vector::apply(std::function<double(double)> func) const {
    Vector result(size_);

    CONDITIONAL_PARALLEL(for (int i = 0; i < size_; i++) { result.values[i] = func(values[i]); });
    return result;
}

std::string Vector::to_string() const {
    std::string result = "[";
    for (size_t i = 0; i < size_; i++) {
        result += std::format("{:.2f}", values[i]);
        if (i < size_ - 1) {
            result += ", ";
        }
    }
    result += "]";
    return result;
}

bool Vector::operator==(const Vector& other) const {
    if (size_ != other.size_) {
        return false;
    }
    for (size_t i = 0; i < size_; i++) {
        if (values[i] != other.values[i]) {
            return false;
        }
    }
    return true;
}