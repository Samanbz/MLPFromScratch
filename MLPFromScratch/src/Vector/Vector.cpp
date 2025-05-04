#include "Vector.cuh"

#include <algorithm>
#include <cstring>
#include <format>

#include "../Matrix/Matrix.cuh"

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

Vector Vector::apply(std::function<double(double)> func) const {
    Vector result(size_);
    for (size_t i = 0; i < size_; i++) {
        result.values[i] = func(values[i]);
    }
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