#include "Vector.cuh"

#include <format>
#include <iostream>

#include "../Matrix/Matrix.h"

Vector::Vector() : values(0) {};

Vector::Vector(size_t size) : values(size) {};

Vector::Vector(size_t size, double value) {
    this->values.reserve(size);
    for (size_t i = 0; i < size; i++) {
        this->values.push_back(value);
    }
};

Vector::Vector(size_t size, const double* values)
    : values(values ? std::vector<double>(values, values + size) : std::vector<double>(size)) {}

Vector::Vector(std::initializer_list<double> init) : Vector(init.size(), init.begin()) {}

Vector::Vector(std::vector<double> values) {
    this->values.reserve(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        this->values.push_back(values[i]);
    }
};

Vector Vector::random(double min, double max) {
    Vector result(1);
    result.values[0] =
        min + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (max - min)));
    return result;
}

double& Vector::operator[](size_t index) { return values[index]; }

const double& Vector::operator[](size_t index) const { return values[index]; }

// double Vector::dot(const Vector& other) const {
//     if (values.size() != other.size()) {
//         throw std::invalid_argument("Vector dimensions must match.");
//     }
//
//     double result = 0;
//     for (size_t i = 0; i < values.size(); i++) {
//         result += values[i] * other[i];
//     }
//     return result;
// }

// Vector Vector::operator+(const Vector& other) const {
//     Vector result(values.size());
//     for (size_t i = 0; i < values.size(); i++) {
//         result[i] = values[i] + other[i];
//     }
//     return result;
// }

Vector Vector::operator-(const Vector& other) const {
    Vector result(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        result[i] = values[i] - other[i];
    }
    return result;
}

Vector Vector::operator*(double scalar) const {
    Vector result(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        result[i] = values[i] * scalar;
    }
    return result;
}

// Vector Vector::elem_mult(const Vector& other) const {
//     if (values.size() != other.size()) {
//         throw std::invalid_argument("Vector dimensions must match.");
//     }
//     Vector result(values.size());
//     for (size_t i = 0; i < values.size(); i++) {
//         result[i] = values[i] * other[i];
//     }
//     return result;
// }

Vector Vector::square() const {
    Vector result(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        result[i] = values[i] * values[i];
    }
    return result;
}

// double Vector::sum() const {
//     double result = 0;
//     for (size_t i = 0; i < values.size(); i++) {
//         result += values[i];
//     }
//     return result;
// }

double Vector::mean() const {
    if (values.size() == 0) {
        throw std::invalid_argument("Cannot compute mean of an empty vector.");
    }
    return sum() / static_cast<double>(values.size());
}

double Vector::norm() const { return std::sqrt(this->square().sum()); }

Vector Vector::apply(std::function<double(double)> func) const {
    Vector result(values.size());

    for (size_t i = 0; i < values.size(); i++) {
        result[i] = func(values[i]);
    }

    return result;
}

Matrix Vector::outer_product(const Vector& other) const {
    Matrix result(values.size(), other.size());
    for (size_t i = 0; i < values.size(); i++) {
        for (size_t j = 0; j < other.size(); j++) {
            result.at(i, j) = values[i] * other[j];
        }
    }
    return result;
}

std::string Vector::to_string() const {
    std::string result = "[";
    for (size_t i = 0; i < values.size(); i++) {
        result += std::format("{:.2f}", values[i]);
        if (i < values.size() - 1) {
            result += ", ";
        }
    }
    result += "]";
    return result;
}

bool Vector::operator==(const Vector& other) const {
    if (values.size() != other.size()) {
        return false;
    }
    for (size_t i = 0; i < values.size(); i++) {
        if (values[i] != other[i]) {
            return false;
        }
    }
    return true;
}