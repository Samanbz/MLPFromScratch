#include "Vector.h"

#include <format>
#include <iostream>

Vector::Vector(size_t size) : values(size){};

Vector::Vector(size_t size, double value) : values(size, value){};

Vector::Vector(size_t size, double* values) : values(values, values + size){};

Vector::Vector(size_t size, std::vector<double> values) : values(values){};

size_t Vector::size() const { return values.size(); }

double& Vector::operator[](size_t index) { return values[index]; }

const double& Vector::operator[](size_t index) const { return values[index]; }

double Vector::dot(const Vector& other) const {
    double result = 0;
    for (size_t i = 0; i < values.size(); i++) {
        result += values[i] * other[i];
    }
    return result;
}

Vector Vector::operator+(const Vector& other) const {
    Vector result(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        result[i] = values[i] + other[i];
    }
    return result;
}

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
