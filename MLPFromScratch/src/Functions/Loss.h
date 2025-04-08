#pragma once

#include <cmath>
#include <stdexcept>

#include "../Vector/Vector.h"

class Loss {
public:
    double static mean_squared_error(const Vector& output, const Vector& target) {
        if (output.size() != target.size()) {
            throw std::invalid_argument("Output and target vectors must have the same size.");
        }
        return (output - target).square().sum() / 2 * output.size();
    }

    Vector static mean_squared_error_derivative(const Vector& output, const Vector& target) {
        if (output.size() != target.size()) {
            throw std::invalid_argument("Output and target vectors must have the same size.");
        }
        return (output - target);
    }
};
