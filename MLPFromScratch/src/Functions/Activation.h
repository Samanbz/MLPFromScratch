#pragma once

#include <cmath>

#include "../Vector/Vector.h"

class Activation {
public:
    static double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
    static double sigmoid_derivative(double x) { return x * (1.0 - x); }
    static double relu(double x) { return std::max(0.0, x); }
    static double relu_derivative(double x) { return (x > 0) ? 1.0 : 0.0; }
};
