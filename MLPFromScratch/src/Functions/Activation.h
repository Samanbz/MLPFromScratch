#pragma once

#include <cmath>
#include <functional>

#include "../Vector/Vector.cuh"

class Activation {
public:
    Activation(std::function<double(double)> f, std::function<double(double)> df) : f(f), df(df) {}

    double operator()(double x) const { return f(x); }

    double derivative(double x) const { return df(x); }

    Vector operator()(const Vector& x) const { return x.apply(f); }

    Vector derivative(const Vector& x) const { return x.apply(df); }

private:
    std::function<double(double)> f;   // Activation function
    std::function<double(double)> df;  // Derivative of the activation function
};

inline double sgmd(double x) { return 1.0 / (1.0 + std::exp(-x)); };

namespace ActivationFunctions {

    // sigmoid
    inline Activation sigmoid([](double x) { return sgmd(x); },
                              [](double x) { return sgmd(x) * (1.0 - sgmd(x)); });

    // tanh
    inline Activation tanh_activation([](double x) { return std::tanh(x); },
                                      [](double x) { return 1.0 - std::pow(std::tanh(x), 2); });

    // ReLU
    inline Activation relu([](double x) { return std::max(0.0, x); },
                           [](double x) { return (x > 0) ? 1.0 : 0.0; });

    // Leaky ReLU
    inline Activation leaky_relu([](double x) { return (x > 0) ? x : 0.01 * x; },
                                 [](double x) { return (x > 0) ? 1.0 : 0.01; });

    // Softmax
    inline Activation softmax([](double x) { return std::exp(x); },
                              [](double x) { return x * (1.0 - x); });

    // Linear
    inline Activation linear([](double x) { return x; }, [](double x) { return 1.0; });

}  // namespace ActivationFunctions