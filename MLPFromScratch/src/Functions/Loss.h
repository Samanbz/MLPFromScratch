#pragma once

#include <functional>

#include "../Vector/Vector.h"

class Loss {
public:
    Loss(std::function<double(const Vector&, const Vector&)> f,
         std::function<Vector(const Vector&, const Vector&)> df)
        : f(f), df(df) {}

    double operator()(const Vector& y_true, const Vector& y_pred) const {
        return f(y_true, y_pred);
    }

    Vector derivative(const Vector& y_true, const Vector& y_pred) const {
        return df(y_true, y_pred);
    }

private:
    std::function<double(Vector, Vector)> f;   // Loss function
    std::function<Vector(Vector, Vector)> df;  // Derivative of the loss function
};

namespace LossFunctions {

    // Mean Squared Error
    inline Loss mse([](const Vector& y_true,
                       const Vector& y_pred) { return ((y_true - y_pred).square()).mean(); },
                    [](const Vector& y_true, const Vector& y_pred) {
                        return ((y_pred - y_true) * 2) * (static_cast<double>(1) / y_true.size());
                    });

    // Mean Absolute Error
    inline Loss mae(
        [](const Vector& y_true, const Vector& y_pred) {
            return (y_true - y_pred).apply([](double x) { return std::abs(x); }).mean();
        },
        [](const Vector& y_true, const Vector& y_pred) {
            return ((y_pred - y_true).apply([](double x) { return (x > 0) ? 1.0 : -1.0; })) *
                   (static_cast<double>(1) / y_true.size());
        });

}  // namespace LossFunctions