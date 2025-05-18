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

    // Cross Entropy
    inline Loss cross_entropy(
        [](const Vector& y_true, const Vector& y_pred) {
            // Compute softmax
            Vector softmax_pred = y_pred.apply([](double x) { return std::exp(x); });
            double sum = softmax_pred.sum();
            softmax_pred = softmax_pred.apply([sum](double x) { return x / sum; });

            // Compute cross-entropy
            double loss = 0.0;
            for (size_t i = 0; i < y_true.size(); ++i) {
                loss -= y_true[i] * std::log(softmax_pred[i]);
            }

            return loss;
        },
        [](const Vector& y_true, const Vector& y_pred) {
            // Compute softmax
            Vector softmax_pred = y_pred.apply([](double x) { return std::exp(x); });
            double sum = softmax_pred.sum();
            softmax_pred = softmax_pred.apply([sum](double x) { return x / sum; });

            // The derivative is simply softmax(y_pred) - y_true
            return softmax_pred - y_true;
        });
}  // namespace LossFunctions