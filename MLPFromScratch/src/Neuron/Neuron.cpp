#include "Neuron.h"

#include <stdexcept>

Neuron::Neuron(size_t input_size) : weights(input_size), bias(0), output(-1), z(-1) {
    // Initialize weights with random values
    for (size_t i = 0; i < input_size; ++i) {
        weights[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

Neuron::Neuron(const Vector& initial_weights, double initial_bias)
    : weights(initial_weights), bias(initial_bias), output(-1), z(-1) {
    if (initial_weights.size() == 0) {
        throw std::invalid_argument("Weights vector cannot be empty.");
    }
}

double Neuron::activate(double z) const {
    // Sigmoid activation function
    return 1.0 / (1.0 + exp(-z));
}

double Neuron::activation_derivative(double z) const {
    // Derivative of the sigmoid function
    double activated_value = activate(z);
    return activated_value * (1 - activated_value);
}

double Neuron::forward(const Vector& input) {
    if (input.size() != weights.size()) {
        throw std::invalid_argument("Input size must match weights size.");
    }
    z = weights.dot(input) + bias;
    output = activate(z);
    return output;
}

double Neuron::get_output() const { return output; }

double Neuron::get_z() const { return z; }

Vector Neuron::get_weights() const { return weights; }

double Neuron::get_bias() const { return bias; }

std::string Neuron::to_string() const {
    std::string result = "Neuron:\n";
    result += "\tWeights: " + weights.to_string() + "\n";
    result += "\tBias: " + std::to_string(bias) + "\n";
    result += "\tOutput: " + std::to_string(output) + "\n";
    return result;
}
