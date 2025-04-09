#include "Neuron.h"

#include <stdexcept>

#include "../Functions/Activation.h"

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

double Neuron::forward(const Vector& input) {
    if (input.size() != weights.size()) {
        throw std::invalid_argument("Input size must match weights size.");
    }
    z = weights.dot(input) + bias;
    output = activate(z);
    return output;
}

double Neuron::get_output() const { return output; }

double Neuron::get_pre_activation() const { return z; }

Vector Neuron::get_weights() const { return weights; }

void Neuron::update_weights(const Vector& weight_delta, double learning_rate) {
    if (weight_delta.size() != weights.size()) {
        throw std::invalid_argument("Weight delta size must match weights size.");
    }
    weights = weights - (weight_delta * learning_rate);
}

double Neuron::get_bias() const { return bias; }

std::string Neuron::to_string() const {
    std::string result = "Neuron:\n";
    result += "\tWeights: " + weights.to_string() + "\n";
    result += "\tBias: " + std::to_string(bias) + "\n";
    result += "\tOutput: " + std::to_string(output) + "\n";
    return result;
}

double Neuron::activate(double z) const { return Activation::sigmoid(z); }

double Neuron::activation_derivative(double z) const { return Activation::sigmoid_derivative(z); }