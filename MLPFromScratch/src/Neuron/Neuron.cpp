#include "Neuron.h"

#include <iostream>
#include <stdexcept>
Neuron::Neuron(size_t input_size, Activation activation)
    : weights(input_size), bias(0), output(0), z(0), activation(activation) {
    // Initialize weights with random values
    for (size_t i = 0; i < input_size; ++i) {
        weights[i] = (rand() / (double)RAND_MAX) * sqrt(2.0 / input_size);
        {
        }
    }
}

Neuron::Neuron(const Vector& initial_weights, double initial_bias, Activation activation)
    : weights(initial_weights), bias(initial_bias), output(0), z(0), activation(activation) {
    if (initial_weights.size() == 0) {
        throw std::invalid_argument("Weights vector cannot be empty.");
    }
}

double Neuron::forward(const Vector& input) {
    if (input.size() != weights.size()) {
        throw std::invalid_argument("Input size must match weights size.");
    }
    z = weights.dot(input) + bias;
    output = activation(z);
    return output;
}

double Neuron::get_output() const { return output; }

double Neuron::get_pre_activation() const { return z; }

Vector Neuron::get_weights() const { return weights; }

void Neuron::update_weights(const Vector& gradient, double learning_rate) {
    if (gradient.size() != weights.size()) {
        throw std::invalid_argument("Weight gradient size must match weights size.");
    }

    Vector clipped_gradient = clip_gradient(gradient, 1);
    this->weight_gradient = clipped_gradient;

    weights = weights - (clipped_gradient * learning_rate);
}

Vector Neuron::clip_gradient(const Vector& gradient, double clip_threshold) {
    double clip_factor = clip_threshold / gradient.norm();
    if (clip_factor < 1.0) {
        Vector clipped_gradient = gradient * clip_factor;
        return clipped_gradient;
    }
    return gradient;
}

double Neuron::get_bias() const { return bias; }

void Neuron::update_bias(double gradient, double learning_rate) {
    this->bias -= gradient * learning_rate;
}

Vector Neuron::get_weight_gradient() const { return weight_gradient; }

std::string Neuron::to_string() const {
    std::string result = "Neuron:\n";
    result += "\tWeights: " + weights.to_string() + "\n";
    result += "\tBias: " + std::to_string(bias) + "\n";
    result += "\tOutput: " + std::to_string(output) + "\n";
    return result;
}