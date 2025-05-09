#include "Layer.h"

#include <cmath>

Layer::Layer(size_t input_size, size_t neuron_count, Activation activation)
    : neuron_count(neuron_count),
      input_size(input_size),
      weights(neuron_count, input_size),
      biases(neuron_count, 0.0),
      outputs(neuron_count, 0.0),
      pre_activations(neuron_count, 0.0),
      gradient(neuron_count, 0.0),
      activation(activation) {
    // Initialize weights with Xavier/Glorot initialization
    double scale = sqrt(2.0 / input_size);
    for (size_t i = 0; i < neuron_count; ++i) {
        for (size_t j = 0; j < input_size; ++j) {
            weights[i][j] = (rand() / (double)RAND_MAX) * scale;
        }
    }
}

Layer::Layer(size_t input_size, size_t neuron_count, const Matrix& init_weights,
             Activation activation)
    : neuron_count(neuron_count),
      input_size(input_size),
      weights(init_weights),
      biases(neuron_count, 0.0),
      outputs(neuron_count, 0.0),
      pre_activations(neuron_count, 0.0),
      gradient(neuron_count, 0.0),
      activation(activation) {
    if (init_weights.rows() != neuron_count || init_weights.cols() != input_size) {
        throw std::invalid_argument(
            "Initial weights size must match layer input size and neuron count.");
    }
}

Vector Layer::forward(const Vector& input) {
    if (input.size() != input_size) {
        throw std::invalid_argument("Input size must match layer input size.");
    }

    pre_activations = (weights * input) + biases;

    outputs = activation(pre_activations);

    return outputs;
}

size_t Layer::get_size() const { return neuron_count; }

size_t Layer::get_input_size() const { return input_size; }

std::string Layer::to_string() const {
    std::string result = "Layer:\n";
    result += "  Weights: " + weights.to_string() + "\n";
    result += "  Biases: " + biases.to_string() + "\n";
    result += "  Outputs: " + outputs.to_string() + "\n";
    return result;
}

Vector Layer::get_outputs() const { return outputs; }

Vector Layer::get_pre_activations() const { return pre_activations; }

Vector Layer::get_gradient() const { return gradient; }

void Layer::set_gradient(const Vector& gradient) {
    if (gradient.size() != neuron_count) {
        throw std::invalid_argument("Gradient size must match layer size.");
    }
    this->gradient = gradient;
}

Matrix Layer::get_weights() const { return weights; }

void Layer::update_weights(const Matrix& weight_gradient, double learning_rate) {
    if (weight_gradient.rows() != neuron_count || weight_gradient.cols() != input_size) {
        throw std::invalid_argument("Weight gradient size must match layer dimensions.");
    }

    weights = weights - (weight_gradient * learning_rate);
}

Vector Layer::get_biases() const { return biases; }

void Layer::update_biases(const Vector& bias_gradient, double learning_rate) {
    if (bias_gradient.size() != neuron_count) {
        throw std::invalid_argument("Bias gradient size must match layer size.");
    }

    biases = biases - (bias_gradient * learning_rate);
}

Vector Layer::activation_derivative(const Vector& input) const {
    return activation.derivative(input);
}

void Layer::clip_gradient(const Vector& gradient, double clip_threshold, Vector& result) {
    double norm = gradient.norm();
    double clip_factor = clip_threshold / norm;

    if (clip_factor < 1.0) {
        result = gradient * clip_factor;
    } else {
        result = gradient;
    }
}