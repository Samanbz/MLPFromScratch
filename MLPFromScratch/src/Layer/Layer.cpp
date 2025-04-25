#include "Layer.h"

Layer::Layer(size_t input_size, size_t neuron_count, Activation activation)
    : neuron_count(neuron_count),
      input_size(input_size),
      gradient(Vector(neuron_count)),
      activation(activation) {
    for (size_t i = 0; i < neuron_count; i++) {
        neurons.push_back(Neuron(input_size, activation));
    }
}

Layer::Layer(size_t input_size, size_t neuron_count, const Matrix& init_weights,
             Activation activation)
    : neuron_count(neuron_count),
      input_size(input_size),
      gradient(Vector(neuron_count)),
      activation(activation) {
    if (init_weights.rows() != neuron_count || init_weights.cols() != input_size) {
        throw std::invalid_argument(
            "Initial weights size must match layer input size and neuron count.");
    }

    for (size_t i = 0; i < neuron_count; i++) {
        neurons.push_back(Neuron(init_weights[i], 0, activation));
    }
}

Vector Layer::forward(const Vector& input) {
    if (input.size() != input_size) {
        throw std::invalid_argument("Input size must match layer input size.");
    }

    Vector output(neuron_count);
    for (size_t i = 0; i < neuron_count; ++i) {
        output[i] = neurons[i].forward(input);
    }
    return output;
}

size_t Layer::get_size() const { return neuron_count; }

size_t Layer::get_input_size() const { return input_size; }

std::string Layer::to_string() const {
    std::string result = "Layer:\n";
    for (size_t i = 0; i < neuron_count; ++i) {
        result += "    " + neurons[i].to_string() + "\n";
    }
    return result;
}

std::vector<Neuron> Layer::get_neurons() const { return neurons; }

Vector Layer::get_outputs() const {
    Vector current_values(neuron_count);
    for (size_t i = 0; i < neuron_count; ++i) {
        current_values[i] = neurons[i].get_output();
    }
    return current_values;
}

Vector Layer::get_pre_activations() const {
    Vector current_values(neuron_count);
    for (size_t i = 0; i < neuron_count; ++i) {
        current_values[i] = neurons[i].get_pre_activation();
    }
    return current_values;
}

Vector Layer::get_gradient() const { return gradient; }

void Layer::set_gradient(const Vector& gradient) {
    if (gradient.size() != neuron_count) {
        throw std::invalid_argument("Gradient size must match layer size.");
    }
    this->gradient = gradient;
}

Matrix Layer::get_weights() const {
    Matrix weights(neuron_count, input_size);
    for (size_t i = 0; i < neuron_count; ++i) {
        weights[i] = neurons[i].get_weights();
    }
    return weights;
}

void Layer::update_weights(const Matrix& gradient, double learning_rate) {
    if (gradient.rows() != neuron_count || gradient.cols() != input_size) {
        throw std::invalid_argument("Weight gradient size must match layer size.");
    }
    for (size_t i = 0; i < neuron_count; i++) {
        neurons[i].update_weights(gradient[i], learning_rate);
    }
}

Vector Layer::get_biases() const {
    Vector biases(neuron_count);
    for (size_t i = 0; i < neuron_count; ++i) {
        biases[i] = neurons[i].get_bias();
    }
    return biases;
}

void Layer::update_biases(const Vector& gradient, double learning_rate) {
    if (gradient.size() != neuron_count) {
        throw std::invalid_argument("Bias gradient size must match layer size.");
    }
    for (size_t i = 0; i < neuron_count; i++) {
        neurons[i].update_bias(gradient[i], learning_rate);
    }
}

Vector Layer::activation_derivative(const Vector& pre_activations) const {
    return activation.derivative(pre_activations);
}
