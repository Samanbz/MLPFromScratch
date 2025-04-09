#include "Layer.h"

Layer::Layer(size_t input_size, size_t neuron_count)
    : neuron_count(neuron_count), input_size(input_size), delta(Vector(neuron_count)) {
    neurons.resize(neuron_count, Neuron(input_size));
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

Vector Layer::get_delta() const { return delta; }

void Layer::set_delta(const Vector& delta) {
    if (delta.size() != neuron_count) {
        throw std::invalid_argument("Delta size must match layer size.");
    }
    this->delta = delta;
}

Matrix Layer::get_weights() const {
    Matrix weights(neuron_count, input_size);
    for (size_t i = 0; i < neuron_count; ++i) {
        weights[i] = neurons[i].get_weights();
    }
    return weights.transpose();
}

void Layer::update_weights(const Matrix& weight_delta, double learning_rate) {
    Matrix weight_delta_t = weight_delta.transpose();
    if (weight_delta_t.rows() != neuron_count || weight_delta_t.cols() != input_size) {
        throw std::invalid_argument("Weight delta size must match layer size.");
    }
    for (size_t i = 0; i < neuron_count; i++) {
        neurons[i].update_weights(weight_delta_t[i], learning_rate);
    }
}