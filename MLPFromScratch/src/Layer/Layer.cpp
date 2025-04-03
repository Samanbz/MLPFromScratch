#include "Layer.h"

Layer::Layer(size_t input_size, size_t neuron_count)
    : neuron_count(neuron_count), input_size(input_size) {
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
        result += "\t" + neurons[i].to_string() + "\n";
    }
    return result;
}