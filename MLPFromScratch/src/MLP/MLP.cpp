#include "MLP.h"

#include <iostream>

#include "../Functions/Activation.h"
#include "../Functions/Loss.h"

MLP::MLP(std::vector<size_t> layer_sizes)
    : input(layer_sizes[0]),  // Initialize using constructor, not assignment
      input_size(layer_sizes[0]),
      output_size(layer_sizes.back()) {
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("MLP must have at least 2 layers (input and output).");
    }
    if (layer_sizes[0] == 0 || layer_sizes.back() == 0) {
        throw std::invalid_argument("Input and output layer sizes must be greater than 0.");
    }

    layers.reserve(layer_sizes.size() - 1);
    for (size_t i = 1; i < layer_sizes.size(); i++) {
        layers.emplace_back(layer_sizes[i - 1], layer_sizes[i]);
    };
}

Vector MLP::forward(const Vector& input) {
    if (input.size() != input_size) {
        throw std::invalid_argument("Input size does not match MLP input size.");
    }

    this->input = input;

    Vector output(input_size);
    for (auto& layer : layers) {
        output = layer.forward(output);
    }

    return output;
}

void MLP::backward(const Vector& target) {
    Layer& output_layer = layers.back();

    Vector output_delta = output_layer.get_pre_activations()
                              .apply(Activation::sigmoid_derivative)
                              .elem_mult(this->loss_derivative(output_layer.get_outputs(), target));

    output_layer.set_delta(output_delta);

    for (size_t i = layers.size() - 1; i-- > 0;) {
        Layer& layer = layers[i];
        Layer& next_layer = layers[i + 1];

        Vector layer_delta =
            (next_layer.get_weights() * next_layer.get_delta())
                .elem_mult(layer.get_pre_activations().apply(Activation::sigmoid_derivative));

        layer.set_delta(layer_delta);

        if (i == 1) {
            Matrix weight_delta = input.outer_product(layer_delta);
        } else {
            Layer& prev_layer = layers[i - 1];
            Matrix weight_delta = prev_layer.get_outputs().outer_product(layer_delta);
        }
    }
}

std::vector<Layer> MLP::get_layers() const { return layers; }

double MLP::loss(const Vector& output, const Vector& target) {
    if (output.size() != target.size()) {
        throw std::invalid_argument("Output size (" + std::to_string(output.size()) +
                                    ") does not match target size (" +
                                    std::to_string(target.size()) + ").");
    }

    return Loss::mean_squared_error(output, target);
}

Vector MLP::loss_derivative(const Vector& output, const Vector& target) {
    if (output.size() != target.size()) {
        throw std::invalid_argument("Output size (" + std::to_string(output.size()) +
                                    ") does not match target size (" +
                                    std::to_string(target.size()) + ").");
    }

    return Loss::mean_squared_error_derivative(output, target);
}
