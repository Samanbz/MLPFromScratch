#include "MLP.h"

MLP::MLP(std::vector<size_t> layer_sizes) {
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("MLP must have at least 2 layers (input and output).");
    }

    input_size = layer_sizes[0];
    output_size = layer_sizes.back();

    for (size_t i = 1; i < layer_sizes.size(); ++i) {
        layers.emplace_back(layer_sizes[i - 1], layer_sizes[i]);
    };
}

Vector MLP::forward(const Vector& input) {
    if (input.size() != input_size) {
        throw std::invalid_argument("Input size does not match MLP input size.");
    }

    Vector output = input;
    for (auto& layer : layers) {
        output = layer.forward(output);
    }

    return output;
}
