#include "MLP.h"

#include <iostream>

#include "../Functions/Activation.h"
#include "../Functions/Loss.h"

MLP::MLP(std::vector<size_t> layer_sizes) {
    if (layer_sizes.empty() || layer_sizes.size() < 2) {
        throw std::invalid_argument("MLP must have at least 2 layers (input and output).");
    }
    if (layer_sizes[0] == 0 || layer_sizes.back() == 0) {
        throw std::invalid_argument("Input and output layer sizes must be greater than 0.");
    }

    input_size = layer_sizes[0];
    output_size = layer_sizes.back();

    input = Vector(input_size);

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
    layers[0].forward(input);
    for (size_t i = 1; i < layers.size(); i++) {
        output = layers[i].forward(layers[i - 1].get_outputs());
    }

    return output;
}

double MLP::backward(const Vector& target) {
    Layer& output_layer = layers.back();

    double loss = this->loss(output_layer.get_outputs(), target);

    Vector output_delta = compute_output_delta(target);

    Matrix output_weight_delta =
        compute_weight_delta(layers[layers.size() - 2].get_outputs(), output_delta);

    output_layer.update_weights(output_weight_delta, learning_rate);

    for (size_t i = layers.size() - 1; i-- > 0;) {
        Layer& layer = layers[i];
        Layer& next_layer = layers[i + 1];

        Vector layer_delta = compute_layer_delta(i);

        Matrix weight_delta;
        if (i > 0) {
            Layer& prev_layer = layers[i - 1];
            weight_delta = compute_weight_delta(prev_layer.get_outputs(), layer_delta);
        } else {
            weight_delta = compute_weight_delta(input, layer_delta);
        }

        layer.update_weights(weight_delta, learning_rate);
    }

    return loss;
}

void MLP::train(const std::vector<Vector>& inputs, const std::vector<Vector>& targets,
                size_t epochs, size_t log_interval) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs and targets must match.");
    }

    double epoch_loss = 0;
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            forward(inputs[i]);
            epoch_loss = backward(targets[i]);
        }
        if (log_interval > 0 && (epoch + 1) % log_interval == 0) {
            std::cout << "Epoch " << epoch + 1 << ", Loss: " << epoch_loss << std::endl;
        }
    }
}

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

Vector MLP::compute_output_delta(const Vector& target) {
    Layer& output_layer = layers.back();

    Vector output_delta = output_layer.get_pre_activations()
                              .apply(Activation::sigmoid_derivative)
                              .elem_mult(this->loss_derivative(output_layer.get_outputs(), target));

    output_layer.set_delta(output_delta);

    return output_delta;
}

Matrix MLP::compute_weight_delta(const Vector& layer_output, const Vector& layer_delta) {
    return layer_output.outer_product(layer_delta);
}

Vector MLP::compute_layer_delta(size_t layer_idx) {
    if (layer_idx >= layers.size() - 1) {
        throw std::out_of_range("Layer index out of range.");
    }

    Layer& layer = layers[layer_idx];
    Layer& next_layer = layers[layer_idx + 1];

    Vector layer_delta =
        (next_layer.get_weights() * next_layer.get_delta())
            .elem_mult(layer.get_pre_activations().apply(Activation::sigmoid_derivative));

    layer.set_delta(layer_delta);

    return layer_delta;
}
