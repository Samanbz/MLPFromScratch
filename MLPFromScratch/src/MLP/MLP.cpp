#include "MLP.h"

#include <iostream>

#include "../Functions/Activation.h"
#include "../Functions/Loss.h"

MLP::MLP(std::vector<size_t> layer_sizes, std::vector<Activation> layer_activations,
         Loss loss_function)
    : loss_function(loss_function) {
    validate_init(layer_sizes, layer_activations, loss_function);

    input_size = layer_sizes[0];
    output_size = layer_sizes.back();

    input = Vector(input_size);
    weight_gradients.resize(layer_sizes.size() - 1);

    layers.reserve(layer_sizes.size() - 1);
    for (size_t i = 1; i < layer_sizes.size(); i++) {
        layers.emplace_back(layer_sizes[i - 1], layer_sizes[i], layer_activations[i - 1]);
    }
}

MLP::MLP(std::vector<size_t> layer_sizes, std::vector<Matrix> init_weights,
         std::vector<Activation> layer_activations, Loss loss_function)
    : loss_function(loss_function) {
    validate_init(layer_sizes, layer_activations, loss_function);
    if (layer_sizes.size() - 1 != init_weights.size()) {
        throw std::invalid_argument("Number of layers must match number of initial weights.");
    }

    input_size = layer_sizes[0];
    output_size = layer_sizes.back();

    input = Vector(input_size);
    weight_gradients.resize(layer_sizes.size() - 1);

    layers.reserve(layer_sizes.size() - 1);
    for (size_t i = 1; i < layer_sizes.size(); i++) {
        layers.emplace_back(layer_sizes[i - 1], layer_sizes[i], init_weights[i - 1],
                            layer_activations[i - 1]);
    }
}

Vector MLP::forward(const Vector& input) {
    if (input.size() != input_size) {
        throw std::invalid_argument("Input size does not match MLP input size.");
    }

    this->input = input;

    Vector current_input = input;
    for (size_t i = 0; i < layers.size(); i++) {
        current_input = layers[i].forward(current_input);
    }

    return current_input;  // Final output
}

double MLP::backward(const Vector& target) {
    Layer& output_layer = layers.back();

    double loss = this->loss_function(target, output_layer.get_outputs());

    // Calculate output layer gradient
    Vector output_gradient = compute_output_gradient(target);

    // Update output layer biases
    output_layer.update_biases(output_gradient, learning_rate);

    // Calculate and store output layer weight gradient
    Matrix output_weight_gradient = compute_weight_gradient(
        layers.size() > 1 ? layers[layers.size() - 2].get_outputs() : input, output_gradient);
    weight_gradients[weight_gradients.size() - 1] = output_weight_gradient;

    // Update output layer weights
    output_layer.update_weights(output_weight_gradient, learning_rate);

    // Backpropagate through the rest of the network
    for (size_t i = layers.size() - 1; i-- > 0;) {
        Layer& current_layer = layers[i];

        // Calculate gradient for the current layer
        Vector layer_gradient = compute_layer_gradient(i);

        // Update biases
        current_layer.update_biases(layer_gradient, learning_rate);

        // Calculate and store weight gradient
        Matrix weight_gradient;
        if (i > 0) {
            weight_gradient = compute_weight_gradient(layers[i - 1].get_outputs(), layer_gradient);
        } else {
            weight_gradient = compute_weight_gradient(input, layer_gradient);
        }
        weight_gradients[i] = weight_gradient;

        // Update weights
        current_layer.update_weights(weight_gradient, learning_rate);
    }

    return loss;
}

void MLP::train(const std::vector<Vector>& inputs, const std::vector<Vector>& targets,
                size_t epochs, size_t log_interval) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs and targets must match.");
    }

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            forward(inputs[i]);
            double loss = backward(targets[i]);
            epoch_loss += loss;
        }
        if (log_interval > 0 && (epoch + 1) % log_interval == 0) {
            std::cout << "Epoch " << epoch + 1 << ", Avg Loss: " << epoch_loss / inputs.size()
                      << std::endl;
        }
    }
}

void MLP::validate_init(std::vector<size_t> layer_sizes, std::vector<Activation> layer_activations,
                        Loss loss_function) {
    if (layer_sizes.empty() || layer_sizes.size() < 2) {
        throw std::invalid_argument("MLP must have at least 2 layers (input and output).");
    }
    if (layer_sizes[0] == 0 || layer_sizes.back() == 0) {
        throw std::invalid_argument("Input and output layer sizes must be greater than 0.");
    }
    if (layer_sizes.size() - 1 != layer_activations.size()) {
        throw std::invalid_argument("Number of layers must match number of activations.");
    }
}

Vector MLP::compute_output_gradient(const Vector& target) {
    Layer& output_layer = layers.back();

    Vector output_gradient =
        output_layer.activation_derivative(output_layer.get_pre_activations())
            .elem_mult(this->loss_function.derivative(target, output_layer.get_outputs()));

    output_layer.set_gradient(output_gradient);
    return output_gradient;
}

Matrix MLP::compute_weight_gradient(const Vector& layer_input, const Vector& layer_gradient) {
    return layer_gradient.outer_product(layer_input);
}

Vector MLP::compute_layer_gradient(size_t layer_idx) {
    if (layer_idx >= layers.size() - 1) {
        throw std::out_of_range("Layer index out of range.");
    }

    Layer& current_layer = layers[layer_idx];
    Layer& next_layer = layers[layer_idx + 1];

    Vector layer_gradient =
        (next_layer.get_weights().transpose() * next_layer.get_gradient())
            .elem_mult(current_layer.activation_derivative(current_layer.get_pre_activations()));

    current_layer.set_gradient(layer_gradient);
    return layer_gradient;
}