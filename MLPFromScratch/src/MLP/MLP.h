#pragma once

#include <functional>
#include <iostream>
#include <vector>

#include "../Functions/Activation.h"
#include "../Functions/Loss.h"
#include "../Layer/Layer.h"
#include "../Matrix/Matrix.h"
#include "../Vector/Vector.h"

class MLP {
public:
    /**
     * @brief Constructs a new MLP object.
     * @param layer_sizes A vector containing the sizes of each layer in the network.
     * @throws std::invalid_argument if the number of layers is less than 2.
     */
    MLP(std::vector<size_t> layer_sizes, std::vector<Activation> layer_activations,
        Loss loss_function);

    /**
     * @brief Initializes a new MLP with specified initial Weights.
     * @param layer_sizes A vector containing the sizes of each layer in the network.
     * @param initial_weights A vector of matrices representing the initial weights for each layer.
     * @throws std::invalid_argument if the number of layers is less than 2 or if the sizes of the
     * matrices does not match.
     */
    MLP(std::vector<size_t> layer_sizes, std::vector<Matrix> init_weights,
        std::vector<Activation> layer_activations, Loss loss_function);

    /**
     * @brief Performs a forward pass through the network.
     * @param input The input vector.
     * @return The output vector after passing through the network.
     */
    Vector forward(const Vector& input);

    /**
     * @brief Performs a backward pass through the network to update weights.
     * @param target The target output vector.
     * @returns the total loss
     */
    double backward(const Vector& target);

    /**
     * @brief Trains the network for a specified number of epochs.
     * @param inputs Vector of input samples.
     * @param targets Vector of target outputs corresponding to inputs.
     * @param epochs Number of training epochs.
     * @param log_interval Print loss every log_interval epochs (0 for no logging).
     */
    void train(const std::vector<Vector>& inputs, const std::vector<Vector>& targets, size_t epochs,
               size_t log_interval = 10);

    /**
     * @brief Tests the network on a single input.
     * @param input The input vector.
     * @return The output vector after passing through the network.
     */
    Vector test(const Vector& input) { return forward(input); }

    /**
     * @brief Returns the layers of the network.
     * @return A vector of layers in the network.
     */
    std::vector<Layer> get_layers() const { return layers; }

    /**
     * @brief Sets the learning rate.
     * @param learning_rate The learning rate to use for the network.
     */
    void set_learning_rate(double learning_rate) { this->learning_rate = learning_rate; }

    /**
     * @brief Gets the weight gradients for testing.
     * @return A vector of weight gradient matrices.
     */
    std::vector<Matrix> get_weight_gradients() const { return weight_gradients; }

private:
    Vector compute_output_gradient(const Vector& target);
    Matrix compute_weight_gradient(const Vector& layer_output, const Vector& layer_gradient);
    Vector compute_layer_gradient(size_t layer_idx);
    void validate_init(std::vector<size_t> layer_sizes, std::vector<Activation> layer_activations,
                       Loss loss_function);

    size_t input_size;
    size_t output_size;

    Vector input;
    std::vector<Layer> layers;
    std::vector<Matrix> weight_gradients;  // Store gradients for testing

    Loss loss_function;
    double learning_rate = 0.01;
};