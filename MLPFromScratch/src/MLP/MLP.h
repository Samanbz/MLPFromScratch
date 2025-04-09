#pragma once

#include <functional>
#include <iostream>
#include <vector>

#include "../Layer/Layer.h"
#include "../Vector/Vector.h"

class MLP {
public:
    /**
     * @brief Constructs a new MLP object.
     * @param layer_sizes A vector containing the sizes of each layer in the network.
     * @throws std::invalid_argument if the number of layers is less than 2.
     */
    MLP(std::vector<size_t> layer_sizes);

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

    Vector test(const Vector& input) { return forward(input); }

    /**
     * @brief Returns the layers of the network.
     * @return A vector of layers in the network.
     */
    std::vector<Layer> get_layers() const { return layers; }

    void set_learning_rate(double learning_rate) { this->learning_rate = learning_rate; }

private:
    /**
     * @brief Computes the loss between the network output and the target output.
     * @param output The output vector from the network.
     * @param target The target output vector.
     * @return The computed loss value.
     */
    double loss(const Vector& output, const Vector& target);

    /**
     * @brief Computes the derivative of the loss with respect to the output.
     * @param output The output vector from the network.
     * @param target The target output vector.
     * @return The vector of loss derivatives.
     */
    Vector loss_derivative(const Vector& output, const Vector& target);

    Vector compute_output_delta(const Vector& target);

    Matrix compute_weight_delta(const Vector& layer_output, const Vector& layer_delta);

    Vector compute_layer_delta(size_t layer_idx);

    size_t input_size;
    size_t output_size;

    Vector input;
    std::vector<Layer> layers;

    double learning_rate = 0.01;
};
