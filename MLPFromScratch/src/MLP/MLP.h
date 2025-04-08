#pragma once

#include <functional>
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
     */
    void backward(const Vector& target);

    /**
     * @brief Returns the layers of the network.
     * @return A vector of layers in the network.
     */
    std::vector<Layer> get_layers() const;

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

    size_t input_size;
    size_t output_size;

    Vector input;
    std::vector<Layer> layers;
};
