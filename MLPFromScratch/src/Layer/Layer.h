#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "../Neuron/Neuron.h"
#include "../Vector/Vector.h"

/**
 * @brief Represents a layer of neurons in a neural network.
 *
 * This class is responsible for managing the neurons in the layer and performing
 * forward passes through the layer.
 */
class Layer {
public:
    /**
     * @brief Constructs a new Layer object.
     *
     * @param input_size The size of the input vector to the layer.
     * @param neuron_count The number of neurons in the layer.
     */
    Layer(size_t input_size, size_t neuron_count);

    /**
     * @brief Performs a forward pass through the layer.
     *
     * @param input The input vector to the layer.
     * @return Vector The output vector from the layer.
     */
    Vector forward(const Vector& input);

    /**
     * @brief Returns the number of neurons in the layer.
     *
     * @return size_t The number of neurons in the layer.
     */
    size_t get_size() const;

    /**
     * @brief Returns the size of the input vector to the layer.
     *
     * @return size_t The size of the input vector to the layer.
     */
    size_t get_input_size() const;

    /**
     * @brief Returns a string representation of the layer.
     *
     * @return std::string A string representation of the layer.
     */
    std::string to_string() const;

private:
    size_t neuron_count;
    size_t input_size;
    std::vector<Neuron> neurons;
};
