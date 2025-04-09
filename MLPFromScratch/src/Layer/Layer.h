#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../Matrix/Matrix.h"
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

    /**
     * @brief Returns the neurons in the layer.
     *
     * @return std::vector<Neuron> The neurons in the layer.
     */
    std::vector<Neuron> get_neurons() const;

    /**
     * @brief Returns a vector representing the current output values of the layer.
     *
     * @return Vector The current output values of the layer.
     */
    Vector get_outputs() const;

    /**
     * @brief Returns a vector representing the current pre-activation values of the layer.
     *
     * @return Vector The current pre-activation values of the layer.
     */
    Vector get_pre_activations() const;

    /**
     * @brief Returns the delta values of the layer.
     *
     * The delta values represent the error signal for each neuron in the layer,
     * which is used during the backpropagation process to update weights.
     *
     * @return Vector The delta values of the layer.
     */
    Vector get_delta() const;

    /**
     * @brief Sets the delta values of the layer.
     *
     * This method allows updating the delta values of the layer, which are
     * used during the backpropagation process to adjust the weights of the neurons.
     *
     * @param delta The new delta values to set for the layer.
     */
    void set_delta(const Vector& delta);

    /**
     * @brief Returns the weights of the layer.
     *
     * The weights are represented as a matrix, where each row corresponds to a neuron
     * and each column corresponds to an input feature.
     *
     * @return Matrix The weights of the layer.
     */
    Matrix get_weights() const;

    /**
     * @brief Updates the weights of the layer based on the provided weight delta.
     *
     * This method applies the weight delta to the weights of the neurons in the layer.
     *
     * @param weight_delta The weight delta to apply to the layer's weights.
     * @param learning_rate The learning rate to use for the weight update.
     */
    void update_weights(const Matrix& weight_delta, double learning_rate);

private:
    size_t neuron_count;
    size_t input_size;
    std::vector<Neuron> neurons;

    Vector delta;
};
