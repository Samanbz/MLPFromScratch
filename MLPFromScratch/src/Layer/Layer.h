#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../Functions/Activation.h"
#include "../Matrix/Matrix.h"
#include "../Neuron/Neuron.h"
#include "../Vector/Vector.cuh"

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
     * @param activation The activation function to be used by the neurons in the layer.
     */
    Layer(size_t input_size, size_t neuron_count, Activation activation);

    /**
     * @brief Constructs a new Layer object with specified initial weights.
     *
     * @param input_size The size of the input vector to the layer.
     * @param neuron_count The number of neurons in the layer.
     * @param init_weights The initial weights of the neurons in the layer.
     * @param activation The activation function to be used by the neurons in the layer.
     */
    Layer(size_t input_size, size_t neuron_count, const Matrix& init_weights,
          Activation activation);

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
     * @brief Returns the gradient of the layer.
     *
     * The gradient values represent the error for each neuron in the layer,
     * which is used during the backpropagation process to update weights.
     *
     * @return Vector The gradient of the layer.
     */
    Vector get_gradient() const;

    /**
     * @brief Sets the gradient values of the layer.
     *
     * This method allows updating the gradient values of the layer, which are
     * used during the backpropagation process to adjust the weights of the neurons.
     *
     * @param gradient The new gradient values to set for the layer.
     */
    void set_gradient(const Vector& gradient);

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
     * @brief Updates the weights of the layer based on the provided weight gradient.
     *
     * This method applies the weight gradient to the weights of the neurons in the layer.
     *
     * @param weight_gradient The gradient to apply to the layer's weights.
     * @param learning_rate The learning rate to use for the weight update.
     */
    void update_weights(const Matrix& weight_gradient, double learning_rate);

    /**
     * @brief Returns the biases of the layer.
     *
     * @return Vector The biases of the layer.
     */
    Vector get_biases() const;
    /**
     * @brief Updates the bias of the layer based on the provided bias gradient.
     *
     * This method applies the bias gradient to the biases of the neurons in the layer.
     *
     * @param bias_gradient The gradient to apply to the layer's biases.
     * @param learning_rate The learning rate to use for the bias update.
     */
    void update_biases(const Vector& bias_gradient, double learning_rate);

    /**
     * @brief evaluates the derivative of the activation function. Used in backprop.
     *
     * @param input The input vector to the layer.
     * @return Vector The derivative of the activation function applied to the input.
     */
    Vector activation_derivative(const Vector& input) const;

private:
    size_t neuron_count;          /// The number of neurons in the layer.
    size_t input_size;            /// The size of the input vector to the layer.
    std::vector<Neuron> neurons;  /// The neurons in the layer.
    Activation activation;        /// The activation function used by the neurons in the layer.

    Vector gradient;  /// The gradient values of the layer, used for backpropagation.
};
