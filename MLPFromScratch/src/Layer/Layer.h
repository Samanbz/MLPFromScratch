#pragma once

#include <functional>
#include <stdexcept>
#include <string>

#include "../Functions/Activation.h"
#include "../Matrix/Matrix.h"
#include "../Vector/Vector.h"

/**
 * @brief Represents a layer in a neural network with optimized memory layout.
 */
class Layer {
public:
    /**
     * @brief Constructs a new Layer with random weights.
     *
     * @param input_size The size of the input vector to the layer.
     * @param neuron_count The number of neurons in the layer.
     * @param activation The activation function to be used by the layer.
     */
    Layer(size_t input_size, size_t neuron_count, Activation activation);

    /**
     * @brief Constructs a new Layer with specified initial weights.
     *
     * @param input_size The size of the input vector to the layer.
     * @param neuron_count The number of neurons in the layer.
     * @param init_weights The initial weights of the layer.
     * @param activation The activation function to be used by the layer.
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
     */
    size_t get_size() const;

    /**
     * @brief Returns the size of the input vector to the layer.
     */
    size_t get_input_size() const;

    /**
     * @brief Returns a string representation of the layer.
     */
    std::string to_string() const;

    /**
     * @brief Returns a vector representing the current output values of the layer.
     */
    Vector get_outputs() const;

    /**
     * @brief Returns a vector representing the current pre-activation values of the layer.
     */
    Vector get_pre_activations() const;

    /**
     * @brief Returns the gradient of the layer.
     */
    Vector get_gradient() const;

    /**
     * @brief Sets the gradient values of the layer.
     */
    void set_gradient(const Vector& gradient);

    /**
     * @brief Returns the weights of the layer.
     */
    Matrix get_weights() const;

    /**
     * @brief Updates the weights of the layer based on the provided weight gradient.
     */
    void update_weights(const Matrix& weight_gradient, double learning_rate);

    /**
     * @brief Returns the biases of the layer.
     */
    Vector get_biases() const;

    /**
     * @brief Updates the bias of the layer based on the provided bias gradient.
     */
    void update_biases(const Vector& bias_gradient, double learning_rate);

    /**
     * @brief Evaluates the derivative of the activation function.
     */
    Vector activation_derivative(const Vector& input) const;

private:
    /**
     * @brief Clips gradients to prevent exploding gradients.
     */
    void clip_gradient(const Vector& gradient, double clip_threshold, Vector& result);

    // Layer dimensions
    size_t neuron_count;  // Number of neurons (output size)
    size_t input_size;    // Input size

    // Layer parameters
    Matrix weights;          // Weight matrix (neuron_count × input_size)
    Vector biases;           // Bias vector (neuron_count)
    Vector outputs;          // Output values after activation (neuron_count)
    Vector pre_activations;  // Values before activation (neuron_count)
    Vector gradient;         // Gradient for backpropagation (neuron_count)

    // Activation function
    Activation activation;  // Activation function used by the layer
};