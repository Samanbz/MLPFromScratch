#pragma once

#include <functional>
#include <string>

#include "../Functions/Activation.h"
#include "../Vector/Vector.h"

/**
 * Represents a neuron in the multi-layer perceptron.
 */
class Neuron {
public:
    /**
     * Initializes the neuron with random initial weights.
     * @param input_size The number of incoming connections to the neuron.
     */
    Neuron(size_t input_size, Activation activation);

    /**
     * Initializes the neuron with initial weights and bias.
     * @param initial_weights The initial weights of the neuron.
     * @param initial_bias The initial bias of the neuron.
     */
    Neuron(const Vector& initial_weights, double initial_bias, Activation activation);

    /**
     * Performs a forward pass through the neuron.
     * @param input The input vector to the neuron.
     * @returns The output of the forward pass.
     */
    double forward(const Vector& input);

    /**
     * Retrieves the output of the neuron after the forward pass.
     * @returns The output of the neuron.
     */
    double get_output() const;

    /**
     * Retrieves the pre-activation value (z) of the neuron.
     * @returns The value of z before activation.
     */
    double get_pre_activation() const;

    /**
     * Retrieves the weights of the neuron.
     * @returns A vector containing the weights of the neuron.
     */
    Vector get_weights() const;

    /**
     * Updates the weights of the neuron using the provided weight gradient and learning rate.
     * @param gradient The gradient to update the weights.
     * @param learning_rate The learning rate to scale the weight update. Defaults to 0.01.
     */
    void update_weights(const Vector& gradient, double learning_rate);

    /**
     * Retrieves the bias of the neuron.
     * @returns The bias of the neuron.
     */
    double get_bias() const;

    /**
     * .
     */
    Vector get_weight_gradient() const;

    /**
     * Converts the neuron to a string representation.
     * @returns A string representation of the neuron's state.
     */
    std::string to_string() const;

private:
    Vector clip_gradient(const Vector& gradient, double clip_threshold);

    Vector weights;         /// The weights of the neuron.
    double bias;            /// The bias of the neuron.
    double output;          /// The output of the neuron after activation. Defaults to -1.
    double z;               /// The pre-activation value of the neuron. Defaults to -1.
    Activation activation;  /// The activation function of the neuron.

    Vector weight_gradient;  /// The gradient of the weights.
};
