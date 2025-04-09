#pragma once

#include <functional>
#include <string>

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
    Neuron(size_t input_size);

    /**
     * Initializes the neuron with initial weights and bias.
     * @param initial_weights The initial weights of the neuron.
     * @param initial_bias The initial bias of the neuron.
     */
    Neuron(const Vector& initial_weights, double initial_bias);

    /**
     * Performs a forward pass through the neuron.
     * @param input The input vector to the neuron.
     * @returns The output of the forward pass.
     */
    double forward(const Vector& input);

    /**
     * Applies the activation function to the neuron's value.
     * @param z The value to be activated.
     * @returns The activated value.
     */
    double activate(double z) const;

    /**
     * Applies the derivative of the activation function to the neuron's value.
     * @param z The value to be differentiated.
     * @returns The value of the derivative at z.
     */
    double activation_derivative(double z) const;

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
     * Updates the weights of the neuron using the provided weight delta and learning rate.
     * @param weight_delta The change to be applied to the weights.
     * @param learning_rate The learning rate to scale the weight update. Defaults to 0.01.
     */
    void update_weights(const Vector& weight_delta, double learning_rate);

    /**
     * Retrieves the bias of the neuron.
     * @returns The bias of the neuron.
     */
    double get_bias() const;

    /**
     * Converts the neuron to a string representation.
     * @returns A string representation of the neuron's state.
     */
    std::string to_string() const;

private:
    Vector weights;  /// The weights of the neuron.
    double bias;     /// The bias of the neuron.
    double output;   /// The output of the neuron after activation. Defaults to -1.
    double z;        /// The pre-activation value of the neuron. Defaults to -1.
};
