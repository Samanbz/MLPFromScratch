#pragma once

#include <string>

#include "../Vector/Vector.h"
/**
 * Represents a neuron in the multi-layer perceptron.
 */
class Neuron {
public:
    /**
     * Initializes the neuron with random initial weights.
     * @param input_size the number of incoming connections to the neuron
     */
    Neuron(size_t input_size);

    /**
     * Initializes the neuron with initial weights and bias.
     */
    Neuron(const Vector& initial_weights, double initial_bias);

    /**
     * Applies the activation function to the neuron's value.
     * @param z the value to be activated
     * @returns the activated value
     */
    double activate(double z) const;

    /**
     * Applies the derivative of the activation function to the neuron's value.
     * @param z the value to be differentiated
     * @return the value of the derivative at z
     */
    double activation_derivative(double z) const;

    /**
     * Forward pass through the neuron.
     * @param neuron input
     * @returns the output of the forward pass
     */
    double forward(const Vector& input);

    /**
     * @returns the output of the neuron if
     */
    double get_output() const;

    /**
     * @returns the value of z
     */
    double get_z() const;

    /**
     * @returns the weights of the neuron
     */
    Vector get_weights() const;

    /**
     * @returns the bias of the neuron
     */
    double get_bias() const;

    /**
     * @returns a string representation of the Neuron.
     */
    std::string to_string() const;

private:
    Vector weights;
    double bias;
    double output;  // defualts to -1
    double z;       // defaults to -1
};
