#include "MLP.h"

#include "../Functions/Activation.h"
#include "../Functions/Loss.h"
#include "catch.hpp"

using namespace ActivationFunctions;
using namespace LossFunctions;

TEST_CASE("MLP construction", "[MLP]") {
    // Test valid construction
    MLP mlp({3, 5, 2}, {sigmoid, sigmoid}, mse);
    REQUIRE(mlp.get_layers().size() == 2);  // 2 layers (hidden and output)

    // Test invalid construction (less than 2 layers)
    REQUIRE_THROWS(MLP({3}, {sigmoid, sigmoid}, mse));
    REQUIRE_THROWS(MLP({}, {sigmoid, sigmoid}, mse));
}

TEST_CASE("MLP construction with initial weights", "[MLP]") {
    // Create initial weights for the layers
    Matrix init_weights1({{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}});
    std::vector<Matrix> init_weights = {init_weights1};
    REQUIRE(init_weights1.rows() == 2);

    MLP mlp({3, 2}, init_weights, {sigmoid}, mse);
    REQUIRE(mlp.get_layers().size() == 1);  // 1 layer (output only)
    REQUIRE(mlp.get_layers()[0].get_weights().rows() == 2);
    REQUIRE(mlp.get_layers()[0].get_weights().cols() == 3);
}

TEST_CASE("MLP construction with invalid initial_weights", "[MLP]") {
    // Create initial weights for the layers
    Matrix init_weights1({{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}});
    std::vector<Matrix> init_weights = {init_weights1};

    // Test invalid construction (less than 2 layers)
    REQUIRE_THROWS(MLP({3}, init_weights, {sigmoid}, mse));
    REQUIRE_THROWS(MLP({}, init_weights, {sigmoid}, mse));

    // Test invalid construction (number of layers does not match number of initial weights)
    std::vector<Matrix> init_weights_invalid = {init_weights1, init_weights1};
    REQUIRE_THROWS(MLP({3, 2}, init_weights_invalid, {sigmoid}, mse));

    // Test invalid construction (initial weights size does not match layer size)
    Matrix init_weights_invalid1({{0.1, 0.2}, {0.4, 0.5}});
    std::vector<Matrix> init_weights_invalid_size = {init_weights_invalid1};
    REQUIRE_THROWS(MLP({3, 2}, init_weights_invalid_size, {sigmoid}, mse));
}

TEST_CASE("MLP forward pass with invalid input size", "[MLP]") {
    MLP mlp({3, 2}, {sigmoid}, mse);
    Vector input(std::vector({1.0, 2.0}));  // Invalid size
    REQUIRE_THROWS(mlp.forward(input));
}

TEST_CASE("MLP forward pass", "[MLP]") {
    // Create a simple MLP with 2 layers: input layer with 3 neurons and output layer with 2 neurons
    MLP mlp({3, 2}, {sigmoid}, mse);

    // Create a sample input vector
    Vector input({1.0, 2.0, 3.0});

    // Perform a forward pass
    Vector output = mlp.forward(input);

    // Check the output size
    REQUIRE(output.size() == 2);

    // Check the output values
    REQUIRE(output[0] != 0);
    REQUIRE(output[1] != 0);
}

TEST_CASE("MLP end to end", "[MLP]") {
    using Catch::Matchers::WithinAbs;

    std::vector<Matrix> init_weights = {Matrix({{0.1}, {0.3}}), Matrix({{0.4, 0.2}})};

    // Create a simple MLP with 3 layers: input(1), hidden(2), and output(1)
    MLP mlp({1, 2, 1}, init_weights, {sigmoid, sigmoid}, mse);

    std::vector<Layer> layers = mlp.get_layers();

    // Check weights directly from the weights matrix
    REQUIRE(layers[0].get_weights()[0][0] == 0.1);
    REQUIRE(layers[0].get_weights()[1][0] == 0.3);
    REQUIRE(layers[1].get_weights()[0][0] == 0.4);
    REQUIRE(layers[1].get_weights()[0][1] == 0.2);

    // Create a sample input vector
    Vector input({1.0});

    Vector output = mlp.forward(input);

    layers = mlp.get_layers();

    // Check pre-activations
    REQUIRE(layers[0].get_pre_activations()[0] == 0.1);
    REQUIRE(layers[0].get_pre_activations()[1] == 0.3);

    // Check layer outputs
    REQUIRE_THAT(layers[0].get_outputs()[0], WithinAbs(0.5249, 1e-4));
    REQUIRE_THAT(layers[0].get_outputs()[1], WithinAbs(0.5744, 1e-4));

    REQUIRE_THAT(layers[1].get_pre_activations()[0], WithinAbs(0.3248, 1e-4));
    REQUIRE_THAT(layers[1].get_outputs()[0], WithinAbs(0.5805, 1e-4));
    REQUIRE_THAT(output[0], WithinAbs(0.5805, 1e-4));

    // Create a sample target vector
    Vector target({0.5});

    // Perform a backward pass
    double loss = mlp.backward(target);

    layers = mlp.get_layers();
    std::vector<Matrix> weight_gradients = mlp.get_weight_gradients();

    REQUIRE_THAT(loss, WithinAbs(0.0064, 1e-4));
    REQUIRE_THAT(layers.back().get_gradient()[0], WithinAbs(0.0392, 1e-4));

    // Check weight gradients
    REQUIRE_THAT(weight_gradients[1][0][0], WithinAbs(0.0205, 1e-4));
    REQUIRE_THAT(weight_gradients[1][0][1], WithinAbs(0.0225, 1e-4));

    REQUIRE_THAT(layers[0].get_gradient()[0], WithinAbs(0.0039, 1e-4));
    REQUIRE_THAT(layers[0].get_gradient()[1], WithinAbs(0.0019, 1e-4));
    REQUIRE_THAT(weight_gradients[0][0][0], WithinAbs(0.0039, 1e-4));
    REQUIRE_THAT(weight_gradients[0][1][0], WithinAbs(0.0019, 1e-4));
}