#include "MLP.h"

#include "catch.hpp"

TEST_CASE("MLP construction", "[MLP]") {
    // Test valid construction
    MLP mlp({3, 5, 2});
    REQUIRE(mlp.get_layers().size() == 2);  // 2 layers (hidden and output)

    // Test invalid construction (less than 2 layers)
    REQUIRE_THROWS(MLP({3}));
    REQUIRE_THROWS(MLP({}));
}

TEST_CASE("MLP forward pass", "[MLP]") {
    // Create a simple MLP with 2 layers: input layer with 3 neurons and output layer with 2 neurons
    MLP mlp({3, 2});

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
