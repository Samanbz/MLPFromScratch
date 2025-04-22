#include "Layer.h"

#include "../Functions/Activation.h"
#include "catch.hpp"

TEST_CASE("Layer Construction", "[Layer]") {
    Layer layer(4, 3, ActivationFunctions::sigmoid);
    REQUIRE(layer.get_size() == 3);
    REQUIRE(layer.get_input_size() == 4);
}

TEST_CASE("Layer Constructionz with initial weights", "[Layer]") {
    Matrix init_weights({{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2}});
    Layer layer(4, 3, init_weights, ActivationFunctions::sigmoid);
    REQUIRE(layer.get_size() == 3);
    REQUIRE(layer.get_input_size() == 4);
}

TEST_CASE("Layer Forward Propogation", "[Layer]") {
    Layer layer(4, 3, ActivationFunctions::sigmoid);
    Vector input({1.0, 2.0, 3.0, 4.0});
    Vector output = layer.forward(input);
    REQUIRE(output.size() == 3);
}