#include "Layer.h"

#include "catch.hpp"

TEST_CASE("Layer Construction", "[Layer]") {
    Layer layer(4, 3);
    REQUIRE(layer.get_size() == 3);
    REQUIRE(layer.get_input_size() == 4);
}

TEST_CASE("Layer Forward Propogation", "[Layer]") {
    Layer layer(4, 3);
    Vector input({1.0, 2.0, 3.0, 4.0});
    Vector output = layer.forward(input);
    REQUIRE(output.size() == 3);
}