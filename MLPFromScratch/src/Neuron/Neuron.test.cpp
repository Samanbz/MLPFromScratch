#include "Neuron.h"

#include "../Functions/Activation.h"
#include "catch.hpp"

using namespace ActivationFunctions;

TEST_CASE("Neuron Construction", "[Neuron]") {
    Neuron n1(3, sigmoid);
    REQUIRE(n1.get_weights().size() == 3);
    REQUIRE(n1.get_bias() == 0);
    REQUIRE(n1.get_output() == 0);
    REQUIRE(n1.get_pre_activation() == 0);

    Vector initial_weights({0.5, 0.5, 0.5});
    Neuron n2(initial_weights, 0.5, sigmoid);
    REQUIRE(n2.get_weights().size() == 3);
    REQUIRE(n2.get_bias() == 0.5);
    REQUIRE(n2.get_output() == 0);
    REQUIRE(n2.get_pre_activation() == 0);
}

TEST_CASE("Neuron Forward Pass - check output", "[Neuron]") {
    Neuron n(3, sigmoid);
    Vector input({1.0, 2.0, 3.0});
    double output = n.forward(input);
    double z = input.dot(n.get_weights()) + n.get_bias();
    double expected_output = sigmoid(z);
    REQUIRE(output == Approx(expected_output).epsilon(0.01));
}

TEST_CASE("Neuron Forward Pass - check bounds", "[Neuron]") {
    Neuron n(3, sigmoid);
    Vector input({1.0, 2.0, 3.0});
    double output = n.forward(input);
    REQUIRE(output > 0);
    REQUIRE(output < 1);
}

TEST_CASE("Neuron Invalid Forward Pass", "[Neuron]") {
    Neuron n(3, sigmoid);
    Vector input({1.0, 2.0, 3.0, 4.0});
    REQUIRE_THROWS_AS(n.forward(input), std::invalid_argument);
}

TEST_CASE("Neuron Getters", "[Neuron]") {
    Neuron n(3, sigmoid);
    Vector input({1.0, 2.0, 3.0});
    n.forward(input);
    REQUIRE(n.get_output() > 0);
    REQUIRE(n.get_pre_activation() > 0);
    REQUIRE(n.get_weights().size() == 3);
    REQUIRE(n.get_bias() == 0);
}

TEST_CASE("Neuron to_string", "[Neuron]") {
    Neuron n(3, sigmoid);
    Vector input({1.0, 2.0, 3.0});
    n.forward(input);
    std::string str = n.to_string();
    REQUIRE(str.find("Neuron:") != std::string::npos);
    REQUIRE(str.find("Weights:") != std::string::npos);
    REQUIRE(str.find("Bias:") != std::string::npos);
    REQUIRE(str.find("Output:") != std::string::npos);
}