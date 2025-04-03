#include "Neuron.h"

#include "catch.hpp"

TEST_CASE("Neuron Construction", "[Neuron]") {
    Neuron n1(3);
    REQUIRE(n1.get_weights().size() == 3);
    REQUIRE(n1.get_bias() == 0);
    REQUIRE(n1.get_output() == -1);
    REQUIRE(n1.get_z() == -1);

    Vector initial_weights({0.5, 0.5, 0.5});
    Neuron n2(initial_weights, 0.5);
    REQUIRE(n2.get_weights().size() == 3);
    REQUIRE(n2.get_bias() == 0.5);
    REQUIRE(n2.get_output() == -1);
    REQUIRE(n2.get_z() == -1);
}

TEST_CASE("Neuron Activation", "[Neuron]") {
    Neuron n(3);
    double z = 0.5;
    double activated_value = n.activate(z);
    REQUIRE(activated_value > 0);
    REQUIRE(activated_value < 1);

    double derivative_value = n.activation_derivative(z);
    REQUIRE(derivative_value > 0);
}

TEST_CASE("Neuron Forward Pass - check output", "[Neuron]") {
    Neuron n(3);
    Vector input({1.0, 2.0, 3.0});
    double output = n.forward(input);
    double z = input.dot(n.get_weights()) + n.get_bias();
    double expected_output = n.activate(z);
    REQUIRE(output == Approx(expected_output).epsilon(0.01));
}

TEST_CASE("Neuron Forward Pass - check bounds", "[Neuron]") {
    Neuron n(3);
    Vector input({1.0, 2.0, 3.0});
    double output = n.forward(input);
    REQUIRE(output > 0);
    REQUIRE(output < 1);
}

TEST_CASE("Neuron Invalid Forward Pass", "[Neuron]") {
    Neuron n(3);
    Vector input({1.0, 2.0, 3.0, 4.0});
    REQUIRE_THROWS_AS(n.forward(input), std::invalid_argument);
}

TEST_CASE("Neuron Getters", "[Neuron]") {
    Neuron n(3);
    Vector input({1.0, 2.0, 3.0});
    n.forward(input);
    REQUIRE(n.get_output() > 0);
    REQUIRE(n.get_z() > 0);
    REQUIRE(n.get_weights().size() == 3);
    REQUIRE(n.get_bias() == 0);
}

TEST_CASE("Neuron to_string", "[Neuron]") {
    Neuron n(3);
    Vector input({1.0, 2.0, 3.0});
    n.forward(input);
    std::string str = n.to_string();
    REQUIRE(str.find("Neuron:") != std::string::npos);
    REQUIRE(str.find("Weights:") != std::string::npos);
    REQUIRE(str.find("Bias:") != std::string::npos);
    REQUIRE(str.find("Output:") != std::string::npos);
}