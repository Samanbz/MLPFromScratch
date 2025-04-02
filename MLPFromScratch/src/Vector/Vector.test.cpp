#include "Vector.h"

#include "catch.hpp"

TEST_CASE("Vector Construction", "[Vector]") {
    Vector v1(3);
    REQUIRE(v1.size() == 3);
    REQUIRE(v1[0] == 0);
    REQUIRE(v1[1] == 0);
    REQUIRE(v1[2] == 0);

    Vector v2(3, 1);
    REQUIRE(v2.size() == 3);
    REQUIRE(v2[0] == 1);
    REQUIRE(v2[1] == 1);
    REQUIRE(v2[2] == 1);

    double values[3] = {1, 2, 3};
    Vector v3(3, values);
    REQUIRE(v3.size() == 3);
    REQUIRE(v3[0] == 1);
    REQUIRE(v3[1] == 2);
    REQUIRE(v3[2] == 3);

    std::vector<double> values2 = {1, 2, 3};
    Vector v4(3, values2);
    REQUIRE(v4.size() == 3);
    REQUIRE(v4[0] == 1);
    REQUIRE(v4[1] == 2);
    REQUIRE(v4[2] == 3);
}

TEST_CASE("Vector Dot Product", "[Vector]") {
    Vector v1(3, 1);
    Vector v2(3, 2);
    REQUIRE(v1.dot(v2) == 6);
}

TEST_CASE("Vector Addition", "[Vector]") {
    Vector v1(3, 1);
    Vector v2(3, 2);
    Vector v3 = v1 + v2;
    REQUIRE(v3.size() == 3);
    REQUIRE(v3[0] == 3);
    REQUIRE(v3[1] == 3);
    REQUIRE(v3[2] == 3);
}

TEST_CASE("Vector Subtraction", "[Vector]") {
    Vector v1(3, 1);
    Vector v2(3, 2);
    Vector v3 = v1 - v2;
    REQUIRE(v3.size() == 3);
    REQUIRE(v3[0] == -1);
    REQUIRE(v3[1] == -1);
    REQUIRE(v3[2] == -1);
}

TEST_CASE("Vector Scalar Multiplication", "[Vector]") {
    Vector v1(3, 1);
    Vector v2 = v1 * 2;
    REQUIRE(v2.size() == 3);
    REQUIRE(v2[0] == 2);
    REQUIRE(v2[1] == 2);
    REQUIRE(v2[2] == 2);
}

TEST_CASE("Vector to_string", "[Vector]") {
    Vector v1(3, 1);
    REQUIRE(v1.to_string() == "[1.00, 1.00, 1.00]");
}
