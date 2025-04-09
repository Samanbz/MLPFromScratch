#include "Vector.h"

#include "../Matrix/Matrix.h"
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
    Vector v4(values2);
    REQUIRE(v4[0] == 1);
    REQUIRE(v4[1] == 2);
    REQUIRE(v4[2] == 3);
}

TEST_CASE("Vector Dot Product", "[Vector]") {
    Vector v1(3, 1);
    Vector v2(3, 2);
    REQUIRE(v1.dot(v2) == 6);
}

TEST_CASE("Vector Invalid Dot Product, ", "[Vector]") {
    Vector v1(3, 1);
    Vector v2(4, 2);
    REQUIRE_THROWS_AS(v1.dot(v2), std::invalid_argument);
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

TEST_CASE("Vector Element-wise Multiplication", "[Vector]") {
    Vector v1(3, 1);
    Vector v2(3, 2);
    Vector v3 = v1.elem_mult(v2);
    REQUIRE(v3.size() == 3);
    REQUIRE(v3[0] == 2);
    REQUIRE(v3[1] == 2);
    REQUIRE(v3[2] == 2);
}

TEST_CASE("Vector Element-wise Multiplication Invalid Size", "[Vector]") {
    Vector v1(3, 1);
    Vector v2(4, 2);
    REQUIRE_THROWS_AS(v1.elem_mult(v2), std::invalid_argument);
}

TEST_CASE("Vector sum", "[Vector]") {
    Vector v1(3, 1);
    double sum = v1.sum();
    REQUIRE(sum == 3);
}

TEST_CASE("Vector outer product", "[Vector]") {
    Vector v1(3, 1);
    Vector v2(3, 2);
    Matrix m = v1.outer_product(v2);

    Matrix expected(3, 3, 2);

    REQUIRE(m == expected);
}

TEST_CASE("Vector apply", "[Vector]") {
    Vector v1(3, 1);
    Vector v2 = v1.apply([](double x) { return x * 2; });
    REQUIRE(v2.size() == 3);
    REQUIRE(v2[0] == 2);
    REQUIRE(v2[1] == 2);
    REQUIRE(v2[2] == 2);
}

TEST_CASE("Vector to_string", "[Vector]") {
    Vector v1(3, 1);
    REQUIRE(v1.to_string() == "[1.00, 1.00, 1.00]");
}