#include "Matrix.h"

#include "catch.hpp"

TEST_CASE("Matrix Construction", "[Matrix]") {
    Matrix m1(2, 3);
    REQUIRE(m1.rows() == 2);
    REQUIRE(m1.cols() == 3);
    REQUIRE(m1[0].size() == 3);
    REQUIRE(m1[1].size() == 3);
    REQUIRE(m1[0][0] == 0);
    REQUIRE(m1[0][1] == 0);
    REQUIRE(m1[0][2] == 0);
    REQUIRE(m1[1][0] == 0);
    REQUIRE(m1[1][1] == 0);
    REQUIRE(m1[1][2] == 0);

    Matrix m2(2, 3, 1);
    REQUIRE(m2.rows() == 2);
    REQUIRE(m2.cols() == 3);
    REQUIRE(m2[0].size() == 3);
    REQUIRE(m2[1].size() == 3);
    REQUIRE(m2[0][0] == 1);
    REQUIRE(m2[0][1] == 1);
    REQUIRE(m2[0][2] == 1);
    REQUIRE(m2[1][0] == 1);
    REQUIRE(m2[1][1] == 1);
    REQUIRE(m2[1][2] == 1);

    std::vector<std::vector<double>> values = {{1, 2, 3}, {4, 5, 6}};
    Matrix m3(values);
    REQUIRE(m3.rows() == 2);
    REQUIRE(m3.cols() == 3);
    REQUIRE(m3[0].size() == 3);
    REQUIRE(m3[1].size() == 3);
    REQUIRE(m3[0][0] == 1);
    REQUIRE(m3[0][1] == 2);
    REQUIRE(m3[0][2] == 3);
    REQUIRE(m3[1][0] == 4);
    REQUIRE(m3[1][1] == 5);
    REQUIRE(m3[1][2] == 6);
}

TEST_CASE("Matrix Addition", "[Matrix]") {
    Matrix m1({{1, 2, 3}, {4, 5, 6}});
    Matrix m2({{1, 2, 3}, {4, 5, 6}});
    Matrix m3 = m1 + m2;
    REQUIRE(m3.rows() == 2);
    REQUIRE(m3.cols() == 3);
    REQUIRE(m3[0][0] == 2);
    REQUIRE(m3[0][1] == 4);
    REQUIRE(m3[0][2] == 6);
    REQUIRE(m3[1][0] == 8);
    REQUIRE(m3[1][1] == 10);
    REQUIRE(m3[1][2] == 12);
}

TEST_CASE("Matrix Invalid Addition", "[Matrix]") {
    Matrix m1({{1, 2, 3}, {4, 5, 6}});
    Matrix m2({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    REQUIRE_THROWS_AS(m1 + m2, std::invalid_argument);
}

TEST_CASE("Matrix Subtraction", "[Matrix]") {
    Matrix m1({{1, 2, 3}, {4, 5, 6}});
    Matrix m2({{1, 2, 3}, {4, 5, 6}});
    Matrix m3 = m1 - m2;
    REQUIRE(m3.rows() == 2);
    REQUIRE(m3.cols() == 3);
    REQUIRE(m3[0][0] == 0);
    REQUIRE(m3[0][1] == 0);
    REQUIRE(m3[0][2] == 0);
    REQUIRE(m3[1][0] == 0);
    REQUIRE(m3[1][1] == 0);
    REQUIRE(m3[1][2] == 0);
}

TEST_CASE("Matrix Invalid Subtraction", "[Matrix]") {
    Matrix m1({{1, 2, 3}, {4, 5, 6}});
    Matrix m2({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    REQUIRE_THROWS_AS(m1 - m2, std::invalid_argument);
}

TEST_CASE("Matrix Multiplication", "[Matrix]") {
    Matrix m1({{1, 2, 3}, {4, 5, 6}});
    Matrix m2({{1, 2}, {3, 4}, {5, 6}});
    Matrix m3 = m1 * m2;
    REQUIRE(m3.rows() == 2);
    REQUIRE(m3.cols() == 2);
    REQUIRE(m3[0][0] == 22);
    REQUIRE(m3[0][1] == 28);
    REQUIRE(m3[1][0] == 49);
    REQUIRE(m3[1][1] == 64);
}

TEST_CASE("Matrix Invalid Multiplication", "[Matrix]") {
    Matrix m1({{1, 2, 3}, {4, 5, 6}});
    Matrix m2({{1, 2, 3}, {4, 5, 6}});
    REQUIRE_THROWS_AS(m1 * m2, std::invalid_argument);
}

TEST_CASE("Matrix Vector Multiplication", "[Matrix]") {
    Matrix m1({{1, 2, 3}, {4, 5, 6}});
    Vector v1({1, 2, 3});
    Vector v2 = m1 * v1;
    REQUIRE(v2.size() == 2);
    REQUIRE(v2[0] == 14);
    REQUIRE(v2[1] == 32);
}

TEST_CASE("Matrix Scalar Multiplication", "[Matrix]") {
    Matrix m1({{1, 2, 3}, {4, 5, 6}});
    Matrix m2 = m1 * 2;
    REQUIRE(m2.rows() == 2);
    REQUIRE(m2.cols() == 3);
    REQUIRE(m2[0][0] == 2);
    REQUIRE(m2[0][1] == 4);
    REQUIRE(m2[0][2] == 6);
    REQUIRE(m2[1][0] == 8);
    REQUIRE(m2[1][1] == 10);
    REQUIRE(m2[1][2] == 12);
}

TEST_CASE("Matrix Transpose", "[Matrix]") {
    Matrix m1({{1, 2, 3}, {4, 5, 6}});
    Matrix m2 = m1.transpose();
    REQUIRE(m2.rows() == 3);
    REQUIRE(m2.cols() == 2);
    REQUIRE(m2[0][0] == 1);
    REQUIRE(m2[0][1] == 4);
    REQUIRE(m2[1][0] == 2);
    REQUIRE(m2[1][1] == 5);
    REQUIRE(m2[2][0] == 3);
    REQUIRE(m2[2][1] == 6);
}

TEST_CASE("Matrix to_string", "[Matrix]") {
    Matrix m1({{1, 2, 3}, {4, 5, 6}});
    std::string expected = "[\n\t[1.00, 2.00, 3.00],\n\t[4.00, 5.00, 6.00]\n]";
    REQUIRE(m1.to_string() == expected);
}
