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

TEST_CASE("Matrix flatten", "[Matrix]") {
    // Create test matrices
    Matrix empty;
    Matrix single_element(1, 1, 5.0);
    Matrix identity({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
    Matrix rectangular({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});

    SECTION("Empty matrix") {
        double* flattened = new double[0];
        empty.flatten(flattened);
        delete[] flattened;
        // No assertions needed; just testing that it doesn't crash
    }

    SECTION("Single element matrix") {
        double* flattened = new double[1];
        single_element.flatten(flattened);
        REQUIRE(flattened[0] == 5.0);
        delete[] flattened;
    }

    SECTION("Square matrix") {
        double* flattened = new double[9];
        identity.flatten(flattened);
        REQUIRE(flattened[0] == 1.0);
        REQUIRE(flattened[1] == 0.0);
        REQUIRE(flattened[2] == 0.0);
        REQUIRE(flattened[3] == 0.0);
        REQUIRE(flattened[4] == 1.0);
        REQUIRE(flattened[5] == 0.0);
        REQUIRE(flattened[6] == 0.0);
        REQUIRE(flattened[7] == 0.0);
        REQUIRE(flattened[8] == 1.0);
        delete[] flattened;
    }

    SECTION("Rectangular matrix") {
        double* flattened = new double[6];
        rectangular.flatten(flattened);
        REQUIRE(flattened[0] == 1.0);
        REQUIRE(flattened[1] == 2.0);
        REQUIRE(flattened[2] == 3.0);
        REQUIRE(flattened[3] == 4.0);
        REQUIRE(flattened[4] == 5.0);
        REQUIRE(flattened[5] == 6.0);
        delete[] flattened;
    }

    SECTION("Verify original matrix unchanged") {
        double* flattened = new double[6];
        rectangular.flatten(flattened);
        REQUIRE(rectangular[0][0] == 1.0);
        REQUIRE(rectangular[0][1] == 2.0);
        REQUIRE(rectangular[0][2] == 3.0);
        REQUIRE(rectangular[1][0] == 4.0);
        REQUIRE(rectangular[1][1] == 5.0);
        REQUIRE(rectangular[1][2] == 6.0);
        delete[] flattened;
    }

    SECTION("Modify flattened array doesn't affect original") {
        double* flattened = new double[6];
        rectangular.flatten(flattened);
        flattened[0] = 99.0;
        REQUIRE(rectangular[0][0] == 1.0);
        delete[] flattened;
    }
}

TEST_CASE("Matrix flatten and from flatten construction", "[Matrix]") {
    SECTION("Empty matrix") {
        Matrix empty(0, 0);
        double* flattened = new double[0];
        empty.flatten(flattened);

        Matrix reconstructed(0, 0, flattened);
        REQUIRE(reconstructed.rows() == 0);
        REQUIRE(reconstructed.cols() == 0);

        delete[] flattened;
    }

    SECTION("Single element matrix") {
        Matrix original(1, 1, 5.0);
        double* flattened = new double[1];
        original.flatten(flattened);

        Matrix reconstructed(1, 1, flattened);
        REQUIRE(reconstructed.rows() == 1);
        REQUIRE(reconstructed.cols() == 1);
        REQUIRE(reconstructed[0][0] == 5.0);

        delete[] flattened;
    }

    SECTION("Square matrix") {
        Matrix original({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
        double* flattened = new double[9];
        original.flatten(flattened);

        Matrix reconstructed(3, 3, flattened);
        REQUIRE(reconstructed.rows() == 3);
        REQUIRE(reconstructed.cols() == 3);

        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                REQUIRE(reconstructed[i][j] == original[i][j]);
            }
        }

        delete[] flattened;
    }

    SECTION("Rectangular matrix") {
        Matrix original({{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}});
        double* flattened = new double[8];
        original.flatten(flattened);

        Matrix reconstructed(2, 4, flattened);
        REQUIRE(reconstructed.rows() == 2);
        REQUIRE(reconstructed.cols() == 4);

        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 4; j++) {
                REQUIRE(reconstructed[i][j] == original[i][j]);
            }
        }

        delete[] flattened;
    }

    SECTION("Round-trip equality") {
        // Create matrices of different sizes and verify round-trip equality
        std::vector<Matrix> test_matrices = {
            Matrix(1, 1, 3.14), Matrix({{1.0, 2.0}, {3.0, 4.0}}),
            Matrix({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}), Matrix(3, 4, 1.5)};

        for (const auto& original : test_matrices) {
            size_t size = original.rows() * original.cols();
            double* flattened = new double[size];
            original.flatten(flattened);

            Matrix reconstructed(original.rows(), original.cols(), flattened);
            REQUIRE(reconstructed == original);

            delete[] flattened;
        }
    }

    SECTION("Modify reconstructed matrix") {
        Matrix original({{1.0, 2.0}, {3.0, 4.0}});
        double* flattened = new double[4];
        original.flatten(flattened);

        Matrix reconstructed(2, 2, flattened);
        reconstructed[0][0] = 99.0;

        // Original should be unchanged
        REQUIRE(original[0][0] == 1.0);
        // But reconstructed should have the new value
        REQUIRE(reconstructed[0][0] == 99.0);

        delete[] flattened;
    }

    SECTION("Modify original flattened array") {
        Matrix original({{1.0, 2.0}, {3.0, 4.0}});
        double* flattened = new double[4];
        original.flatten(flattened);

        // Modify the flattened array after creating the reconstructed matrix
        flattened[0] = 99.0;

        Matrix reconstructed(2, 2, flattened);

        // Reconstructed should have the modified value
        REQUIRE(reconstructed[0][0] == 99.0);
        // Original should be unchanged
        REQUIRE(original[0][0] == 1.0);

        delete[] flattened;
    }
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
