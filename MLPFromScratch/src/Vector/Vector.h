#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

// Forward declaration of Matrix class
class Matrix;

/**
 * @brief Represents a vector of real numbers.
 */
class Vector {
public:
    /**
     * @brief default constructor that initializes an empty vector.
     */
    Vector();

    /**
     * @brief Creates a new vector with the given size.
     *
     * @param size The size of the vector.
     */
    Vector(size_t size);

    /**
     * @brief Creates a new vector with the given size and initializes all elements to the given
     * value.
     *
     * @param size The size of the vector.
     * @param value The value to initialize all elements to.
     */
    Vector(size_t size, double value);

    /**
     * @brief Creates a new vector with the given size and initializes all elements to the given
     * values.
     *
     * @param size The size of the vector.
     * @param values The values to initialize all elements to.
     */
    Vector(size_t size, double* values);

    /**
     * @brief Creates a new vector with the given size and initializes all elements to the given
     * values.
     *
     * @param values The values to initialize all elements to.
     */
    Vector(std::vector<double> values);

    /**
     * @brief initializes a random vector with values within a given range
     *
     * @param range of the random values
     */
    Vector static random(double min, double max);

    /**
     * @brief Returns the size of the vector.
     *
     * @returns The size of the vector.
     */
    size_t size() const;

    /**
     * @brief Returns the element at the given index.
     *
     * @param index The index of the element to return.
     * @returns The element at the given index.
     * @throws std::out_of_range if the index is out of bounds.
     */
    double& operator[](size_t index);

    /**
     * @brief Returns the element at the given index.
     *
     * @param index The index of the element to return.
     * @returns The element at the given index.
     * @throws std::out_of_range if the index is out of bounds.
     */
    const double& operator[](size_t index) const;

    /**
     * @brief Computes the dot product of this vector and another vector.
     *
     * @param other The other vector to compute the dot product with.
     * @returns The dot product of the two vectors.
     * @throws std::invalid_argument if the vectors have different sizes.
     */
    double dot(const Vector& other) const;

    /**
     * @brief Computes the sum of this vector and another vector.
     *
     * @param other The other vector to compute the sum with.
     * @returns The sum of the two vectors.
     * @throws std::invalid_argument if the vectors have different sizes.
     */
    Vector operator+(const Vector& other) const;

    /**
     * @brief Computes the difference of this vector and another vector.
     *
     * @param other The other vector to compute the difference with.
     * @returns The difference of the two vectors.
     * @throws std::invalid_argument if the vectors have different sizes.
     */
    Vector operator-(const Vector& other) const;

    /**
     * @brief Computes the product of this vector and a scalar.
     *
     * @param scalar The scalar to compute the product with.
     * @returns The product of the vector and the scalar.
     */
    Vector operator*(double scalar) const;

    /**
     * @brief returns the values of the vector.
     *
     * @returns the underlying vector.
     */
    std::vector<double> get_values() const;

    /**
     * @brief Computes the element-wise multiplication of this vector with another vector.
     *
     * @param other The other vector to compute the element-wise multiplication with.
     * @returns A new vector containing the element-wise product.
     * @throws std::invalid_argument if the vectors have different sizes.
     */
    Vector elem_mult(const Vector& other) const;

    /**
     * @brief returns the element-wise squared Vector.
     *
     * @param vector to be squared.
     * @returns The Vector squared element-wise.
     */
    Vector square() const;

    /**
     * @brief Computes the sum of all elements in the vector.
     *
     * @returns The sum of all elements in the vector.
     */
    double sum() const;

    /**
     * @brief Applies a function to all elements in the vector.
     *
     * @returns A new vector containing the results of applying the function to each element.
     */
    Vector apply(std::function<double(double)> func) const;

    /**
     * @brief Computes the outer product of this vector with another vector.
     *
     * @param other The other vector to compute the outer product with.
     * @returns A new matrix representing the outer product of the two vectors.
     */
    Matrix outer_product(const Vector& other) const;

    /**
     * @brief Returns the string representation of this vector.
     *
     * @returns The string representation of this vector.
     */
    std::string to_string() const;

    /**
     * @brief Checks if two vectors are equal.
     *
     * @param other vector to compare with.
     * @returns true if the vectors are equal, false otherwise.
     */
    bool operator==(const Vector& other) const;

private:
    std::vector<double> values;
};
