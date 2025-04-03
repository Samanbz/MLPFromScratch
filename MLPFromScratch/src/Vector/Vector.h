#pragma once

#include <string>
#include <vector>

/**
 * @brief Represents a vector of real numbers.
 */
class Vector {
public:
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
     */
    double& operator[](size_t index);

    /**
     * @brief Returns the element at the given index.
     *
     * @param index The index of the element to return.
     * @returns The element at the given index.
     */
    const double& operator[](size_t index) const;

    /**
     * @brief Computes the dot product of this vector and another vector.
     *
     * @param other The other vector to compute the dot product with.
     * @returns The dot product of the two vectors.
     */
    double dot(const Vector& other) const;

    /**
     * @brief Computes the sum of this vector and another vector.
     *
     * @param other The other vector to compute the sum with.
     * @returns The sum of the two vectors.
     */
    Vector operator+(const Vector& other) const;

    /**
     * @brief Computes the difference of this vector and another vector.
     *
     * @param other The other vector to compute the difference with.
     * @returns The difference of the two vectors.
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
     * @brief Returns the string representation of this vector.
     *
     * @returns The string representation of this vector.
     */
    std::string to_string() const;

private:
    std::vector<double> values;
};