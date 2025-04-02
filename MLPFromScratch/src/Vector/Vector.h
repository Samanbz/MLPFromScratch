#pragma once

#include <vector>
#include <string>

/**
 * Represents a vector of real numbers.
 */
class Vector {
public:
    /**
     * Creates a new vector with the given size.
     * @param size The size of the vector.
     */
    Vector(size_t size);

    /**
     * Creates a new vector with the given size and initializes all elements to the given value.
     * @param size The size of the vector.
     * @param value The value to initialize all elements to.
     */
    Vector(size_t size, double value);

    /**
     * Creates a new vector with the given size and initializes all elements to the given values.
     * @param size The size of the vector.
     * @param values The values to initialize all elements to.
     */
    Vector(size_t size, double* values);

    /**
     * Creates a new vector with the given size and initializes all elements to the given values.
     * @param size The size of the vector.
     * @param values The values to initialize all elements to.
     */
    Vector(size_t size, std::vector<double> values);

    /**
     * @returns The size of the vector.
     */
    size_t size() const;

    /**
     * @param index The index of the element to return.
     * @returns The element at the given index.
     */
    double& operator[](size_t index);

    /**
     * @param index The index of the element to return.
     * @returns The element at the given index.
     */
    const double& operator[](size_t index) const;

    /**
     * Computes the dot product of this vector and another vector.
     * @param other The other vector to compute the dot product with.
     * @returns The dot product of the two vectors.
     */
    double dot(const Vector& other) const;

	/**
	 * Computes the sum of this vector and another vector.
	 * @param other The other vector to compute the sum with.
	 * @returns The sum of the two vectors.
	 */
	Vector operator+(const Vector& other) const;

	/**
	 * Computes the difference of this vector and another vector.
	 * @param other The other vector to compute the difference with.
	 * @returns The difference of the two vectors.
	 */
	Vector operator-(const Vector& other) const;

	/**
	 * Computes the product of this vector and a scalar.
	 * @param scalar The scalar to compute the product with.
	 * @returns The product of the vector and the scalar.
	 */
	Vector operator*(double scalar) const;

    /**
     * @returns the string representation of this vector.
     */
	std::string to_string() const;

private:
	std::vector<double> values;
};