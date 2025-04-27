#pragma once

#include <device_launch_parameters.h>

#include <cassert>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

// Forward declaration of Matrix class
class Matrix;

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
    Vector(size_t size, const double* values);

    /**
     * @brief Creates a vector from an initializer list.
     * @param init The initializer list of values.
     */
    Vector(std::initializer_list<double> init);

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
    __host__ __device__ size_t size() const { return values.size(); }

    /**
     * @brief Returns the element at the given index.
     *
     * @param index The index of the element to return.
     * @returns The element at the given index.
     * @throws std::out_of_range if the index is out of bounds.
     */
    __host__ __device__ double& at(size_t index) {
        assert(index < size());
        return values[index];
    }

    /**
     * @brief Returns the element at the given index.
     *
     * @param index The index of the element to return.
     * @returns The element at the given index.
     * @throws std::out_of_range if the index is out of bounds.
     */
    __host__ __device__ double at(size_t index) const {
        assert(index < size());
        return values[index];
    }

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
     * @brief return the underlying values of the vector.
     *
     * @returns the underlying vector.
     */
    std::vector<double>& get_values() { return values; }

    /**
     * @brief return the underlying values of the vector.
     *
     * @returns the underlying vector.
     */
    const std::vector<double> get_values() const { return values; }

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
     * @brief Computes the mean of all elements in the vector.
     *
     * @returns The mean of all elements in the vector.
     */
    double mean() const;

    /**
     * @brief Computes the Euclidean norm (magnitude) of the vector.
     *
     * The norm is calculated as the square root of the sum of the squares
     * of all elements in the vector.
     *
     * @returns The Euclidean norm of the vector.
     */
    double norm() const;

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
