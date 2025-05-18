#pragma once

#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

// Forward declaration of Matrix class
class Matrix;

/**
 * @brief A class representing a mathematical vector with CUDA support
 *
 * This Vector class provides operations for mathematical vectors with CUDA
 * support using __host__ and __device__ qualifiers for GPU compatibility.
 */
class Vector {
public:
    /**
     * @brief Default constructor
     *
     * Creates an empty vector with size 0
     */
    Vector();

    /**
     * @brief Constructs a vector of specified size initialized to zeros
     *
     * @param size Number of elements in the vector
     */
    Vector(size_t size);

    /**
     * @brief Constructs a vector of specified size with all elements set to a given value
     *
     * @param size Number of elements in the vector
     * @param value Value to initialize all elements with
     */
    Vector(size_t size, double value);

    /**
     * @brief Constructs a vector from a raw array of values
     *
     * @param size Number of elements in the vector
     * @param values Pointer to the array of values to copy
     */
    Vector(size_t size, const double* values);

    /**
     * @brief Constructs a vector from an initializer list
     *
     * @param init Initializer list containing the values
     */
    Vector(std::initializer_list<double> init);

    /**
     * @brief Constructs a vector from a std::vector
     *
     * @param values std::vector containing the values to copy
     */
    Vector(std::vector<double> values);

    /**
     * @brief Copy constructor
     *
     * @param other Vector to be copied
     */
    Vector(const Vector& other);

    /**
     * @brief Copy assignment operator
     *
     * @param other Vector to be copied
     * @return Reference to the assigned vector
     */
    Vector& operator=(const Vector& other);

    /**
     * @brief Destructor
     *
     * Frees the memory allocated for the vector elements
     */
    ~Vector();

    /**
     * @brief Creates a vector with random values
     *
     * @param min Lower bound for random values
     * @param max Upper bound for random values
     * @return A new Vector with random values between min and max
     */
    static Vector random(double min, double max);

    /**
     * @brief Gets the size of the vector
     *
     * @return Number of elements in the vector
     */
    size_t size() const { return size_; }

    /**
     * @brief Access operator for modifiable elements
     *
     * @param index Position of the element to access
     * @return Reference to the element at the specified position
     */
    double& operator[](size_t index);

    /**
     * @brief Access operator for read-only elements
     *
     * @param index Position of the element to access
     * @return Constant reference to the element at the specified position
     */
    const double& operator[](size_t index) const;

    /**
     * @brief Calculates the dot product of two vectors
     *
     * @param other Vector to calculate dot product with
     * @return Dot product result
     */
    double dot(const Vector& other) const;

    /**
     * @brief Adds two vectors element-wise
     *
     * @param other Vector to add
     * @return New vector containing the sum
     */
    Vector operator+(const Vector& other) const;

    /**
     * @brief Subtracts two vectors element-wise
     *
     * @param other Vector to subtract
     * @return New vector containing the difference
     */
    Vector operator-(const Vector& other) const;

    /**
     * @brief Multiplies the vector by a scalar
     *
     * @param scalar Value to multiply each element by
     * @return New vector containing the scaled values
     */
    Vector operator*(double scalar) const;

    /**
     * @brief Performs element-wise multiplication with another vector
     *
     * @param other Vector to multiply element-wise
     * @return New vector containing the element-wise product
     */
    Vector elem_mult(const Vector& other) const;

    /**
     * @brief Squares each element of the vector
     *
     * @return New vector with squared elements
     */
    Vector square() const;

    /**
     * @brief Calculates the sum of all elements
     *
     * @return Sum of all elements in the vector
     */
    double sum() const;

    /**
     * @brief Calculates the mean of all elements
     *
     * @return Mean value of the vector
     */
    double mean() const;

    /**
     * @brief Calculates the Euclidean norm (L2 norm) of the vector
     *
     * @return Euclidean norm of the vector
     */
    double norm() const;

    /**
     * @brief Provides direct access to the underlying data array
     *
     * @return Pointer to the first element of the vector
     */
    double* data() { return values; }

    /**
     * @brief Provides direct read-only access to the underlying data array
     *
     * @return Constant pointer to the first element of the vector
     */
    const double* data() const { return values; }

    /**
     * @brief Gets vector values as a std::vector
     *
     * @return std::vector containing a copy of all elements
     */
    std::vector<double> get_values() const;

    /**
     * @brief Applies a function to each element of the vector
     *
     * @param func Function to apply to each element
     * @return New vector with transformed elements
     */
    Vector apply(std::function<double(double)> func) const;

    /**
     * @brief Computes the outer product with another vector
     *
     * @param other Vector to compute outer product with
     * @return Matrix representing the outer product
     */
    Matrix outer_product(const Vector& other) const;

    /**
     * @brief Converts the vector to a string representation
     *
     * @return String representation of the vector
     */
    std::string to_string() const;

    /**
     * @brief Equality comparison operator
     *
     * @param other Vector to compare with
     * @return true if vectors are equal, false otherwise
     */
    bool operator==(const Vector& other) const;

private:
    double* values;  // Raw pointer to array
    size_t size_;    // Size of vector
};