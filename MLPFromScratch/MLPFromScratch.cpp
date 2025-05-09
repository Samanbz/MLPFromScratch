// MLPFromScratch.cpp : Defines the entry point for the application.
//

#include "MLPFromScratch.h"

#include "./examples/MNIST/mnist_example.cpp"

int main() {
    // Run forward on 3 different MLP instances with different layer sizes, 500, 1000, and 4000
    // and measure the time it takes to run each one

    MLP mlp1({500, 250, 10}, {leaky_relu, leaky_relu}, mse);

    MLP mlp2({1000, 500, 10}, {leaky_relu, leaky_relu}, mse);

    MLP mlp3({4000, 2000, 10}, {leaky_relu, leaky_relu}, mse);

    // Create fake random data for all three
    std::vector<Vector> inputs1(1000, Vector(500, 0.5));
    std::vector<Vector> inputs2(1000, Vector(1000, 0.5));
    std::vector<Vector> inputs3(1000, Vector(4000, 0.5));

    std::vector<Vector> targets1(1000, Vector(10, 0.5));
    std::vector<Vector> targets2(1000, Vector(10, 0.5));
    std::vector<Vector> targets3(1000, Vector(10, 0.5));

    // auto start1 = std::chrono::high_resolution_clock::now();
    // mlp1.forward(inputs1[0]);
    // auto end1 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed1 = end1 - start1;

    // auto start2 = std::chrono::high_resolution_clock::now();
    // mlp2.forward(inputs2[0]);
    // auto end2 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed2 = end2 - start2;

    auto start3 = std::chrono::high_resolution_clock::now();
    mlp3.forward(inputs3[0]);
    auto end3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed3 = end3 - start3;

    // std::cout << "Elapsed time for mlp1: " << elapsed1.count() * 1000 << " ms\n";
    // std::cout << "Elapsed time for mlp2: " << elapsed2.count() * 1000 << " ms\n";
    std::cout << "Elapsed time for mlp3: " << elapsed3.count() * 1000 << " ms\n";

    return 0;
}
