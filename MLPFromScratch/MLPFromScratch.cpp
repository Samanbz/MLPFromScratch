// MLPFromScratch.cpp : Defines the entry point for the application.
//

#include "MLPFromScratch.h"

#include "./src/Functions/Activation.h"
#include "./src/MLP/MLP.h"

using namespace std;

int main() {
    MLP mlp({1, 2, 1});

    mlp.set_learning_rate(0.01);

    // generate a std::vector with Vector.random as generator
    std::vector<Vector> inputs;
    std::vector<Vector> targets;

    for (int i = 0; i < 1000; i++) {
        Vector input = Vector::random(0, 1);
        Vector target = input.apply([](double x) { return 2 * x; });
        inputs.push_back(input);
        targets.push_back(target);
    }

    mlp.train(inputs, targets, 1000, 10);

    // Test the MLP with a new input
    Vector test_input = Vector::random(0, 1);
    Vector test_output = mlp.test(test_input);
    std::cout << "Test input: " << test_input.to_string() << std::endl;
    std::cout << "Test output: " << test_output.to_string() << std::endl;
}
