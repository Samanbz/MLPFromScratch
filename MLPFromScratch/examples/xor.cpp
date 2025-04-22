#include "example.h"

static void xor_example() {
    // Create the training data for XOR function
    std::vector<Vector> inputs = {Vector({0, 0}), Vector({0, 1}), Vector({1, 0}), Vector({1, 1})};
    std::vector<Vector> targets = {Vector({0}), Vector({1}), Vector({1}), Vector({0})};

    // Create MLP model with 2-input, hidden layer with 4 nodes, and 1 output
    // Input layer (2 nodes) -> Hidden layer (4 nodes) -> Output layer (1 node)
    std::vector<size_t> architecture = {2, 4, 1};
    MLP mlp(architecture, {sigmoid, sigmoid}, mse);
    mlp.set_learning_rate(0.15);

    // Training parameters
    int epochs = 100000;
    int log_interval = 10000;  // Print loss every 1000 epochs

    // Train the model
    mlp.train(inputs, targets, epochs, log_interval);

    // Test the model
    std::cout << "Testing XOR function:" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Vector prediction = mlp.forward(inputs[i]);
        std::cout << "Input: [" << inputs[i].get_values()[0] << ", " << inputs[i].get_values()[1]
                  << "], Target: " << targets[i].get_values()[0]
                  << ", Prediction: " << prediction.get_values()[0] << std::endl;
    }
}
