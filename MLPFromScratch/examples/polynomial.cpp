#include "example.h"

// Function to generate a polynomial of any degree and of any dimension

static void polynomial_example() {
    // Generate training data for a simple polynomial y = x^2
    std::vector<Vector> inputs;
    std::vector<Vector> targets;
    for (int x = 0; x <= 5; ++x) {
        double x_normalized = (static_cast<double>(x) / 5.0);
        double y_normalized = static_cast<double>(x * x) / 25.0;

        inputs.push_back(Vector({x_normalized}));
        targets.push_back(Vector({y_normalized}));
    }

    // Revised architecture (1-8-8-1) with tanh activations
    std::vector<size_t> architecture = {1, 100, 100, 1};
    MLP mlp(architecture, {leaky_relu, leaky_relu, leaky_relu}, mse);
    mlp.set_learning_rate(0.1);

    // Double training duration with progress checks
    int epochs = 10000;
    int log_interval = 1000;

    // Train the model
    mlp.train(inputs, targets, epochs, log_interval);

    // Test the model
    std::cout << "Testing polynomial y = x^2:" << std::endl;
    double x = 2.5 / 5.0;

    Vector prediction = mlp.forward(Vector({x}));
    std::cout << "x: " << 2.5 << ", Target: " << 6.25
              << ", Prediction: " << prediction.get_values()[0] * 25 << std::endl;

    // Extrapolate
    x = 6 / 5.0;
    prediction = mlp.forward(Vector({x}));
    std::cout << "x: " << 6 << ", Target: " << 36
              << ", Prediction: " << prediction.get_values()[0] * 25 << std::endl;
}
