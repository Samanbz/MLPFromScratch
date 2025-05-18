#include <algorithm>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include "../example.h"
#include "mnist_reader.cpp"

namespace fs = std::filesystem;

static size_t get_max_idx(std::vector<double> values) {
    size_t max_index = 0;

    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[max_index]) {
            max_index = i;
        }
    }

    return max_index;
}

static void mnist_example() {
    std::srand(2);

    fs::path current_path = fs::current_path();
    fs::path project_dir = current_path.parent_path().parent_path().parent_path().parent_path();

    fs::path training_image_filepath = project_dir / "input" / "mnist" / "train-images.idx3-ubyte";
    fs::path training_label_filepath = project_dir / "input" / "mnist" / "train-labels.idx1-ubyte";
    fs::path test_image_filepath = project_dir / "input" / "mnist" / "t10k-images.idx3-ubyte";
    fs::path test_label_filepath = project_dir / "input" / "mnist" / "t10k-labels.idx1-ubyte";

    std::cout << "Loading MNIST dataset from:\n"
              << "Training images: " << training_image_filepath << "\n"
              << "Training labels: " << training_label_filepath << "\n"
              << "Test images: " << test_image_filepath << "\n"
              << "Test labels: " << test_label_filepath << "\n";

    MNISTReader mnist_reader(training_image_filepath.string(), training_label_filepath.string(),
                             test_image_filepath.string(), test_label_filepath.string());

    auto [train_data, test_data] = mnist_reader.read_data();
    auto [train_images, train_labels] = train_data;
    auto [test_images, test_labels] = test_data;

    auto train_labels_one_hot = MNISTReader::one_hot_encode(train_labels, 10);
    auto test_labels_one_hot = MNISTReader::one_hot_encode(test_labels, 10);

    MLP mlp({784, 128, 10}, {relu, relu}, cross_entropy);

    mlp.set_learning_rate(0.001);

    // sample 10% of training and test data
    size_t train_sample_size = static_cast<size_t>(train_images.size() * 0.05);
    size_t test_sample_size = static_cast<size_t>(test_images.size() * 0.05);

    train_images.resize(train_sample_size);
    train_labels_one_hot.resize(train_sample_size);

    // train the model
    mlp.train(train_images, train_labels_one_hot, 100, 5);

    // test the model
    int correct_predictions = 0;
    int limit = 5;
    for (size_t i = 0; i < test_images.size(); ++i) {
        auto output = mlp.forward(test_images[i]);
        size_t predicted_label = get_max_idx(output.get_values());
        size_t actual_label = get_max_idx(test_labels_one_hot[i].get_values());

        if (predicted_label == actual_label) {
            ++correct_predictions;
        }

        // if incorrectly predicted, display the image and the predicted and actual labels
        if (predicted_label != actual_label && limit > 0) {
            std::cout << "Incorrect prediction for image " << i << "\n";
            std::cout << "Predicted: " << predicted_label << ", Actual: " << actual_label << "\n";

            // display the image as ascii art
            for (size_t j = 0; j < test_images[i].size(); ++j) {
                if (j % 28 == 0 && j != 0) {
                    std::cout << "\n";
                }
                static const char* shades = " .:-=+*#%@";
                int shade_levels = 10;
                double pixel = test_images[i][j];
                int idx = static_cast<int>(pixel * (shade_levels - 1));
                if (idx < 0) idx = 0;
                if (idx >= shade_levels) idx = shade_levels - 1;
                std::cout << shades[idx];
            }
            limit--;
        }
    }
    double accuracy = static_cast<double>(correct_predictions) / test_images.size();
    std::cout << "Model accuracy: " << accuracy * 100 << "%" << std::endl;
}
