#include <filesystem>

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
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

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

    MLP mlp({784, 128, 10}, {leaky_relu, leaky_relu}, mse);

    // time it

    auto start = std::chrono::high_resolution_clock::now();

    Vector output = mlp.forward(train_images[0]);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Forward pass time: " << elapsed.count() * 1000 << " ms\n";

    start = std::chrono::high_resolution_clock::now();

    double loss = mlp.backward(train_labels_one_hot[0]);

    end = std::chrono::high_resolution_clock::now();

    elapsed = end - start;

    std::cout << "Backward pass time: " << elapsed.count() * 1000 << " ms\n";
}
