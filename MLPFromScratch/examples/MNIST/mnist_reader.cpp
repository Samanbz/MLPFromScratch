#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../src/Vector/Vector.h"

using namespace std;

class MNISTReader {
private:
    string training_images_filepath;
    string training_labels_filepath;
    string test_images_filepath;
    string test_labels_filepath;

    // Helper to read big-endian integers
    int read_int(ifstream& file) {
        unsigned char buffer[4];
        file.read(reinterpret_cast<char*>(buffer), 4);
        return ((buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3]);
    }

    pair<vector<Vector>, vector<int>> read_images_lables(const string& images_filepath,
                                                         const string& labels_filepath) {
        vector<int> labels;
        ifstream labels_file(labels_filepath, ios::binary);
        if (!labels_file.is_open()) {
            throw runtime_error("Could not open labels file: " + labels_filepath);
        }

        int magic = read_int(labels_file);
        if (magic != 2049) {
            throw runtime_error("Invalid labels file format");
        }

        int labels_count = read_int(labels_file);
        labels.reserve(labels_count);

        unsigned char label_value;
        for (int i = 0; i < labels_count; ++i) {
            labels_file.read(reinterpret_cast<char*>(&label_value), 1);
            labels.push_back(static_cast<int>(label_value));
        }
        labels_file.close();

        ifstream images_file(images_filepath, ios::binary);
        if (!images_file.is_open()) {
            throw runtime_error("Could not open images file: " + images_filepath);
        }

        magic = read_int(images_file);
        if (magic != 2051) {
            throw runtime_error("Invalid images file format");
        }

        int images_count = read_int(images_file);
        int rows = read_int(images_file);
        int cols = read_int(images_file);
        int image_size = rows * cols;

        if (labels_count != images_count) {
            throw runtime_error("Mismatch between number of labels and images");
        }

        vector<Vector> images;
        images.reserve(images_count);

        vector<unsigned char> buffer(image_size);

        for (int i = 0; i < images_count; ++i) {
            images_file.read(reinterpret_cast<char*>(buffer.data()), image_size);

            Vector img(image_size);
            for (int j = 0; j < image_size; ++j) {
                img[j] = static_cast<float>(buffer[j]) / 255.0f;  // Normalize pixel values
            }

            images.push_back(img);
        }
        images_file.close();

        return {images, labels};
    };

public:
    MNISTReader(const string& training_images_filepath, const string& training_labels_filepath,
                const string& test_images_filepath, const string& test_labels_filepath)
        : training_images_filepath(training_images_filepath),
          training_labels_filepath(training_labels_filepath),
          test_images_filepath(test_images_filepath),
          test_labels_filepath(test_labels_filepath) {}

    pair<pair<vector<Vector>, vector<int>>, pair<vector<Vector>, vector<int>>> read_data() {
        auto training_data = read_images_lables(training_images_filepath, training_labels_filepath);
        auto test_data = read_images_lables(test_images_filepath, test_labels_filepath);

        return {training_data, test_data};
    }

    static std::vector<Vector> one_hot_encode(const std::vector<int>& labels,
                                              int num_classes = 10) {
        std::vector<Vector> encoded_labels;
        encoded_labels.reserve(labels.size());

        for (int label : labels) {
            Vector one_hot(num_classes, 0.0);
            if (label >= 0 && label < num_classes) {
                one_hot[label] = 1.0;
            }
            encoded_labels.push_back(std::move(one_hot));
        }

        return encoded_labels;
    }
};
