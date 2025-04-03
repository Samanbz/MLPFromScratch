#pragma once

#include <vector>

#include "../Layer/Layer.h"
#include "../Vector/Vector.h"

class MLP {
public:
    MLP(std::vector<size_t> layer_sizes);

    Vector forward(const Vector& input);

private:
    size_t input_size;
    size_t output_size;

    std::vector<Layer> layers;
};
