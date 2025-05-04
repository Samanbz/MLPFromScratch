#pragma once

// The threshold below which we will not use gpu parallelization to avoid overhead
constexpr double PARALLEL_THRESHOLD = 256e20;
