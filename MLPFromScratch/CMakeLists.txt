cmake_minimum_required(VERSION 3.10)
project(MLPFromScratch LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define common source files
set(COMMON_SOURCES
    "src/Vector/Vector.cpp"
    "src/Vector/Vector.h"
    "src/Matrix/Matrix.h"
    "src/Matrix/Matrix.cpp"
    "src/Neuron/Neuron.h"
    "src/Neuron/Neuron.cpp"
    "src/Layer/Layer.h"
    "src/Layer/Layer.cpp"
    "src/MLP/MLP.h"
    "src/MLP/MLP.cpp"
    "src/Functions/Activation.h"
    "src/Functions/Loss.h"
    "src/config.h"
)

# Define main executable specific sources
set(MAIN_SOURCES
    "MLPFromScratch.cpp"
    "MLPFromScratch.h"
)

# Define test executable specific sources
set(TEST_SOURCES
    "MLPFromScratchTest.cpp"
    "src/Vector/Vector.test.cpp"
    "src/Matrix/Matrix.test.cpp"
    "src/Neuron/Neuron.test.cpp"
    "src/Layer/Layer.test.cpp"
    "src/MLP/MLP.test.cpp"
)

# Add main executable (without test files)
add_executable(MLPFromScratch ${MAIN_SOURCES} ${COMMON_SOURCES})

# Add test executable
add_executable(MLPFromScratchTest ${TEST_SOURCES} ${COMMON_SOURCES})

# Add include directories for both targets
target_include_directories(MLPFromScratch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(MLPFromScratchTest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add include directories for finding Vector.h
target_include_directories(MLPFromScratch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(MLPFromScratchTest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

set_target_properties(MLPFromScratch PROPERTIES
    CUDA_ARCHITECTURES "86-real"
    CUDA_SEPARABLE_COMPILATION ON
)

find_package(CUDAToolkit REQUIRED)

target_compile_options(MLPFromScratch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(MLPFromScratchTest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Add CUDA includes to BOTH targets
target_include_directories(MLPFromScratch PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(MLPFromScratchTest PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Explicitly link against CUDA runtime (required for cudaMalloc/cudaFree)
target_link_libraries(MLPFromScratch PRIVATE CUDA::cudart)
target_link_libraries(MLPFromScratchTest PRIVATE CUDA::cudart)
