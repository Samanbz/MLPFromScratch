cmake_minimum_required(VERSION 3.10)
project(MLPFromScratch)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add main executable (without test files)
add_executable(MLPFromScratch 
    "MLPFromScratch.cpp" 
    "MLPFromScratch.h"
    "src/Vector/Vector.cpp" 
    "src/Vector/Vector.h"
    "src/Matrix/Matrix.h" 
    "src/Matrix/Matrix.cpp" 
    "src/Neuron/Neuron.h" 
    "src/Neuron/Neuron.cpp" "src/Layer/Layer.h" "src/Layer/Layer.cpp" "src/MLP/MLP.h" "src/MLP/MLP.cpp" "src/Functions/Activation.h" "src/Functions/Loss.h")

# Add test executable (separate)
add_executable(MLPFromScratchTest 
    "MLPFromScratchTest.cpp"
    "src/Vector/Vector.cpp" 
    "src/Vector/Vector.h"
    "src/Vector/Vector.test.cpp"
    "src/Matrix/Matrix.h" 
    "src/Matrix/Matrix.cpp" 
    "src/Matrix/Matrix.test.cpp" 
    "src/Neuron/Neuron.h" 
    "src/Neuron/Neuron.cpp" "src/Neuron/Neuron.test.cpp" "src/Layer/Layer.h" "src/Layer/Layer.cpp" "src/Layer/Layer.test.cpp" "src/MLP/MLP.h" "src/MLP/MLP.cpp" "src/MLP/MLP.test.cpp" "src/Functions/Activation.h" "src/Functions/Loss.h")

# Add include directories for both targets
target_include_directories(MLPFromScratch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(MLPFromScratchTest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add include directories for finding Vector.h
target_include_directories(MLPFromScratch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(MLPFromScratchTest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
