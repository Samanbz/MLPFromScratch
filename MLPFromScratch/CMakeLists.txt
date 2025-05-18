cmake_minimum_required(VERSION 3.10)  
project(MLPFromScratch LANGUAGES CXX)  

# Set C++ standard  
set(CMAKE_CXX_STANDARD 20)  
set(CMAKE_CXX_STANDARD_REQUIRED ON)  

# Enable OpenMP  
find_package(OpenMP REQUIRED)  

# Define common source files  
set(COMMON_SOURCES  
   "src/Vector/Vector.cpp"  
   "src/Vector/Vector.h"  
   "src/Matrix/Matrix.h"  
   "src/Matrix/Matrix.cpp"  
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
   "src/Layer/Layer.test.cpp"  
   "src/MLP/MLP.test.cpp"  
)  

set(EXAMPLES_SOURCES  
   "examples/MNIST/mnist_reader.cpp"  
   "examples/MNIST/mnist_example.cpp")  

# Add main executable (without test files)  
add_executable(MLPFromScratch ${MAIN_SOURCES} ${COMMON_SOURCES} ${EXAMPLES_SOURCES})  

# Add test executable  
add_executable(MLPFromScratchTest ${TEST_SOURCES} ${COMMON_SOURCES})  

# Add include directories for both targets  
target_include_directories(MLPFromScratch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)  
target_include_directories(MLPFromScratchTest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)  

# Add include directories for finding Vector.h  
target_include_directories(MLPFromScratch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)  
target_include_directories(MLPFromScratchTest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)  

# Link OpenMP to both executables  
target_link_libraries(MLPFromScratch PUBLIC OpenMP::OpenMP_CXX)  
target_link_libraries(MLPFromScratchTest PUBLIC OpenMP::OpenMP_CXX)
