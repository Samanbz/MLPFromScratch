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
)

# Add test executable (separate)
add_executable(MLPFromScratchTest 
    "MLPFromScratchTest.cpp"
    "src/Vector/Vector.cpp" 
    "src/Vector/Vector.h"
    "src/Vector/Vector.test.cpp"
)

# Add include directories for both targets
target_include_directories(MLPFromScratch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(MLPFromScratchTest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add include directories for finding Vector.h
target_include_directories(MLPFromScratch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(MLPFromScratchTest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
