cmake_minimum_required(VERSION 3.4...3.18)
project(cathalTest)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Include the header files
include_directories(include/)

# Add the module
pybind11_add_module(cathalTest src/functionDefs.cpp src/myPBmodule.cpp)

# Optionally set the C++ standard
set_target_properties(cathalTest PROPERTIES CXX_STANDARD 17)

message(STATUS "C++ compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ flags, Debug configuration: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "C++ flags, Release configuration: ${CMAKE_CXX_FLAGS_RELEASE}")
