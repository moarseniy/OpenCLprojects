cmake_minimum_required(VERSION 3.18.2)

project(OpenCL_square C CXX CUDA)     

#OpenCL
find_package(OpenCL REQUIRED)
add_executable(OpenCL_square square.cpp)
target_link_libraries(OpenCL_square OpenCL::OpenCL)

