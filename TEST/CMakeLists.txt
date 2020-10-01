cmake_minimum_required(VERSION 3.18.2)

project(CUDAreduction LANGUAGES C CUDA)

add_definitions(-D_GPU_CUDA_RUN)
set_source_files_properties(kernel.cu PROPERTIES LANGUAGE "CUDA")
add_executable(kernel kernel.cu)
set_target_properties(kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)