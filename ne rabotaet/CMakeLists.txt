cmake_minimum_required(VERSION 3.18.2)

project(kernel)     


set_source_files_properties(kernel.cu PROPERTIES LANGUAGE "CUDA")


add_executable(kernel kernel.cu)

#add_definitions(-DGPU_CUDA_RUN) #-D_GPU_OPENCL_RUN

set_target_properties(kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)




