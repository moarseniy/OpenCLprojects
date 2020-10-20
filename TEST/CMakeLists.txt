cmake_minimum_required(VERSION 3.18.2 FATAL_ERROR)

project(CUDAreduction LANGUAGES C CXX CUDA )


#set_source_files_properties(kernel.cu PROPERTIES LANGUAGE "CUDA")

#enable_language("CUDA")
add_executable(CUDAreduction kernel.cu)


#set_target_properties(CUDAreduction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)



add_definitions(-D_GPU_CUDA_RUN)
