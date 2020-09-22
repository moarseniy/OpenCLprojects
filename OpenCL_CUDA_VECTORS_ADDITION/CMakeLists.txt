cmake_minimum_required(VERSION 3.18.2)


project(OpenCL_CUDAsum)     

if(false)
	#CUDA
	set_source_files_properties(kernel.cu PROPERTIES LANGUAGE "CUDA")
	add_executable(OpenCL_CUDAsum kernel.cu)
	set_target_properties(OpenCL_CUDAsum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
	#OpenCL
	find_package(OpenCL REQUIRED)
	add_executable(OpenCL_CUDAsum OpenCLsum.cpp)
	target_link_libraries(OpenCL_CUDAsum OpenCL::OpenCL)
endif()