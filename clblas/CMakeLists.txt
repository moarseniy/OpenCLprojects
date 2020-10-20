cmake_minimum_required(VERSION 3.18.2)

project(OpenCL_cblas_test1 C CXX CUDA)     

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${clblast_SOURCE_DIR}/cmake/Modules/")




find_package(OpenCL REQUIRED)
find_package(clBLAS REQUIRED)
#set(API_LIBRARIES ${OPENCL_LIBRARIES})
#set(API_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
add_executable(OpenCL_cblas_test1 cblas_test1.cpp)
target_link_libraries(OpenCL_cblas_test1 OpenCL::OpenCL)
target_link_libraries(OpenCL_cblas_test1 /home/usr/lib64/libclBLAS.so)

