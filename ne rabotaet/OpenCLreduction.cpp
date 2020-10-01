
#include <CL\cl.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <ctime>

#define MAX_SOURCE_SIZE (0x100000)
#define SIZE 256


int main(void) {

	int n = 1 << 16;

	float* A = (float*)malloc(sizeof(float) * n);
	float* B = (float*)malloc(sizeof(float) * n);

	//float* C = (float*)malloc(sizeof(float) * SIZE);

	
	for (int i = 0; i < n; i++) 
	{
		A[i] = 1;
	}

	
	FILE* kernelFile;
	char* kernelSource;
	size_t kernelSize;

	kernelFile = fopen("vecAddKernel.cl", "r");

	if (!kernelFile) {

		fprintf(stderr, "file not found\n");
		exit(-1);
	}

	kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
	kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
	fclose(kernelFile);

	///////////////////////////////

	cl_platform_id platformId = NULL;
	cl_device_id deviceID = NULL;
	cl_uint retNumDevices;
	cl_uint retNumPlatforms;
	cl_int ret = clGetPlatformIDs(1, &platformId, &retNumPlatforms);
	ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceID, &retNumDevices);


	cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &ret);


	cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, 0, &ret);


	cl_mem aMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &ret);
	cl_mem bMemObj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &ret);
	//cl_mem cMemObj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE * sizeof(float), NULL, &ret);


	
	ret = clEnqueueWriteBuffer(commandQueue, aMemObj, CL_TRUE, 0, n * sizeof(float), A, 0, NULL, NULL);;
	//ret = clEnqueueWriteBuffer(commandQueue, bMemObj, CL_TRUE, 0, SIZE * sizeof(float), B, 0, NULL, NULL);


	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, (const size_t*)&kernelSize, &ret);

	unsigned int start_time = clock();

	ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

	cl_kernel kernel = clCreateKernel(program, "addVectors", &ret);


	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&aMemObj);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bMemObj);
	//ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&cMemObj);


	size_t globalItemSize = n;
	size_t localItemSize = 256; // 65536/256 = 256
	ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);

	//clFinish(queue);

	unsigned int end_time = clock();

	ret = clEnqueueReadBuffer(commandQueue, bMemObj, CL_TRUE, 0, n * sizeof(float), B, 0, NULL, NULL);

	printf("result= %f \n", B[0]);

	printf("TIME = %d", end_time - start_time);

	ret = clFlush(commandQueue);
	ret = clFinish(commandQueue);
	ret = clReleaseCommandQueue(commandQueue);
	ret = clReleaseKernel(kernel);
	clReleaseProgram(program);
	ret = clReleaseMemObject(aMemObj);
	ret = clReleaseMemObject(bMemObj);
	//ret = clReleaseMemObject(cMemObj);
	ret = clReleaseContext(context);
	free(A);
	free(B);
	//free(C);

	return 0;

}
