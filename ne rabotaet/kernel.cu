#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>

#define SIZE 256




cudaEvent_t start, stop;

__global__ void sum_reduction(float* a, float* b)
{
	__shared__ float temp[SIZE]; 

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	temp[threadIdx.x] = a[id];
	
	__syncthreads();


	for (int i = 1; i < blockDim.x; i *= 2)
	{
		if (2*i* threadIdx.x <blockDim.x)
		{
			temp[2*i*threadIdx.x] += temp[2*i* threadIdx.x + i];
		}
		__syncthreads();
	}


	if (threadIdx.x == 0) 
	{
		b[blockIdx.x] = temp[0];
	}
}

void init_vector(float *a, int n) 
{
	for (int i = 0; i < n; i++) 
	{
		a[i] = 1.0;//rand() % 10;
	}
}

int main() 
{
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float gpuTime = 0.0;

	int n = 1 << 16;
	size_t bytes = n * sizeof(float);
	
	cudaSetDevice(0);

	float *h_a, *h_b;
	float *d_a, *d_b;

	h_a = (float*)malloc(bytes);
	h_b= (float*)malloc(bytes);
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);

	init_vector(h_a, n);

	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

	int numBlocks = n / SIZE;

	cudaEventRecord(start, 0);
	

	

	sum_reduction <<<numBlocks, SIZE >>> (d_a, d_b);

	


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	printf("GPU time = %.4f \n", gpuTime);

	cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);

	float result = 0;
	for (int i = 0; i < numBlocks; i++)
	{
		result += h_b[i];
	}

	printf("Result= %f \n", result);


	free(h_a);
	free(h_b);
	cudaFree(d_a);
	cudaFree(d_b);

	return 0;
}

