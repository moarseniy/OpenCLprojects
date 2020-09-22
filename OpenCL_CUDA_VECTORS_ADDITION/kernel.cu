#include <stdlib.h>
#include <stdio.h>

cudaEvent_t start, stop;

__global__ void vectorAdd(float* a, float* b, float* c)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	c[index] = a[index] + b[index];
}

void randomInit(float* a, int n)
{
	for (int i = 0; i < n; i++)
		a[i] = rand()%10;
}

int main(void)
{
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float gpuTime = 0.0;

	const unsigned int blockSize = 512;
	const unsigned int numBlocks = 3;
	const unsigned int numItems = numBlocks*blockSize;

	cudaSetDevice(0);

	float* a = new float[numItems];
	float* b = new float[numItems];
	float* c = new float[numItems];

	randomInit(a, numItems);
	randomInit(b, numItems);

	float* aDev, * bDev, * cDev;

	cudaMalloc((void**)&aDev, numItems * sizeof(float));
	cudaMalloc((void**)&bDev, numItems * sizeof(float));
	cudaMalloc((void**)&cDev, numItems * sizeof(float));

	cudaMemcpy(aDev, a, numItems * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bDev, b, numItems * sizeof(float), cudaMemcpyHostToDevice);


	cudaEventRecord(start, 0);

	vectorAdd << <numBlocks, blockSize >> > (aDev, bDev, cDev);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	printf("GPU time = %.4f \n", gpuTime);

	cudaMemcpy((void*) c, cDev, numItems * sizeof(float), cudaMemcpyDeviceToHost);

	

	delete[] a;
	delete[] b;
	delete[] c;

	cudaFree(aDev);
	cudaFree(bDev);
	cudaFree(cDev);
}