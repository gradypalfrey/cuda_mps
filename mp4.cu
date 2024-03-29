#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string.h>
#include <stdio.h>

#include <iostream>
#include <memory>
#include <string>

#include <stdlib.h>
#include <time.h>

using namespace std;

#define TILEWIDTH 2
#define arrLen 8

__global__ void tileMult(int *a, int *b, int *c, int length)
{
	__shared__ int m1[TILEWIDTH][TILEWIDTH];
	__shared__ int m2[TILEWIDTH][TILEWIDTH];

	int i = threadIdx.x + TILEWIDTH * blockIdx.x;
	int n = threadIdx.y + TILEWIDTH * blockIdx.y;

	int curr = 0;
	for (int j = 0; j < arrLen / TILEWIDTH; j++)
	{
		if (i < arrLen && n < arrLen)
		{
			m1[threadIdx.y][threadIdx.x] = a[n * arrLen + (j * TILEWIDTH + threadIdx.x)];
			m2[threadIdx.y][threadIdx.x] = b[i + arrLen * (j * TILEWIDTH + threadIdx.y)];
		}
		else
		{
			m1[threadIdx.y][threadIdx.x] = 0;
			m2[threadIdx.y][threadIdx.x] = 0;
		}
		__syncthreads();
		for (int x = 0; x < TILEWIDTH; x++)
		{
			curr += m1[threadIdx.y][x] * m2[x][threadIdx.x];
			__syncthreads();
		}
	}
	c[n * arrLen + i] = curr;
}

void printMatrix(int *arr1)
{
	for (int i = 0; i < arrLen; i++)
	{
		for (int n = 0; n < arrLen; n++)
		{
			int idx = i + n * arrLen;
			cout << arr1[idx] << " ";
		}
		cout << endl;
	}
}

int checkMatrix(int *arr1, int *arr2)
{
	for (int i = 0; i < arrLen * arrLen; i++)
	{
		if (arr1[i] != arr2[i])
		{
			return -1;
		}
	}
	return 1;
}

int matrix_addition()
{

	int *a;
	int *b;
	int *c;
	int *cTmp;

	size_t size = arrLen * arrLen * sizeof(int);

	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);
	cTmp = (int *)malloc(size);

	srand(time(NULL));

	for (int i = 0; i < arrLen * arrLen; i++)
	{
		a[i] = rand() % 10 + 1;
		b[i] = rand() % 10 + 1;
		c[i] = 0;
		c[i] = 0;
	}

	int *pA, *pB, *pC;

	cudaMalloc((void **)&pA, size);
	cudaMalloc((void **)&pB, size);
	cudaMalloc((void **)&pC, size);

	cudaMemcpy(pA, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(pB, b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(pC, c, size, cudaMemcpyHostToDevice);

	cout << "Start CPU" << endl;

	float time = 0;
	cudaEvent_t start, end;

	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
	
	for (int i = 0; i < arrLen; i++) {
		for (int n = 0; n < arrLen; n++) {
			int curr = 0;
			for (int k = 0; k < arrLen; k++) {
				curr += a[k + i * arrLen] * b[k * arrLen + n];
			}
			cTmp[i * arrLen + n] = curr;
		}
	}

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	cout << "CPU Element Time: " << time << endl;

	time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	dim3 threadsPerBlock(TILEWIDTH, TILEWIDTH);
	dim3 numBlocks((int)ceil(arrLen / (float)TILEWIDTH), (int)ceil(arrLen / (float)TILEWIDTH));
	tileMult << <numBlocks, threadsPerBlock >> >(pA, pB, pC, arrLen);

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	cout << "GPU Element Time: " << time << endl;

	cudaMemcpy(c, pC, (arrLen * arrLen) * sizeof(float), cudaMemcpyDeviceToHost);

	int rslt = checkMatrix(c, cTmp);
	cout << rslt << endl;

	if (rslt == -1)
	{
		cout << "Test Failed" << endl;
	}
	else
	{
		cout << "Test Passed" << endl;
	}

	cudaFree(pA);
	cudaFree(pB);
	cudaFree(pC);
	free(a);
	free(b);
	free(c);
	free(cTmp);

	cudaDeviceReset();

	return 0;
}

int main(int argc, char *argv[])
{

	matrix_addition();

	cout << "Done" << endl;

	return 0;
}