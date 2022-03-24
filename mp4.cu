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

__global__ void tileMult(int *a, int *b, int *c, int length) {
	int row = threadIdx.x + TILEWIDTH * blockIdx.x;
	int col = threadIdx.y + TILEWIDTH * blockIdx.y;
	__shared__ int sharedM[TILEWIDTH][TILEWIDTH];
	__shared__ int sharedN[TILEWIDTH][TILEWIDTH];
	int temp = 0;
	int l = col * arrLen + row;
	for (int k = 0; k < arrLen / TILEWIDTH; ++k) {
		if (row < arrLen && col < arrLen) {
			sharedM[threadIdx.y][threadIdx.x] = a[col*arrLen + (k* TILEWIDTH + threadIdx.x)];
			sharedN[threadIdx.y][threadIdx.x] = b[row + arrLen * (k* TILEWIDTH + threadIdx.y)];
		}
		else {
			sharedM[threadIdx.y][threadIdx.x] = 0;
			sharedN[threadIdx.y][threadIdx.x] = 0;
		}
		__syncthreads();
		for (int h = 0; h < TILEWIDTH; h++) {
			temp += sharedM[threadIdx.y][h] * sharedN[h][threadIdx.x];
			__syncthreads();
		}
	}
	c[l] = temp;

}

void printMatrix(int *arr1) {
	for (int i = 0; i < arrLen; i++) {
		for (int n = 0; n < arrLen; n++) {
			int idx = i + n*arrLen;
			cout << arr1[idx] << " ";
		}
		cout << endl;
	}
}

int checkMatrix(int *arr1, int *arr2) {
	for (int i = 0; i < arrLen*arrLen; i++) {
		if (arr1[i] != arr2[i]) {
			return -1;
		}
	}
	return 1;
}

int cpuMult(int* a, int *b, int *c) {
	for (int i = 0; i < arrLen; i++) {
		for (int n = 0; n < arrLen; n++) {
			int current = 0;
			for (int k = 0; k < arrLen; k++) {
				int midx = k + i * arrLen;
				int midy = k * arrLen + n;
				current += a[k + i * arrLen] * b[k * arrLen + n];
			}
			c[i * arrLen + n] = current;
		}
	}
	return 0;
}


int matrix_addition() {

	int *a;
	int *b;
	int *c;

	size_t size = arrLen * arrLen * sizeof(int);

	a = (int*)malloc(size);
	b = (int*)malloc(size);
	c = (int*)malloc(size);

	srand(time(NULL));

	for (int i = 0; i < arrLen*arrLen; i++) {
		a[i] = rand() % 10 + 1;
		b[i] = rand() % 10 + 1;
		c[i] = 0;
	}

	int *pA, *pB, *pC;

	cudaMalloc((void**)&pA, size);
	cudaMalloc((void**)&pB, size);
	cudaMalloc((void**)&pC, size);

	cudaMemcpy(pA, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(pB, b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(pC, c, size, cudaMemcpyHostToDevice);

	cout << "Start CPU" << endl;

	float time = 0;
	cudaEvent_t start, end;

	clock_t before = clock();
    int *cTmp = (int*)malloc(size);
    cpuMult(a, b, cTmp);
	clock_t difference = clock() - before;
	float msec = difference * 1000 / CLOCKS_PER_SEC;
	printf("CPU calculation took %f s\n", msec / 1000);

	int rslt = checkMatrix(c, cTmp);
	cout << rslt << endl;

	if (rslt == -1) {
		cout << "Test Failed" << endl;
	}
	else {
		cout << "Test Passed" << endl;
	}


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

	cudaMemcpy(c, pC, (arrLen*arrLen)*sizeof(float), cudaMemcpyDeviceToHost);

	// printMatrix(c);
	// cout << endl;

	rslt = checkMatrix(c, cTmp);
	cout << rslt << endl;

	if (rslt == -1) {
		cout << "Test Failed" << endl;
	}
	else {
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