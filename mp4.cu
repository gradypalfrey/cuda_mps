
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

// #define BLOCKDIM 16
#define TILEWIDTH 2
#define arrLen 8

typedef int matrix[];

__global__ void tileMult(matrix *a, matrix *b, matrix *c, int length) {
	int row = threadIdx.x + TILEWIDTH * blockIdx.x;
	int col = threadIdx.y + TILEWIDTH * blockIdx.y;
	__shared__ int sharedM[TILEWIDTH][TILEWIDTH];
	__shared__ int sharedN[TILEWIDTH][TILEWIDTH];
	int temp = 0;
	int l = col * arrLen + row;
	for (int k = 0; k < arrLen / TILEWIDTH; ++k) {
		if (row < arrLen && col < arrLen) {
			sharedM[threadIdx.y][threadIdx.x] = (*a)[col*arrLen + (k* TILEWIDTH + threadIdx.x)];
			sharedN[threadIdx.y][threadIdx.x] = (*b)[row + arrLen * (k* TILEWIDTH + threadIdx.y)];
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
	(*c)[l] = temp;
}

int printMatrix(matrix *matrix) {
	for (int x = 0; x < arrLen; x++)
	{
		for (int y = 0; y < arrLen; y++)
		{
			int i = y + x * arrLen;
			printf("(%d)", (*matrix)[i]);
		}
		printf("\n");
	}
	return 0;
}

int checkMatrix(matrix *arr1, matrix *arr2) {
	for (int x = 0; x < arrLen; x++) {
		for (int y = 0; y < arrLen; y++) {
			int i = x + y * arrLen;
			if ((*arr1)[i] != (*arr2)[i]) {
				return -1;
			}
		}
	}
	return 1;
}

int initRandomMatrix(matrix *m) {
	for (int x = 0; x < arrLen; x++) {
		for (int y = 0; y < arrLen; y++) {
			int i = x + y * arrLen;
			(*m)[i] = rand() % 10;
		}
	}
	return 0;
}

int multMatrixCPU(matrix* M, matrix *N, matrix *P) {
	for (int x = 0; x < arrLen; x++) {
		for (int y = 0; y < arrLen; y++) {
			int l = 0;
			for (int k = 0; k < arrLen; k++) {
				int midx = k + x * arrLen;
				int midy = k * arrLen + y;
				l += (*M)[midx] * (*N)[midy];

			}
			(*P)[x * arrLen + y] = l;
		}
	}
	return 0;
}


int matrix_addition() {

	matrix *a;
	matrix *b;
	matrix *c;

	size_t size = arrLen * arrLen * sizeof(int);

	a = (matrix*)malloc(size);
	b = (matrix*)malloc(size);
	c = (matrix*)malloc(size);

	srand(time(NULL));

	initRandomMatrix(a);
	initRandomMatrix(b);
	initRandomMatrix(c);

	cudaEvent_t start, end;
	float time = 0;

	matrix *pA, *pB, *pC;

	cudaMalloc((void**)&pA, size);
	cudaMalloc((void**)&pB, size);
	cudaMalloc((void**)&pC, size);

	cudaMemcpy(pA, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(pB, b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(pC, c, size, cudaMemcpyHostToDevice);

	cout << "Start CPU" << endl;

	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);

	dim3 threadsPerBlock(TILEWIDTH, TILEWIDTH);
	dim3 numBlocks((int)ceil(arrLen / (float)TILEWIDTH), (int)ceil(arrLen / (float)TILEWIDTH));
	tileMult << <numBlocks, threadsPerBlock >> >(pA, pB, pC, arrLen);

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	cout << "GPU Element Time: " << time << endl;

	cudaMemcpy(c, pC, (arrLen*arrLen)*sizeof(int), cudaMemcpyDeviceToHost);

	printMatrix(c);
	cout << endl;


	clock_t before = clock();
	matrix *cTmp = (matrix*)malloc(size);
	multMatrixCPU(a, b, cTmp);
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

	printMatrix(cTmp);
	cout << endl;

	cudaFree(pA);
	cudaFree(pB);
	cudaFree(pC);
	free(a);
	free(b);
	free(c);
	free(cTmp);

	return 0;
}

int main(int argc, char *argv[])
{

	matrix_addition();

	cout << "Done" << endl;

	return 0;
}