
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

#define BLOCKDIM 16
#define arrLen 16

__global__ void multElement(int *a, int *b, int *c, int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;
    for (int j = 0; j < length; j++) {
	    if (i < length && n < length) {
            int idx = i + n*arrLen;
            c[idx] += a[i + j*length] * b[j + n*length];
        }
	}
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


int matrix_addition() {

	int *a;
	int *b;
	int *c;
	int *cTmp;

	size_t size = arrLen * arrLen * sizeof(int);

	a = (int*)malloc(size);
	b = (int*)malloc(size);
	c = (int*)malloc(size);
	cTmp = (int*)malloc(size);

	srand(time(NULL));

	for (int i = 0; i < arrLen*arrLen; i++) {
		a[i] = rand() % 10 + 1;
		b[i] = rand() % 10 + 1;
		c[i] = 0;
		cTmp[i] = 0;
	}

	int *pA, *pB, *pC;

    float time1 = 0;
	cudaEvent_t start1, end1;
    cudaEventCreate(&start1);
	cudaEventCreate(&end1);
	cudaEventRecord(start1);

	cudaMalloc((void**)&pA, size);
	cudaMalloc((void**)&pB, size);
	cudaMalloc((void**)&pC, size);

	cudaMemcpy(pA, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(pB, b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(pC, c, size, cudaMemcpyHostToDevice);

    cudaEventRecord(end1);
	cudaEventSynchronize(end1);
	cudaEventElapsedTime(&time1, start1, end1);
	cudaEventDestroy(start1);
	cudaEventDestroy(end1);
	cout << "Transfer Time: " << time1 << endl;

	cout << "Start CPU" << endl;

	float time = 0;
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	for (int i = 0; i < arrLen; i++) {
		for (int n = 0; n < arrLen; n++) {
			for (int j = 0; j < arrLen; j++) {
				cTmp[i + n*arrLen] += a[i + j*arrLen] * b[j + n*arrLen];
			}
		}
	}

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	cout << "CPU Time: " << time << endl;

	time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
	dim3 numBlocks((int)ceil(arrLen / (float)threadsPerBlock.x), (int)ceil(arrLen / (float)threadsPerBlock.y));
	multElement << <numBlocks, threadsPerBlock >> >(pA, pB, pC, arrLen);

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	cout << "GPU Element Time: " << time << endl;

	cudaMemcpy(c, pC, (arrLen*arrLen)*sizeof(float), cudaMemcpyDeviceToHost);

	int rslt = checkMatrix(c, cTmp);
    cout << rslt << endl;

	if (rslt == -1) {
		cout << "Test Failed" << endl;
	}
	else {
		cout << "Test Passed" << endl;
	}

    // printMatrix(a);
    // cout << endl;

    // printMatrix(b);
    // cout << endl;

    // printMatrix(cTmp);
    // cout << endl;

    // printMatrix(c);
    // cout << endl;

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