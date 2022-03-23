
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
#define arrLen 256

__global__ void addElement(int a[][arrLen], int b[][arrLen], int c[][arrLen], int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < length && n < length) {
		c[i][n] = a[i][n] + b[i][n];
	}
}

__global__ void addRow(int a[][arrLen], int b[][arrLen], int c[][arrLen], int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int n = 0; n < length; n++) {
		if (i < length && n < length)
			c[i][n] = a[i][n] + b[i][n];
	}
}

__global__ void addCol(int a[][arrLen], int b[][arrLen], int c[][arrLen], int length) {
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < length; i++) {
		if (i < length && n < length)
			c[i][n] = a[i][n] + b[i][n];
	}
}

void printMatrix(int a[][arrLen]) {
	for (int i = 0; i < arrLen; i++) {
		for (int n = 0; n < arrLen; n++) {
			cout << a[i][n] << " ";
		}
		cout << endl;
	}
}

int checkMatrix(int arr1[][arrLen], int arr2[][arrLen]) {
	for (int i = 0; i < arrLen; i++) {
		for (int n = 0; n < arrLen; n++) {
			if (arr1[i][n] != arr2[i][n]) {
				return -1;
			}
		}
	}
    return 1;
}

tyedef int matrix[];

size_t dsize;

int matrix_addition() {

	int a[arrLen][arrLen];
	int b[arrLen][arrLen];
	int c[arrLen][arrLen];

	srand(time(NULL));

	for (int i = 0; i < arrLen; i++) {
		for (int n = 0; n < arrLen; n++) {
			a[i][n] = rand() % 10 + 1;
			b[i][n] = rand() % 10 + 1;
			c[i][n] = 0;
		}
	}

	int(*pA)[arrLen], (*pB)[arrLen], (*pC)[arrLen];

	cudaMalloc((void**)&pA, (arrLen*arrLen)*sizeof(int));
	cudaMalloc((void**)&pB, (arrLen*arrLen)*sizeof(int));
	cudaMalloc((void**)&pC, (arrLen*arrLen)*sizeof(int));

	cudaMemcpy(pA, a, (arrLen*arrLen)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pB, b, (arrLen*arrLen)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pC, c, (arrLen*arrLen)*sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(c, pC, (arrLen*arrLen)*sizeof(int), cudaMemcpyDeviceToHost);

	float time = 0;
    cudaEvent_t start, end;
	// cudaEventCreate(&start);
	// cudaEventCreate(&end);
	// cudaEventRecord(start);

	// auto cTmp = new int[arrLen][arrLen];


	// for (int i = 0; i < arrLen; i++) {
	// 	for (int n = 0; n < arrLen; n++) {
	// 		cTmp[i][n] = a[i][n] + b[i][n];
	// 	}
	// }

    // cudaEventRecord(end);
	// cudaEventSynchronize(end);
	// cudaEventElapsedTime(&time, start, end);
	// cudaEventDestroy(start);
	// cudaEventDestroy(end);
    // cout << "CPU Time: " << time << endl;

    // time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
	dim3 numBlocks((int)ceil(arrLen / (float)threadsPerBlock.x), (int)ceil(arrLen / (float)threadsPerBlock.y));
	addElement << <numBlocks, threadsPerBlock >> >(pA, pB, pC, arrLen);

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	cout << "GPU Element Time: " << time << endl;

    // cudaMemcpy(c, pC, (arrLen*arrLen)*sizeof(float), cudaMemcpyDeviceToHost);

    // int rslt = checkMatrix(c, cTmp);

    // if (rslt) {
    //     cout << "Test Passed" << endl;
    // } else {
    //     cout << "Test Failed" << endl;
    // }

    // time = 0;
	// cudaEventCreate(&start);
	// cudaEventCreate(&end);
	// cudaEventRecord(start);

	// addRow << <numBlocks, threadsPerBlock >> >(pA, pB, pC, arrLen);

	// cudaEventRecord(end);
	// cudaEventSynchronize(end);
	// cudaEventElapsedTime(&time, start, end);
	// cudaEventDestroy(start);
	// cudaEventDestroy(end);
	// cout << "GPU Row Time: " << time << endl;

    // cudaMemcpy(c, pC, (arrLen*arrLen)*sizeof(float), cudaMemcpyDeviceToHost);

    // rslt = checkMatrix(c, cTmp);

    // if (rslt) {
    //     cout << "Test Passed" << endl;
    // } else {
    //     cout << "Test Failed" << endl;
    // }

    // time = 0;
	// cudaEventCreate(&start);
	// cudaEventCreate(&end);
	// cudaEventRecord(start);

	// addCol << <numBlocks, threadsPerBlock >> >(pA, pB, pC, arrLen);

	// cudaEventRecord(end);
	// cudaEventSynchronize(end);
	// cudaEventElapsedTime(&time, start, end);
	// cudaEventDestroy(start);
	// cudaEventDestroy(end);
	// cout << "GPU Col Time: " << time << endl;

	// cudaMemcpy(c, pC, (arrLen*arrLen)*sizeof(float), cudaMemcpyDeviceToHost);

    // rslt = checkMatrix(c, cTmp);

    // if (rslt) {
    //     cout << "Test Passed" << endl;
    // } else {
    //     cout << "Test Failed" << endl;
    // }


	cudaFree(pA);
	cudaFree(pB);
	cudaFree(pC);

	return 0;
}

int main(int argc, char *argv[])
{

	matrix_addition();

	cout << "Done" << endl;

	return 0;
}