
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
#define arrLen 4096

__global__ void addElement(int *a, int *b, int *c, int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < length && n < length) {
		int idx = i + n*arrLen;
		c[idx] = a[idx] + b[idx];
	}
}

__global__ void addRow(int *a, int *b, int *c, int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (int n = 0; n < length; n++) {
		if (i < length && n < length) {
			int idx = i + n*arrLen;
			c[idx] = a[idx] + b[idx];
		}
	}
}

__global__ void addCol(int *a, int *b, int *c, int length) {
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < length; i++) {
		if (i < length && n < length) {
			int idx = i + n*arrLen;
			c[idx] = a[idx] + b[idx];
		}
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

	cudaMalloc((void**)&pA, size);
	cudaMalloc((void**)&pB, size);
	cudaMalloc((void**)&pC, size);

	cudaMemcpy(pA, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(pB, b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(pC, c, size, cudaMemcpyHostToDevice);

	// cudaMemcpy(c, pC, size, cudaMemcpyDeviceToHost);

	cout << "Start CPU" << endl;

	float time = 0;
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);


	for (int i = 0; i < arrLen*arrLen; i++) {
		cTmp[i] = a[i] + b[i];
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
	addElement << <numBlocks, threadsPerBlock >> >(pA, pB, pC, arrLen);

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	cout << "GPU Element Time: " << time << endl;

	cudaMemcpy(c, pC, (arrLen*arrLen)*sizeof(float), cudaMemcpyDeviceToHost);

	int rslt = checkMatrix(c, cTmp);

	if (rslt) {
		cout << "Test Passed" << endl;
	}
	else {
		cout << "Test Failed" << endl;
	}

	time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	dim3 numBlocks1((int)ceil(arrLen / (float)threadsPerBlock.x), 1);
	addRow << <numBlocks1, threadsPerBlock >> >(pA, pB, pC, arrLen);

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	cout << "GPU Row Time: " << time << endl;

	cudaMemcpy(c, pC, (arrLen*arrLen)*sizeof(float), cudaMemcpyDeviceToHost);

	rslt = checkMatrix(c, cTmp);

	if (rslt) {
		cout << "Test Passed" << endl;
	}
	else {
		cout << "Test Failed" << endl;
	}

	time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	dim3 numBlocks2(1, (int)ceil(arrLen / (float)threadsPerBlock.x));
	addCol << <numBlocks2, threadsPerBlock >> >(pA, pB, pC, arrLen);

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	cout << "GPU Col Time: " << time << endl;

	cudaMemcpy(c, pC, (arrLen*arrLen)*sizeof(float), cudaMemcpyDeviceToHost);

	rslt = checkMatrix(c, cTmp);

	if (rslt) {
		cout << "Test Passed" << endl;
	}
	else {
		cout << "Test Failed" << endl;
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