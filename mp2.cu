
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string.h> 
#include <stdio.h>

#include <iostream>
#include <memory>
#include <string>

#include <stdlib.h>

using namespace std;

#define BLOCKDIM 16
#define arrLen 16

__global__ void addElement(int a[][arrLen], int b[][arrLen], int c[][arrLen], int length) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < length && j < length)
		c[i][j] = a[i][j] + b[i][j];
}

__global__ void addRow(int a[][arrLen], int b[][arrLen], int c[][arrLen], int length) {


}

__global__ void addCol(int a[][arrLen], int b[][arrLen], int c[][arrLen], int length) {

}

int matrix_addition() {

	int a[arrLen][arrLen];
	int b[arrLen][arrLen];
	int c[arrLen][arrLen];

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


	dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
	dim3 numBlocks((int)ceil(arrLen / (float)threadsPerBlock.x), (int)ceil(arrLen / (float)threadsPerBlock.y));
	addElement << <numBlocks, threadsPerBlock >> >(pA, pB, pC, arrLen);

	cudaMemcpy(c, pC, (arrLen*arrLen)*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < arrLen; i++) {
		for (int n = 0; n < arrLen; n++) {
			cout << a[i][n] << " ";
		}
		cout << endl;
	}
	cout << "end a" << endl;

	for (int i = 0; i < arrLen; i++) {
		for (int n = 0; n < arrLen; n++) {
			cout << b[i][n] << " ";
		}
		cout << endl;
	}
	cout << "end b" << endl;

	for (int i = 0; i < arrLen; i++) {
		for (int n = 0; n < arrLen; n++) {
			cout << c[i][n] << " ";
		}
		cout << endl;
	}

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