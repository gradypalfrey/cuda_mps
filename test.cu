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
__global__ void tile(int *a, int *b, int *c, int length)
{
 __shared__ int m1[TILEWIDTH][TILEWIDTH];
 __shared__ int m2[TILEWIDTH][TILEWIDTH];
 int i = threadIdx.x + TILEWIDTH * blockIdx.x;
 int n = threadIdx.y + TILEWIDTH * blockIdx.y;
 int curr = 0;
 for (int j = 0; j < arrLen / TILEWIDTH; j)
 {
 if (i < arrLen && n < arrLen)
 {
 m1[threadIdx.y][threadIdx.x] = a[n * arrLen + (j * TILEWIDTH + threadIdx.x)];
 m2[threadIdx.y][threadIdx.x] = b[i + arrLen * (j * TILEWIDTH + threadIdx.y)];
 __syncthreads();
 }
 else
 {
 m1[threadIdx.y][threadIdx.x] = 0;
 m2[threadIdx.y][threadIdx.x] = 0;
 __syncthreads();
 }
 for (int h = 0; h < TILEWIDTH; h++)
 {
 curr += m1[threadIdx.y][h] * m2[h][threadIdx.x];
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
int mult(int *a, int *b, int *c)
{
 for (int i = 0; i < arrLen; i++) {
 for (int n = 0; n < arrLen; n++) {
 for (int j = 0; j < arrLen; j++) {
 c[i + n*arrLen] += a[i + j*arrLen] * b[j + n*arrLen];
 }
 }
 }
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
 cTmp[i] = 0;
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
 mult(a, b, cTmp);
 cudaEventRecord(end);
 cudaEventSynchronize(end);
 cudaEventElapsedTime(&time, start, end);
 cudaEventDestroy(start);
 cudaEventDestroy(end);
 cout << "CPU Element Time: " << time << endl;
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
 time = 0;
 cudaEventCreate(&start);
 cudaEventCreate(&end);
 cudaEventRecord(start);
 dim3 threadsPerBlock(TILEWIDTH, TILEWIDTH);
 dim3 numBlocks((int)ceil(arrLen / (float)TILEWIDTH), (int)ceil(arrLen / (float)TILEWIDTH));
 tile<<<numBlocks, threadsPerBlock>>>(pA, pB, pC, arrLen);
 cudaEventRecord(end);
 cudaEventSynchronize(end);
 cudaEventElapsedTime(&time, start, end);
 cudaEventDestroy(start);
 cudaEventDestroy(end);
 cout << "GPU Element Time: " << time << endl;
 cudaMemcpy(c, pC, (arrLen * arrLen) * sizeof(float), cudaMemcpyDeviceToHost);
 rslt = checkMatrix(c, cTmp);
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