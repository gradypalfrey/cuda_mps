#include "cuda_runtime.h"

#include <string.h> 
#include <stdio.h>

#include <iostream>
#include <memory>
#include <string>

__global__ void increment_kernel(int *g_data, int inc_value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	g_data[idx] = g_data[idx] + inc_value;
}

int correct_output(int *data, const int n, const int x)
{
	for (int i = 0; i < n; i++)
		if (data[i] != x)
			return 0;
	return 1;
}

int main(int argc, char *argv[])
{
	cudaDeviceProp deviceProps;

	// get device name  
	cudaGetDeviceProperties(&deviceProps, 0);
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
	}
	else {
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	for (int d = 0; d < deviceCount; d++)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, d);
		printf("  Clock Rate:               %0.f gHz\n",
			deviceProp.clockRate * 1e-6f);
		printf("  Total number of streaming multiprocessors:        %zu bytes\n",
			deviceProp.multiProcessorCount);
		printf("  Total number of cores:        %03d cores\n",
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
		printf("  Total amount of constant memory:               %zu bytes\n",
			deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %zu bytes\n",
			deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n",
			deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n",
			deviceProp.warpSize);
		printf("  Maximum number of threads per block:           %d\n",
			deviceProp.maxThreadsPerBlock);
		printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
	}

	int n = 16 * 1024 * 1024;
	int nbytes = n * sizeof(int);
	int value = 26;

	// allocate host memory 
	int *a = 0;
	cudaMallocHost((void**)&a, nbytes);
	//    memset(a, 0, nbytes); 

	// allocate device memory 
	int *d_a = 0;
	cudaMalloc((void**)&d_a, nbytes);
	cudaMemset(d_a, 255, nbytes);
	// set kernel launch configuration 
	dim3 threads = dim3(512, 1);
	dim3 blocks = dim3(n / threads.x, 1);

	// create cuda event handles 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaDeviceSynchronize();
	float gpu_time = 0.0f;

	// asynchronously issue work to the GPU (all to stream 0) 
	cudaEventRecord(start, 0);
	cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
	increment_kernel <<<blocks, threads, 0, 0 >> > (d_a, value);
	cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
	cudaEventRecord(stop, 0);

	// have CPU do some work while waiting for GPU to finish 
	unsigned long int counter = 0;
	while (cudaEventQuery(stop) == cudaErrorNotReady)
	{
		counter++; // Indicates that the CPU is running asynchronously while GPU is executing  
	}

	cudaEventSynchronize(stop);   // stop is updated here 
	cudaEventElapsedTime(&gpu_time, start, stop);   //time difference between start and stop 

	   // print the GPU times 
	printf("time spent executing by the GPU: %.2f\n", gpu_time);
	printf("CPU executed %d iterations while waiting for GPU to finish\n", counter);

	// check the output for correctness 
	bool bFinalResults = (bool)correct_output(a, n, value);



	// release resources 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFreeHost(a);
	cudaFree(d_a);
	cudaDeviceReset();

	return 0;
}