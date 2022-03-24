
#include "cuda_runtime.h"

#include <string.h> 
#include <stdio.h>

#include <iostream>
#include <memory>
#include <string>

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
		printf("  Total number of streaming multiprocessors:        %zu\n",
			deviceProp.multiProcessorCount);
		printf("  Total number of cores:       %d cores\n", _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
               deviceProp.multiProcessorCount);
		printf("  Total amount of constant memory:               %zu bytes\n",
			deviceProp.totalConstMem);
		printf("  Total amount of global memory:       %zu bytes\n",
			deviceProp.totalGlobalMem);
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
	return 0;
}