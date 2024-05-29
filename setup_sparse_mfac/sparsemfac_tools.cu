#include "utils.h"

__global__ void TestKernel(int n, int *v) {}

int info_cuda(int numThreads, int sharedMemSize, bool verbose) {
	int numBlocksPerSm = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0); // device 0
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, (void*)TestKernel, numThreads, sharedMemSize);

	if(verbose) {
		cout << "Float epsilon: " << std::numeric_limits<float>::epsilon()
			 << "\nDouble epsilon: " << std::numeric_limits<double>::epsilon()
			 << "\n";
		cout << "CUDA Device Properties:"
			 << "\n\tmultiProcessorCount = " << deviceProp.multiProcessorCount
			 << "\n\tnumBlocksPerSm = " << numBlocksPerSm
			 << "\n\ttotal threads = " << numBlocksPerSm * deviceProp.multiProcessorCount * numThreads
			 << "\n"
			 << "\n\tsharedMemPerBlock = " << deviceProp.sharedMemPerBlock
			 << "\n\tsharedMemPerMultiprocessor = " << deviceProp.sharedMemPerMultiprocessor
			 << "\n\twarpSize = " << deviceProp.warpSize
			 << "\n";
	}

	return numBlocksPerSm * deviceProp.multiProcessorCount;
}

