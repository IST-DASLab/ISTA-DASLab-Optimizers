#include "../utils.h"

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

int get_max_floats_for_shared_memory_per_thread_block_cuda() {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0); // device 0

	// deviceProp.sharedMemPerBlock returns the maximum size (in bytes) for the shared memory for a single thread block
	// int maxSharedMemSize_floats = deviceProp.sharedMemPerBlock / sizeof(float); // retrieves shared memory size for static allocation
	int maxSharedMemSizePerSM_bytes = deviceProp.sharedMemPerMultiprocessor;
	int maxSharedMemSizePerSM_kilobytes = maxSharedMemSizePerSM_bytes / 1024;
	int floats_count = (maxSharedMemSizePerSM_kilobytes - 1) * 1024 / sizeof(float); // retrieves shared memory size for dynamic allocation
	// check https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf page 449: Maximum amount of shared memory per thread block^32
	printf("[CUDA Kernel] maxSharedMemSizePerSM_bytes = %d, maxSharedMemSizePerSM_kilobytes = %d, floats_count = %d\n",
		maxSharedMemSizePerSM_bytes, maxSharedMemSizePerSM_kilobytes, floats_count);
	return floats_count;
}

int get_sm_count_cuda() {
	int numBlocksPerSm = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0); // device 0
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, (void*)TestKernel, 128, 0);
	return deviceProp.multiProcessorCount;
}

__global__ void zerorize_block_components_kernel_bf16(bfloat16 *vector, int16 *indices, LL d, LL k, LL d_block_size, LL k_block_size) {
// 	const LL B = gridDim.x; // number of blocks
	const LL Bid = blockIdx.x; // block id
// 	const LL T = blockDim.x; // number of threads
	const LL Tid = threadIdx.x; // thread id

    LL d_index_start = Bid * d_block_size;
//     LL d_index_end = min(d_index_start + d_block_size, d);

    LL k_index_start = Bid * k_block_size;
    LL k_index_end = min(k_index_start + k_block_size, k);

    bfloat16 zero = __float2bfloat16(0.0f);

    LL ki = Tid + k_index_start;

    if(ki < k_index_end) { // Tid is the index for indices
        // to obtain the index to be used in vector, we have to add the block offset given by d_index_start
        vector[indices[ki] + d_index_start] = zero;
    }
}
__global__ void zerorize_block_components_kernel_f32(float *vector, int16 *indices, LL d, LL k, LL d_block_size, LL k_block_size) {
// 	const LL B = gridDim.x; // number of blocks
	const LL Bid = blockIdx.x; // block id
// 	const LL T = blockDim.x; // number of threads
	const LL Tid = threadIdx.x; // thread id

    LL d_index_start = Bid * d_block_size;
//     LL d_index_end = min(d_index_start + d_block_size, d);

    LL k_index_start = Bid * k_block_size;
    LL k_index_end = min(k_index_start + k_block_size, k);

    float zero = 0.0f;

    LL ki = Tid + k_index_start;

    if(ki < k_index_end) { // Tid is the index for indices
        // to obtain the index to be used in vector, we have to add the block offset given by d_index_start
        vector[indices[ki] + d_index_start] = zero;
    }
}
void zerorize_block_components_cuda(torch::Tensor vector, torch::Tensor indices, LL d, LL k, LL d_block_size, LL k_block_size) {
    assert(k_block_size <= 1024);
    LL blocks = 1 + (LL)(k / k_block_size);

    // determine the number of threads as the first power of 2 larger than or equal to k_block_size
    LL threads = 1;
    while(threads < k_block_size) {
        threads <<= 1;
    }
//     LL threads = k_block_size;
//     printf("Using %ld blocks and %ld threads.\n", blocks, threads);
    switch(vector.scalar_type()) {
        case torch::ScalarType::BFloat16:
            zerorize_block_components_kernel_bf16<<<blocks, threads>>>((bfloat16*) vector.data_ptr(),
                                                                      (int16*) indices.data_ptr(),
                                                                      d,
                                                                      k,
                                                                      d_block_size,
                                                                      k_block_size);
            break;
        case torch::ScalarType::Float:
            zerorize_block_components_kernel_f32<<<blocks, threads>>>((float*) vector.data_ptr(),
                                                                      (int16*) indices.data_ptr(),
                                                                      d,
                                                                      k,
                                                                      d_block_size,
                                                                      k_block_size);
            break;
    }
    // error checks
	gpuErrorCheck(cudaGetLastError());
	gpuErrorCheck(cudaPeekAtLastError());
// 	gpuErrorCheck(cudaDeviceSynchronize());
}

__global__ void copy_values_large_to_small_kernel_bf16(LL d, LL k, LL d_block_size, LL k_block_size, int16 *indices, bfloat16 *vector, bfloat16 *out) {
    /*
        This kernel performs the operation out = vector[indices], where `indices` contains int16 values representing the
    relative indices in each block of size d_block_size, having k_block_size (last block might be shorter).
        Dimensions:
        - indices: size k
        - vector: size d
        - out: size k
    */
    // const LL B = gridDim.x; // number of blocks
	const LL Bid = blockIdx.x; // block id
// 	const LL T = blockDim.x; // number of threads
	const LL Tid = threadIdx.x; // thread id

    LL d_index_start = Bid * d_block_size;
    //LL d_index_end = min(d_index_start + d_block_size, d);

    LL k_index_start = Bid * k_block_size;
    LL k_index_end = min(k_index_start + k_block_size, k);

    LL ki = Tid + k_index_start;

    if(ki < k_index_end) { // Tid is the index for indices
        out[indices[ki] + d_index_start] = vector[ki];
//         printf("Bid=%ld, Tid=%ld, d_index_start=%ld, k_index_start=%ld, k_index_end=%ld, i=%ld, pos=%ld\n",
//             Bid, Tid, d_index_start, k_index_start, k_index_end, i, i + d_index_start);
    }
}
__global__ void copy_values_large_to_small_kernel_f32(LL d, LL k, LL d_block_size, LL k_block_size, int16 *indices, float *vector, bfloat16 *out) {
    /*
        This kernel performs the operation out = vector[indices], where `indices` contains int16 values representing the
    relative indices in each block of size d_block_size, having k_block_size (last block might be shorter).
        Dimensions:
        - indices: size k
        - vector: size d
        - out: size k
    */
    // const LL B = gridDim.x; // number of blocks
	const LL Bid = blockIdx.x; // block id
// 	const LL T = blockDim.x; // number of threads
	const LL Tid = threadIdx.x; // thread id

    LL d_index_start = Bid * d_block_size;
    //LL d_index_end = min(d_index_start + d_block_size, d);

    LL k_index_start = Bid * k_block_size;
    LL k_index_end = min(k_index_start + k_block_size, k);

    LL ki = Tid + k_index_start;

    if(ki < k_index_end) { // Tid is the index for indices
        out[indices[ki] + d_index_start] = vector[ki];
//         printf("Bid=%ld, Tid=%ld, d_index_start=%ld, k_index_start=%ld, k_index_end=%ld, i=%ld, pos=%ld\n",
//             Bid, Tid, d_index_start, k_index_start, k_index_end, i, i + d_index_start);
    }
}
void copy_values_large_to_small_cuda(LL d, LL k, LL d_block_size, LL k_block_size, torch::Tensor indices, torch::Tensor vector, torch::Tensor out) {
    assert(k_block_size <= 1024);
    LL blocks = 1 + (LL)(k / k_block_size);

    // determine the number of threads as the first power of 2 larger than or equal to k_block_size
    LL threads = 1;
    while(threads < k_block_size) {
        threads <<= 1;
    }

    switch(vector.scalar_type()) {
        case torch::ScalarType::BFloat16:
            copy_values_large_to_small_kernel_bf16<<<blocks, threads>>>(d,
                                                                            k,
                                                                            d_block_size,
                                                                            k_block_size,
                                                                            (int16*) indices.data_ptr(),
                                                                            (bfloat16*) vector.data_ptr(),
                                                                            (bfloat16*) out.data_ptr());
            break;
        case torch::ScalarType::Float:
            copy_values_large_to_small_kernel_f32<<<blocks, threads>>>(d,
                                                                       k,
                                                                       d_block_size,
                                                                       k_block_size,
                                                                       (int16*) indices.data_ptr(),
                                                                       (float*) vector.data_ptr(),
                                                                       (bfloat16*) out.data_ptr());
            break;
    }
    // error checks
	gpuErrorCheck(cudaGetLastError());
	gpuErrorCheck(cudaPeekAtLastError());
// 	gpuErrorCheck(cudaDeviceSynchronize());
}

__global__ void copy_values_small_to_large_kernel_bf16(LL d, LL k, LL d_block_size, LL k_block_size, int16 *indices, bfloat16 *vector,  bfloat16 *out) {
    /*
        Explaining the kernel name:
        - the values from a vector of size k (small) (vector) will be copied to a vector of size d (large) (out)
        - this method is used in SparseMFAC to get the d-dimensional vector that contains the Top-K values and the zeros used
        for preconditioning

        This kernel performs the operation out = vector[indices], where `indices` contains int16 values representing the
    relative indices in each block of size d_block_size, having k_block_size (last block might be shorter).
        Dimensions:
        - indices: size k
        - vector: size k
        - out: size d
    */
    // const LL B = gridDim.x; // number of blocks
	const LL Bid = blockIdx.x; // block id
 	// const LL T = blockDim.x; // number of threads
	const LL Tid = threadIdx.x; // thread id

    LL d_index_start = Bid * d_block_size;
    //LL d_index_end = min(d_index_start + d_block_size, d);

    LL k_index_start = Bid * k_block_size;
    LL k_index_end = min(k_index_start + k_block_size, k);

    LL ki = Tid + k_index_start;

    if(ki < k_index_end) { // Tid is the index for indices
        out[ki] = vector[indices[ki] + d_index_start];
    }
}
void copy_values_small_to_large_cuda(LL d, LL k, LL d_block_size, LL k_block_size, torch::Tensor indices, torch::Tensor vector, torch::Tensor out) {
    assert(k_block_size <= 1024);
    LL blocks = 1 + (LL)(k / k_block_size);

    // determine the number of threads as the first power of 2 larger than or equal to k_block_size
    LL threads = 1;
    while(threads < k_block_size) {
        threads <<= 1;
    }

    switch(vector.scalar_type()) {
        case torch::ScalarType::BFloat16:
            copy_values_small_to_large_kernel_bf16<<<blocks, threads>>>(d,
                                                                        k,
                                                                        d_block_size,
                                                                        k_block_size,
                                                                        (int16*) indices.data_ptr(),
                                                                        (bfloat16*) vector.data_ptr(),
                                                                        (bfloat16*) out.data_ptr());
            break;
        case torch::ScalarType::Float:
            printf("copy_values_small_to_large_cuda was not implemented for float32!\n");
            exit(666);
//             copy_values_small_to_large_kernel_f32<<<blocks, threads>>>(d,
//                                                                             k,
//                                                                             d_block_size,
//                                                                             k_block_size,
//                                                                             (int16*) indices.data_ptr(),
//                                                                             (float*) vector.data_ptr(),
//                                                                             (bfloat16*) out.data_ptr());
            break;
    }
    // error checks
	gpuErrorCheck(cudaGetLastError());
	gpuErrorCheck(cudaPeekAtLastError());
// 	gpuErrorCheck(cudaDeviceSynchronize());
}
