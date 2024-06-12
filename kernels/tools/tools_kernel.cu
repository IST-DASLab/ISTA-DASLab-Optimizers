#include "../utils.h"

__global__ void TestKernel(int n, int *v) { /* empty */}

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

__global__ void zerorize_block_components_kernel_bf16(bfloat16 *vector,
                                                      int16 *indices,
                                                      LL d,
                                                      LL k,
                                                      LL d_block_size,
                                                      LL k_block_size) {
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
            printf("zerorize_block_components was not implemented for float32!\n");
            exit(666);
            break;
    }
    // error checks
	GPU_ERROR_CHECK(cudaGetLastError());
	GPU_ERROR_CHECK(cudaPeekAtLastError());
// 	GPU_ERROR_CHECK(cudaDeviceSynchronize());
}

__global__ void copy_values_large_to_small_kernel_bf16(LL d,
                                                       LL k,
                                                       LL d_block_size,
                                                       LL k_block_size,
                                                       int16 *indices,
                                                       bfloat16 *vector,
                                                       bfloat16 *out) {
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
        out[indices[ki] + d_index_start] = vector[ki]; // V[index,:] = error[I[index, :]]
    }
}
void copy_values_large_to_small_cuda(LL d, LL k, LL d_block_size, LL k_block_size, torch::Tensor indices, torch::Tensor vector, torch::Tensor out) {
//     ASSERT_BF16(vector);
//     ASSERT_BF16(out);
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
            printf("copy_values_large_to_small was not implemented for float32!\n");
            exit(666);
            break;
    }
    // error checks
	GPU_ERROR_CHECK(cudaGetLastError());
	GPU_ERROR_CHECK(cudaPeekAtLastError());
// 	GPU_ERROR_CHECK(cudaDeviceSynchronize());
}

__global__ void copy_values_small_to_large_kernel_bf16(LL d,
                                                       LL k,
                                                       LL d_block_size,
                                                       LL k_block_size,
                                                       int16 *indices,
                                                       bfloat16 *vector,
                                                       bfloat16 *out) {
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
        out[ki] = vector[indices[ki] + d_index_start]; // out[I[index]] = vector
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
            printf("copy_values_small_to_large was not implemented for float32!\n");
            exit(666);
            break;
    }
    // error checks
	GPU_ERROR_CHECK(cudaGetLastError());
	GPU_ERROR_CHECK(cudaPeekAtLastError());
// 	GPU_ERROR_CHECK(cudaDeviceSynchronize());
}

// __global__ void copy_values_kernel_inp_bf16_out_bf16(LL d,
//                                                      LL k,
//                                                      LL d_block_size,
//                                                      LL k_block_size,
//                                                      int16* indices,
//                                                      bfloat16* inp,
//                                                      bfloat16* out,
//                                                      int direction) {
//     /*
//         direction == COPY_DIRECTION_k2d: input is k-dim and output is d-dim
//         direction == COPY_DIRECTION_d2k: input is d-dim and output is k-dim
//     */
// 	const LL Bid = blockIdx.x; // block id
// 	const LL Tid = threadIdx.x; // thread id
//
//     LL k_index_start = Bid * k_block_size;
//     LL k_index_end = min(k_index_start + k_block_size, k);
//     LL ki = k_index_start + Tid;
//     // we create the number of threads based on k_block_size and we have to test if thread index (ki) is not outside of the bounds
//     if(ki < k_index_end) { // Tid is the index for indices
//         LL d_index_start = Bid * d_block_size;
//         LL di = d_index_start + indices[ki];
//         LL inp_index = (direction == COPY_DIRECTION_k2d) ? ki : di;
//         LL out_index = (direction == COPY_DIRECTION_k2d) ? di : ki;
//         out[out_index] = inp[inp_index];
//     }
// }
__global__ void copy_values_kernel(LL d,
                                   LL k,
                                   LL d_block_size,
                                   LL k_block_size,
                                   int16* indices,
                                   void* inp,
                                   void* out,
                                   int direction,
                                   int inp_bits,
                                   int out_bits) {
    /*
        direction == COPY_DIRECTION_k2d: input is k-dim and output is d-dim
        direction == COPY_DIRECTION_d2k: input is d-dim and output is k-dim
    */
	const LL Bid = blockIdx.x; // block id
	const LL Tid = threadIdx.x; // thread id

    LL k_index_start = Bid * k_block_size;
    LL k_index_end = min(k_index_start + k_block_size, k);
    LL ki = k_index_start + Tid;
    // we create the number of threads based on k_block_size and we have to test if thread index (ki) is not outside of the bounds
    if(ki < k_index_end) { // Tid is the index for indices
        LL d_index_start = Bid * d_block_size;
        LL di = d_index_start + indices[ki];
        LL inp_index = (direction == COPY_DIRECTION_k2d) ? ki : di;
        LL out_index = (direction == COPY_DIRECTION_k2d) ? di : ki;

        dynamically_assign(out, inp, out_index, inp_index, out_bits, inp_bits);
    }
}
void copy_values_cuda(LL d,
                      LL k,
                      LL d_block_size,
                      LL k_block_size,
                      torch::Tensor indices,
                      torch::Tensor inp,
                      torch::Tensor out,
                      int direction) {

    LL blocks = div_inc(k, k_block_size);
    LL threads = get_threads(k_block_size);

    // we know that inp and out have either Float or BFloat16 type
    int inp_bits = (torch::ScalarType::BFloat16 == inp.scalar_type())? 16 : 32;
    int out_bits = (torch::ScalarType::BFloat16 == out.scalar_type())? 16 : 32;

    copy_values_kernel<<<blocks, threads>>>(d,
                                            k,
                                            d_block_size,
                                            k_block_size,
                                            (int16*) indices.data_ptr(),
                                            inp.data_ptr(),
                                            out.data_ptr(),
                                            direction,
                                            inp_bits,
                                            out_bits);

//     copy_values_kernel_inp_bf16_out_bf16<<<blocks, threads>>>(d,
//                                                               k,
//                                                               d_block_size,
//                                                               k_block_size,
//                                                               (int16*) indices.data_ptr(),
//                                                               (bfloat16*) inp.data_ptr(),
//                                                               (bfloat16*) out.data_ptr(),
//                                                               direction);

    // error checks
	GPU_ERROR_CHECK(cudaGetLastError());
	GPU_ERROR_CHECK(cudaPeekAtLastError());
// 	GPU_ERROR_CHECK(cudaDeviceSynchronize());
}