#include "../utils.h"

__device__ void parallel_reduce(float *mem, long T, long logT, long Tid, long offset=0, bool zerorize=false) {
	/*
		Perform parallel reduce in logarithmic time over the vector mem with T threads (mem has T components).
		If zerorize=true, then set the components of mem to zero after accumulation.
		Use offset > 0 to perform the parallel reduction over a different sequence of size T in mem
		Tid is the current thread id and logT is log2(T).
		Return the sum, which will be located at mem[offset]

		Resources:
			https://shreeraman-ak.medium.com/parallel-reduction-with-cuda-d0ae10c1ae2c
	*/
	long mid = T >> 1; // half of number of threads
	long offset_PLUS_Tid = offset + Tid;
//	if(offset_PLUS_Tid >= T) printf("[parallel_reduce] offset_PLUS_Tid=%d >= T=%d", offset_PLUS_Tid, T);
	for(long i = 0; i < logT; ++i) { // perform log2(T) rounds of accumulation
		__syncthreads();
		if(Tid < mid) { // left half accumulates, right half sends to left half
//			if(offset_PLUS_Tid+mid >= T) printf("[parallel_reduce] offset_PLUS_Tid+mid=%d+%d=%d >= T=%d", offset_PLUS_Tid, mid, offset_PLUS_Tid+mid, T);
			mem[offset_PLUS_Tid] += mem[offset_PLUS_Tid + mid];
			if(zerorize) {
				mem[offset_PLUS_Tid + mid] = 0.;
			}
		}
		mid >>= 1;
	}
}

__global__ void SP_v23(long d,
                       long m,
                       long k,
                       long d_block_size,
                       long k_block_size,
                       float* g,
                       int16* indices,
                       float* values,
                       float* out) {
	/*
		Scalar Products version #2.3:
			Float type for values
			1 block / row
			CMA
			Logarithmic Reduction
		This kernel uses int16 for the indices and we need to compute an offset to convert the relative indices (int16) to absolute indices.
	*/
	extern __shared__ float mem[]; // has T elements
	// const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id
	const long logT = log_threads(T);
	// const long num_threads = B * T; // total number of threads across all blocks
	// const long id = Bid * T + Tid; // this is the thread id when counting the threads from 0 to num_threads-1 from all blocks

	long i;
	long row = Bid; // each block processes one row
	long thread_processing_size = div_inc(k, T); // this is how many elements a thread processes per row

	// compute the pointer for the current row in indices and values to save up some indexing computations (compute offsets to current row)
	long row_start = row * k;
	long row_end = row_start + k;
	// long thread_start_index = row_start + Tid * thread_processing_size;
	// long thread_end_index = thread_start_index + thread_processing_size;

	double sum = 0.0f;

	/*
	    Explanation for how we should compute an offset to be added to an int16 index:
	    Let's suppose we have the following settings:
	    - d = 20: model size
	    - d_block_size = 10: apply top-k in blocks if this size, let's say with density 50%
	    - k_block_size = 5
	    - k = 10: 2 blocks in total
	    - this is how a row from the indices matrix (I) looks like, where the values are in the interval [0, d_block_size=10]:

	       index i       : 0 1 2 3 4 | 5   6  7  8  9 (in the main for loop where we compute the sum)
	    indices[i]       : 0 2 4 6 8 | 1   3  5  7  9
	    indices[i]+offset: 0 2 4 6 8 | 11 13 15 17 19

	    absolute indices of 1st block:     0 1 2 3 4 5 6 7 8 9
	    selected indices in the 1st block: ^   ^   ^   ^   ^

	    absolute indices of 2nd block:     10 11 12 13 14 15 16 17 18 19
	    selected indices in the 2nd block:     ^     ^     ^     ^     ^

	    indices[i] will always be between [0, d_block_size] and we have to compute an offset to convert them between [0, d]

	    We can see that the offset is topk_block_id * d_block_size
	    How to determine topk_block_id given the index i that's between [0, k]:

	        topk_block_id = i / k_block_size

	    Now, we have to account for row number because i loops through a particular row:
	        topk_block_id = (i - row_start) / k_block_size

	    In the end, the offset is:
	        offset = topk_block_id * d_block_size
	*/
	long offset;

	for(i = row_start + Tid; i < row_end; i += T) { // start from Tid and increase with T because we have T threads per block
        offset = ((i - row_start) / k_block_size) * d_block_size;
		sum += (values[i] * g[indices[i] + offset]);
	}
	mem[Tid] = sum;

	parallel_reduce(mem, T, logT, Tid, 0, false);

	if(Tid == 0) {
		out[Bid] = mem[0];
	}
}

__global__ void
__launch_bounds__(1024) // 3 possible params, in this order: maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster
SP_v23_bf16(long d,
            long m,
            long k,
            long d_block_size,
            long k_block_size,
            bfloat16* g,
            int16* indices,
            bfloat16* values,
            float* out) {
	/*
		Scalar Products version #2.3:
			bfloat16 for values
			1 block / row
			CMA
			Logarithmic Reduction
	*/
	extern __shared__ float mem[]; // has T elements
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id
	const long logT = log_threads(T);

	long i;
	long row = Bid; // each block processes one row

	// compute the pointer for the current row in indices and values
	long row_start = row * k;
	long row_end = row_start + k;

	long blocks_count = div_inc(d, d_block_size);

	double sum = 0.0f;

	/*
	    Explanation for how we should compute an offset to be added to an int16 index:
	    Let's suppose we have the following settings:
	    - d = 20: model size
	    - d_block_size = 10: apply top-k in blocks if this size, let's say with density 50%
	    - k_block_size = 5
	    - k = 10: 2 blocks in total
	    - this is how a row from the indices matrix (I) looks like, where the values are in the interval [0, d_block_size=10]:

	       index i       : 0 1 2 3 4 | 5   6  7  8  9 (in the main for loop where we compute the sum)
	    indices[i]       : 0 2 4 6 8 | 1   3  5  7  9
	    indices[i]+offset: 0 2 4 6 8 | 11 13 15 17 19

	    absolute indices of 1st block:     0 1 2 3 4 5 6 7 8 9
	    selected indices in the 1st block: ^   ^   ^   ^   ^

	    absolute indices of 2nd block:     10 11 12 13 14 15 16 17 18 19
	    selected indices in the 2nd block:     ^     ^     ^     ^     ^

	    indices[i] will always be between [0, d_block_size] and we have to compute an offset to convert them between [0, d]

	    We can see that the offset is topk_block_id * d_block_size
	    How to determine topk_block_id given the index i that's between [0, k]:

	        topk_block_id = i / k_block_size

	    Now, we have to account for row number because i loops through a particular row:
	        topk_block_id = (i - row_start) / k_block_size

	    In the end, the offset is:
	        offset = topk_block_id * d_block_size
	*/
	long offset;
	int16 ind;
	for(i = row_start + Tid; i < row_end; i += T) { // start from Tid and increase with T because we have T threads per block
        offset = ((i - row_start) / k_block_size) * d_block_size;
        ind = indices[i];
        if(ind + offset < d) {
		    sum += (__bfloat162float(values[i]) * __bfloat162float(g[ind + offset]));
		}
	}
	/*
	for(i = row_start + Tid; i < row_end; i += T) { // start from Tid and increase with T because we have T threads per block
        offset = ((i - row_start) / k_block_size) * d_block_size;
		sum += (__bfloat162float(values[i]) * __bfloat162float(g[indices[i] + offset]));
	}
    */
	mem[Tid] = sum;

	parallel_reduce(mem, T, logT, Tid, 0, false);
	if(Tid == 0) {
		out[Bid] = mem[0];
	}
}

void SP_cuda(int blocks,
			 int threads,
			 int version,
			 long d,
			 long m,
			 long k,
			 long d_block_size,
			 long k_block_size,
			 torch::Tensor g,
			 torch::Tensor indices,
			 torch::Tensor values,
			 torch::Tensor out,
			 int use_bf16) {
	assert(version == 23);
    long sharedMemSize = threads * sizeof(float);

//     printf("[KERNEL use_bf16=%d] sharedMemSize=%ld, d=%ld, m=%ld, k=%ld, d_block_size=%ld, k_block_size=%ld\n",
//         use_bf16, sharedMemSize, d, m, k, d_block_size, k_block_size);

	if (use_bf16 == 1) {
// 	    printf("[KERNEL] using SP_v23_bf16\n");
        SP_v23_bf16<<<blocks, threads, sharedMemSize>>>(d,
                                                        m,
                                                        k,
                                                        d_block_size,
                                                        k_block_size,
                                                        (bfloat16*) g.data_ptr(),
                                                        (int16*) indices.data_ptr(),
                                                        (bfloat16*) values.data_ptr(),
                                                        (float*) out.data_ptr());
	} else { // float values
// 	    printf("[KERNEL] using SP_v23\n");
        SP_v23<<<blocks, threads, sharedMemSize>>>(d,
                                                   m,
                                                   k,
                                                   d_block_size,
                                                   k_block_size,
                                                   (float*) g.data_ptr(),
                                                   (int16*) indices.data_ptr(),
                                                   (float*) values.data_ptr(),
                                                   (float*) out.data_ptr());
	}

	GPU_ERROR_CHECK(cudaGetLastError());
	GPU_ERROR_CHECK(cudaPeekAtLastError());
	GPU_ERROR_CHECK(cudaDeviceSynchronize());
}
