#include "../utils.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////// LINEAR COMBINATION OF GRADIENTS (LCG)
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void LCG_v51(long d,
                        long m,
                        long k,
                        long d_block_size,
                        long k_block_size,
                        float* coef,
                        int16* indices,
                        float* values,
                        float* out) {
	/*
		Linear Combination of Gradients v5.1: (v51)
			One CUDA block processes a slice of size (m, k_block_size) columns and computes the linear combination of all m gradients using Shared Memory
			This way, after one thread block processes its current slice, the shared memory will contain the LCG for a slice of size k_block_size
			The number of thread blocks is chosen dynamically such that the GPU is filled and there are no thread blocks waiting in the queue
			A few more details:
				Top-K is applied in blocks of size d_block_size in each row and it results in k_block_size values (k_block_size = 1% of d_block_size)
				We know that in any range of size d_block_size there are k_block_size non-zero values, scattered more or less randomly
				We use the shared memory of size d_block_size to accumulate linear combinations in it and in the end write the result to the output at the right index locations
	*/
	extern __shared__ float mem[];

	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id

	for(long i = Tid; i < d_block_size; i += T) { // mem init because we will add values to this buffer
		mem[i] = 0;
	}
	__syncthreads();

	long blocks_count = div_inc(d, d_block_size); // how many k-blocks and d-blocks we have (the same number for both)
	long blocks_per_thread_block = div_inc(blocks_count, B);

	long idx_slice; // used in the first for loop to indicate the current block id (k-block or d-block)
	long d_block_start; // start index for a d-block
	long d_block_end; // end index of a d-block

	long row; // the row index
	long row_mul_k; // holds row * k by starting with 0 and adding k at each row (addition takes less cycles than multiplication)

	float c; // the coefficient for each row TODO: save it in the shared memory using 4-element reads, at the beginning (adjust shared memory size for this)
	long k_block_start; // start index for a k-block
	long k_block_end; // end index of a k-block

	long k_col; // column index to extract data from indices and values at the current row
	long index; // the 1-D index to extract data from indices and values at the current row (row * k + k_col)
	long ind; // the data from indices at the index "index"
	float val; // the data from values at the index "index"

	// save the following quantities to be used next
	long Bid_MUL_blocksPerThreadBlock = Bid * blocks_per_thread_block;
	long dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = d_block_size * Bid_MUL_blocksPerThreadBlock;
	long kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = k_block_size * Bid_MUL_blocksPerThreadBlock;

	long d_sub_d_block_size;

	for(idx_slice = 0; idx_slice < blocks_per_thread_block; ++idx_slice) { // iterate through all slices (or blocks) of size (m, k_block_size) that will be processed by the current thread block
		d_block_start = dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + d_block_size * idx_slice; // start index for the current d-block
		d_block_end = min(d, d_block_start + d_block_size); // end index for the current d-block
		k_block_start = kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + k_block_size * idx_slice; // start index for the current k-block
		k_block_end = min(k, k_block_start + k_block_size); // end index for the current k-block
		d_sub_d_block_size = d - d_block_start; // perform this subtraction only once, it will be used below to compute some offsets

		for(row = 0, row_mul_k = 0; row < m; ++row, row_mul_k += k) { // iterate through all rows in the current slice of size (m, k_block_size) to compute the LCG
			c = coef[row]; // maybe this read can be improved by saving it into the shared memory? see LCG_v53_bf16_cached_coef

			for(k_col = k_block_start + Tid; k_col < k_block_end && k_col < d_block_end; k_col += T) { // iterate through all k_block_size columns in the slice of size (m, k_block_size)
				index = row_mul_k + k_col;
				ind = indices[index];
				val = values[index];

				/*old (int32)*/ // mem[ind - d_block_start] += c * val; // mem update: this is where the LCG is performed
				/*new (int16)*/ mem[ind] += c * val; // mem update: this is where the LCG is performed
			}
			__syncthreads(); // make sure all threads finish one row to avoid race conditions
		}

		// now, mem contains the LCG for the slice (m, k_block_size). Actually, the results were scattered in a range of size d_block_size in which we only had k_block_size values (k_block_size = 1% of d_block_size)
		for(long i = Tid; i < d_block_size && i < d_sub_d_block_size; i += T) { // mem dump: i < d_sub_d_block_size is required to avoid illegal memory access
			out[i + d_block_start] = mem[i]; // write the content of mem to the right location in out
			mem[i] = 0; // then zerorize mem to prepare for the next block/slice
		}
		__syncthreads(); // make sure all threads finish writing to out
	}
}

__global__ void LCG_v51_bf16(long d,
                             long m,
                             long k,
                             long d_block_size,
                             long k_block_size,
                             float* coef,
                             int16* indices,
                             bfloat16* values,
                             bfloat16* out) {
	/*
		Linear Combination of Gradients v5.1: (v51)
			One CUDA block processes a slice of size (m, k_block_size) columns and computes the linear combination of all m gradients using Shared Memory
			This way, after one thread block processes its current slice, the shared memory will contain the LCG for a slice of size k_block_size
			The number of thread blocks is chosen dynamically such that the GPU is filled and there are no thread blocks waiting in the queue
			A few more details:
				Top-K is applied in blocks of size d_block_size in each row and it results in k_block_size values (k_block_size = 1% of d_block_size)
				We know that in any range of size d_block_size there are k_block_size non-zero values, scattered more or less randomly
				We use the shared memory of size d_block_size to accumulate linear combinations in it and in the end write the result to the output at the right index locations
	*/
	extern __shared__ float mem[];

	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id

	for(long i = Tid; i < d_block_size; i += T) { // mem init because we will add values to this buffer
		mem[i] = 0;
	}
	__syncthreads();

	// printf("In the kernel:\n");
	// printf("d=%ld, m=%ld, k=%ld, d_block_size=%ld, k_block_size=%ld\n", d, m, k, d_block_size, k_block_size);

	long blocks_count = div_inc(d, d_block_size); // how many k-blocks and d-blocks we have (the same number for both)
	long blocks_per_thread_block = div_inc(blocks_count, B);

    if(Bid == 0 && Tid == 0) {
        // printf("B=%ld, blocks_count=%ld, blocks_per_thread_block=%ld\n\n", B, blocks_count, blocks_per_thread_block);
	}

	long idx_slice; // used in the first for loop to indicate the current block id (k-block or d-block)
	long d_block_start; // start index for a d-block
	long d_block_end; // end index of a d-block

	long row; // the row index
	long row_mul_k; // holds row * k by starting with 0 and adding k at each row (addition takes less cycles than multiplication)

	float c; // the coefficient for each row TODO: save it in the shared memory using 4-element reads, at the beginning (adjust shared memory size for this)
	long k_block_start; // start index for a k-block
	long k_block_end; // end index of a k-block

    long i; // vector index (for indices and values)
	long k_col; // column index to extract data from indices and values at the current row
// 	long index; // the 1-D index to extract data from indices and values at the current row (row * k + k_col)
	int16 ind; // the data from indices at the index "index"
	bfloat16 val; // the data from values at the index "index"

	// save the following quantities to be used next
	long Bid_MUL_blocksPerThreadBlock = Bid * blocks_per_thread_block;
	long dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = d_block_size * Bid_MUL_blocksPerThreadBlock;
	long kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = k_block_size * Bid_MUL_blocksPerThreadBlock;

	long d_sub_d_block_size;

	for(idx_slice = 0; idx_slice < blocks_per_thread_block; ++idx_slice) { // iterate through all slices (or blocks) of size (m, k_block_size) that will be processed by the current thread block
		d_block_start = dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + d_block_size * idx_slice; // start index for the current d-block
		d_block_end = min(d, d_block_start + d_block_size); // end index for the current d-block
		k_block_start = kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + k_block_size * idx_slice; // start index for the current k-block
		k_block_end = min(k, k_block_start + k_block_size); // end index for the current k-block
		d_sub_d_block_size = d - d_block_start; // perform this subtraction only once, it will be used below to compute some offsets

        // printf("idx_slice=%ld d_block_start=%ld d_block_end=%ld k_block_start=%ld k_block_end=%ld\n", idx_slice, d_block_start, d_block_end, k_block_start, k_block_end);

		for(row = 0, row_mul_k = 0; row < m; ++row, row_mul_k += k) { // iterate through all rows in the current slice of size (m, k_block_size) to compute the LCG
			c = coef[row]; // maybe this read can be improved by saving it into the shared memory? see LCG_v53_bf16_cached_coef
			// printf("\trow=%ld c=%.2f\n", row, c);

			for(k_col = k_block_start + Tid; k_col < k_block_end && k_col < d_block_end; k_col += T) { // iterate through all k_block_size columns in the slice of size (m, k_block_size)
				i = row_mul_k + k_col;

				ind = indices[i];
				val = values[i];
				// printf("\t\ti=%ld row_mul_k=%ld k_col=%ld ind=%hi, val=%.2f\n", i, row_mul_k, k_col, ind, __bfloat162float(val));

				/*old (int32)*/ //mem[ind - d_block_start] += c * __bfloat162float(val); // mem update: this is where the LCG is performed
				/*new (int16)*/ mem[ind] += c * __bfloat162float(val); // mem update: this is where the LCG is performed
			}
			__syncthreads(); // make sure all threads finish one row to avoid race conditions
			// printf("\n");
		}

		// printf("mem: ");
		// for(i=0; i < d_block_size; ++i) {
		//     printf("%.2f ", mem[i]);
		// }
		// printf("\n");

		// now, mem contains the LCG for the slice (m, k_block_size). Actually, the results were scattered in a range of size d_block_size in which we only had k_block_size values (k_block_size = 1% of d_block_size)
		for(i = Tid; i < d_block_size && i < d_sub_d_block_size; i += T) { // mem dump: i < d_sub_d_block_size is required to avoid illegal memory access
			out[i + d_block_start] = __float2bfloat16(mem[i]); // write the content of mem to the right location in out
			mem[i] = 0; // then zerorize mem to prepare for the next block/slice
		}
		__syncthreads(); // make sure all threads finish writing to out
	}
}

void LCG_cuda(int blocks,
              int threads,
              int version,
              long d,
              long m,
              long k,
              long d_block_size,
              long k_block_size,
              torch::Tensor c,
              torch::Tensor indices,
              torch::Tensor values,
              torch::Tensor out,
              int use_bf16) {
	assert(version == 51);
    long sharedMemSize = d_block_size * sizeof(float);

	if (use_bf16) {
        // see https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf, chapter 19.7. Compute Capability 8.x and the example above it
        cudaFuncSetAttribute(LCG_v51_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
        // printf("Before calling the kernel: d=%ld, m=%ld, k=%ld, d_block_size=%ld, k_block_size=%ld\n", d, m, k, d_block_size, k_block_size);
        LCG_v51_bf16<<<blocks, threads, sharedMemSize>>>(d,
                                                         m,
                                                         k,
                                                         d_block_size,
                                                         k_block_size,
                                                         (float*) c.data_ptr(),
                                                         (int16*) indices.data_ptr(),
                                                         (bfloat16*) values.data_ptr(),
                                                         (bfloat16*) out.data_ptr());
	} else { // float values
        // see https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf, chapter 19.7. Compute Capability 8.x and the example above it
        cudaFuncSetAttribute(LCG_v51, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
        LCG_v51<<<blocks, threads, sharedMemSize>>>(d,
                                                    m,
                                                    k,
                                                    d_block_size,
                                                    k_block_size,
                                                    (float*) c.data_ptr(),
                                                    (int16*) indices.data_ptr(),
                                                    (float*) values.data_ptr(),
                                                    (float*) out.data_ptr());
	}

	GPU_ERROR_CHECK(cudaGetLastError());
	GPU_ERROR_CHECK(cudaPeekAtLastError());
	GPU_ERROR_CHECK(cudaDeviceSynchronize());
}
