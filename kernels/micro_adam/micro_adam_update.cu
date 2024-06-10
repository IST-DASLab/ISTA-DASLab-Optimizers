#include "../utils.h"

__global__ void compute_microadam_update_kernel_bf16(LL t,
                                                 float beta1,
                                                 float beta2,
                                                 float c1,
                                                 float c2,
                                                 float eps,
								                 LL d_block_size,
								                 LL k_block_size,
                                                 LL d,
                                                 LL m,
                                                 LL k,
                                                 int16* indices,
                                                 bfloat16* values,
                                                 bfloat16* out) {
    /*
		This kernel computes the update (m / (1-beta1**t)) / (sqrt(v / (q-beta2**t))+eps) for Compressed Adam:
		    - Use a shared memory of size 2*d_block_size where the first d_block_size components store m_t and the last
		d_block_size components store v_t
			- One CUDA block processes a slice of size (m, k_block_size) and computes the linear combination of all m
        gradients using Shared Memory
			- This way, after one thread block processes its current slice, the shared memory will contain the LCG for a
		slice of size k_block_size
			- The number of thread blocks is chosen dynamically such that the GPU is filled and there are no thread
		blocks waiting in the queue
			- A few more details:
				* Top-K is applied in blocks of size d_block_size in each row and it results in k_block_size values
			(k_block_size = 1% of d_block_size)
				* We know that in any range of size d_block_size there are k_block_size non-zero values, scattered more
			or less randomly
				* We use the shared memory of size d_block_size to accumulate linear combinations in it and in the end
			write the result to the output at the right index locations
	*/
	extern __shared__ float mem[];

	float *mem_m = mem; // m starts at the beginning of mem
	float *mem_v = mem + d_block_size; // v starts in the center of mem

	const LL B = gridDim.x; // number of blocks
	const LL Bid = blockIdx.x; // block id
	const LL T = blockDim.x; // number of threads
	const LL Tid = threadIdx.x; // thread id

	for(LL i = Tid; i < 2 * d_block_size; i += T) { // mem init because we will add values to this buffer
		mem[i] = 0;
	}
	__syncthreads();

	LL blocks_count = div_inc(d, d_block_size); // how many k-blocks and d-blocks we have (the same number for both)
	LL blocks_per_thread_block = div_inc(blocks_count, B);

	LL idx_block; // used in the first for loop to indicate the current block id (k-block or d-block)
	LL d_block_start; // start index for a d-block
	LL d_block_end; // end index of a d-block

	LL row; // the row index
	LL row_mul_k; // holds row * k by starting with 0 and adding k at each row (addition takes less cycles than multiplication)

	LL k_block_start; // start index for a k-block
	LL k_block_end; // end index of a k-block

	LL col; // column index to extract data from indices and values at the current row
	LL index; // the 1-D index to extract data from indices and values at the current row (row * k + col)
	LL i; // the data from indices at the index "index"
	float fval; // the data from values at the index "index", but converted to float
// 	bfloat16 fval; // the data from values at the index "index", but converted to float

	// save the following quantities to be used next
	LL Bid_MUL_blocksPerThreadBlock = Bid * blocks_per_thread_block;
	LL dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = d_block_size * Bid_MUL_blocksPerThreadBlock;
	LL kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = k_block_size * Bid_MUL_blocksPerThreadBlock;

	LL d_sub_d_block_size;

	float beta1_exp, beta2_exp;
	int exp;

	// iterate through all slices (or blocks) of size (m, k_block_size) that will be processed by the current thread block
	for(idx_block = 0; idx_block < blocks_per_thread_block; ++idx_block) {
		d_block_start = dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + d_block_size * idx_block; // start index for the current d-block
		d_block_end = min(d, d_block_start + d_block_size); // end index for the current d-block
		k_block_start = kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + k_block_size * idx_block; // start index for the current k-block
		k_block_end = min(k, k_block_start + k_block_size); // end index for the current k-block
		d_sub_d_block_size = d - d_block_start; // perform this subtraction only once, it will be used below to compute some offsets

		// iterate through all rows in the current slice of size (m, k_block_size) to compute the LCG
		for(row = 0, row_mul_k = 0; row < min(t, m); ++row, row_mul_k += k) {
			exp = (t - 1 - row) % m;
			beta1_exp = my_pow(beta1, exp);
			beta2_exp = my_pow(beta2, exp);

			// iterate through all k_block_size columns in the slice of size (m, k_block_size)
			for(col = k_block_start + Tid; col < k_block_end && col < d_block_end; col += T) {
				index = row_mul_k + col;
				i = indices[index];
				fval = __bfloat162float(values[index]);
				mem_m[i] += beta1_exp * fval;
				mem_v[i] += beta2_exp * fval * fval;
			}
			__syncthreads(); // make sure all threads finish one row to avoid race conditions
		} // at this point, mem_m and mem_v contain the linear combinations with coefficients beta1,2^((t-i)%m)

		// now, mem contains the LCG for the slice (m, k_block_size). Actually, the results were scattered in a range of size
		// d_block_size in which we only had k_block_size values (k_block_size = 1% of d_block_size)
		// mem dump: i < d_sub_d_block_size is required to avoid illegal memory access
		// apply bias correction, then compute sqrt, then add epsilon using coalesced memory access
		for(i = Tid; i < d_block_size && i < d_sub_d_block_size; i += T) {
			out[i + d_block_start] = __float2bfloat16((c1 * mem_m[i]) / (sqrt(c2 *  mem_v[i]) + eps));
		}
		__syncthreads(); // make sure all threads finish writing to out

		for(i = Tid; i < 2 * d_block_size; i += T) {
		    mem[i] = 0;
		}
		__syncthreads();
	}
}

void compute_microadam_update_cuda(int blocks, int threads, int carveout,
                               LL t, float beta1, float beta2, float eps,
                               LL d_block_size, LL k_block_size,
                               LL d, LL m, LL k,
                               torch::Tensor indices, torch::Tensor values, torch::Tensor out) {

    assert((carveout == 25) || (carveout == 50) || (carveout == 100));

	float c1 = (1. - beta1) / (1. - my_pow(beta1, t));
	float c2 = (1. - beta2) / (1. - my_pow(beta2, t));

	// d_block_size is chosen as the max_floats_possible / 2 and thsi value will be used in top-k
	// we multiply by 2 here because in the first half we store m_t and in the second half we store v_t
    LL sharedMemSize_bytes = d_block_size * 2 * sizeof(float);


    // kernel call
    switch(out.scalar_type()) {
        case torch::ScalarType::BFloat16:
            //// see https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf or https://docs.nvidia.com/cuda/cuda-c-programming-guide/
            //// chapter 19.7. Compute Capability 8.x and the example above it

            // cudaFuncSetAttribute(compute_microadam_update_kernel_bf16, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
            if(sharedMemSize_bytes > 48 * 1024) {
                //// if we want to allocate more than 48KB, then we have to call this method
                cudaFuncSetAttribute(compute_microadam_update_kernel_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize_bytes);
            }
            compute_microadam_update_kernel_bf16<<<blocks, threads, sharedMemSize_bytes>>>(t, beta1, beta2, c1, c2, eps,
                                                                            d_block_size, k_block_size,
                                                                            d, m, k,
                                                                            (int16*) indices.data_ptr(),
                                                                            (bfloat16*) values.data_ptr(),
                                                                            (bfloat16*) out.data_ptr());
            break;
        case torch::ScalarType::Float:
            printf("compute_microadam_update was not implemented for float32!\n");
            exit(666);
            break;
    }
	// error checks
	GPU_ERROR_CHECK(cudaGetLastError());
	GPU_ERROR_CHECK(cudaPeekAtLastError());
// 	GPU_ERROR_CHECK(cudaDeviceSynchronize());
}


