#include "utils.h"

__device__ long div_inc_SP(long a, long b) {
	long r = a / b;
	return (a % b > 0) ? (r + 1) : r;
}

__device__ long log_threads(long T) {
	if(T == 2) return 1;
	if(T == 4) return 2;
	if(T == 8) return 3;
	if(T == 16) return 4;
	if(T == 32) return 5;
	if(T == 64) return 6;
	if(T == 128) return 7;
	if(T == 256) return 8;
	if(T == 512) return 9;
	if(T == 1024) return 10;
}

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


__device__ void group_parallel_reduce(float *mem, long T, long group_size, long num_reds, long Tid, long offset=0, bool zerorize=false) {
	/*
		Perform parallel reduce in logarithmic time over the vector mem with T threads (mem has T components).
		If zerorize=true, then set the components of mem to zero after accumulation.
		Use offset > 0 to perform the parallel reduction over a different sequence of size T in mem
		Tid is the current thread id and logT is log2(T).
		Return the sum, which will be located at mem[offset]

		Resources:
			https://shreeraman-ak.medium.com/parallel-reduction-with-cuda-d0ae10c1ae2c
	*/

	long local_Tid = Tid % group_size;
	long mid = group_size >> 1; // half of number of threads
	long offset_PLUS_Tid = offset + Tid;
	for(long i = 0; i < num_reds; ++i) { // perform log2(T) rounds of accumulation
		__syncthreads();
		if(local_Tid < mid) { // left half accumulates, right half sends to left half
			mem[offset_PLUS_Tid] += mem[offset_PLUS_Tid + mid];
			if(zerorize) {
				mem[offset_PLUS_Tid + mid] = 0.;
			}
		}
		mid >>= 1;
	}
}

__global__ void SP_v1(long m, long k, float* g, int* indices, float* values, float* out) {
	/*
		Scalar Products version #1:
			1 block / all rows
			1 thread / row
			Non-CMA
	*/
	long i, row = threadIdx.x;
	double dot = 0.0f;

	// compute the pointer for the current row in indices and values to save up some indexing computations (compute offsets to current row)
	long row_mul_k = row * k;
	int *crt_index_row = indices + row_mul_k;
	float *crt_val_row = values + row_mul_k;

	for(i = 0; i < k; ++i) {
		dot += crt_val_row[i] * g[crt_index_row[i]];
	}
	out[row] = dot;
}

__global__ void SP_v21(long m, long k, float* g, int* indices, float* values, float* out) {
	/*
		Scalar Products version #2.1:
			Float type for values
			One block per row (1024 threads / row)
			Shared memory to store partial dot products of each thread slice
			Non-CMA
			Linear Reduction (inefficient)
			This version is equivalent to dot_prods_kernel_v1 from the first trial (nov 2022)
	*/
	extern __shared__ float mem[];
	long i;
	long row = blockIdx.x; // each block processes one row
	long thread_processing_size = div_inc_SP(k, blockDim.x); // this is how many elements a thread processes per row

	// compute the pointer for the current row in indices and values to save up some indexing computations (compute offsets to current row)
	long row_start = row * k;
	long row_end = row_start + k;
	long thread_start_index = row_start + threadIdx.x * thread_processing_size;
	long thread_end_index = thread_start_index + thread_processing_size;

	double sum = 0.0f;
	for(i = thread_start_index; i < thread_end_index && i < row_end; ++i) { // NON-COALESCED MEMORY ACCESS
		sum += (values[i] * g[indices[i]]);
	}
	mem[threadIdx.x] = sum;

	__syncthreads();

	if(threadIdx.x == 0) {
		sum = 0.0f;
		for(i = 0; i < blockDim.x; ++i) {
			sum += mem[i];
		}
		out[blockIdx.x] = sum;
	}
}

__global__ void SP_v22(long m, long k, float* g, int* indices, float* values, float* out) {
	/*
		Scalar Products version #2.2:
			Float type for values
			One block per row (1024 threads / row)
			Shared memory to store partial dot products of each thread slice
			CMA
			Linear Reduction (inefficient)
	*/
	extern __shared__ float mem[]; // has T elements
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id

	long i;
	long row = Bid; // each block processes one row
	long thread_processing_size = div_inc_SP(k, T); // this is how many elements a thread processes per row

	// compute the pointer for the current row in indices and values to save up some indexing computations (compute offsets to current row)
	long row_start = row * k;
	long row_end = row_start + k;
	// long thread_start_index = row_start + Tid * thread_processing_size;
	// long thread_end_index = thread_start_index + thread_processing_size;

	double sum = 0.0f;
	for(i = row_start + Tid; i < row_end; i += T) { // COALESCED MEMORY ACCESS
		sum += (values[i] * g[indices[i]]);
	}
	mem[Tid] = sum;

	__syncthreads();

	if(threadIdx.x == 0) { // TODO: diagonal reduction
		sum = 0.0f;
		for(i = 0; i < T; ++i) {
			sum += mem[i];
		}
		out[Bid] = sum;
	}
}

__global__ void SP_v23(long m, long k, float* g, int* indices, float* values, float* out) {
	/*
		Scalar Products version #2.3:
			Float type for values
			1 block / row
			CMA
			Logarithmic Reduction
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
	long thread_processing_size = div_inc_SP(k, T); // this is how many elements a thread processes per row

	// compute the pointer for the current row in indices and values to save up some indexing computations (compute offsets to current row)
	long row_start = row * k;
	long row_end = row_start + k;
	// long thread_start_index = row_start + Tid * thread_processing_size;
	// long thread_end_index = thread_start_index + thread_processing_size;

	double sum = 0.0f;
	for(i = row_start + Tid; i < row_end; i += T) { // start from Tid and increase with T because we have T threads per block
		sum += (values[i] * g[indices[i]]);
	}
	mem[Tid] = sum;

	parallel_reduce(mem, T, logT, Tid, 0, false);
//	long mid = T / 2; // half of number of threads
//	for(i = 0; i < 10; ++i) { // perform log2(1024) = 10 rounds of accumulation
//		__syncthreads();
//		if(Tid < mid) { // left half accumulates, right half sends to left half
//			mem[Tid] += mem[Tid + mid];
//		}
//		mid >>= 1; // reduce by 2 number of threads that we interact with
//	}

	if(Tid == 0) {
		out[Bid] = mem[0];
	}
}

__global__ void
__launch_bounds__(1024) // 3 possible params, in this order: maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster
SP_v23_bf16(long d, long m, long k, float* g, int* indices, bfloat16* values, float* out) {
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

	double sum = 0.0f;
	bfloat16 val;
	long ind;
	for(i = row_start + Tid; i < row_end; i += T) { // start from Tid and increase with T because we have T threads per block
		val = values[i];
		ind = indices[i];
		sum += (__bfloat162float(val) * g[ind]);
	}

	mem[Tid] = sum;

	parallel_reduce(mem, T, logT, Tid, 0, false);
	if(Tid == 0) {
		out[Bid] = mem[0];
	}
}

__global__ void SP_v24_bf16_vectorized(long m, long k, float* g, int* indices, bfloat16* values, float* out) {
	/*
		Scalar Products version #2.4:
			bfloat16 for values
			1 block / row (as many threads ar rows: m)
			CMA
			Logarithmic Reduction
			VMA-4 for indices and values
	*/
	extern __shared__ float mem[]; // has T elements

	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id
	const long logT = log_threads(T);

	int4 *indices4 = reinterpret_cast<int4*>(indices);
	bfloat164 *values4 = reinterpret_cast<bfloat164*>(values);
	long k_div4 = k >> 2;

	long i;
	long row = Bid; // each block processes one row

	// compute the pointer for the current row in indices and values to save up some indexing computations (compute offsets to current row)
	long row_start = row * k_div4;
	long row_end = row_start + k_div4;

	int4 ind;
	bfloat164 val;

	float sum = 0.0f;
	for(i = row_start + Tid; i < row_end; i += T) {
		ind = indices4[i];
		val = values4[i];

		sum += __bfloat162float(val.x) * g[ind.x] +
			   __bfloat162float(val.y) * g[ind.y] +
			   __bfloat162float(val.z) * g[ind.z] +
			   __bfloat162float(val.w) * g[ind.w];
	}
	mem[Tid] = sum;

	parallel_reduce(mem, T, logT, Tid, 0, false);
//	long mid = T / 2; // half of number of threads
//	for(i = 0; i < 10; ++i) { // perform log2(1024) = 10 rounds of accumulation
//		__syncthreads();
//		if(Tid < mid) { // left half accumulates, right half sends to left half
//			mem[Tid] += mem[Tid + mid];
//		}
//		mid >>= 1; // reduce by 2 number of threads that are queried at the next step
//	}

	if(Tid == 0) {
		out[Bid] = mem[0];
	}
}

__global__ void SP_v251_bf16(long d_block_size, long k_block_size, long d, long m, long k, float* g, int* indices, bfloat16* values, float* out) {
	/*
		Scalar Products version #2.5.1:
			Gradient slices in shared memory (m, d_block_size)
			1 block, 1 thread / row
			Non-CMA
			!!! THIS IS TOO SLOW BECAUSE 1) TOO FEW THREADS and 2) NON-COALESCED MEMORY ACCESS
	*/
	extern __shared__ float mem[];

	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id

	long blocks_count = div_inc_SP(d, d_block_size); // how many k-blocks and d-blocks we have (the same number for both)
	long blocks_per_thread_block = div_inc_SP(blocks_count, B);

	long idx_block; // used in the first for loop to indicate the current block id (k-block or d-block)
	long d_block_start; // start index for a d-block
	long d_block_end; // end index of a d-block

	long row = Tid; // the row index
	long row_mul_k = row * k; // holds the starting index of the current row "row"

//	float c; // the coefficient for each row TODO: save it in the shared memory using 4-element reads, at the beginning (adjust shared memory size for this)
	long k_block_start; // start index for a k-block
	long k_block_end; // end index of a k-block

	long col; // column index to extract data from indices and values at the current row
	long index; // the 1-D index to extract data from indices and values at the current row (row * k + col)
	long ind; // the data from indices at the index "index"
	float val; // the data from values at the index "index"

	// save the following quantities to be used next
	long Bid_MUL_blocksPerThreadBlock = Bid * blocks_per_thread_block;
	long dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = d_block_size * Bid_MUL_blocksPerThreadBlock;
	long kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = k_block_size * Bid_MUL_blocksPerThreadBlock;

	double dot = 0.; // saves the dot product between the current row and gradient (final result)

	for(idx_block = 0; idx_block < blocks_per_thread_block; ++idx_block) {
		// copy a block of size d_block_size from g to mem
		d_block_start = dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + d_block_size * idx_block; // start index for the current d-block
		k_block_start = kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + k_block_size * idx_block; // start index for the current k-block
		d_block_end = min(d, d_block_start + d_block_size); // end index for the current d-block
		k_block_end = min(k, k_block_start + k_block_size); // end index for the current k-block

		for(long i = d_block_start + Tid; i < d_block_end; i += T) { // COALESCED MEMORY ACCESS TO COPY g TO mem
			mem[i - d_block_start] = g[i];
		}
		__syncthreads();
		for(col = k_block_start; col < k_block_end && col < d_block_end; ++col) { // NON-COALESCED MEMORY ACCESS FOR indices & values
			index = row_mul_k + col;
			ind = indices[index];
			val = __bfloat162float(values[index]);
			dot += mem[ind - d_block_start] * val;
		}
		__syncthreads();
	}
	atomicAdd(out + Tid, dot); // we should have as many threads as components in out (here, we have m=1024)
//	out[row] = dot; // write scalar product once in each thread (row = Tid)
}

__device__ float safe_prod(float *mem, long index, long offset, bfloat16 val) {
	long diff = index - offset;
	return (diff >= 0) ? mem[diff] * __bfloat162float(val) : 0.;
	// mem[ind4.x - d_block_start_mul4] * __bfloat162float(val4.x)
}

__global__ void SP_v252_bf16_vectorized(long d_block_size, long k_block_size, long d, long m, long k, float* g, int* indices, bfloat16* values, float* out) {
	/*
		Scalar Products version #2.5.2:
			Gradient slices in shared memory (m, d_block_size)
			1 block, 1 thread / row
			CMA
			VMA-4
	*/
	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id

	extern __shared__ float mem[];
	int4 *indices4 = reinterpret_cast<int4*>(indices);
	bfloat164 *values4 = reinterpret_cast<bfloat164*>(values);
	float4 *g4 = reinterpret_cast<float4*>(g);
//	float4 *out4 = reinterpret_cast<float4*>(out);
	float4 *mem4 = reinterpret_cast<float4*>(mem);
	long d_block_size_div4 = d_block_size >> 2;
	long k_block_size_div4 = k_block_size >> 2;
//	long m_div4 = m >> 2;
	long k_div4 = k >> 2;
    long d_div4 = d >> 2;
//	long row_mul_k_div4; // holds row * k by starting with 0 and adding k at each row (addition takes less cycles than multiplication)
	int4 ind4; // the data from indices at the index "index"
	bfloat164 val4; // the data from values at the index "index"
//	float4 zero4;
//	zero4.x = zero4.y = zero4.z = zero4.w = 0;

//	if(Bid == 0 && Tid == 0) {
//		printf("d=%ld, k=%ld, d_div4=%ld, k_div4=%ld, d_block_size=%ld, k_block_size=%ld, d_block_size_div4=%ld, k_block_size_div4=%ld\n",
//			   d, k, d_div4, k_div4, d_block_size, k_block_size, d_block_size_div4, k_block_size_div4);
//	}

	long blocks_count = div_inc_SP(d, d_block_size); // how many k-blocks and d-blocks we have (the same number for both)
	long blocks_per_thread_block = div_inc_SP(blocks_count, B);

	long idx_block; // used in the first for loop to indicate the current block id (k-block or d-block)
	long d_block_start_div4; // start index for a d-block
	long d_block_end_div4; // end index of a d-block
	long d_block_start_mul4; // start index for a d-block for vectorized memory access

	long row = Tid; // the row index
	long row_mul_k_div4 = row * k_div4; // holds the starting index of the current row "row"

//	float c; // the coefficient for each row TODO: save it in the shared memory using 4-element reads, at the beginning (adjust shared memory size for this)
	long k_block_start_div4; // start index for a k-block
	long k_block_end_div4; // end index of a k-block

	long col; // column index to extract data from indices and values at the current row
	long index; // the 1-D index to extract data from indices and values at the current row (row * k + col)
//	long ind; // the data from indices at the index "index"
//	float val; // the data from values at the index "index"
	long diff; // the difference for indices

	// save the following quantities to be used next
	long Bid_MUL_blocksPerThreadBlock = Bid * blocks_per_thread_block;
	long dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = d_block_size_div4 * Bid_MUL_blocksPerThreadBlock;
	long kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = k_block_size_div4 * Bid_MUL_blocksPerThreadBlock;

	double dot = 0.; // saves the dot product between the current row and gradient (final result)

	for(idx_block = 0; idx_block < blocks_per_thread_block; ++idx_block) {
		// copy a block of size d_block_size from g to mem
		d_block_start_div4 = dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + d_block_size_div4 * idx_block; // start index for the current d-block
		k_block_start_div4 = kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + k_block_size_div4 * idx_block; // start index for the current k-block
		d_block_end_div4 = min(d_div4, d_block_start_div4 + d_block_size_div4); // end index for the current d-block
		k_block_end_div4 = min(k_div4, k_block_start_div4 + k_block_size_div4); // end index for the current k-block
		d_block_start_mul4 = d_block_start_div4 << 2;

		for(long i = d_block_start_div4 + Tid; i < d_block_end_div4; i += T) {
			diff = i - d_block_start_div4;
//			if(diff < 0 || diff >= m * k_div4) {
//				printf("[Bid=%ld] [Tid=%ld] diff=%ld, m * k_div4 = %ld\n", Bid, Tid, diff, m * k_div4);
//			}
			mem4[i - d_block_start_div4] = g4[i];
		}
		__syncthreads();
		for(col = k_block_start_div4; col < k_block_end_div4 && col < d_block_end_div4; ++col) { // NON-COALESCED MEMORY ACCESS
			index = row_mul_k_div4 + col;
			ind4 = indices4[index];
			val4 = values4[index];
			dot += safe_prod(mem, ind4.x, d_block_start_mul4, val4.x) +
				   safe_prod(mem, ind4.y, d_block_start_mul4, val4.y) +
				   safe_prod(mem, ind4.z, d_block_start_mul4, val4.z) +
				   safe_prod(mem, ind4.w, d_block_start_mul4, val4.w);
//			dot += mem[ind4.x - d_block_start_mul4] * __bfloat162float(val4.x) +
//				   mem[ind4.y - d_block_start_mul4] * __bfloat162float(val4.y) +
//				   mem[ind4.z - d_block_start_mul4] * __bfloat162float(val4.z) +
//				   mem[ind4.w - d_block_start_mul4] * __bfloat162float(val4.w);
		}
		__syncthreads();
	}
	atomicAdd(out + Tid, dot); // we should have as many threads as components in out (here, we have m=1024)
//	out[row] = dot; // write scalar product once in each thread (row = Tid)
}

__global__ void SP_v261_bf16(long d_block_size, long k_block_size, long d, long m, long k, float* g, int* indices, bfloat16* values, float* out) {
	/*
		Scalar Products version #2.6.1:
			Gradient slices in shared memory (m, d_block_size)
			Max number of blocks to fill in GPU, 1 block / slice
			CMA for grad to mem
			CMA for I & V
			Each thread accumulates the dot product in its own dot variable and uses atomic add to update the output
	*/
	extern __shared__ float mem[];

	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id

	long blocks_count = div_inc_SP(d, d_block_size); // how many k-blocks and d-blocks we have (the same number for both)
	long blocks_per_thread_block = div_inc_SP(blocks_count, B);

	long idx_block; // used in the first for loop to indicate the current block id (k-block or d-block)
	long d_block_start; // start index for a d-block
	long d_block_end; // end index of a d-block

	long row = Tid; // the row index
	long row_mul_k = row * k; // holds the starting index of the current row "row"

	long k_block_start; // start index for a k-block
	long k_block_end; // end index of a k-block

	long col; // column index to extract data from indices and values at the current row
	long index; // the 1-D index to extract data from indices and values at the current row (row * k + col)
	long ind; // the data from indices at the index "index"
	float val; // the data from values at the index "index"

	// save the following quantities to be used next
	long Bid_MUL_blocksPerThreadBlock = Bid * blocks_per_thread_block;
	long dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = d_block_size * Bid_MUL_blocksPerThreadBlock;
	long kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = k_block_size * Bid_MUL_blocksPerThreadBlock;

	double dot = 0.; // saves the dot product between the current row and gradient (final result)

	for(idx_block = 0; idx_block < blocks_per_thread_block; ++idx_block) { // iterate through slices
		// copy a block of size d_block_size from g to mem
		d_block_start = dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + d_block_size * idx_block; // start index for the current d-block
		k_block_start = kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + k_block_size * idx_block; // start index for the current k-block
		d_block_end = min(d, d_block_start + d_block_size); // end index for the current d-block
		k_block_end = min(k, k_block_start + k_block_size); // end index for the current k-block

		for(long i = d_block_start + Tid; i < d_block_end; i += T) { // all threads copy g from global memory to shared memory
			mem[i - d_block_start] = g[i];
		}
		__syncthreads();

		for(col = k_block_start; col < k_block_end && col < d_block_end; ++col) { // NON-CMA for I & V because we have only one thread per row and the thread should go through all elements
			index = row_mul_k + col;
			ind = indices[index];
			val = __bfloat162float(values[index]);
			dot += mem[ind - d_block_start] * val;
//			printf("[Bid=%ld] [Tid=%ld] idx_block=%ld, blocks_per_thread_block=%ld, d_block_start=%ld, d_block_end=%ld, d_block_size=%ld, row=%ld, col=%ld==> grad=%f, val=%f @ dot=%lf\n",
//			   Bid, Tid, idx_block, blocks_per_thread_block, d_block_start, d_block_end, d_block_size, row, col, mem[ind-d_block_start], val, dot);
		}
		__syncthreads();
	}
	atomicAdd(out + Tid, dot); // we should have as many threads as components in out (here, we have m=1024)
}

__global__ void SP_v262_bf16_vectorized(long d_block_size, long k_block_size, long d, long m, long k, float* g, int* indices, bfloat16* values, float* out) {
	/*
		Scalar Products version #2.6.2:
			Gradient slices in shared memory (m, d_block_size)
			Max number of blocks to fill in GPU, 1 block / slice
			CMA for mem <- grad
			NON-CMA + VMA-4 for I & V
			Each thread accumulates the dot product in its own dot variable and uses atomic add to update the output
	*/
	extern __shared__ float mem[];
	int4 *indices4 = reinterpret_cast<int4*>(indices);
	bfloat164 *values4 = reinterpret_cast<bfloat164*>(values);
	float4 *g4 = reinterpret_cast<float4*>(g);
//	float4 *out4 = reinterpret_cast<float4*>(out);
	float4 *mem4 = reinterpret_cast<float4*>(mem);
	long d_block_size_div4 = d_block_size >> 2;
	long k_block_size_div4 = k_block_size >> 2;
//	long m_div4 = m >> 2;
	long k_div4 = k >> 2;
    long d_div4 = d >> 2;
//	long row_mul_k_div4; // holds row * k by starting with 0 and adding k at each row (addition takes less cycles than multiplication)
	int4 ind4; // the data from indices at the index "index"
	bfloat164 val4; // the data from values at the index "index"
//	float4 zero4;
//	zero4.x = zero4.y = zero4.z = zero4.w = 0;

	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id

	long blocks_count = div_inc_SP(d, d_block_size); // how many k-blocks and d-blocks we have (the same number for both)
	long blocks_per_thread_block = div_inc_SP(blocks_count, B);

	long idx_block; // used in the first for loop to indicate the current block id (k-block or d-block)
	long d_block_start_div4; // start index for a d-block
	long d_block_end_div4; // end index of a d-block
	long d_block_start_mul4; // start index for a d-block for vectorized memory access

	long row = Tid; // the row index
	long row_mul_k_div4 = row * k_div4; // holds the starting index of the current row "row"

//	float c; // the coefficient for each row TODO: save it in the shared memory using 4-element reads, at the beginning (adjust shared memory size for this)
	long k_block_start_div4; // start index for a k-block
	long k_block_end_div4; // end index of a k-block

	long col; // column index to extract data from indices and values at the current row
	long index; // the 1-D index to extract data from indices and values at the current row (row * k + col)
//	long ind; // the data from indices at the index "index"
//	float val; // the data from values at the index "index"

	// save the following quantities to be used next
	long Bid_MUL_blocksPerThreadBlock = Bid * blocks_per_thread_block;
	long dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = d_block_size_div4 * Bid_MUL_blocksPerThreadBlock;
	long kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = k_block_size_div4 * Bid_MUL_blocksPerThreadBlock;

	double dot; // saves the dot product between the current row and gradient (final result)

	for(idx_block = 0; idx_block < blocks_per_thread_block; ++idx_block) {
		// copy a block of size d_block_size from g to mem
		d_block_start_div4 = dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + d_block_size_div4 * idx_block; // start index for the current d-block
		k_block_start_div4 = kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + k_block_size_div4 * idx_block; // start index for the current k-block
		d_block_end_div4 = min(d_div4, d_block_start_div4 + d_block_size_div4); // end index for the current d-block
		k_block_end_div4 = min(k_div4, k_block_start_div4 + k_block_size_div4); // end index for the current k-block
		d_block_start_mul4 = d_block_start_div4 << 2;

		for(long i = d_block_start_div4 + Tid; i < d_block_end_div4; i += T) { // COALESCED MEMORY ACCESS
			mem4[i - d_block_start_div4] = g4[i];
		}
		__syncthreads();
		dot = 0.;
		for(col = k_block_start_div4; col < k_block_end_div4 && col < d_block_end_div4; ++col) { // NON-CMA for I & V because we have only one thread per row and the thread should go through all elements
			index = row_mul_k_div4 + col;
			ind4 = indices4[index];
			val4 = values4[index];
			dot += safe_prod(mem, ind4.x, d_block_start_mul4, val4.x) +
				   safe_prod(mem, ind4.y, d_block_start_mul4, val4.y) +
				   safe_prod(mem, ind4.z, d_block_start_mul4, val4.z) +
				   safe_prod(mem, ind4.w, d_block_start_mul4, val4.w);
//			dot += mem[ind4.x - d_block_start_mul4] * __bfloat162float(val4.x) +
//				   mem[ind4.y - d_block_start_mul4] * __bfloat162float(val4.y) +
//				   mem[ind4.z - d_block_start_mul4] * __bfloat162float(val4.z) +
//				   mem[ind4.w - d_block_start_mul4] * __bfloat162float(val4.w);

//			if(fabs(mem[ind4.x - d_block_start_mul4]) > FLT_EPSILON) dot += mem[ind4.x - d_block_start_mul4] * __bfloat162float(val4.x);
//			if(fabs(mem[ind4.y - d_block_start_mul4]) > FLT_EPSILON) dot += mem[ind4.y - d_block_start_mul4] * __bfloat162float(val4.y);
//			if(fabs(mem[ind4.z - d_block_start_mul4]) > FLT_EPSILON) dot += mem[ind4.z - d_block_start_mul4] * __bfloat162float(val4.z);
//			if(fabs(mem[ind4.w - d_block_start_mul4]) > FLT_EPSILON) dot += mem[ind4.w - d_block_start_mul4] * __bfloat162float(val4.w);
		}
		__syncthreads();

		atomicAdd(out + Tid, dot); // we should have as many threads as components in out (here, we have m=1024)
		__syncthreads();
	}
}

__global__ void SP_v271_bf16(long d_block_size, long k_block_size, long d, long m, long k, float* g, int* indices, bfloat16* values, float* out) {
	/*
		Scalar Products version #2.7.1:
			Similar to SPv261, but:
			Caches both gradient slice (first d_block_size floats) and a buffer in the shared memory (additional T floats)
			Use the same methodology as in LCGv51:
				- read a slice of g once
				- iterate row by row to perform the scalar product with slice of g (save element-wise products in the shared memory - right side of g slice)
				- perform parallel reduce the buffer to have the result in mem[d_block_size] (which is the first component of the buffer)
				- go to the next row
				!!! This trick should ensure memory coalescing when accessing a row, which version 2.6.2 doesn't have
					Version 2.6.2 performs VMA-4 access with one thread per row, processing k_block_size elements.
	*/
	extern __shared__ float mem[]; // gradient slice is between 0 and d_block_size-1 and dot products are between d_block_size and d_block_size+T
	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id
	const long logT = log_threads(T);

	// zerorize mem between d_block_size and d_block_size+T: each thread zerorizes its component
	mem[d_block_size + Tid] = 0.;
	__syncthreads();

	long blocks_count = div_inc_SP(d, d_block_size); // how many k-blocks and d-blocks we have (the same number for both)
	long blocks_per_thread_block = div_inc_SP(blocks_count, B);

	long idx_block; // used in the first for loop to indicate the current block id (k-block or d-block)
	long d_block_start; // start index for a d-block
	long d_block_end; // end index of a d-block

	long row; // the row index
	long row_mul_k; // holds row * k by starting with 0 and adding k at each row (addition takes less cycles than multiplication)

//	float c; // the coefficient for each row TODO: save it in the shared memory using 4-element reads, at the beginning (adjust shared memory size for this)
	long k_block_start; // start index for a k-block
	long k_block_end; // end index of a k-block

	long col; // column index to extract data from indices and values at the current row
	long index; // the 1-D index to extract data from indices and values at the current row (row * k + col)
	long ind; // the data from indices at the index "index"
	bfloat16 val; // the data from values at the index "index"
	float g_val; // gradient value

	// save the following quantities to be used next
	long Bid_MUL_blocksPerThreadBlock = Bid * blocks_per_thread_block;
	long dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = d_block_size * Bid_MUL_blocksPerThreadBlock;
	long kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = k_block_size * Bid_MUL_blocksPerThreadBlock;

	for(idx_block = 0; idx_block < blocks_per_thread_block; ++idx_block) {
		d_block_start = dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + d_block_size * idx_block; // start index for the current d-block
		d_block_end = min(d, d_block_start + d_block_size); // end index for the current d-block

		k_block_start = kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + k_block_size * idx_block; // start index for the current k-block
		k_block_end = min(k, k_block_start + k_block_size); // end index for the current k-block

		for(long i = d_block_start + Tid; i < d_block_end; i += T) { // COALESCED MEMORY ACCESS
			mem[i - d_block_start] = g[i];
		}
		__syncthreads();

		for(row = 0, row_mul_k = 0; row < m; ++row, row_mul_k += k) {
			for(col = k_block_start + Tid; col < k_block_end && col < d_block_end; col += T) {
				index = row_mul_k + col;
				ind = indices[index];
				val = values[index];
				g_val = mem[ind - d_block_start];
				if(fabs(g_val) > FLT_EPSILON) {
					mem[d_block_size + Tid] +=  g_val * __bfloat162float(val); // mem update
				}
			}

			// compute sum(mem[d_block_size : d_block_size + T]) using parallel reduce
			parallel_reduce(mem, T, logT, Tid, d_block_size, true);
			if(Tid == 0) {
				atomicAdd(out + row, mem[d_block_size]);
				mem[d_block_size] = 0.;
			}
			__syncthreads();
		}
	}
}

__global__ void SP_v272_bf16(long d_block_size, long k_block_size, long d, long m, long k, float* g, int* indices, bfloat16* values, float* out) {
	/*
		Scalar Products version #2.7.2:
			Same as to SPv272, but:
				- warps are properly aligned with memory
	*/
	extern __shared__ float mem[]; // gradient slice is between 0 and d_block_size-1 and dot products are between d_block_size and d_block_size+T
	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id
	const long logT = log_threads(T);

	// zerorize mem between d_block_size and d_block_size+T: each thread zerorizes its component
	mem[d_block_size + Tid] = 0.;
	__syncthreads();

	long blocks_count = div_inc_SP(d, d_block_size); // how many k-blocks and d-blocks we have (the same number for both)
	long blocks_per_thread_block = div_inc_SP(blocks_count, B);

	long idx_block; // used in the first for loop to indicate the current block id (k-block or d-block)
	long d_block_start; // start index for a d-block
	long d_block_end; // end index of a d-block

	long row; // the row index
	long row_mul_k; // holds row * k by starting with 0 and adding k at each row (addition takes less cycles than multiplication)

//	float c; // the coefficient for each row TODO: save it in the shared memory using 4-element reads, at the beginning (adjust shared memory size for this)
	long k_block_start; // start index for a k-block
	long k_block_end; // end index of a k-block

	long col; // column index to extract data from indices and values at the current row
	long index; // the 1-D index to extract data from indices and values at the current row (row * k + col)
	long ind; // the data from indices at the index "index"
	bfloat16 val; // the data from values at the index "index"
	float g_val; // gradient value

	// save the following quantities to be used next
	long Bid_MUL_blocksPerThreadBlock = Bid * blocks_per_thread_block;
	long dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = d_block_size * Bid_MUL_blocksPerThreadBlock;
	long kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = k_block_size * Bid_MUL_blocksPerThreadBlock;



	long group_size = min(T, 32L);
	long slots_per_row = div_inc_SP(k_block_size, group_size);
	long padded_row_length = slots_per_row * group_size;
	long reduction_steps = log_threads(group_size);
	long total_elements = padded_row_length * m;


	for(idx_block = 0; idx_block < blocks_per_thread_block; ++idx_block) {
		d_block_start = dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + d_block_size * idx_block; // start index for the current d-block
		d_block_end = min(d, d_block_start + d_block_size); // end index for the current d-block

		k_block_start = kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + k_block_size * idx_block; // start index for the current k-block
		k_block_end = min(k, k_block_start + k_block_size); // end index for the current k-block

		for(long i = d_block_start + Tid; i < d_block_end; i += T) { // COALESCED MEMORY ACCESS
			mem[i - d_block_start] = g[i];
		}
		__syncthreads();

		for (long cur_id = Tid; cur_id < total_elements; cur_id += T) {
			row = cur_id / padded_row_length;
			col = k_block_start +  cur_id % padded_row_length;
			if (col >= k_block_end || col >= d_block_end) {
				val = __float2bfloat16(0);
				g_val = 0.0;
			} else {
				index = row * k + col;
				ind = indices[index];
				val = values[index];
				g_val = mem[ind - d_block_start];
			}
			mem[d_block_size + Tid] +=  g_val * __bfloat162float(val); // mem update

			group_parallel_reduce(mem, T, group_size, reduction_steps, Tid, d_block_size, true);
			if(Tid % group_size == 0) {
				atomicAdd(out + row, mem[d_block_size + Tid]);
				mem[d_block_size + Tid] = 0.;
			}
			__syncthreads();
		}
	}
}

void SP_cuda(int blocks, int threads, int version, long d, long m, long k, torch::Tensor g, torch::Tensor indices, torch::Tensor values, torch::Tensor out, int use_bf16, long d_block_size, long k_block_size) {
	int *i_ptr = indices.data_ptr<int>();
	float *o_ptr = out.data_ptr<float>();
	float *g_ptr = g.data_ptr<float>();

	if (use_bf16 == 1) {
		bfloat16 *v_ptr_bf16 = (bfloat16*) values.data_ptr();
		if(version == 23) {
//			printf("blocks=%d, threads=%d, version=%d, d=%d, m=%d, k=%d, use_bf16=%d, d_block_size=%d, k_block_size=%d\n",
//				   blocks, threads, version, d, m, k, use_bf16, d_block_size, k_block_size);
			SP_v23_bf16<<<blocks, threads, threads * sizeof(float)>>>(d, m, k, g_ptr, i_ptr, v_ptr_bf16, o_ptr);
		} else if(version == 24) {
			SP_v24_bf16_vectorized<<<blocks, threads, threads * sizeof(float)>>>(m, k, g_ptr, i_ptr, v_ptr_bf16, o_ptr);
		} else if(version == 251) {
			// search for "cudaFuncSetAttribute(MyKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);" in LCG_kernel.cu
			SP_v251_bf16<<<blocks, threads, d_block_size * sizeof(float)>>>(d_block_size, k_block_size, d, m, k, g_ptr, i_ptr, v_ptr_bf16, o_ptr);
		} else if(version == 252) {
			// m < threads ? m : threads
			// search for "cudaFuncSetAttribute(MyKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);" in LCG_kernel.cu
			SP_v252_bf16_vectorized<<<blocks, threads, d_block_size * sizeof(float)>>>(d_block_size, k_block_size, d, m, k, g_ptr, i_ptr, v_ptr_bf16, o_ptr);
		} else if(version == 261) {
		// search for "cudaFuncSetAttribute(MyKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);" in LCG_kernel.cu
			SP_v261_bf16<<<blocks, threads, d_block_size * sizeof(float)>>>(d_block_size, k_block_size, d, m, k, g_ptr, i_ptr, v_ptr_bf16, o_ptr);
		} else if(version == 262) {
			// if seems like number of blocks per SM is 2 for 1..768 threads and 1 for 769...1024
			// search for "cudaFuncSetAttribute(MyKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);" in LCG_kernel.cu
			SP_v262_bf16_vectorized<<<blocks, threads, d_block_size * sizeof(float)>>>(d_block_size, k_block_size, d, m, k, g_ptr, i_ptr, v_ptr_bf16, o_ptr);
		} else if(version == 271) {
			// search for "cudaFuncSetAttribute(MyKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);" in LCG_kernel.cu
			SP_v271_bf16<<<blocks, threads, (d_block_size + threads) * sizeof(float)>>>(d_block_size, k_block_size, d, m, k, g_ptr, i_ptr, v_ptr_bf16, o_ptr);
		} else if(version == 272) {
			// search for "cudaFuncSetAttribute(MyKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);" in LCG_kernel.cu
			SP_v272_bf16<<<blocks, threads, (d_block_size + threads) * sizeof(float)>>>(d_block_size, k_block_size, d, m, k, g_ptr, i_ptr, v_ptr_bf16, o_ptr);
		} else {
			fprintf(stderr, "Version %d does not have bf16 support for SP!\n", version);
		}
	} else { // float values
		float *v_ptr = values.data_ptr<float>();
		if(version == 1) {
			SP_v1<<<blocks, threads>>>(m, k, g_ptr, i_ptr, v_ptr, o_ptr);
		} else if(version == 21) {
			SP_v21<<<blocks, threads, 1024 * sizeof(float)>>>(m, k, g_ptr, i_ptr, v_ptr, o_ptr);
		} else if(version == 22) {
			SP_v22<<<blocks, threads, 1024 * sizeof(float)>>>(m, k, g_ptr, i_ptr, v_ptr, o_ptr);
		} else if(version == 23) {
			SP_v23<<<blocks, threads, 1024 * sizeof(float)>>>(m, k, g_ptr, i_ptr, v_ptr, o_ptr);
		}
	}
	gpuErrorCheck(cudaGetLastError());
	gpuErrorCheck(cudaPeekAtLastError());
	gpuErrorCheck(cudaDeviceSynchronize());
}


/*
	if(Tid == 0) {
		printf("[Tid=%d] idx_block=%d, blocks_per_thread_block=%d, d_block_start=%d, d_block_end=%d, d_block_size=%d\n",
			   Tid, idx_block, blocks_per_thread_block, d_block_start, d_block_end, d_block_size);
		for(long i = 0; i < d_block_size; ++i) {
			printf("%f ", mem[i]);
		}
		printf("\n\n");
	}

	printf("[Tid=%d] col=%d, index=%d, ind=%d, val=%f\n", Tid, col, index, ind, val);


	//// Aleksei's parallel reduce code
//	long mid = T / 2; // half of number of threads
//	for(i = 0; i < 10; ++i) { // perform log2(1024) = 10 rounds of accumulation
//		__syncthreads();
//		if(Tid < mid) { // left half accumulates, right half sends to left half
//			mem[Tid] += mem[Tid + mid];
//		}
//		mid >>= 1; // reduce by 2 number of threads that are queried at the next step
//	}
*/