#include "utils.h"

__device__ long div_inc_LCG(long a, long b) {
	long r = a / b;
	return (a % b > 0) ? (r + 1) : r;
}

__device__ long binary_search(int *haystack, long needle, long start_pos, long end_pos) {
	/*
		Binary Search:
			* Find the position of the needle in the haystack
			* The haystack has "size" elements
	*/
	long left = start_pos, right = end_pos - 1;
	if((needle < haystack[left]) || (haystack[right] < needle)) return -1;
	if(needle == haystack[left]) return left;
	if(needle == haystack[right]) return right;

	long mid, h;
	while(left <= right) {
		mid = (left + right) / 2;
		h = haystack[mid];

		if(h == needle) {
			return mid;
		} else if(needle < h) {
			right = mid - 1;
		} else { // if (h < needle)
			left = mid + 1;
		}
	}
	return -1;
}

__device__ void LCG_process_column(long col, long m, long k, float *coef, int *indices, float *values, float *out) {
	/*
		Iterate through all rows for a specific column "col" and perform Binary Search for the value "col" on each row
			- search for value "col" in all rows on the matrix "indices"
			- when the value is found, add coef[row] * values[row][col] to the sum
	*/
	double sum = 0.0f;
	long row, pos, row_mul_k;
	for(row = 0, row_mul_k = 0; row < m; ++row, row_mul_k += k) {
		pos = binary_search(indices + row_mul_k, col, 0, k);
		if(pos != -1) {
			sum += coef[row] * values[row_mul_k + pos];
		}
	}
	out[col] = sum;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////// LINEAR COMBINATION OF GRADIENTS (LCG)
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void LCG_v1(long d, long m, long k, float *coef, int *indices, float *values, float *out) {
	/*
		Linear Combination of Gradients version #1:
			Each thread is processing column c and then it has to search for the value c across all rows using binary search.
	*/
	long col = blockDim.x * blockIdx.x + threadIdx.x;
	if(col < d) { // the column that the thread processes is inside the matrix
		LCG_process_column(col, m, k, coef, indices, values, out);
	}
}

__global__ void LCG_v2(long d, long m, long k, float* coef, int* indices, float* values, float* out) {
	/*
		Linear Combination of Gradients version #2:
			Since the number of blocks is limited, each thread will process more than one column to limit how many threads access the global memory.
			Steps taken:
				1) compute how many columns are processed by a block
				2) compute how many columns within the block are processed by each thread
				3) once the column bounds are computed for each thread, iterate through the columns and perform the same steps as in LCG_v1
	*/
	long cols_per_block = div_inc_LCG(d, gridDim.x); // how many columns to process per block
	long cols_per_thread = div_inc_LCG(cols_per_block, blockDim.x); // how many columns to process per thread inside each block
	long block_start_col = blockIdx.x * cols_per_block; // starting column for the block
	long block_end_col = block_start_col + cols_per_block; // ending column for the block
	long thread_start_col = block_start_col + threadIdx.x * cols_per_thread; // starting column for the thread (some may start outside the matrix - larger than d)
	long thread_end_col = std::min(thread_start_col + cols_per_thread, block_end_col); // ending column for the thread (some may end outside the matrix - larger than d)
	long col;
	for(col = thread_start_col; col < thread_end_col && col < d; ++col) {
		LCG_process_column(col, m, k, coef, indices, values, out);
	}
	// printf("[block %d / thread # %d] cols_per_block = %d, cols_per_thread = %d, thread_start_col = %d, thread_end_col = %d\n", blockIdx.x, threadIdx.x, cols_per_block, cols_per_thread, thread_start_col, thread_end_col);
}

__global__ void LCG_v3(long d, long m, long k, float* coef, int* indices, float* values, float* out, int *mem) {
	/*
		Linear Combination of Gradients version #3:
			Use an external memory of size T x m, where T is number of threads to change the left side of the interval for the binary search
			Here, mem[i,j] = the starting position for the binary search for the current column on i-th row when processed by j-th thread
			Initially, this matrix should contain 0 everywhere and each thread initializes one row
	*/
	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id

//	long right[1024]; // m=1024 rows
//	for(long i=0; i<1024; ++i) {
//		right[i] = k;
//	}
//	__syncthreads();

	double sum = 0.0f;
	long col;
	long row;
	long pos;
	long row_mul_k;
	long bs_row_start_pos; // left end for binary search
	long bs_row_end_pos; // right end for binary search
//	long min_left_end; // used to compute the min position across all rows and then update bs_row_start_pos
	long crt_thread_row_index; //, next_thread_row_index; // current/next index in shared memory mem for current block, thread and row
//	long next_thread_row_value; // the value at the same row for the next thread that serves as a candidate for right[row]
	long crt_thread_row_start = Bid * T * m + Tid * m; // start index for the row of current thread
//	long next_thread_row_start;
//	if(Tid == T-1) {
//		next_thread_row_start = -1;
//	} else {
//		next_thread_row_start = Bid * T * m + (Tid + 1) * m;
//	}
//	long mem_row_end = crt_thread_row_start + m; // end index for the row of current thread

	// the following variables are copy-pasted from LCG_v2
	long cols_per_block = div_inc_LCG(d, B); // how many columns to process per block
	long cols_per_thread = div_inc_LCG(cols_per_block, T); // how many columns to process per thread inside each block
	long block_start_col = blockIdx.x * cols_per_block; // starting column for the block
	long block_end_col = block_start_col + cols_per_block; // ending column for the block
	long thread_start_col = block_start_col + threadIdx.x * cols_per_thread; // starting column for the thread (some may start outside the matrix - larger than d)
	long thread_end_col = std::min(thread_start_col + cols_per_thread, block_end_col); // ending column for the thread (some may end outside the matrix - larger than d)

	for(col = thread_start_col; col < thread_end_col && col < d; ++col) {
		sum = 0.0f;

		for(row = 0, row_mul_k = 0; row < m; ++row, row_mul_k += k) {
			// logic for the left end of Binary Search
			crt_thread_row_index = crt_thread_row_start + row;
			bs_row_start_pos = mem[crt_thread_row_index]; // save the value mem[Bid][Tid][row]
			bs_row_end_pos = k;

//			// logic for the right end of Binary Search
//			if(next_thread_row_start == -1) { // this is the last thread and the right end stays k
//				bs_row_end_pos = k;
//			} else {
//				next_thread_row_index = next_thread_row_start + row;
//				next_thread_row_value = mem[next_thread_row_index];
//
//				if((next_thread_row_value > 0) && // the next thread wrote something to the memory
//				   (next_thread_row_value < k) &&// the next thread did not end processing
//				   (right[row] == k)) // the right end of the current thread was not updated yet on this thread
//				{
//					right[row] = next_thread_row_value; // update it once because the next thread will increase its value in mem
//				}
//				bs_row_end_pos = right[row]; // set the right side of the interval
//			}

			pos = binary_search(indices + row_mul_k, col, bs_row_start_pos, bs_row_end_pos);
			if(pos != -1) {
				sum += coef[row] * values[row_mul_k + pos];
				//// update the matrix mem by doing mem[Tid][row] = pos+1
				mem[crt_thread_row_index] = pos + 1; // save pos+1 because we already know that at current position pos we have an element
			}
		}
		out[col] = sum;
	}
}

__global__ void LCG_v41_coop(long m, long k, float* coef, int* indices, float* values, float* out) {
	/*
		Linear Combination of Gradients version #4.1:
			Similar to lin_comb_grads_kernel_v1, but iterate over rows in this method instead of calling the kernel m times and sync grid
	*/
	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id

	double c;
	long row;
	long col;
	long row_mul_k;
	long idx;

	// cols (per block/thread) refers to columns from indices/values matrices, not from 1 to d
	long cols_per_block = div_inc_LCG(k, B); // how many columns to process per block !!!!!!!!!!!!!!!!!!!
	long cols_per_thread = div_inc_LCG(cols_per_block, T); // how many columns to process per thread inside each block
	long block_start_col = Bid * cols_per_block; // starting column for the block
	long block_end_col = block_start_col + cols_per_block; // ending column for the block
	long thread_start_col = block_start_col + Tid * cols_per_thread; // starting column for the thread (some may start outside the matrix - larger than d)
	long thread_end_col = std::min(thread_start_col + cols_per_thread, block_end_col); // ending column for the thread (some may end outside the matrix - larger than d)

	grid_group grid = this_grid();
	for(row = 0, row_mul_k = 0; row < m; ++row, row_mul_k += k) {
		c = coef[row];
		for(col = thread_start_col; col < thread_end_col && col < k; ++col) { // col refers to a column index between 0 and k-1
			idx = row_mul_k + col;
			out[indices[idx]] += (c * values[idx]);
//			printf("Bid=%d, Tid=%d, cols_per_block=%d, cols_per_thread=%d, block_start_col=%d, block_end_col=%d, thread_start_col=%d, thread_end_col=%d, row=%d, col=%d, c=%f, index=%d, value=%f, \n",
//					Bid,    Tid,    cols_per_block,    cols_per_thread,    block_start_col,    block_end_col,    thread_start_col,    thread_end_col,    row,    col,    c,    indices[idx], values[idx]);
		}
//		printf("sync...\n");
		grid.sync();
	}
}

__global__ void LCG_v42_coop(long m, long k, float* coef, int* indices, float* values, float* out) {
	/*
		Linear Combination of Gradients version #4.2:
			Similar to LCG_v41_coop, but uses memory coalescing: consecutive threads process consecutive
	*/
	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id
	const long num_threads = B * T; // total number of threads across all blocks
	const long id = Bid * T + Tid; // this is the thread id when counting the threads from 0 to num_threads-1 from all blocks

	grid_group grid = this_grid();
	long col; // iterates through columns making sure that memory is coalesced
	long row; // iterates through rows of matrices indices and values
	long row_mul_k; // saves the memory location of each row in matrices indices/values
	long idx; // saves the exact index in matrices indices/values for element at position (row, col)
	float c; // saves the coefficient at each row to avoid accessing the global memory often
	for(row = 0, row_mul_k = 0; row < m; ++row, row_mul_k += k) {
		c = coef[row];
		for(col = id; col < k; col += num_threads) {
			idx = row_mul_k + col;
			out[indices[idx]] += (c * values[idx]);
		}
		grid.sync();
	}
}

__global__ void LCG_v42_coop_bf16(long m, long k, float* coef, int* indices, bfloat16* values, float* out) {
	/*
		Linear Combination of Gradients version #4.2:
			Similar to LCG_v42_coop, but uses bf16 type for values
	*/
	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id
	const long num_threads = B * T; // total number of threads across all blocks
	const long id = Bid * T + Tid; // this is the thread id when counting the threads from 0 to num_threads-1 from all blocks

	grid_group grid = this_grid();
	long col; // iterates through columns making sure that memory is coalesced
	long row; // iterates through rows of matrices indices and values
	long row_mul_k; // saves the memory location of each row in matrices indices/values
	long idx; // saves the exact index in matrices indices/values for element at position (row, col)
	float c; // saves the coefficient at each row to avoid accessing the global memory often
	for(row = 0, row_mul_k = 0; row < m; ++row, row_mul_k += k) {
		c = coef[row];
		for(col = id; col < k; col += num_threads) {
			idx = row_mul_k + col;
			out[indices[idx]] += c * __bfloat162float(values[idx]);
		}
		grid.sync();
	}
}

__device__ void set_value_if_non_zero(float value, long index, float *out) {
	if(fabs(value) > FLT_EPSILON) {
		out[index] = value;
	}
}

__global__ void LCG_v43_coop_bf16_vectorized(long m, long k, float* coef, int* indices, bfloat16* values, float* out) {
	/*
		Linear Combination of Gradients version #4.2:
			Similar to LCG_v43_coop_bf16, but retrieves indices as int4 (vectorized types)
	*/
	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id
	const long num_threads = B * T; // total number of threads across all blocks
	const long id = Bid * T + Tid; // this is the thread id when counting the threads from 0 to num_threads-1 from all blocks

	long col; // iterates through columns making sure that memory is coalesced
	long row; // iterates through rows of matrices indices and values
	long row_mul_kVec; // saves the memory location of each row in matrices indices/values
	long idx; // saves the exact index in matrices indices/values for element at position (row, col)
	long kVec = k >> 2; // new limit for vectorized implementation: k / 4
	float c; // saves the coefficient at each row to avoid accessing the global memory often
//	float v; // saves the value to be written to out
	int4 i4, *indices4 = reinterpret_cast<int4*> (indices);
	bfloat164 v4, *values4 = reinterpret_cast<bfloat164*>(values);

	grid_group grid = this_grid();
	for(row = 0, row_mul_kVec = 0; row < m; ++row, row_mul_kVec += kVec) {
		c = coef[row];
		for(col = id; col < kVec; col += num_threads) {
			idx = row_mul_kVec + col;
			i4 = indices4[idx];
			v4 = values4[idx];

			set_value_if_non_zero(c * __bfloat162float(v4.x), i4.x, out);
			set_value_if_non_zero(c * __bfloat162float(v4.y), i4.y, out);
			set_value_if_non_zero(c * __bfloat162float(v4.z), i4.z, out);
			set_value_if_non_zero(c * __bfloat162float(v4.w), i4.w, out);
//			out[i4.x] += c * __bfloat162float(v4.x);
//			out[i4.y] += c * __bfloat162float(v4.y);
//			out[i4.z] += c * __bfloat162float(v4.z);
//			out[i4.w] += c * __bfloat162float(v4.w);
		}
		grid.sync();
	}
}


__global__ void LCG_v43_coop_float_vectorized(long m, long k, float* coef, int* indices, float* values, float* out) {
	/*
		Linear Combination of Gradients version #4.2:
			Similar to LCG_v43_coop_bf16, but retrieves indices as int4 (vectorized types)
	*/
	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id
	const long num_threads = B * T; // total number of threads across all blocks
	const long id = Bid * T + Tid; // this is the thread id when counting the threads from 0 to num_threads-1 from all blocks

	long col; // iterates through columns making sure that memory is coalesced
	long row; // iterates through rows of matrices indices and values
	long row_mul_kVec; // saves the memory location of each row in matrices indices/values
	long idx; // saves the exact index in matrices indices/values for element at position (row, col)
	long kVec = k >> 2; // new limit for vectorized implementation: k / 4
	float c; // saves the coefficient at each row to avoid accessing the global memory often
//	float v; // saves the value to be written to out
	int4 i4, *indices4 = reinterpret_cast<int4*> (indices);
	float4 v4, *values4 = reinterpret_cast<float4*>(values);

	grid_group grid = this_grid();
	for(row = 0, row_mul_kVec = 0; row < m; ++row, row_mul_kVec += kVec) {
		c = coef[row];
		for(col = id; col < kVec; col += num_threads) {
			idx = row_mul_kVec + col;
			i4 = indices4[idx];
			v4 = values4[idx];

			set_value_if_non_zero(c * v4.x, i4.x, out);
			set_value_if_non_zero(c * v4.y, i4.y, out);
			set_value_if_non_zero(c * v4.z, i4.z, out);
			set_value_if_non_zero(c * v4.w, i4.w, out);
//			out[i4.x] += c * __bfloat162float(v4.x);
//			out[i4.y] += c * __bfloat162float(v4.y);
//			out[i4.z] += c * __bfloat162float(v4.z);
//			out[i4.w] += c * __bfloat162float(v4.w);
		}
		grid.sync();
	}
}

__global__ void LCG_v51_bf16(long d_block_size, long k_block_size, long d, long m, long k, float* coef, int* indices, bfloat16* values, float* out) {
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
//		if (i < 0 || i >= d_block_size) {
//			printf("[LCG_v51_bf16] [init mem w zero] Bid=%ld, Tid=%ld, i=%ld, d_block_size=%ld\n", Bid, Tid, i, d_block_size);
//			return;
//		}
		mem[i] = 0;
	}
	__syncthreads();

	long blocks_count = div_inc_LCG(d, d_block_size); // how many k-blocks and d-blocks we have (the same number for both)
	long blocks_per_thread_block = div_inc_LCG(blocks_count, B);
//	if(Bid * blocks_per_thread_block > blocks_count) {
//		// if the current block would process more blocks than required for the current model,
//		return;
//	}
//	if(Bid == 0 && Tid == 0) {
//		printf("blocks_count=%ld, blocks_per_thread_block=%ld\n", blocks_count, blocks_per_thread_block);
//	}

	long idx_block; // used in the first for loop to indicate the current block id (k-block or d-block)
	long d_block_start; // start index for a d-block
	long d_block_end; // end index of a d-block

	long row; // the row index
	long row_mul_k; // holds row * k by starting with 0 and adding k at each row (addition takes less cycles than multiplication)

	float c; // the coefficient for each row TODO: save it in the shared memory using 4-element reads, at the beginning (adjust shared memory size for this)
	long k_block_start; // start index for a k-block
	long k_block_end; // end index of a k-block

	long col; // column index to extract data from indices and values at the current row
	long index; // the 1-D index to extract data from indices and values at the current row (row * k + col)
	long ind; // the data from indices at the index "index"
	bfloat16 val; // the data from values at the index "index"

	// save the following quantities to be used next
	long Bid_MUL_blocksPerThreadBlock = Bid * blocks_per_thread_block;
	long dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = d_block_size * Bid_MUL_blocksPerThreadBlock;
	long kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = k_block_size * Bid_MUL_blocksPerThreadBlock;

	long d_sub_d_block_size;

	for(idx_block = 0; idx_block < blocks_per_thread_block; ++idx_block) { // iterate through all slices (or blocks) of size (m, k_block_size) that will be processed by the current thread block
		d_block_start = dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + d_block_size * idx_block; // start index for the current d-block
		d_block_end = min(d, d_block_start + d_block_size); // end index for the current d-block
		k_block_start = kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + k_block_size * idx_block; // start index for the current k-block
		k_block_end = min(k, k_block_start + k_block_size); // end index for the current k-block
		d_sub_d_block_size = d - d_block_start; // perform this subtraction only once, it will be used below to compute some offsets

		for(row = 0, row_mul_k = 0; row < m; ++row, row_mul_k += k) { // iterate through all rows in the current slice of size (m, k_block_size) to compute the LCG
//			if (row < 0 || row >= m) {
//				printf("[LCG_v51_bf16] [read coef[row]] Bid=%ld, Tid=%ld, row=%ld\n", Bid, Tid, row);
//				return;
//			}
			c = coef[row]; // maybe this read can be improved by saving it into the shared memory? see LCG_v53_bf16_cached_coef

			for(col = k_block_start + Tid; col < k_block_end && col < d_block_end; col += T) { // iterate through all k_block_size columns in the slice of size (m, k_block_size)
				index = row_mul_k + col;
//				if (index < 0 || index >= k * m) {
//					printf("[LCG_v51_bf16] [read indices & values] Bid=%ld, Tid=%ld, index=%ld, m*k=%ld\n", Bid, Tid, index, m*k);
//					return;
//				}
				ind = indices[index];
				val = values[index];
//				if (ind - d_block_start < 0 || ind - d_block_start >= d_block_size) {
//					printf("[LCG_v51_bf16] [mem += c*v] Bid=%ld, Tid=%ld, ind=%ld, d_block_start=%ld, ind - d_block_start=%ld, d_block_size=%ld\n", Bid, Tid, ind, d_block_start, ind - d_block_start, d_block_size);
//					return;
//				}
				mem[ind - d_block_start] += c * __bfloat162float(val); // mem update: this is where the LCG is performed

//				printf("[Bid=%d, Tid=%d] blocks_count = %d, blocks_per_thread_block = %d, idx_block = %d, d_block_start = %d, d_block_end = %d, row = %d, coef = %f, row_mul_k = %d, k_block_start = %d, k_block_end = %d, col = %d, index = %d, ind = %d, val = %f\n",
//					   Bid, Tid, blocks_count, blocks_per_thread_block, idx_block, d_block_start, d_block_end, row, c, row_mul_k, k_block_start, k_block_end, col, index, ind, __bfloat162float(val));

			}
			__syncthreads(); // make sure all threads finish one row to avoid race conditions
		}

		// now, mem contains the LCG for the slice (m, k_block_size). Actually, the results were scattered in a range of size d_block_size in which we only had k_block_size values (k_block_size = 1% of d_block_size)
		for(long i = Tid; i < d_block_size && i < d_sub_d_block_size; i += T) { // mem dump: i < d_sub_d_block_size is required to avoid illegal memory access
//			if (i + d_block_start < 0 || i + d_block_start >= d) {
//				printf("[LCG_v51_bf16] [write out] Bid=%ld, Tid=%ld, i=%ld, d_block_start=%ld, i+d_block_start=%ld, d=%ld\n", Bid, Tid, i, d_block_start, i+d_block_start, d);
//				return;
//			}
			out[i + d_block_start] = mem[i]; // write the content of mem to the right location in out
			mem[i] = 0; // then zerorize mem to prepare for the next block/slice
		}
		__syncthreads(); // make sure all threads finish writing to out
	}
}

__global__ void LCG_v51(long d_block_size, long k_block_size, long d, long m, long k, float* coef, int* indices, float* values, float* out) {
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
//		if (i < 0 || i >= d_block_size) {
//			printf("[LCG_v51] [init mem w zero] Bid=%ld, Tid=%ld, i=%ld, d_block_size=%ld\n", Bid, Tid, i, d_block_size);
//			return;
//		}
		mem[i] = 0;
	}
	__syncthreads();

	long blocks_count = div_inc_LCG(d, d_block_size); // how many k-blocks and d-blocks we have (the same number for both)
	long blocks_per_thread_block = div_inc_LCG(blocks_count, B);
//	if(Bid * blocks_per_thread_block > blocks_count) {
//		// if the current block would process more blocks than required for the current model,
//		return;
//	}
//	if(Bid == 0 && Tid == 0) {
//		printf("blocks_count=%ld, blocks_per_thread_block=%ld\n", blocks_count, blocks_per_thread_block);
//	}

	long idx_block; // used in the first for loop to indicate the current block id (k-block or d-block)
	long d_block_start; // start index for a d-block
	long d_block_end; // end index of a d-block

	long row; // the row index
	long row_mul_k; // holds row * k by starting with 0 and adding k at each row (addition takes less cycles than multiplication)

	float c; // the coefficient for each row TODO: save it in the shared memory using 4-element reads, at the beginning (adjust shared memory size for this)
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

	long d_sub_d_block_size;

	for(idx_block = 0; idx_block < blocks_per_thread_block; ++idx_block) { // iterate through all slices (or blocks) of size (m, k_block_size) that will be processed by the current thread block
		d_block_start = dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + d_block_size * idx_block; // start index for the current d-block
		d_block_end = min(d, d_block_start + d_block_size); // end index for the current d-block
		k_block_start = kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + k_block_size * idx_block; // start index for the current k-block
		k_block_end = min(k, k_block_start + k_block_size); // end index for the current k-block
		d_sub_d_block_size = d - d_block_start; // perform this subtraction only once, it will be used below to compute some offsets

		for(row = 0, row_mul_k = 0; row < m; ++row, row_mul_k += k) { // iterate through all rows in the current slice of size (m, k_block_size) to compute the LCG
//			if (row < 0 || row >= m) {
//				printf("[LCG_v51] [read coef[row]] Bid=%ld, Tid=%ld, row=%ld\n", Bid, Tid, row);
//				return;
//			}
			c = coef[row]; // maybe this read can be improved by saving it into the shared memory? see LCG_v53_bf16_cached_coef

			for(col = k_block_start + Tid; col < k_block_end && col < d_block_end; col += T) { // iterate through all k_block_size columns in the slice of size (m, k_block_size)
				index = row_mul_k + col;
//				if (index < 0 || index >= k * m) {
//					printf("[LCG_v51] [read indices & values] Bid=%ld, Tid=%ld, index=%ld, m*k=%ld\n", Bid, Tid, index, m*k);
//					return;
//				}
				ind = indices[index];
				val = values[index];
//				if (ind - d_block_start < 0 || ind - d_block_start >= d_block_size) {
//					printf("[LCG_v51] [mem += c*v] Bid=%ld, Tid=%ld, ind=%ld, d_block_start=%ld, ind - d_block_start=%ld, d_block_size=%ld\n", Bid, Tid, ind, d_block_start, ind - d_block_start, d_block_size);
//					return;
//				}
				mem[ind - d_block_start] += c * val; // mem update: this is where the LCG is performed

//				printf("[Bid=%d, Tid=%d] blocks_count = %d, blocks_per_thread_block = %d, idx_block = %d, d_block_start = %d, d_block_end = %d, row = %d, coef = %f, row_mul_k = %d, k_block_start = %d, k_block_end = %d, col = %d, index = %d, ind = %d, val = %f\n",
//					   Bid, Tid, blocks_count, blocks_per_thread_block, idx_block, d_block_start, d_block_end, row, c, row_mul_k, k_block_start, k_block_end, col, index, ind, __bfloat162float(val));

			}
			__syncthreads(); // make sure all threads finish one row to avoid race conditions
		}

		// now, mem contains the LCG for the slice (m, k_block_size). Actually, the results were scattered in a range of size d_block_size in which we only had k_block_size values (k_block_size = 1% of d_block_size)
		for(long i = Tid; i < d_block_size && i < d_sub_d_block_size; i += T) { // mem dump: i < d_sub_d_block_size is required to avoid illegal memory access
//			if (i + d_block_start < 0 || i + d_block_start >= d) {
//				printf("[LCG_v51] [write out] Bid=%ld, Tid=%ld, i=%ld, d_block_start=%ld, i+d_block_start=%ld, d=%ld\n", Bid, Tid, i, d_block_start, i+d_block_start, d);
//				return;
//			}
			out[i + d_block_start] = mem[i]; // write the content of mem to the right location in out
			mem[i] = 0; // then zerorize mem to prepare for the next block/slice
		}
		__syncthreads(); // make sure all threads finish writing to out
	}
}

__global__ void LCG_v52_bf16_vectorized2(long d_block_size, long k_block_size, long d, long m, long k, float* coef, int* indices, bfloat16* values, float* out) {
	/*
		Linear Combination of Gradients v5.2: (v522)
			Similar to LCG_v51, but uses vectorized memory access (2 values)
	*/
	extern __shared__ float mem[];
	int2 *indices2 = reinterpret_cast<int2*>(indices);
	bfloat162 *values2 = reinterpret_cast<bfloat162*>(values);
	float2 *out2 = reinterpret_cast<float2*>(out);
	float2 *mem2 = reinterpret_cast<float2*>(mem);
	long d_block_size_div2 = d_block_size >> 1;
	long k_block_size_div2 = k_block_size >> 1;
	long k_div2 = k >> 1;
    long d_div2 = d >> 1;
	long row_mul_k_div2; // holds row * k by starting with 0 and adding k at each row (addition takes less cycles than multiplication)
	int2 ind2; // the data from indices at the index "index"
	bfloat162 val2; // the data from values at the index "index"
	float2 zero2;
	zero2.x = zero2.y = 0;

	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id

	for(long i = Tid; i < d_block_size_div2; i += T) { // mem init
		mem2[i] = zero2;
	}
	__syncthreads();

	long blocks_count = div_inc_LCG(d, d_block_size); // how many k-blocks and d-blocks we have (the same number for both)
	long blocks_per_thread_block = div_inc_LCG(blocks_count, B);

	long idx_block; // used in the first for loop to indicate the current block id (k-block or d-block)
	long d_block_start_div2; // start index for a d-block
	long d_block_end_div2; // end index of a d-block
	long d_block_start_mul2; // start index for a d-block for vectorized memory access

	long row; // the row index

	float c; // the coefficient for each row TODO: save it in the shared memory using 2-element reads, at the beginning (adjust shared memory size for this)
	long k_block_start_div2; // start index for a k-block
	long k_block_end_div2; // end index of a k-block

	long col; // column index to extract data from indices and values at the current row
	long index; // the 1-D index to extract data from indices and values at the current row (row * k + col)

	// save the following quantities to be used next
	long Bid_MUL_blocksPerThreadBlock = Bid * blocks_per_thread_block;
	long dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = d_block_size_div2 * Bid_MUL_blocksPerThreadBlock;
	long kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = k_block_size_div2 * Bid_MUL_blocksPerThreadBlock;

	for(idx_block = 0; idx_block < blocks_per_thread_block; ++idx_block) {
		d_block_start_div2 = dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + d_block_size_div2 * idx_block; // start index for the current d-block
		d_block_end_div2 = min(d_div2, d_block_start_div2 + d_block_size_div2); // end index for the current d-block
		d_block_start_mul2 = d_block_start_div2 << 1;

		for(row = 0, row_mul_k_div2 = 0; row < m; ++row, row_mul_k_div2 += k_div2) {
			c = coef[row];
			k_block_start_div2 = kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + k_block_size_div2 * idx_block; // start index for the current k-block
			k_block_end_div2 = min(k_div2, k_block_start_div2 + k_block_size_div2); // end index for the current k-block

			for(col = k_block_start_div2 + Tid; col < k_block_end_div2 && col < d_block_end_div2; col += T) {
				index = row_mul_k_div2 + col;
				ind2 = indices2[index];
				val2 = values2[index];
				// mem update
				mem[ind2.x - d_block_start_mul2] += c * __bfloat162float(val2.x);
				mem[ind2.y - d_block_start_mul2] += c * __bfloat162float(val2.y);

//				printf("[Bid=%d, Tid=%d] blocks_count = %d, blocks_per_thread_block = %d, idx_block = %d, d_block_start_div2 = %d, d_block_end_div2 = %d, row = %d, coef = %f, row_mul_k_div2 = %d, k_block_start_div2 = %d, k_block_end_div2 = %d, col = %d, index = %d, ind = (%d,%d), val = (%f,%f)\n",
//					   Bid, Tid, blocks_count, blocks_per_thread_block, idx_block, d_block_start_div2, d_block_end_div2, row, c, row_mul_k_div2, k_block_start_div2, k_block_end_div2, col, index, ind2.x, ind2.y, __bfloat162float(val2.x), __bfloat162float(val2.y));

			}
			__syncthreads(); // make sure all threads finish one row to avoid race conditions
		}

		for(long i = Tid; i < d_block_size_div2; i += T) { // mem dump
			out2[i + d_block_start_div2] = mem2[i];
			mem2[i] = zero2;
		}
		__syncthreads(); // make sure all threads finish writing to out
	}
}

__global__ void LCG_v52_bf16_vectorized4(long d_block_size, long k_block_size, long d, long m, long k, float* coef, int* indices, bfloat16* values, float* out) {
	/*
		Linear Combination of Gradients v5.2: (v524)
			Similar to LCG_v51, but uses vectorized memory access (4 values)
	*/
	extern __shared__ float mem[];
	int4 *indices4 = reinterpret_cast<int4*>(indices);
	bfloat164 *values4 = reinterpret_cast<bfloat164*>(values);
	float4 *out4 = reinterpret_cast<float4*>(out);
	float4 *mem4 = reinterpret_cast<float4*>(mem);
	long d_block_size_div4 = d_block_size >> 2;
	long k_block_size_div4 = k_block_size >> 2;
	long k_div4 = k >> 2;
    long d_div4 = d >> 2;
	long row_mul_k_div4; // holds row * k by starting with 0 and adding k at each row (addition takes less cycles than multiplication)
	int4 ind4; // the data from indices at the index "index"
	bfloat164 val4; // the data from values at the index "index"
	float4 zero4;
	zero4.x = zero4.y = zero4.z = zero4.w = 0;

	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id

	for(long i = Tid; i < d_block_size_div4; i += T) { // mem init
		mem4[i] = zero4;
	}
	__syncthreads();

	long blocks_count = div_inc_LCG(d, d_block_size); // how many k-blocks and d-blocks we have (the same number for both)
	long blocks_per_thread_block = div_inc_LCG(blocks_count, B);

	long idx_block; // used in the first for loop to indicate the current block id (k-block or d-block)
	long d_block_start_div4; // start index for a d-block
	long d_block_end_div4; // end index of a d-block
	long d_block_start_mul4; // start index for a d-block for vectorized memory access

	long row; // the row index

	float c; // the coefficient for each row TODO: save it in the shared memory using 4-element reads, at the beginning (adjust shared memory size for this)
	long k_block_start_div4; // start index for a k-block
	long k_block_end_div4; // end index of a k-block

	long col; // column index to extract data from indices and values at the current row
	long index; // the 1-D index to extract data from indices and values at the current row (row * k + col)

	// save the following quantities to be used next
	long Bid_MUL_blocksPerThreadBlock = Bid * blocks_per_thread_block;
	long dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = d_block_size_div4 * Bid_MUL_blocksPerThreadBlock;
	long kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = k_block_size_div4 * Bid_MUL_blocksPerThreadBlock;

	for(idx_block = 0; idx_block < blocks_per_thread_block; ++idx_block) {
		d_block_start_div4 = dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + d_block_size_div4 * idx_block; // start index for the current d-block
		d_block_end_div4 = min(d_div4, d_block_start_div4 + d_block_size_div4); // end index for the current d-block
		d_block_start_mul4 = d_block_start_div4 << 2;

		for(row = 0, row_mul_k_div4 = 0; row < m; ++row, row_mul_k_div4 += k_div4) {
			c = coef[row];
			k_block_start_div4 = kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + k_block_size_div4 * idx_block; // start index for the current k-block
			k_block_end_div4 = min(k_div4, k_block_start_div4 + k_block_size_div4); // end index for the current k-block

			for(col = k_block_start_div4 + Tid; col < k_block_end_div4 && col < d_block_end_div4; col += T) {
				index = row_mul_k_div4 + col;
				ind4 = indices4[index];
				val4 = values4[index];
				mem[ind4.x - d_block_start_mul4] += c * __bfloat162float(val4.x); // mem update
				mem[ind4.y - d_block_start_mul4] += c * __bfloat162float(val4.y);
				mem[ind4.z - d_block_start_mul4] += c * __bfloat162float(val4.z);
				mem[ind4.w - d_block_start_mul4] += c * __bfloat162float(val4.w);

//				printf("[Bid=%d, Tid=%d] blocks_count = %d, blocks_per_thread_block = %d, idx_block = %d, d_block_start_div4 = %d, d_block_end_div4 = %d, row = %d, coef = %f, row_mul_k_div4 = %d, k_block_start_div4 = %d, k_block_end_div4 = %d, col = %d, index = %d, ind = (%d,%d,%d,%d), val = (%f,%f,%f,%f)\n",
//					   Bid, Tid, blocks_count, blocks_per_thread_block, idx_block, d_block_start_div4, d_block_end_div4, row, c, row_mul_k_div4, k_block_start_div4, k_block_end_div4, col, index, ind4.x, ind4.y, ind4.z, ind4.w, __bfloat162float(val4.x), __bfloat162float(val4.y), __bfloat162float(val4.z), __bfloat162float(val4.w));

			}
			__syncthreads(); // make sure all threads finish one row to avoid race conditions
		}

		for(long i = Tid; i < d_block_size_div4; i += T) { // mem dump
			out4[i + d_block_start_div4] = mem4[i];
			mem4[i] = zero4;
		}
		__syncthreads(); // make sure all threads finish writing to out
	}
}

__global__ void LCG_v53_bf16_cached_coef(long d_block_size, long k_block_size, long d, long m, long k, float* coef, int* indices, bfloat16* values, float* out) {
	/*
		Linear Combination of Gradients v5.3 (v53)
			Similar to LCG_v51, but caches coef in the last m components of mem
	*/
	extern __shared__ float mem[];
	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id

	//// new version: coef at the end of mem
	for(long i = Tid; i < d_block_size; i += T) { // 0...d_block_size-1 contains zeros (to save LCG)
		mem[i] = 0;
	}
	__syncthreads();
	for(long i = d_block_size + Tid; i < d_block_size + m; i += T) { // d_block_size...d_block_size+m (to cache coef)
		mem[i] = coef[i - d_block_size];
	}
	__syncthreads();
	//// old version: coef at the beginning of mem
//	for(long i = Tid; i < m; i += T) {
//		mem[i] = coef[i];
//	}
//	__syncthreads();
//	for(long i = m + Tid; i < m + d_block_size; i += T) { // mem init
//		mem[i] = 0;
//	}
//	__syncthreads();

	long blocks_count = div_inc_LCG(d, d_block_size); // how many k-blocks and d-blocks we have (the same number for both)
	long blocks_per_thread_block = div_inc_LCG(blocks_count, B);

	long idx_block; // used in the first for loop to indicate the current block id (k-block or d-block)
	long d_block_start; // start index for a d-block
	long d_block_end; // end index of a d-block

	long row; // the row index
	long row_mul_k; // holds row * k by starting with 0 and adding k at each row (addition takes less cycles than multiplication)

	float c; // the coefficient for each row TODO: save it in the shared memory using 4-element reads, at the beginning (adjust shared memory size for this)
	long k_block_start; // start index for a k-block
	long k_block_end; // end index of a k-block

	long col; // column index to extract data from indices and values at the current row
	long index; // the 1-D index to extract data from indices and values at the current row (row * k + col)
	long ind; // the data from indices at the index "index"
	bfloat16 val; // the data from values at the index "index"

	// save the following quantities to be used next
	long Bid_MUL_blocksPerThreadBlock = Bid * blocks_per_thread_block;
	long dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = d_block_size * Bid_MUL_blocksPerThreadBlock;
	long kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = k_block_size * Bid_MUL_blocksPerThreadBlock;

	for(idx_block = 0; idx_block < blocks_per_thread_block; ++idx_block) {
		d_block_start = dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + d_block_size * idx_block; // start index for the current d-block
		d_block_end = min(d, d_block_start + d_block_size); // end index for the current d-block
		k_block_start = kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + k_block_size * idx_block; // start index for the current k-block
		k_block_end = min(k, k_block_start + k_block_size); // end index for the current k-block

		for(row = 0, row_mul_k = 0; row < m; ++row, row_mul_k += k) {
			c = mem[d_block_size + row]; //// new version: coef at the end of mem
			// c = mem[row]; //// old version: coef at the beginning of mem

			for(col = k_block_start + Tid; col < k_block_end && col < d_block_end; col += T) {
				index = row_mul_k + col;
				ind = indices[index];
				val = values[index];

				//// new version: coef at the end of mem
				mem[ind - d_block_start] += c * __bfloat162float(val); // mem update

				//// old version: coef at the beginning of mem
				//mem[m + ind - d_block_start] += c * __bfloat162float(val); // mem update

				//printf("[Bid=%d, Tid=%d] blocks_count = %d, blocks_per_thread_block = %d, idx_block = %d, d_block_start = %d, d_block_end = %d, row = %d, coef = %f, row_mul_k = %d, k_block_start = %d, k_block_end = %d, col = %d, index = %d, ind = %d, val = %f\n",
				//	   Bid, Tid, blocks_count, blocks_per_thread_block, idx_block, d_block_start, d_block_end, row, c, row_mul_k, k_block_start, k_block_end, col, index, ind, __bfloat162float(val));

			}
			__syncthreads(); // make sure all threads finish one row to avoid race conditions
		}

		//// new version: coef at the end of mem
		for(long i = Tid; i < d_block_size; i += T) { // mem dump
			out[d_block_start + i] = mem[i];
			mem[i] = 0;
		}

		//// old version: coef at the beginning of mem
//		for(long i = m + Tid; i < m + d_block_size; i += T) { // mem dump
//			out[i - m + d_block_start] = mem[i];
//			mem[i] = 0;
//		}
		__syncthreads(); // make sure all threads finish writing to out
	}
}

__global__ void LCG_v54_bf16_cached_coef_vectorized4(long d_block_size, long k_block_size, long d, long m, long k, float* coef, int* indices, bfloat16* values, float* out) {
	/*
		Linear Combination of Gradients v5.4: (v54)
			Similar to LCG_v524, but stores coef in cache and uses vectorized memory access (4 values)
	*/
	extern __shared__ float mem[];
	int4 *indices4 = reinterpret_cast<int4*>(indices);
	bfloat164 *values4 = reinterpret_cast<bfloat164*>(values);
	float4 *coef4 = reinterpret_cast<float4*>(coef);
	float4 *out4 = reinterpret_cast<float4*>(out);
	float4 *mem4 = reinterpret_cast<float4*>(mem);
	long d_block_size_div4 = d_block_size >> 2;
	long k_block_size_div4 = k_block_size >> 2;
	long m_div4 = m >> 2;
	long k_div4 = k >> 2;
    long d_div4 = d >> 2;
	long row_mul_k_div4; // holds row * k by starting with 0 and adding k at each row (addition takes less cycles than multiplication)
	int4 ind4; // the data from indices at the index "index"
	bfloat164 val4; // the data from values at the index "index"
	float4 zero4;
	zero4.x = zero4.y = zero4.z = zero4.w = 0;

	const long B = gridDim.x; // number of blocks
	const long Bid = blockIdx.x; // block id
	const long T = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id

	for(long i = Tid; i < m_div4; i += T) {
		mem4[i] = coef4[i];
	}
	__syncthreads();
	for(long i = m_div4 + Tid; i < m_div4 + d_block_size_div4; i += T) { // mem init
		mem4[i] = zero4;
	}
	__syncthreads();

	long blocks_count = div_inc_LCG(d, d_block_size); // how many k-blocks and d-blocks we have (the same number for both)
	long blocks_per_thread_block = div_inc_LCG(blocks_count, B);

	long idx_block; // used in the first for loop to indicate the current block id (k-block or d-block)
	long d_block_start_div4; // start index for a d-block
	long d_block_end_div4; // end index of a d-block
	long d_block_start_mul4; // start index for a d-block for vectorized memory access

	long row; // the row index

	float c; // the coefficient for each row TODO: save it in the shared memory using 4-element reads, at the beginning (adjust shared memory size for this)
	long k_block_start_div4; // start index for a k-block
	long k_block_end_div4; // end index of a k-block

	long col; // column index to extract data from indices and values at the current row
	long index; // the 1-D index to extract data from indices and values at the current row (row * k + col)

	// save the following quantities to be used next
	long Bid_MUL_blocksPerThreadBlock = Bid * blocks_per_thread_block;
	long dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = d_block_size_div4 * Bid_MUL_blocksPerThreadBlock;
	long kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock = k_block_size_div4 * Bid_MUL_blocksPerThreadBlock;

	for(idx_block = 0; idx_block < blocks_per_thread_block; ++idx_block) {
		d_block_start_div4 = dBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + d_block_size_div4 * idx_block; // start index for the current d-block
		d_block_end_div4 = min(d_div4, d_block_start_div4 + d_block_size_div4); // end index for the current d-block
		d_block_start_mul4 = d_block_start_div4 << 2;

		for(row = 0, row_mul_k_div4 = 0; row < m; ++row, row_mul_k_div4 += k_div4) {
			c = mem[row];
			k_block_start_div4 = kBlockSize_MUL_Bid_MUL_blocksPerThreadBlock + k_block_size_div4 * idx_block; // start index for the current k-block
			k_block_end_div4 = min(k_div4, k_block_start_div4 + k_block_size_div4); // end index for the current k-block

			for(col = k_block_start_div4 + Tid; col < k_block_end_div4 && col < d_block_end_div4; col += T) {
				index = row_mul_k_div4 + col;
				ind4 = indices4[index];
				val4 = values4[index];
				mem[m + ind4.x - d_block_start_mul4] += c * __bfloat162float(val4.x); // mem update
				mem[m + ind4.y - d_block_start_mul4] += c * __bfloat162float(val4.y);
				mem[m + ind4.z - d_block_start_mul4] += c * __bfloat162float(val4.z);
				mem[m + ind4.w - d_block_start_mul4] += c * __bfloat162float(val4.w);

//				printf("[Bid=%d, Tid=%d] blocks_count = %d, blocks_per_thread_block = %d, idx_block = %d, d_block_start_div4 = %d, d_block_end_div4 = %d, row = %d, coef = %f, row_mul_k_div4 = %d, k_block_start_div4 = %d, k_block_end_div4 = %d, col = %d, index = %d, ind = (%d,%d,%d,%d), val = (%f,%f,%f,%f)\n",
//					   Bid, Tid, blocks_count, blocks_per_thread_block, idx_block, d_block_start_div4, d_block_end_div4, row, c, row_mul_k_div4, k_block_start_div4, k_block_end_div4, col, index, ind4.x, ind4.y, ind4.z, ind4.w, __bfloat162float(val4.x), __bfloat162float(val4.y), __bfloat162float(val4.z), __bfloat162float(val4.w));

			}
			__syncthreads(); // make sure all threads finish one row to avoid race conditions
		}

		for(long i = m_div4 + Tid; i < m_div4 + d_block_size_div4; i += T) { // mem dump
			out4[i - m_div4 + d_block_start_div4] = mem4[i];
			mem4[i] = zero4;
		}
		__syncthreads(); // make sure all threads finish writing to out
	}
}

void LCG_cuda(int blocks, int threads, int version, long d, long m, long k, torch::Tensor c, torch::Tensor indices, torch::Tensor values, torch::Tensor out, int use_bf16, long d_block_size, long k_block_size) {
	int *i_ptr = indices.data_ptr<int>();
	float *c_ptr = c.data_ptr<float>();
	float *o_ptr = out.data_ptr<float>();

	if (use_bf16) {
		// bfloat16 *c_ptr_bf16 = (bfloat16*) c.data_ptr();
		bfloat16 *v_ptr_bf16 = (bfloat16*) values.data_ptr();
		void *kernelArgs[] = { &m, &k, &c_ptr, &i_ptr, &v_ptr_bf16, &o_ptr };
		dim3 dimGrid(blocks, 1, 1);
		dim3 dimBlock(threads, 1, 1);
		if(version == 42) {
			cudaLaunchCooperativeKernel((void*)LCG_v42_coop_bf16, dimGrid, dimBlock, kernelArgs);
		} else if (version == 43) {
			cudaLaunchCooperativeKernel((void*)LCG_v43_coop_bf16_vectorized, dimGrid, dimBlock, kernelArgs);
		} else if (version == 51) {
			long sharedMemSize = d_block_size * sizeof(float);
			// see https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf, chapter 19.7. Compute Capability 8.x and the example above it
			cudaFuncSetAttribute(LCG_v51_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
			LCG_v51_bf16<<<blocks, threads, sharedMemSize>>>(d_block_size, k_block_size, d, m, k, c_ptr, i_ptr, v_ptr_bf16, o_ptr);
		} else if (version == 522 || version == 524 || version == 53 || version == 54) {
			if(version == 522) {
//				printf("Using VMA-2: ");
				LCG_v52_bf16_vectorized2<<<blocks, threads, d_block_size * sizeof(float)>>>(d_block_size, k_block_size, d, m, k, c_ptr, i_ptr, v_ptr_bf16, o_ptr);
			} else if (version == 524) {
//				printf("Using VMA-4: ");
				LCG_v52_bf16_vectorized4<<<blocks, threads, d_block_size * sizeof(float)>>>(d_block_size, k_block_size, d, m, k, c_ptr, i_ptr, v_ptr_bf16, o_ptr);
			} else if (version == 53) {
//				printf("Using Cached Coef: "); //                           v
				LCG_v53_bf16_cached_coef<<<blocks, threads, (d_block_size + m) * sizeof(float)>>>(d_block_size, k_block_size, d, m, k, c_ptr, i_ptr, v_ptr_bf16, o_ptr);
			} else if (version == 54) {
//				printf("Using Cached Coef & VMA-4: "); //                   v
				LCG_v54_bf16_cached_coef_vectorized4<<<blocks, threads, (d_block_size + m) * sizeof(float)>>>(d_block_size, k_block_size, d, m, k, c_ptr, i_ptr, v_ptr_bf16, o_ptr);
			}
		} else {
			fprintf(stderr, "Version %d does not have bf16 support for LCG!\n", version);
		}
	} else { // float values
		float *v_ptr = values.data_ptr<float>();
		if(version == 1) {
			blocks = 1 + d / 1024;
			threads = 1024;
			LCG_v1<<<blocks, threads>>>(d, m, k, c_ptr, i_ptr, v_ptr, o_ptr);
		} else if (version == 2) {
			LCG_v2<<<blocks, threads>>>(d, m, k, c_ptr, i_ptr, v_ptr, o_ptr);
		} else if (version == 3) {
			//LCG_v3<<<blocks, threads>>>(d, m, k, c_ptr, i_ptr, v_ptr, o_ptr, mem_ptr);
			fprintf(stderr, "The memory tensor was removed, please add it to use LCG_v3!\n");
		} else if (version == 41 || version == 42 || version == 43) {
			//// pdf page 293 from the CUDA Bible: https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf

			void *kernelArgs[] = { &m, &k, &c_ptr, &i_ptr, &v_ptr, &o_ptr };
			dim3 dimGrid(blocks, 1, 1);
			dim3 dimBlock(threads, 1, 1);

			if(version == 41) {
				cudaLaunchCooperativeKernel((void*)LCG_v41_coop, dimGrid, dimBlock, kernelArgs);
			} else if (version == 42) {
				cudaLaunchCooperativeKernel((void*)LCG_v42_coop, dimGrid, dimBlock, kernelArgs);
			} else if (version == 43) {
				cudaLaunchCooperativeKernel((void*)LCG_v43_coop_float_vectorized, dimGrid, dimBlock, kernelArgs);
			}
		} else if (version == 51) {
			long sharedMemSize = d_block_size * sizeof(float);
			// see https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf, chapter 19.7. Compute Capability 8.x and the example above it
			cudaFuncSetAttribute(LCG_v51, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
			LCG_v51<<<blocks, threads, sharedMemSize>>>(d_block_size, k_block_size, d, m, k, c_ptr, i_ptr, v_ptr, o_ptr);
		}
	}
	gpuErrorCheck(cudaGetLastError());
	gpuErrorCheck(cudaPeekAtLastError());
	gpuErrorCheck(cudaDeviceSynchronize());
//	cout << "coop kernel status =" << status << "\n";
}

