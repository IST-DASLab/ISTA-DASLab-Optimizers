#include "../utils.h"

__global__ void asymm_block_quant_kernel_bf16_bf16(LL d, LL q_block_size, uint8_t *xq, bfloat16 *xmin, bfloat16 *xmax, bfloat16 *x) {
    /*
		This kernel computes xq = Q(x, x_min, x_max) for 4 bits (implements point 4 from PhD notebook page 55)
		In contrast to "globally" kernel, this kernel works with a single block
		Make sure block_size is always divisible by 2

		We have to read:
		- q_block_size values from x
		- one value from ranges
		- q_block_size / 2 values from xq
	*/
	bfloat162 *x2 = reinterpret_cast<bfloat162*>(x); // we will read two values from x at once

	const LL B = gridDim.x; // number of blocks
	const LL Bid = blockIdx.x; // block id
	const LL T = blockDim.x; // number of threads
	const LL Tid = threadIdx.x; // thread id

	LL half_d = (d >> 1);
	LL half_q_block_size = (q_block_size >> 1); // block size in xq
	LL half_start_index = Bid * half_q_block_size; // start index in vector x
	LL half_end_index = min(half_start_index + half_q_block_size, half_d); // end index in vector x
	float m = __bfloat162float(xmin[Bid]);
	float M = __bfloat162float(xmax[Bid]);
    float u = (M - m) / 15.0f; // 15 = 16 - 1 = 2^4 - 1

	bfloat162 vx2; // the value that will store x2[index]
	uint8_t msb; // the MSB of a xq component
	uint8_t lsb; // the LSB of a xq component

    for(LL half_index = half_start_index + Tid; half_index < half_end_index; half_index += T) {
		vx2 = x2[half_index];
		msb = (uint8_t) floorf((__bfloat162float(vx2.x) - m) / u + 0.5f);
		lsb = (uint8_t) floorf((__bfloat162float(vx2.y) - m) / u + 0.5f);
		xq[half_index] = (msb << 4) | lsb;
    }

    if((d & 1) && (Bid == B-1) && (Tid == T-1)) {
        msb = (uint8_t) floorf((__bfloat162float(x[d - 1]) - m) / u + 0.5f);
        xq[half_d] = (msb << 4);
    }
}

void asymm_block_quant_cuda(LL d, LL q_block_size, torch::Tensor xq, torch::Tensor xmin, torch::Tensor xmax, torch::Tensor x) {
    torch::ScalarType bf16 = torch::ScalarType::BFloat16;
    assert(xmin.scalar_type() == bf16 && xmax.scalar_type() == bf16 && x.scalar_type() == torch::ScalarType::BFloat16);

    LL blocks = 1 + (LL)(d / q_block_size);
    uint8_t *ptr_xq = (uint8_t*) xq.data_ptr();

    asymm_block_quant_kernel_bf16_bf16<<<blocks, 1024>>>(d,
                                                         q_block_size,
                                                         ptr_xq,
                                                         (bfloat16*) xmin.data_ptr(),
                                                         (bfloat16*) xmax.data_ptr(),
                                                         (bfloat16*) x.data_ptr());

    // error checks
	GPU_ERROR_CHECK(cudaGetLastError());
	GPU_ERROR_CHECK(cudaPeekAtLastError());
// 	GPU_ERROR_CHECK(cudaDeviceSynchronize());
}
