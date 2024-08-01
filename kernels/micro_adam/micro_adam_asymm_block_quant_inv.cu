#include "../utils.h"

__global__ void asymm_block_quant_inv_kernel_bf16_bf16(LL d, LL q_block_size, uint8_t *xq, bfloat16 *xmin, bfloat16 *xmax, bfloat16 *x, float alpha) {
	/*
		This kernel computes x += alpha * Q_inv(xq, xmin, xmax) for 4 bits (implements point 1 from PhD #9 notebook page 55)
		Here, x is the output buffer and will already contain the dense gradient
		The output buffer x has d components and xq has d/2 components because one uint8_t stores two 4-bit values
		In contrast to "globally" kernel, this kernel works with a single block
		Make sure block_size is always divisible by 2

		We have to read:
		- q_block_size values from x
		- one value from ranges
		- q_block_size / 2 values from xq
	*/
	bfloat162 *x2 = reinterpret_cast<bfloat162*>(x); // we will read two values from x at once

	const LL B = (LL) gridDim.x; // number of blocks
	const LL Bid = (LL) blockIdx.x; // block id
	const LL T = (LL) blockDim.x; // number of threads
	const LL Tid = (LL) threadIdx.x; // thread id

	LL half_d = (d >> 1);
	LL half_q_block_size = (q_block_size >> 1); // block size in xq
	LL half_start_index = Bid * half_q_block_size; // start index in vector x
	LL half_end_index = minLL(half_start_index + half_q_block_size, half_d); // end index in vector x
// 	if (Bid == 0 && Tid == 0) {
// 	    printf("\n\n\n\t\t\t&&&&&&&&&& half_d=%lld, half_q_block_size=%lld, half_start_index=%lld, half_end_index=%lld\n\n\n");
// 	}
	float m = __bfloat162float(xmin[Bid]);
	float M = __bfloat162float(xmax[Bid]);
    float u = (M - m) / 15.0f; // 15 = 16 - 1 = 2^4 - 1
	bfloat162 vx2; // the value that will store x2[index]
	uint8_t vq; // the value that will store xq[index]
	uint8_t msb; // the MSB of a xq component
	uint8_t lsb; // the LSB of a xq component

	for(LL half_index = half_start_index + Tid; half_index < half_end_index; half_index += T) {
        vx2 = x2[half_index];
        vq = xq[half_index];

		msb = (vq & 0xF0) >> 4;
		lsb = (vq & 0x0F);

        // += operation happens here
		vx2.x += __float2bfloat16((msb * u + m) * alpha);
		vx2.y += __float2bfloat16((lsb * u + m) * alpha);
		x2[half_index] = vx2;
	}
	if((d & 1) && (Bid == B-1) && (Tid == T-1)) {
		bfloat16 vx = x[d - 1];
		vq = xq[half_d];
		msb = (vq & 0xF0) >> 4;
		// += operation happens here
		vx += __float2bfloat16((msb * u + m) * alpha);
		x[d - 1] = vx;
	}
}

void asymm_block_quant_inv_cuda(LL d, LL q_block_size, torch::Tensor xq, torch::Tensor xmin, torch::Tensor xmax, torch::Tensor x, float alpha) {
    ASSERT_BF16(xmin);
    ASSERT_BF16(xmax);
    ASSERT_BF16(x);

    LL blocks = 1 + (LL)(d / q_block_size);
    dim3 B(blocks, 1, 1);
    dim3 T(1024, 1, 1);

    uint8_t *ptr_xq = (uint8_t*) xq.data_ptr();

    asymm_block_quant_inv_kernel_bf16_bf16<<<B, T>>>(d,
                                                     q_block_size,
                                                     (uint8_t*) xq.data_ptr(),
                                                     (bfloat16*) xmin.data_ptr(),
                                                     (bfloat16*) xmax.data_ptr(),
                                                     (bfloat16*) x.data_ptr(),
                                                     alpha);

    // error checks
	GPU_ERROR_CHECK(cudaGetLastError());
	GPU_ERROR_CHECK(cudaPeekAtLastError());
// 	GPU_ERROR_CHECK(cudaDeviceSynchronize());
}
