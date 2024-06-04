#include "../utils.h"

__global__ void symm_block_quant_kernel(LL d, LL q_block_size, uint8_t *xq, bfloat16 *ranges, bfloat16 *x) {
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
	float r = __bfloat162float(ranges[Bid]);
    float S = 2.0f * r / 15.0f; // 15 = 16 - 1 = 2^4 - 1

	bfloat162 vx2; // the value that will store x2[index]
	uint8_t msb; // the MSB of a xq component
	uint8_t lsb; // the LSB of a xq component
//     uint8_t eight = 8;

    for(LL half_index = half_start_index + Tid; half_index < half_end_index; half_index += T) {
		vx2 = x2[half_index];
// 		msb = ((int8_t) floorf((__bfloat162float(vx2.x) + r) / S)) - eight; // xq in [-2^(b-1), 2^(b-1)-1]
// 		lsb = ((int8_t) floorf((__bfloat162float(vx2.y) + r) / S)) - eight;
		msb = (uint8_t) floorf((__bfloat162float(vx2.x) + r) / S); // xq in [0, 2^b-1]
		lsb = (uint8_t) floorf((__bfloat162float(vx2.y) + r) / S);
		xq[half_index] = (msb << 4) | lsb;
// 		printf("[Q] Bid=%ld, Tid=%ld, half_index=%ld, r=%f, S=%f, vx2.x=%f, vx2.y=%f, msb=%d, lsb=%d, xq=%d\n",
// 		    Bid, Tid, half_index, r, S, __bfloat162float(vx2.x), __bfloat162float(vx2.y), msb, lsb, xq[half_index]);
    }

    if((d & 1) && (Bid == B-1) && (Tid == T-1)) {
        bfloat16 vx = x[d - 1];
//         msb = ((int8_t) floorf((__bfloat162float(vx) + r) / S)) - eight; // xq in [-2^(b-1), 2^(b-1)-1]
        msb = (uint8_t) floorf((__bfloat162float(vx) + r) / S); // xq in [0, 2^b-1]
        xq[half_d] = (msb << 4);
//         printf("[Q] Bid=%ld, Tid=%ld, last, r=%f, S=%f, vx=%f, msb=%d, lsb=%d, xq=%d\n",
//             Bid, Tid, r, S, __bfloat162float(vx), msb, 0, xq[half_d]);
    }
}
void symm_block_quant_cuda(LL d, LL q_block_size, torch::Tensor xq, torch::Tensor ranges, torch::Tensor x) {
    LL blocks = 1 + (LL)(d / q_block_size);
    symm_block_quant_kernel<<<blocks, 1024>>>(d,
                                                q_block_size,
                                                (uint8_t*) xq.data_ptr(),
                                                (bfloat16*) ranges.data_ptr(),
                                                (bfloat16*) x.data_ptr());
    // error checks
	gpuErrorCheck(cudaGetLastError());
	gpuErrorCheck(cudaPeekAtLastError());
// 	gpuErrorCheck(cudaDeviceSynchronize());
}