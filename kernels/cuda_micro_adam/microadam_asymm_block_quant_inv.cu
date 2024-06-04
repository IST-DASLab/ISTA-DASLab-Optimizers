#include "../utils.h"

__device__ LL minLL(LL a, LL b) {
    return (a < b) ? a : b;
}

__global__ void asymm_block_quant_inv_kernel_bf16_bf16(LL d, LL q_block_size, uint8_t *xq, bfloat16 *xmin, bfloat16 *xmax, bfloat16 *x) {
	/*
		This kernel computes x += Q_inv(xq, xmin, xmax) for 4 bits (implements point 1 from PhD notebook page 55)
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
		vx2.x += __float2bfloat16(msb * u + m);
		vx2.y += __float2bfloat16(lsb * u + m);
		x2[half_index] = vx2;
	}
	if((d & 1) && (Bid == B-1) && (Tid == T-1)) {
		bfloat16 vx = x[d - 1];
		vq = xq[half_d];
		msb = (vq & 0xF0) >> 4;
		vx += __float2bfloat16(msb * u + m);
		x[d - 1] = vx;
	}
}

__global__ void asymm_block_quant_inv_kernel_bf16_f32(LL d, LL q_block_size, uint8_t *xq, bfloat16 *xmin, bfloat16 *xmax, float *x) {
	/*
		This kernel computes x += Q_inv(xq, xmin, xmax) for 4 bits (implements point 1 from PhD notebook page 55)
		Here, x is the output buffer and will already contain the dense gradient
		The output buffer x has d components and xq has d/2 components because one uint8_t stores two 4-bit values
		In contrast to "globally" kernel, this kernel works with a single block
		Make sure block_size is always divisible by 2

		We have to read:
		- q_block_size values from x
		- one value from ranges
		- q_block_size / 2 values from xq
	*/
	float2 *x2 = reinterpret_cast<float2*>(x); // we will read two values from x at once

	const LL B = gridDim.x; // number of blocks
	const LL Bid = blockIdx.x; // block id
	const LL T = blockDim.x; // number of threads
	const LL Tid = threadIdx.x; // thread id

	LL half_d = (d >> 1);
	LL half_q_block_size = (q_block_size >> 1); // block size in xq
	LL half_start_index = Bid * half_q_block_size; // start index in vector x
	LL half_end_index = minLL(half_start_index + half_q_block_size, half_d); // end index in vector x
	float m = __bfloat162float(xmin[Bid]);
	float M = __bfloat162float(xmax[Bid]);
    float u = (M - m) / 15.0f; // 15 = 16 - 1 = 2^4 - 1

	float2 vx2; // the value that will store x2[index]
	uint8_t vq; // the value that will store xq[index]
	uint8_t msb; // the MSB of a xq component
	uint8_t lsb; // the LSB of a xq component

	for(LL half_index = half_start_index + Tid; half_index < half_end_index; half_index += T) {
        vx2 = x2[half_index];
        vq = xq[half_index];

		msb = (vq & 0xF0) >> 4;
		lsb = (vq & 0x0F);

        // += operation happens here
		vx2.x += msb * u + m;
		vx2.y += lsb * u + m;
		x2[half_index] = vx2;
	}
	if((d & 1) && (Bid == B-1) && (Tid == T-1)) {
		float vx = x[d - 1];
		vq = xq[half_d];
		msb = (vq & 0xF0) >> 4;
		vx += (msb * u + m);
		x[d - 1] = vx;
	}
}

__global__ void asymm_block_quant_inv_kernel_f32_bf16(LL d, LL q_block_size, uint8_t *xq, float *xmin, float *xmax, bfloat16 *x) {
	/*
		This kernel computes x += Q_inv(xq, xmin, xmax) for 4 bits (implements point 1 from PhD notebook page 55)
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

	const LL B = gridDim.x; // number of blocks
	const LL Bid = blockIdx.x; // block id
	const LL T = blockDim.x; // number of threads
	const LL Tid = threadIdx.x; // thread id

	LL half_d = (d >> 1);
	LL half_q_block_size = (q_block_size >> 1); // block size in xq
	LL half_start_index = Bid * half_q_block_size; // start index in vector x
	LL half_end_index = minLL(half_start_index + half_q_block_size, half_d); // end index in vector x
	float m = xmin[Bid];
	float M = xmax[Bid];
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
		vx2.x += __float2bfloat16(msb * u + m);
		vx2.y += __float2bfloat16(lsb * u + m);
		x2[half_index] = vx2;
	}
	if((d & 1) && (Bid == B-1) && (Tid == T-1)) {
		bfloat16 vx = x[d - 1];
		vq = xq[half_d];
		msb = (vq & 0xF0) >> 4;
		vx += __float2bfloat16(msb * u + m);
		x[d - 1] = vx;
	}
}

__global__ void asymm_block_quant_inv_kernel_f32_f32(LL d, LL q_block_size, uint8_t *xq, float *xmin, float *xmax, float *x) {
	/*
		This kernel computes x += Q_inv(xq, xmin, xmax) for 4 bits (implements point 1 from PhD notebook page 55)
		Here, x is the output buffer and will already contain the dense gradient
		The output buffer x has d components and xq has d/2 components because one uint8_t stores two 4-bit values
		In contrast to "globally" kernel, this kernel works with a single block
		Make sure block_size is always divisible by 2

		We have to read:
		- q_block_size values from x
		- one value from ranges
		- q_block_size / 2 values from xq
	*/
	float2 *x2 = reinterpret_cast<float2*>(x); // we will read two values from x at once

	const LL B = gridDim.x; // number of blocks
	const LL Bid = blockIdx.x; // block id
	const LL T = blockDim.x; // number of threads
	const LL Tid = threadIdx.x; // thread id

	LL half_d = (d >> 1);
	LL half_q_block_size = (q_block_size >> 1); // block size in xq
	LL half_start_index = Bid * half_q_block_size; // start index in vector x
	LL half_end_index = minLL(half_start_index + half_q_block_size, half_d); // end index in vector x
	float m = xmin[Bid];
	float M = xmax[Bid];
    float u = (M - m) / 15.0f; // 15 = 16 - 1 = 2^4 - 1

	float2 vx2; // the value that will store x2[index]
	uint8_t vq; // the value that will store xq[index]
	uint8_t msb; // the MSB of a xq component
	uint8_t lsb; // the LSB of a xq component

	for(LL half_index = half_start_index + Tid; half_index < half_end_index; half_index += T) {
        vx2 = x2[half_index];
        vq = xq[half_index];

		msb = (vq & 0xF0) >> 4;
		lsb = (vq & 0x0F);

        // += operation happens here
		vx2.x += (msb * u + m);
		vx2.y += (lsb * u + m);
		x2[half_index] = vx2;
	}
	if((d & 1) && (Bid == B-1) && (Tid == T-1)) {
		float vx = x[d - 1];
		vq = xq[half_d];
		msb = (vq & 0xF0) >> 4;
		vx += (msb * u + m);
		x[d - 1] = vx;
	}
}

void asymm_block_quant_inv_cuda(LL d, LL q_block_size, torch::Tensor xq, torch::Tensor xmin, torch::Tensor xmax, torch::Tensor x) {
    LL blocks = 1 + (LL)(d / q_block_size);
    dim3 B(blocks, 1, 1);
    dim3 T(1024, 1, 1);

    torch::ScalarType type_stat = xmin.scalar_type();
    torch::ScalarType type_x = x.scalar_type();

    uint8_t *ptr_xq = (uint8_t*) xq.data_ptr();
//     printf("\n\n\n\t\t\t\t\t********** min/max type = %s, x type = %s\n\n\n", type_stat, type_x);
//     cout << "\n\n\n\t\t\t\t@@@@@min/max type " << type_stat << " x type " << type_x << ", blocks = " << blocks << "\n\n\n";

//     asymm_block_quant_inv_kernel_bf16_bf16<<<B, T>>>(d, q_block_size, ptr_xq, (bfloat16*) xmin.data_ptr(), (bfloat16*) xmax.data_ptr(), (bfloat16*) x.data_ptr());
    switch(type_stat) {
        case torch::ScalarType::BFloat16: // bf16
            switch(type_x) {
                case torch::ScalarType::BFloat16: // bf16
                    asymm_block_quant_inv_kernel_bf16_bf16<<<B, T>>>(d, q_block_size, ptr_xq, (bfloat16*) xmin.data_ptr(), (bfloat16*) xmax.data_ptr(), (bfloat16*) x.data_ptr());
                    break;
                case torch::ScalarType::Float: // f32
                    asymm_block_quant_inv_kernel_bf16_f32<<<B, T>>>(d, q_block_size, ptr_xq, (bfloat16*) xmin.data_ptr(), (bfloat16*) xmax.data_ptr(), (float*) x.data_ptr());
                    break;
            }
            break;
        case torch::ScalarType::Float: // f32
            switch(type_x) {
                case torch::ScalarType::BFloat16: // bf16
                    asymm_block_quant_inv_kernel_f32_bf16<<<B, T>>>(d, q_block_size, ptr_xq, (float*) xmin.data_ptr(), (float*) xmax.data_ptr(), (bfloat16*) x.data_ptr());
                    break;
                case torch::ScalarType::Float: // f32
                    asymm_block_quant_inv_kernel_f32_f32<<<B, T>>>(d, q_block_size, ptr_xq, (float*) xmin.data_ptr(), (float*) xmax.data_ptr(), (float*) x.data_ptr());
                    break;
            }
            break;
    }

    // error checks
	gpuErrorCheck(cudaGetLastError());
	gpuErrorCheck(cudaPeekAtLastError());
// 	gpuErrorCheck(cudaDeviceSynchronize());
}
