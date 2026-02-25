#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <curand_kernel.h>
#include "renderer_cuda/cu_renderer_math.cuh"

/*
========================
NoiseKernel
========================
*/
__global__ __forceinline__ void NoiseKernel(
	float* framebuffer,
	unsigned char* outputBuffer,
	int width,
	int height,
	unsigned int frameSeed
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	int idx = y * width + x;

	unsigned int seed = (unsigned int)(idx) ^ frameSeed;

	float r = cupt_random(seed);
	float g = cupt_random(seed * 2654435761u);
	float b = cupt_random(seed * 2246822519u);

	int fb = idx * 4;
	framebuffer[fb + 0] = r;
	framebuffer[fb + 1] = g;
	framebuffer[fb + 2] = b;
	framebuffer[fb + 3] = 1.0f;

	outputBuffer[fb + 0] = (unsigned char)(fminf(r, 1.0f) * 255.0f);
	outputBuffer[fb + 1] = (unsigned char)(fminf(g, 1.0f) * 255.0f);
	outputBuffer[fb + 2] = (unsigned char)(fminf(b, 1.0f) * 255.0f);
	outputBuffer[fb + 3] = 255;
}
