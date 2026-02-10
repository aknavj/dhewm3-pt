
#include <cuda_runtime.h>
#include <stdint.h>

/*
=================================================================================
Simple hash-based PRNG for per-pixel noise generation.
=================================================================================
*/

__device__ unsigned int cupt_hash(unsigned int seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

__device__ float cupt_random(unsigned int seed) {
    return float(cupt_hash(seed)) / float(0xFFFFFFFFu);
}

/*
========================
NoiseKernel
========================
*/
__global__ void NoiseKernel(
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

    // unique per-pixel seed mixed with a frame counter
    unsigned int seed = (unsigned int)(idx) ^ frameSeed;

    float r = cupt_random(seed);
    float g = cupt_random(seed * 2654435761u);
    float b = cupt_random(seed * 2246822519u);

    // write HDR framebuffer (RGBA float)
    int fb = idx * 4;
    framebuffer[fb + 0] = r;
    framebuffer[fb + 1] = g;
    framebuffer[fb + 2] = b;
    framebuffer[fb + 3] = 1.0f;

    // write LDR output buffer (RGBA8) – simple clamp tonemap
    outputBuffer[fb + 0] = (unsigned char)(fminf(r, 1.0f) * 255.0f);
    outputBuffer[fb + 1] = (unsigned char)(fminf(g, 1.0f) * 255.0f);
    outputBuffer[fb + 2] = (unsigned char)(fminf(b, 1.0f) * 255.0f);
    outputBuffer[fb + 3] = 255;
}

/*
========================
CUDA_LaunchRenderView
========================
*/
static unsigned int s_frameCounter = 0;

extern "C" void CUDA_LaunchRenderView(
    float* framebuffer,
    unsigned char* outputBuffer,
    int width,
    int height
) {
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width  + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    NoiseKernel<<<gridSize, blockSize>>>(
        framebuffer,
        outputBuffer,
        width,
        height,
        s_frameCounter
    );

    s_frameCounter++;
}
