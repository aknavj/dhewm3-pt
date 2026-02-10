
#include <cuda_runtime.h>
#include <stdint.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "renderer_cuda/cu_renderer.h"

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
TriangleDrawKernel
========================
*/
__global__ void TriangleDrawKernel(
	const cudaVertex_t* vertices,
	const cudaTriangle_t* triangles,
	const int* triIndices,
	float* framebuffer,
	unsigned char* outputBuffer,
	int width,
	int height,
	int numTriangles,
    const float* cameraPos,
	const float* cameraForward,
	const float* cameraRight,
	const float* cameraUp,
	float fovX,
	float fovY
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x >= width || y >= height) {
		return;
	}
	
	int pixelIndex = (y * width + x) * 4;

	// generate camera ray through this pixel
	float halfTanX = tanf(fovX * 0.5f * 3.14159265f / 180.0f);
	float halfTanY = tanf(fovY * 0.5f * 3.14159265f / 180.0f);

	float u = (2.0f * ((float)x + 0.5f) / (float)width  - 1.0f) * halfTanX;
	float v = (2.0f * ((float)y + 0.5f) / (float)height - 1.0f) * halfTanY;

	float dirX = cameraForward[0] + u * cameraRight[0] + v * cameraUp[0];
	float dirY = cameraForward[1] + u * cameraRight[1] + v * cameraUp[1];
	float dirZ = cameraForward[2] + u * cameraRight[2] + v * cameraUp[2];

	// normalize direction
	float invLen = rsqrtf(dirX * dirX + dirY * dirY + dirZ * dirZ);
	dirX *= invLen;
	dirY *= invLen;
	dirZ *= invLen;

	float origX = cameraPos[0];
	float origY = cameraPos[1];
	float origZ = cameraPos[2];

	// brute-force test all triangles (Moller-Trumbore)
	float nearestT = 1e30f;
	int   hitTriIdx = -1;

	for (int i = 0; i < numTriangles; i++) {
		int i0 = triangles[i].vertexIndices[0];
		int i1 = triangles[i].vertexIndices[1];
		int i2 = triangles[i].vertexIndices[2];

		float v0x = vertices[i0].position[0];
		float v0y = vertices[i0].position[1];
		float v0z = vertices[i0].position[2];

		// edge vectors
		float e1x = vertices[i1].position[0] - v0x;
		float e1y = vertices[i1].position[1] - v0y;
		float e1z = vertices[i1].position[2] - v0z;

		float e2x = vertices[i2].position[0] - v0x;
		float e2y = vertices[i2].position[1] - v0y;
		float e2z = vertices[i2].position[2] - v0z;

		// h = dir x e2
		float hx = dirY * e2z - dirZ * e2y;
		float hy = dirZ * e2x - dirX * e2z;
		float hz = dirX * e2y - dirY * e2x;

		float a = e1x * hx + e1y * hy + e1z * hz;
		if (fabsf(a) < 1e-7f) {
			continue;
		}

		float f = 1.0f / a;
		float sx = origX - v0x;
		float sy = origY - v0y;
		float sz = origZ - v0z;

		float ub = f * (sx * hx + sy * hy + sz * hz);
		if (ub < 0.0f || ub > 1.0f) {
			continue;
		}

		// q = s x e1
		float qx = sy * e1z - sz * e1y;
		float qy = sz * e1x - sx * e1z;
		float qz = sx * e1y - sy * e1x;

		float vb = f * (dirX * qx + dirY * qy + dirZ * qz);
		if (vb < 0.0f || ub + vb > 1.0f) {
			continue;
		}

		float t = f * (e2x * qx + e2y * qy + e2z * qz);
		if (t > 0.001f && t < nearestT) {
			nearestT = t;
			hitTriIdx = i;
		}
	}

	// shade pixel
	float r, g, b;

	if (hitTriIdx >= 0) {
		// deterministic random color per triangle (for debugging)
		unsigned int hash = cupt_hash((unsigned int)hitTriIdx * 7919u + 1u);
		r = float((hash >>  0) & 0xFF) / 255.0f;
		g = float((hash >>  8) & 0xFF) / 255.0f;
		b = float((hash >> 16) & 0xFF) / 255.0f;

		// boost minimum brightness so dark triangles are still visible
		r = r * 0.7f + 0.3f;
		g = g * 0.7f + 0.3f;
		b = b * 0.7f + 0.3f;
	} else {
		// background - dark gray
		r = 0.05f;
		g = 0.05f;
		b = 0.05f;
	}

	// write HDR framebuffer (float4 RGBA)
	framebuffer[pixelIndex + 0] = r;
	framebuffer[pixelIndex + 1] = g;
	framebuffer[pixelIndex + 2] = b;
	framebuffer[pixelIndex + 3] = 1.0f;

	// write LDR output buffer (RGBA8)
	outputBuffer[pixelIndex + 0] = (unsigned char)(r * 255.0f);
	outputBuffer[pixelIndex + 1] = (unsigned char)(g * 255.0f);
	outputBuffer[pixelIndex + 2] = (unsigned char)(b * 255.0f);
	outputBuffer[pixelIndex + 3] = 255;
}

/*
========================
CUDA_LaunchRenderView
========================
*/
static unsigned int s_frameCounter = 0;

extern "C" void CUDA_LaunchRenderView(
    const cudaVertex_t* d_vertices,
	const cudaTriangle_t* d_triangles,
	const int* d_triIndices,
    float* framebuffer,
    unsigned char* outputBuffer,
    int width,
    int height,
    int numTriangles,
    const float* cam_pos,
	const float* cam_forward,
	const float* cam_right,
	const float* cam_up,
	float fovX,
	float fovY,
    int renderMode
) {
    
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
	dim3 gridSize(
        (width + TILE_SIZE - 1) / TILE_SIZE, 
        (height + TILE_SIZE - 1) / TILE_SIZE
    );
	

    if (renderMode == 1) {

        NoiseKernel<<<gridSize, blockSize>>>(
            framebuffer,
            outputBuffer,
            width,
            height,
            s_frameCounter
        );

    } else {

        // draw triangles
        TriangleDrawKernel<<<gridSize, blockSize>>>(
            d_vertices,
            d_triangles,
            d_triIndices,
            framebuffer,
            outputBuffer,
            width,
            height,
            numTriangles,
            cam_pos,
            cam_forward,
            cam_right,
            cam_up,
            fovX,
            fovY
        ); 

    }

    s_frameCounter++;
}
