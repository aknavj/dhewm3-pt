#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "renderer_cuda/cu_renderer.h"
#include "renderer_cuda/cu_renderer_math.cuh"

/*
========================
TriangleDrawKernel
========================
*/
__global__ __forceinline__ void TriangleDrawKernel(
	const cudaVertex_t* vertices,
	const cudaTriangle_t* triangles,
	const int* triIndices,
	const cudaBVHNode_t* bvhNodes,
	int numBVHNodes,
	float* framebuffer,
	unsigned char* outputBuffer,
	int width,
	int height,
	int numTriangles,
	const float* cameraPos,
	const float* cameraForward,
	const float* cameraRight,
	const float* cameraUp,
	float fov_x,
	float fov_y
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x >= width || y >= height) {
		return;
	}
	
	int pixelIndex = (y * width + x) * 4;

	float halfTanX = tanf(fov_x * 0.5f * 3.14159265f / 180.0f);
	float halfTanY = tanf(fov_y * 0.5f * 3.14159265f / 180.0f);

	float u = (2.0f * ((float)x + 0.5f) / (float)width  - 1.0f) * halfTanX;
	float v = (2.0f * ((float)y + 0.5f) / (float)height - 1.0f) * halfTanY;

	float dirX = cameraForward[0] + u * cameraRight[0] + v * cameraUp[0];
	float dirY = cameraForward[1] + u * cameraRight[1] + v * cameraUp[1];
	float dirZ = cameraForward[2] + u * cameraRight[2] + v * cameraUp[2];

	float invLen = rsqrtf(dirX * dirX + dirY * dirY + dirZ * dirZ);
	dirX *= invLen;
	dirY *= invLen;
	dirZ *= invLen;

	float origX = cameraPos[0];
	float origY = cameraPos[1];
	float origZ = cameraPos[2];

	float invDirX = 1.0f / (fabsf(dirX) > 1e-8f ? dirX : copysignf(1e-8f, dirX));
	float invDirY = 1.0f / (fabsf(dirY) > 1e-8f ? dirY : copysignf(1e-8f, dirY));
	float invDirZ = 1.0f / (fabsf(dirZ) > 1e-8f ? dirZ : copysignf(1e-8f, dirZ));

	float nearestT = 1e30f;
	int   hitTriIdx = -1;

	if (numBVHNodes > 0) {
		int stack[64];
		int stackPtr = 0;
		stack[stackPtr++] = 0;

		while (stackPtr > 0) {
			int nodeIdx = stack[--stackPtr];
			const cudaBVHNode_t& node = bvhNodes[nodeIdx];

			if (!IntersectAABB(origX, origY, origZ, invDirX, invDirY, invDirZ,
							   node.bounds, nearestT)) {
				continue;
			}

			if (node.primitive_count > 0) {
				for (int i = 0; i < node.primitive_count; i++) {
					int triIdx = triIndices[node.first_primitive + i];
					float t = IntersectTriangle(origX, origY, origZ,
												dirX, dirY, dirZ,
												vertices, triangles[triIdx]);
					if (t > 0.0f && t < nearestT) {
						nearestT = t;
						hitTriIdx = triIdx;
					}
				}
			} else {
				if (node.l_child >= 0 && stackPtr < 63) {
					stack[stackPtr++] = node.l_child;
				}
				if (node.r_child >= 0 && stackPtr < 63) {
					stack[stackPtr++] = node.r_child;
				}
			}
		}
	} else {
		for (int i = 0; i < numTriangles; i++) {
			float t = IntersectTriangle(origX, origY, origZ,
										dirX, dirY, dirZ,
										vertices, triangles[i]);
			if (t > 0.0f && t < nearestT) {
				nearestT = t;
				hitTriIdx = i;
			}
		}
	}

	float r, g, b;

	if (hitTriIdx >= 0) {
		unsigned int hash = cupt_hash((unsigned int)hitTriIdx * 7919u + 1u);
		r = float((hash >>  0) & 0xFF) / 255.0f;
		g = float((hash >>  8) & 0xFF) / 255.0f;
		b = float((hash >> 16) & 0xFF) / 255.0f;

		r = r * 0.7f + 0.3f;
		g = g * 0.7f + 0.3f;
		b = b * 0.7f + 0.3f;
	} else {
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