
#include <cuda_runtime.h>
#include <stdint.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "renderer_cuda/cu_renderer.h"
#include "renderer_cuda/cu_renderer_math.cuh"

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

/*
========================
TriangleDrawKernel
========================
*/
__global__ void TriangleDrawKernel(
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

static unsigned int s_frameCounter = 0;


/*
========================
TriangleDrawTexturedKernel
========================
*/
__global__ void TriangleDrawTexturedKernel(
	const cudaVertex_t* vertices,
	const cudaTriangle_t* triangles,
	const cudaMaterial_t* materials,
	const cudaTexture_t* textures,
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
	float hitU = 0.0f;
	float hitV = 0.0f;

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
					float bu, bv;
					float t = IntersectTriangleUV(origX, origY, origZ,
												  dirX, dirY, dirZ,
												  vertices, triangles[triIdx],
												  bu, bv);
					if (t > 0.0f && t < nearestT) {
						nearestT = t;
						hitTriIdx = triIdx;
						hitU = bu;
						hitV = bv;
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
			float bu, bv;
			float t = IntersectTriangleUV(origX, origY, origZ,
										  dirX, dirY, dirZ,
										  vertices, triangles[i],
										  bu, bv);
			if (t > 0.0f && t < nearestT) {
				nearestT = t;
				hitTriIdx = i;
				hitU = bu;
				hitV = bv;
			}
		}
	}

	float r, g, b;

	if (hitTriIdx >= 0) {
		const cudaTriangle_t& tri = triangles[hitTriIdx];
		int i0 = tri.vertexIndices[0];
		int i1 = tri.vertexIndices[1];
		int i2 = tri.vertexIndices[2];

		float w0 = 1.0f - hitU - hitV;
		float w1 = hitU;
		float w2 = hitV;

		float nx = w0 * vertices[i0].normal[0] + w1 * vertices[i1].normal[0] + w2 * vertices[i2].normal[0];
		float ny = w0 * vertices[i0].normal[1] + w1 * vertices[i1].normal[1] + w2 * vertices[i2].normal[1];
		float nz = w0 * vertices[i0].normal[2] + w1 * vertices[i1].normal[2] + w2 * vertices[i2].normal[2];

		float nInvLen = rsqrtf(nx * nx + ny * ny + nz * nz + 1e-12f);
		nx *= nInvLen;
		ny *= nInvLen;
		nz *= nInvLen;

		float tu = w0 * vertices[i0].texcoord[0] + w1 * vertices[i1].texcoord[0] + w2 * vertices[i2].texcoord[0];
		float tv = w0 * vertices[i0].texcoord[1] + w1 * vertices[i1].texcoord[1] + w2 * vertices[i2].texcoord[1];

		const cudaMaterial_t& mat = materials[tri.materialIndex];
		float albedoR = mat.albedo[0];
		float albedoG = mat.albedo[1];
		float albedoB = mat.albedo[2];

		if (mat.albedoTexture >= 0) {
			float texR, texG, texB;
			SampleTexture(textures[mat.albedoTexture], tu, tv, texR, texG, texB);
			albedoR *= texR;
			albedoG *= texG;
			albedoB *= texB;
		}

        // fixed light direction for simple Lambertian shading
		float lightDirX = 0.8f;
		float lightDirY = 0.8f;
		float lightDirZ = 0.8f;

		float NdotL = nx * lightDirX + ny * lightDirY + nz * lightDirZ;
		NdotL = fmaxf(NdotL, 0.0f);

		float ambient = 0.5f;
		float diffuse = 0.9f * NdotL;
		float lighting = ambient + diffuse;

		r = albedoR * lighting;
		g = albedoG * lighting;
		b = albedoB * lighting;

		r = fminf(r, 1.0f);
		g = fminf(g, 1.0f);
		b = fminf(b, 1.0f);
	} else {
		r = 0.05f;
		g = 0.05f;
		b = 0.05f;
	}

	framebuffer[pixelIndex + 0] = r;
	framebuffer[pixelIndex + 1] = g;
	framebuffer[pixelIndex + 2] = b;
	framebuffer[pixelIndex + 3] = 1.0f;

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
extern "C" void CUDA_LaunchRenderView(
	const cudaVertex_t* d_vertices,
	const cudaTriangle_t* d_triangles,
	const int* d_triIndices,
    const cudaMaterial_t* d_materials,
    const cudaTexture_t* d_textures,
	const cudaBVHNode_t* d_bvhNodes,
	int numBVHNodes,
	float* framebuffer,
	unsigned char* outputBuffer,
	int width,
	int height,
	int numTriangles,
	const float* cam_pos,
	const float* cam_forward,
	const float* cam_right,
	const float* cam_up,
	float fov_x,
	float fov_y,
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

	} else if (renderMode == 2) {

		TriangleDrawKernel<<<gridSize, blockSize>>>(
			d_vertices,
			d_triangles,
			d_triIndices,
			d_bvhNodes,
			numBVHNodes,
			framebuffer,
			outputBuffer,
			width,
			height,
			numTriangles,
			cam_pos,
			cam_forward,
			cam_right,
			cam_up,
			fov_x,
			fov_y
		); 

	} else {

        TriangleDrawTexturedKernel<<<gridSize, blockSize>>>(
            d_vertices,
            d_triangles,
            d_materials,
            d_textures,
            d_triIndices,
            d_bvhNodes,
            numBVHNodes,
            framebuffer,
            outputBuffer,
            width,
            height,
            numTriangles,
            cam_pos,
            cam_forward,
            cam_right,
            cam_up,
            fov_x,
            fov_y
        );

    }

	s_frameCounter++;
}
