
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
TraceShadowRay
========================
*/
__device__ __forceinline__ bool TraceShadowRay(
	float oX, float oY, float oZ,
	float dX, float dY, float dZ,
	float maxDist,
	const cudaVertex_t* vertices,
	const cudaTriangle_t* triangles,
	const int* triIndices,
	const cudaBVHNode_t* bvhNodes,
	int numBVHNodes
) {
	float idX = 1.0f / (fabsf(dX) > 1e-8f ? dX : copysignf(1e-8f, dX));
	float idY = 1.0f / (fabsf(dY) > 1e-8f ? dY : copysignf(1e-8f, dY));
	float idZ = 1.0f / (fabsf(dZ) > 1e-8f ? dZ : copysignf(1e-8f, dZ));

	int stack[64];
	int sp = 0;
	stack[sp++] = 0;

	while (sp > 0) {
		int ni = stack[--sp];
		const cudaBVHNode_t& nd = bvhNodes[ni];

		if (!IntersectAABB(oX, oY, oZ, idX, idY, idZ, nd.bounds, maxDist)) {
			continue;
		}

		if (nd.primitive_count > 0) {
			for (int i = 0; i < nd.primitive_count; i++) {
				int ti = triIndices[nd.first_primitive + i];
				float t = IntersectTriangle(oX, oY, oZ, dX, dY, dZ,
										   vertices, triangles[ti]);
				if (t > 0.001f && t < maxDist) {
					return true;
				}
			}
		} else {
			if (nd.l_child >= 0 && sp < 63) stack[sp++] = nd.l_child;
			if (nd.r_child >= 0 && sp < 63) stack[sp++] = nd.r_child;
		}
	}
	return false;
}

/*
========================
TraceRay
========================
*/
__device__ __forceinline__ float TraceRay(
	float oX, float oY, float oZ,
	float dX, float dY, float dZ,
	const cudaVertex_t* vertices,
	const cudaTriangle_t* triangles,
	const int* triIndices,
	const cudaBVHNode_t* bvhNodes,
	int numBVHNodes,
	int numTriangles,
	int& outTriIdx,
	float& outU,
	float& outV
) {
	float idX = 1.0f / (fabsf(dX) > 1e-8f ? dX : copysignf(1e-8f, dX));
	float idY = 1.0f / (fabsf(dY) > 1e-8f ? dY : copysignf(1e-8f, dY));
	float idZ = 1.0f / (fabsf(dZ) > 1e-8f ? dZ : copysignf(1e-8f, dZ));

	float nearest = 1e30f;
	outTriIdx = -1;

	if (numBVHNodes > 0) {
		int stack[64];
		int sp = 0;
		stack[sp++] = 0;

		while (sp > 0) {
			int ni = stack[--sp];
			const cudaBVHNode_t& nd = bvhNodes[ni];

			if (!IntersectAABB(oX, oY, oZ, idX, idY, idZ, nd.bounds, nearest)) {
				continue;
			}

			if (nd.primitive_count > 0) {
				for (int i = 0; i < nd.primitive_count; i++) {
					int ti = triIndices[nd.first_primitive + i];
					float bu, bv;
					float t = IntersectTriangleUV(oX, oY, oZ, dX, dY, dZ,
												 vertices, triangles[ti], bu, bv);
					if (t > 0.0f && t < nearest) {
						nearest = t;
						outTriIdx = ti;
						outU = bu;
						outV = bv;
					}
				}
			} else {
				if (nd.l_child >= 0 && sp < 63) stack[sp++] = nd.l_child;
				if (nd.r_child >= 0 && sp < 63) stack[sp++] = nd.r_child;
			}
		}
	} else {
		for (int i = 0; i < numTriangles; i++) {
			float bu, bv;
			float t = IntersectTriangleUV(oX, oY, oZ, dX, dY, dZ,
										 vertices, triangles[i], bu, bv);
			if (t > 0.0f && t < nearest) {
				nearest = t;
				outTriIdx = i;
				outU = bu;
				outV = bv;
			}
		}
	}
	return nearest;
}

/*
========================
TriangleDrawRayTracedKernel
========================
*/
__global__ void TriangleDrawRayTracedKernel(
	const cudaVertex_t* vertices,
	const cudaTriangle_t* triangles,
	const cudaMaterial_t* materials,
	const cudaTexture_t* textures,
	const cudaLight_t* lights,
	int numLights,
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

	// generate primary ray
	float halfTanX = tanf(fov_x * 0.5f * 3.14159265f / 180.0f);
	float halfTanY = tanf(fov_y * 0.5f * 3.14159265f / 180.0f);

	float su = (2.0f * ((float)x + 0.5f) / (float)width  - 1.0f) * halfTanX;
	float sv = (2.0f * ((float)y + 0.5f) / (float)height - 1.0f) * halfTanY;

	float rayDirX = cameraForward[0] + su * cameraRight[0] + sv * cameraUp[0];
	float rayDirY = cameraForward[1] + su * cameraRight[1] + sv * cameraUp[1];
	float rayDirZ = cameraForward[2] + su * cameraRight[2] + sv * cameraUp[2];

	float il = rsqrtf(rayDirX * rayDirX + rayDirY * rayDirY + rayDirZ * rayDirZ);
	rayDirX *= il;
	rayDirY *= il;
	rayDirZ *= il;

	float rayOrigX = cameraPos[0];
	float rayOrigY = cameraPos[1];
	float rayOrigZ = cameraPos[2];

	// whitted-style iterative bounce loop
	float colorR = 0.0f, colorG = 0.0f, colorB = 0.0f;
	float throughR = 1.0f, throughG = 1.0f, throughB = 1.0f;

	const int MAX_BOUNCES = 4;
	const float SPECULAR_POWER = 64.0f;
	const float REFLECTIVITY = 0.15f;

	for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {

		int hitIdx;
		float hitU, hitV;
		float hitT = TraceRay(rayOrigX, rayOrigY, rayOrigZ,
							  rayDirX, rayDirY, rayDirZ,
							  vertices, triangles, triIndices, bvhNodes,
							  numBVHNodes, numTriangles,
							  hitIdx, hitU, hitV);

		if (hitIdx < 0) {
			// sky / miss
			colorR += throughR * 0.02f;
			colorG += throughG * 0.02f;
			colorB += throughB * 0.03f;
			break;
		}

		const cudaTriangle_t& tri = triangles[hitIdx];
		int i0 = tri.vertexIndices[0];
		int i1 = tri.vertexIndices[1];
		int i2 = tri.vertexIndices[2];

		float w0 = 1.0f - hitU - hitV;
		float w1 = hitU;
		float w2 = hitV;

		// interpolate surface normal
		float nx = w0 * vertices[i0].normal[0] + w1 * vertices[i1].normal[0] + w2 * vertices[i2].normal[0];
		float ny = w0 * vertices[i0].normal[1] + w1 * vertices[i1].normal[1] + w2 * vertices[i2].normal[1];
		float nz = w0 * vertices[i0].normal[2] + w1 * vertices[i1].normal[2] + w2 * vertices[i2].normal[2];
		float nIL = rsqrtf(nx * nx + ny * ny + nz * nz + 1e-12f);
		nx *= nIL;
		ny *= nIL;
		nz *= nIL;

		// interpolate texcoord
		float tu = w0 * vertices[i0].texcoord[0] + w1 * vertices[i1].texcoord[0] + w2 * vertices[i2].texcoord[0];
		float tv = w0 * vertices[i0].texcoord[1] + w1 * vertices[i1].texcoord[1] + w2 * vertices[i2].texcoord[1];

		// hit position
		float pX = rayOrigX + rayDirX * hitT;
		float pY = rayOrigY + rayDirY * hitT;
		float pZ = rayOrigZ + rayDirZ * hitT;

		// material lookup
		const cudaMaterial_t& mat = materials[tri.materialIndex];
		float albR = mat.albedo[0];
		float albG = mat.albedo[1];
		float albB = mat.albedo[2];

		if (mat.albedoTexture >= 0) {
			float tR, tG, tB;
			SampleTexture(textures[mat.albedoTexture], tu, tv, tR, tG, tB);
			albR *= tR;
			albG *= tG;
			albB *= tB;
		}

		// view direction (points toward camera)
		float vX = -rayDirX;
		float vY = -rayDirY;
		float vZ = -rayDirZ;

		// accumulate direct lighting from all point lights
		float diffR = 0.0f, diffG = 0.0f, diffB = 0.0f;
		float specR = 0.0f, specG = 0.0f, specB = 0.0f;

		for (int li = 0; li < numLights; li++) {
			const cudaLight_t& light = lights[li];

			float lX = light.position[0] - pX;
			float lY = light.position[1] - pY;
			float lZ = light.position[2] - pZ;

			float dSq = lX * lX + lY * lY + lZ * lZ;
			float d = sqrtf(dSq + 1e-12f);
			float iD = 1.0f / d;
			lX *= iD;
			lY *= iD;
			lZ *= iD;

			// skip if outside light radius
			float R = light.radius;
			if (d >= R) {
				continue;
			}

			// shadow ray
			float sOx = pX + nx * 0.1f;
			float sOy = pY + ny * 0.1f;
			float sOz = pZ + nz * 0.1f;

			if (numBVHNodes > 0 &&
				TraceShadowRay(sOx, sOy, sOz, lX, lY, lZ, d,
							   vertices, triangles, triIndices, bvhNodes, numBVHNodes)) {
				continue;
			}

			// smooth radius-based attenuation: (1 - (d/R)^2)^2
			float ratio = d / R;
			float falloff = 1.0f - ratio * ratio;
			falloff = falloff * falloff;
			float atten = light.intensity * falloff;

			// Lambertian diffuse
			float NdotL = fmaxf(nx * lX + ny * lY + nz * lZ, 0.0f);

			diffR += light.color[0] * NdotL * atten;
			diffG += light.color[1] * NdotL * atten;
			diffB += light.color[2] * NdotL * atten;

			// Blinn-Phong specular
			float hX = lX + vX;
			float hY = lY + vY;
			float hZ = lZ + vZ;
			float hIL = rsqrtf(hX * hX + hY * hY + hZ * hZ + 1e-12f);
			hX *= hIL;
			hY *= hIL;
			hZ *= hIL;

			float NdotH = fmaxf(nx * hX + ny * hY + nz * hZ, 0.0f);
			float spec = powf(NdotH, SPECULAR_POWER);

			specR += light.color[0] * spec * atten;
			specG += light.color[1] * spec * atten;
			specB += light.color[2] * spec * atten;
		}

		// combine: ambient + diffuse * albedo + specular
		float ambient = 0.012f;
		float localR = albR * (ambient + diffR) + REFLECTIVITY * specR;
		float localG = albG * (ambient + diffG) + REFLECTIVITY * specG;
		float localB = albB * (ambient + diffB) + REFLECTIVITY * specB;

		// add local contribution weighted by current throughput
		float oneMinusRefl = 1.0f - REFLECTIVITY;
		colorR += throughR * localR * oneMinusRefl;
		colorG += throughG * localG * oneMinusRefl;
		colorB += throughB * localB * oneMinusRefl;

		// reflection ray for next bounce
		float RdotN = rayDirX * nx + rayDirY * ny + rayDirZ * nz;
		rayDirX = rayDirX - 2.0f * RdotN * nx;
		rayDirY = rayDirY - 2.0f * RdotN * ny;
		rayDirZ = rayDirZ - 2.0f * RdotN * nz;

		rayOrigX = pX + nx * 0.01f;
		rayOrigY = pY + ny * 0.01f;
		rayOrigZ = pZ + nz * 0.01f;

		// attenuate throughput by reflectivity
		throughR *= REFLECTIVITY * albR;
		throughG *= REFLECTIVITY * albG;
		throughB *= REFLECTIVITY * albB;

		// early out if throughput is negligible
		if (throughR + throughG + throughB < 0.001f) {
			break;
		}
	}

	// tonemap and clamp
	float r = fminf(colorR, 1.0f);
	float g = fminf(colorG, 1.0f);
	float b = fminf(colorB, 1.0f);

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
PathTracingKernel
========================
*/
__global__ void PathTracingKernel(
	const cudaVertex_t* vertices,
	const cudaTriangle_t* triangles,
	const cudaMaterial_t* materials,
	const cudaTexture_t* textures,
	const cudaLight_t* lights,
	int numLights,
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
	float fov_y,
	unsigned int frameNumber
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	int pixelIndex = (y * width + x) * 4;

	// per-pixel RNG state seeded from pixel position and frame number
	unsigned int rng = cupt_hash((unsigned int)(y * width + x) ^ cupt_hash(frameNumber + 1u));

	// generate primary ray with sub-pixel jitter for anti-aliasing
	float halfTanX = tanf(fov_x * 0.5f * 3.14159265f / 180.0f);
	float halfTanY = tanf(fov_y * 0.5f * 3.14159265f / 180.0f);

	float jx = cupt_random(rng); rng = cupt_hash(rng);
	float jy = cupt_random(rng); rng = cupt_hash(rng);

	float su = (2.0f * ((float)x + jx) / (float)width  - 1.0f) * halfTanX;
	float sv = (2.0f * ((float)y + jy) / (float)height - 1.0f) * halfTanY;

	float rayDirX = cameraForward[0] + su * cameraRight[0] + sv * cameraUp[0];
	float rayDirY = cameraForward[1] + su * cameraRight[1] + sv * cameraUp[1];
	float rayDirZ = cameraForward[2] + su * cameraRight[2] + sv * cameraUp[2];

	float il = rsqrtf(rayDirX * rayDirX + rayDirY * rayDirY + rayDirZ * rayDirZ);
	rayDirX *= il;
	rayDirY *= il;
	rayDirZ *= il;

	float rayOrigX = cameraPos[0];
	float rayOrigY = cameraPos[1];
	float rayOrigZ = cameraPos[2];

	// path tracing loop
	float colorR = 0.0f, colorG = 0.0f, colorB = 0.0f;
	float throughR = 1.0f, throughG = 1.0f, throughB = 1.0f;

	// sky ambient parameters
	// Doom 3 uses Z-up; the sky gradient is based on ray direction Z component
	const float SKY_INTENSITY   = 0.35f;
	const float SKY_ZENITH_R    = 0.15f, SKY_ZENITH_G = 0.18f, SKY_ZENITH_B = 0.30f;
	const float SKY_HORIZON_R   = 0.20f, SKY_HORIZON_G = 0.20f, SKY_HORIZON_B = 0.22f;
	const float SKY_GROUND_R    = 0.08f, SKY_GROUND_G = 0.07f, SKY_GROUND_B = 0.06f;

	const int MAX_BOUNCES = 6;

	for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {

		int hitIdx;
		float hitU, hitV;
		float hitT = TraceRay(rayOrigX, rayOrigY, rayOrigZ,
							  rayDirX, rayDirY, rayDirZ,
							  vertices, triangles, triIndices, bvhNodes,
							  numBVHNodes, numTriangles,
							  hitIdx, hitU, hitV);

		if (hitIdx < 0) {
			// sky ambient: gradient based on ray direction
			float skyT = rayDirZ * 0.5f + 0.5f; // map [-1,1] -> [0,1]
			skyT = fmaxf(0.0f, fminf(1.0f, skyT));

			float skyR, skyG, skyB;
			if (skyT > 0.5f) {
				// upper hemisphere: horizon -> zenith
				float s = (skyT - 0.5f) * 2.0f;
				skyR = SKY_HORIZON_R + (SKY_ZENITH_R - SKY_HORIZON_R) * s;
				skyG = SKY_HORIZON_G + (SKY_ZENITH_G - SKY_HORIZON_G) * s;
				skyB = SKY_HORIZON_B + (SKY_ZENITH_B - SKY_HORIZON_B) * s;
			} else {
				// lower hemisphere: ground -> horizon
				float s = skyT * 2.0f;
				skyR = SKY_GROUND_R + (SKY_HORIZON_R - SKY_GROUND_R) * s;
				skyG = SKY_GROUND_G + (SKY_HORIZON_G - SKY_GROUND_G) * s;
				skyB = SKY_GROUND_B + (SKY_HORIZON_B - SKY_GROUND_B) * s;
			}

			colorR += throughR * skyR * SKY_INTENSITY;
			colorG += throughG * skyG * SKY_INTENSITY;
			colorB += throughB * skyB * SKY_INTENSITY;
			break;
		}

		const cudaTriangle_t& tri = triangles[hitIdx];
		int i0 = tri.vertexIndices[0];
		int i1 = tri.vertexIndices[1];
		int i2 = tri.vertexIndices[2];

		float w0 = 1.0f - hitU - hitV;
		float w1 = hitU;
		float w2 = hitV;

		// interpolate shading normal
		float nx = w0 * vertices[i0].normal[0] + w1 * vertices[i1].normal[0] + w2 * vertices[i2].normal[0];
		float ny = w0 * vertices[i0].normal[1] + w1 * vertices[i1].normal[1] + w2 * vertices[i2].normal[1];
		float nz = w0 * vertices[i0].normal[2] + w1 * vertices[i1].normal[2] + w2 * vertices[i2].normal[2];
		float nIL = rsqrtf(nx * nx + ny * ny + nz * nz + 1e-12f);
		nx *= nIL;
		ny *= nIL;
		nz *= nIL;

		// ensure normal faces the incoming ray direction
		if (nx * rayDirX + ny * rayDirY + nz * rayDirZ > 0.0f) {
			nx = -nx; ny = -ny; nz = -nz;
		}

		// interpolate texcoord
		float tu = w0 * vertices[i0].texcoord[0] + w1 * vertices[i1].texcoord[0] + w2 * vertices[i2].texcoord[0];
		float tv = w0 * vertices[i0].texcoord[1] + w1 * vertices[i1].texcoord[1] + w2 * vertices[i2].texcoord[1];

		// hit position
		float pX = rayOrigX + rayDirX * hitT;
		float pY = rayOrigY + rayDirY * hitT;
		float pZ = rayOrigZ + rayDirZ * hitT;

		// material lookup
		const cudaMaterial_t& mat = materials[tri.materialIndex];
		float albR = mat.albedo[0];
		float albG = mat.albedo[1];
		float albB = mat.albedo[2];

		if (mat.albedoTexture >= 0) {
			float tR, tG, tB;
			SampleTexture(textures[mat.albedoTexture], tu, tv, tR, tG, tB);
			albR *= tR;
			albG *= tG;
			albB *= tB;
		}

		// direct light sampling
		if (numLights > 0) {
			// randomly pick one light and weight by numLights
			int li = (int)(cupt_random(rng) * (float)numLights);
			rng = cupt_hash(rng);
			if (li >= numLights) li = numLights - 1;

			const cudaLight_t& light = lights[li];

			float lX = light.position[0] - pX;
			float lY = light.position[1] - pY;
			float lZ = light.position[2] - pZ;

			float dSq = lX * lX + lY * lY + lZ * lZ;
			float d   = sqrtf(dSq + 1e-12f);
			float iD  = 1.0f / d;
			lX *= iD;
			lY *= iD;
			lZ *= iD;

			float R = light.radius;
			if (d < R) {
				float NdotL = nx * lX + ny * lY + nz * lZ;

				if (NdotL > 0.0f) {
					float sOx = pX + nx * 0.1f;
					float sOy = pY + ny * 0.1f;
					float sOz = pZ + nz * 0.1f;

					bool inShadow = (numBVHNodes > 0) &&
						TraceShadowRay(sOx, sOy, sOz, lX, lY, lZ, d,
									   vertices, triangles, triIndices, bvhNodes, numBVHNodes);

					if (!inShadow) {
						// smooth radius-based attenuation: (1 - (d/R)^2)^2
						float ratio  = d / R;
						float falloff = 1.0f - ratio * ratio;
						falloff = falloff * falloff;
						float atten = light.intensity * falloff;

						// Lambertian BRDF = albedo / pi
						// weight by numLights to compensate for random selection
						float scale = NdotL * atten * (float)numLights / 3.14159265f;

						colorR += throughR * albR * light.color[0] * scale;
						colorG += throughG * albG * light.color[1] * scale;
						colorB += throughB * albB * light.color[2] * scale;
					}
				}
			}
		}

		// cosine-weighted hemisphere sampling for indirect bounce
		float r1 = cupt_random(rng); rng = cupt_hash(rng);
		float r2 = cupt_random(rng); rng = cupt_hash(rng);

		float sinTheta = sqrtf(r1);
		float cosTheta = sqrtf(1.0f - r1);
		float phi = 2.0f * 3.14159265f * r2;

		// build orthonormal basis (tangent, bitangent) from normal
		float upX, upY, upZ;
		if (fabsf(nx) < 0.9f) { upX = 1.0f; upY = 0.0f; upZ = 0.0f; }
		else                   { upX = 0.0f; upY = 1.0f; upZ = 0.0f; }

		// tangent = normalize(cross(up, n))
		float tX = upY * nz - upZ * ny;
		float tY = upZ * nx - upX * nz;
		float tZ = upX * ny - upY * nx;
		float tIL = rsqrtf(tX * tX + tY * tY + tZ * tZ + 1e-12f);
		tX *= tIL; tY *= tIL; tZ *= tIL;

		// bitangent = cross(n, tangent)
		float bX = ny * tZ - nz * tY;
		float bY = nz * tX - nx * tZ;
		float bZ = nx * tY - ny * tX;

		// sample direction in local frame -> world
		float cosP = cosf(phi);
		float sinP = sinf(phi);

		rayDirX = sinTheta * cosP * tX + sinTheta * sinP * bX + cosTheta * nx;
		rayDirY = sinTheta * cosP * tY + sinTheta * sinP * bY + cosTheta * ny;
		rayDirZ = sinTheta * cosP * tZ + sinTheta * sinP * bZ + cosTheta * nz;

		// offset origin along normal to avoid self-intersection
		rayOrigX = pX + nx * 0.01f;
		rayOrigY = pY + ny * 0.01f;
		rayOrigZ = pZ + nz * 0.01f;

		// for cosine-weighted sampling  PDF = cosTheta / pi
		// Lambertian BRDF = albedo / pi
		// weight = (BRDF * cosTheta) / PDF = albedo
		throughR *= albR;
		throughG *= albG;
		throughB *= albB;

		// russian roulette after bounce 2
		if (bounce >= 2) {
			float p = fmaxf(throughR, fmaxf(throughG, throughB));
			if (p < 0.01f) break;
			float rrRoll = cupt_random(rng); rng = cupt_hash(rng);
			if (rrRoll > p) break;
			float invP = 1.0f / p;
			throughR *= invP;
			throughG *= invP;
			throughB *= invP;
		}
	}

	// progressive accumulation
	// Blend new sample with running average stored in framebuffer
	if (frameNumber > 0) {
		float oldR = framebuffer[pixelIndex + 0];
		float oldG = framebuffer[pixelIndex + 1];
		float oldB = framebuffer[pixelIndex + 2];

		float t = 1.0f / (float)(frameNumber + 1);
		colorR = oldR + (colorR - oldR) * t;
		colorG = oldG + (colorG - oldG) * t;
		colorB = oldB + (colorB - oldB) * t;
	}

	// store accumulated HDR
	framebuffer[pixelIndex + 0] = colorR;
	framebuffer[pixelIndex + 1] = colorG;
	framebuffer[pixelIndex + 2] = colorB;
	framebuffer[pixelIndex + 3] = 1.0f;

	// tonemap (clamp) and write LDR output
	float r = fminf(colorR, 1.0f);
	float g = fminf(colorG, 1.0f);
	float b = fminf(colorB, 1.0f);

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
    const cudaLight_t* lights,
    int numLights,
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
	int renderMode,
	unsigned int frameNumber
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
			frameNumber
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

	} else if (renderMode == 3) {

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

    } else if (renderMode == 4) {

		TriangleDrawRayTracedKernel<<<gridSize, blockSize>>>(
			d_vertices,
			d_triangles,
			d_materials,
			d_textures,
			lights,
			numLights,
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

        PathTracingKernel<<<gridSize, blockSize>>>(
            d_vertices,
            d_triangles,
            d_materials,
            d_textures,
            lights,
            numLights,
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
            fov_y,
            frameNumber
        );

    }
}
