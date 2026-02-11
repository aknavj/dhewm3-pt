#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "renderer_cuda/cu_renderer.h"
#include "renderer_cuda/cu_renderer_math.cuh"

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
__global__ __forceinline__ void TriangleDrawRayTracedKernel(
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
