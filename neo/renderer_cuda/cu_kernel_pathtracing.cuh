#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "renderer_cuda/cu_renderer.h"
#include "renderer_cuda/cu_renderer_math.cuh"
#include "renderer_cuda/cu_renderer_material.cuh"

/*
========================
PathTracingKernel
========================
*/
__global__ __forceinline__ void PathTracingKernel(
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

		// interpolate tangent
		float geoTangent[3];
		geoTangent[0] = w0 * vertices[i0].tangent[0] + w1 * vertices[i1].tangent[0] + w2 * vertices[i2].tangent[0];
		geoTangent[1] = w0 * vertices[i0].tangent[1] + w1 * vertices[i1].tangent[1] + w2 * vertices[i2].tangent[1];
		geoTangent[2] = w0 * vertices[i0].tangent[2] + w1 * vertices[i1].tangent[2] + w2 * vertices[i2].tangent[2];
		float tILen = rsqrtf(geoTangent[0] * geoTangent[0] + geoTangent[1] * geoTangent[1] + geoTangent[2] * geoTangent[2] + 1e-12f);
		geoTangent[0] *= tILen; geoTangent[1] *= tILen; geoTangent[2] *= tILen;

		// interpolate bitangent
		float geoBitangent[3];
		geoBitangent[0] = w0 * vertices[i0].bitangent[0] + w1 * vertices[i1].bitangent[0] + w2 * vertices[i2].bitangent[0];
		geoBitangent[1] = w0 * vertices[i0].bitangent[1] + w1 * vertices[i1].bitangent[1] + w2 * vertices[i2].bitangent[1];
		geoBitangent[2] = w0 * vertices[i0].bitangent[2] + w1 * vertices[i1].bitangent[2] + w2 * vertices[i2].bitangent[2];
		float bILen = rsqrtf(geoBitangent[0] * geoBitangent[0] + geoBitangent[1] * geoBitangent[1] + geoBitangent[2] * geoBitangent[2] + 1e-12f);
		geoBitangent[0] *= bILen; geoBitangent[1] *= bILen; geoBitangent[2] *= bILen;

		// interpolate texcoord
		float tu = w0 * vertices[i0].texcoord[0] + w1 * vertices[i1].texcoord[0] + w2 * vertices[i2].texcoord[0];
		float tv = w0 * vertices[i0].texcoord[1] + w1 * vertices[i1].texcoord[1] + w2 * vertices[i2].texcoord[1];

		// hit position
		float pX = rayOrigX + rayDirX * hitT;
		float pY = rayOrigY + rayDirY * hitT;
		float pZ = rayOrigZ + rayDirZ * hitT;

		// evaluate material (albedo, normal map, specular)
		const cudaMaterial_t& mat = materials[tri.materialIndex];
		float geoNormal[3] = { nx, ny, nz };
		float albedo[3], shadingNormal[3], specColor[3];
		EvaluateMaterial(mat, textures, tu, tv, geoNormal, geoTangent, geoBitangent,
						 albedo, shadingNormal, specColor);

		// use perturbed shading normal for lighting
		nx = shadingNormal[0];
		ny = shadingNormal[1];
		nz = shadingNormal[2];

		// re-ensure perturbed normal faces the incoming ray
		if (nx * rayDirX + ny * rayDirY + nz * rayDirZ > 0.0f) {
			nx = -nx; ny = -ny; nz = -nz;
		}

		float albR = albedo[0];
		float albG = albedo[1];
		float albB = albedo[2];

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