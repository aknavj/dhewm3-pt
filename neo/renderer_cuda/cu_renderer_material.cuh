#ifndef __CU_RENDERER_MATERIAL_CUH__
#define __CU_RENDERER_MATERIAL_CUH__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "renderer_cuda/cu_renderer.h"
#include "renderer_cuda/cu_renderer_math.cuh"

/*
=================================================================================
Texture Sampling (bilinear filtering)
=================================================================================
*/
__device__ __forceinline__ void SampleTexture(const cudaTexture_t* textures, int texIndex, float u, float v, float* color) {
	if (texIndex < 0 || texIndex >= MAX_TEXTURES || !textures[texIndex].data
		|| textures[texIndex].width <= 0 || textures[texIndex].height <= 0) {
		// Return white (1,1,1,1) - allows material color modulation
		color[0] = color[1] = color[2] = color[3] = 1.0f;
		return;
	}
	
	const cudaTexture_t& tex = textures[texIndex];
	
	// wrap texture coordinates
	u = u - floorf(u);
	v = v - floorf(v);
	
	// convert to texel coordinates
	float x = u * (float)tex.width;
	float y = v * (float)tex.height;
	
	int x0 = (int)x % tex.width;
	int y0 = (int)y % tex.height;
	int x1 = (x0 + 1) % tex.width;
	int y1 = (y0 + 1) % tex.height;
	
	float fx = x - floorf(x);
	float fy = y - floorf(y);
	
	// bilinear interpolation
	int idx00 = (y0 * tex.width + x0) * 4;
	int idx10 = (y0 * tex.width + x1) * 4;
	int idx01 = (y1 * tex.width + x0) * 4;
	int idx11 = (y1 * tex.width + x1) * 4;
	
	// sample all 4 channels (RGBA)
	for (int c = 0; c < 4; c++) {
		float v0 = tex.data[idx00 + c] * (1.0f - fx) + tex.data[idx10 + c] * fx;
		float v1 = tex.data[idx01 + c] * (1.0f - fx) + tex.data[idx11 + c] * fx;
		color[c] = (v0 * (1.0f - fy) + v1 * fy) / 255.0f;
	}
}

/*
=================================================================================
EvaluateMaterial
=================================================================================
*/
__device__  __forceinline__ void EvaluateMaterial(
	const cudaMaterial_t& mat,
	const cudaTexture_t* textures,
	float tu, float tv,
	const float* normal,
	const float* tangent,
	const float* bitangent,
	float* outAlbedo,
	float* outNormal,
	float* outSpecular,
	float* outEmission,
	float& outAlpha
) {
	// emission from ambient
	outEmission[0] = mat.emission[0];
	outEmission[1] = mat.emission[1];
	outEmission[2] = mat.emission[2];

	// sample additive glow map if present
	if (mat.emissionTexture >= 0) {
		float emTex[4];
		SampleTexture(textures, mat.emissionTexture, tu, tv, emTex);
		outEmission[0] *= emTex[0];
		outEmission[1] *= emTex[1];
		outEmission[2] *= emTex[2];
	}

	outAlbedo[0] = mat.albedo[0];
	outAlbedo[1] = mat.albedo[1];
	outAlbedo[2] = mat.albedo[2];
	outAlpha = 1.0f;

	if (mat.albedoTexture >= 0) {
		float texColor[4];
		SampleTexture(textures, mat.albedoTexture, tu, tv, texColor);
		outAlbedo[0] *= texColor[0];
		outAlbedo[1] *= texColor[1];
		outAlbedo[2] *= texColor[2];
		outAlpha = texColor[3];
	}

	outNormal[0] = normal[0];
	outNormal[1] = normal[1];
	outNormal[2] = normal[2];

	if (mat.normalTexture >= 0) {
		float texNormal[4];
		SampleTexture(textures, mat.normalTexture, tu, tv, texNormal);

		// RXGB normal map compression (Doom 3 convention)
		// interaction.vfp: "MOV localNormal.x, localNormal.a"
		float tnX = texNormal[3] * 2.0f - 1.0f;  // alpha -> X
		float tnY = texNormal[1] * 2.0f - 1.0f;  // green -> Y
		float tnZ = texNormal[2] * 2.0f - 1.0f;  // blue  -> Z

		// worldN = T * tnX + B * tnY + N * tnZ
		outNormal[0] = tangent[0] * tnX + bitangent[0] * tnY + normal[0] * tnZ;
		outNormal[1] = tangent[1] * tnX + bitangent[1] * tnY + normal[1] * tnZ;
		outNormal[2] = tangent[2] * tnX + bitangent[2] * tnY + normal[2] * tnZ;

		// normalize the perturbed normal
		float len = sqrtf(outNormal[0] * outNormal[0] + outNormal[1] * outNormal[1] + outNormal[2] * outNormal[2]);
		if (len > 1e-6f) {
			float invLen = 1.0f / len;
			outNormal[0] *= invLen;
			outNormal[1] *= invLen;
			outNormal[2] *= invLen;
		} else {
			outNormal[0] = normal[0];
			outNormal[1] = normal[1];
			outNormal[2] = normal[2];
		}
	}

	outSpecular[0] = mat.specular[0];
	outSpecular[1] = mat.specular[1];
	outSpecular[2] = mat.specular[2];

	if (mat.specularTexture >= 0) {
		float texSpec[4];
		SampleTexture(textures, mat.specularTexture, tu, tv, texSpec);
		// interaction.vfp doubles the specular map: "ADD R2, R2, R2"
		outSpecular[0] *= texSpec[0] * 2.0f;
		outSpecular[1] *= texSpec[1] * 2.0f;
		outSpecular[2] *= texSpec[2] * 2.0f;
	}
}

#endif // __CU_RENDERER_MATERIAL_CUH__