#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "renderer_cuda/cu_renderer.h"
#include "renderer_cuda/cu_renderer_math.cuh"

/*
=================================================================================
Texture Sampling (bilinear filtering)
=================================================================================
*/
__device__ inline void SampleTexture(const cudaTexture_t* textures, int texIndex, float u, float v, float* color) {
	if (texIndex < 0 || texIndex >= MAX_TEXTURES || !textures[texIndex].data) {
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
Texture Sampling (bilinear filtering)
=================================================================================
*/
__device__ void EvaluateMaterial(const float* albedo, float *brdf) {	
	brdf[0] = albedo[0];
	brdf[1] = albedo[1];
	brdf[2] = albedo[2];
}