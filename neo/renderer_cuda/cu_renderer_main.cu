
#include <cuda_runtime.h>
#include <stdint.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "renderer_cuda/cu_renderer.h"
#include "renderer_cuda/cu_kernel_triangles.cuh"
#include "renderer_cuda/cu_kernel_triangles_textured.cuh"
#include "renderer_cuda/cu_kernel_triangles_raytraced.cuh"
#include "renderer_cuda/cu_kernel_pathtracing.cuh"

/*
========================
LaunchPathTracingKernel
========================
*/
extern "C" void LaunchPathTracingKernel(
	const cudaVertex_t* vertices,
	const cudaTriangle_t* triangles,
	const cudaBVHNode_t* bvhNodes,
	const int* triIndices,
	const cudaMaterial_t* materials,
	const cudaTexture_t* textures,
	const cudaLight_t* lights,
	int numLights,
	int renderMode,
	float* framebuffer,
	int width,
	int height,
	const float* cameraPos,
	const float* cameraForward,
	const float* cameraRight,
	const float* cameraUp,
	float fov,
	float fovY,
	int samplesPerPixel,
	int maxDepth,
	int maxLightSamples,
	int frameIndex,
	float emissionBoost,
	float indirectProb,
	int rrEnabled,
	int rrMinBounces,
	float rrSurvivalMin,
	float earlyTermThreshold,
	float fireflyClamp,
	float throughputClamp,
	float rayOffset,
	float specularBoost,
	float skyIntensity,
	const float* skyColorZenith,
	const float* skyColorHorizon,
	const float* skyColorGround,
	float volumetricDensity,
	int volumetricSteps,
	float volumetricAnisotropy,
	float volFalloff,
	float volMaxDist,
	float softShadowScale,
	cudaStream_t stream
) {
	// upload device-side constants
	cudaMemcpyToSymbolAsync(g_softShadowScale, &softShadowScale, sizeof(float), 0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(g_rayOffset, &rayOffset, sizeof(float), 0, cudaMemcpyHostToDevice, stream);

	dim3 blockSize(TILE_SIZE, TILE_SIZE);
	dim3 gridSize(
		(width + TILE_SIZE - 1) / TILE_SIZE,
		(height + TILE_SIZE - 1) / TILE_SIZE
	);

	if (renderMode == 1) {
		
		// triangle rasterization (for BVH visualization, etc)
		TriangleDrawKernel<<<gridSize, blockSize, 0, stream>>>(
			vertices,
			triangles,
			bvhNodes,
			triIndices,
			materials,
			textures,
			lights,
			numLights,
			framebuffer,
			width,
			height,
			cameraPos,
			cameraForward,
			cameraRight,
			cameraUp,
			fov,
			fovY,
			samplesPerPixel,
			maxDepth,
			maxLightSamples,
			frameIndex,
			emissionBoost,
			indirectProb,
			rrEnabled,
			rrMinBounces,
			rrSurvivalMin,
			earlyTermThreshold,
			fireflyClamp,
			throughputClamp,
			rayOffset,
			specularBoost,
			skyIntensity,
			skyColorZenith,
			skyColorHorizon,
			skyColorGround,
			volumetricDensity,
			volumetricSteps,
			volumetricAnisotropy,
			volFalloff,
			volMaxDist
		);
	} else if (renderMode >= 2 && renderMode <= 6) {

		// textured debug visualization
		// renderMode 2 = combined (albedo tex * base color, Lambertian lit)
		// renderMode 3 = base color only (material albedo, no texture)
		// renderMode 4 = albedo texture only (raw diffuse texture)
		// renderMode 5 = normal texture (visualized as RGB)
		// renderMode 6 = specular texture (visualized as grayscale)
		// sub-mode is passed via maxLightSamples parameter slot
		int texSubMode = renderMode - 2; // 0=combined, 1=basecolor, 2=albedo, 3=normal, 4=specular

		TriangleDrawTexturedKernel<<<gridSize, blockSize, 0, stream>>>(
			vertices,
			triangles,
			bvhNodes,
			triIndices,
			materials,
			textures,
			lights,
			numLights,
			framebuffer,
			width,
			height,
			cameraPos,
			cameraForward,
			cameraRight,
			cameraUp,
			fov,
			fovY,
			samplesPerPixel,
			maxDepth,
			texSubMode,
			frameIndex,
			emissionBoost,
			indirectProb,
			rrEnabled,
			rrMinBounces,
			rrSurvivalMin,
			earlyTermThreshold,
			fireflyClamp,
			throughputClamp,
			rayOffset,
			specularBoost,
			skyIntensity,
			skyColorZenith,
			skyColorHorizon,
			skyColorGround,
			volumetricDensity,
			volumetricSteps,
			volumetricAnisotropy,
			volFalloff,
			volMaxDist
		);

	} else if (renderMode == 7) { 
	
		TriangleDrawRayTracedKernel<<<gridSize, blockSize, 0, stream>>>(
			vertices,
			triangles,
			bvhNodes,
			triIndices,
			materials,
			textures,
			lights,
			numLights,
			framebuffer,
			width,
			height,
			cameraPos,
			cameraForward,
			cameraRight,
			cameraUp,
			fov,
			fovY,
			samplesPerPixel,
			maxDepth,
			maxLightSamples,
			frameIndex,
			emissionBoost,
			indirectProb,
			rrEnabled,
			rrMinBounces,
			rrSurvivalMin,
			earlyTermThreshold,
			fireflyClamp,
			throughputClamp,
			rayOffset,
			specularBoost,
			skyIntensity,
			skyColorZenith,
			skyColorHorizon,
			skyColorGround,
			volumetricDensity,
			volumetricSteps,
			volumetricAnisotropy,
			volFalloff,
			volMaxDist
		);

	} else {

		// path tracing
		PathTracingKernel<<<gridSize, blockSize, 0, stream>>>(
			vertices,
			triangles,
			bvhNodes,
			triIndices,
			materials,
			textures,
			lights,
			numLights,
			framebuffer,
			width,
			height,
			cameraPos,
			cameraForward,
			cameraRight,
			cameraUp,
			fov,
			fovY,
			samplesPerPixel,
			maxDepth,
			maxLightSamples,
			frameIndex,
			emissionBoost,
			indirectProb,
			rrEnabled,
			rrMinBounces,
			rrSurvivalMin,
			earlyTermThreshold,
			fireflyClamp,
			throughputClamp,
			rayOffset,
			specularBoost,
			skyIntensity,
			skyColorZenith,
			skyColorHorizon,
			skyColorGround,
			volumetricDensity,
			volumetricSteps,
			volumetricAnisotropy,
			volFalloff,
			volMaxDist
		);

	}
}

/*
========================
LaunchToneMappingKernel
========================
*/
extern "C" void LaunchToneMappingKernel(
	const float* hdrBuffer,
	unsigned char* ldrBuffer,
	int width,
	int height,
	float exposure,
	float gamma,
	int toneMapMode,
	int frameCount,
	cudaStream_t stream
) {
	dim3 blockSize(TILE_SIZE, TILE_SIZE);
	dim3 gridSize(
		(width + TILE_SIZE - 1) / TILE_SIZE,
		(height + TILE_SIZE - 1) / TILE_SIZE
	);

	ToneMappingKernel<<<gridSize, blockSize, 0, stream>>>(
		hdrBuffer,
		ldrBuffer,
		width,
		height,
		exposure,
		gamma,
		toneMapMode,
		frameCount
	);
}
