
#include <cuda_runtime.h>
#include <stdint.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "renderer_cuda/cu_renderer.h"
#include "renderer_cuda/cu_kernel_noise.cuh"
#include "renderer_cuda/cu_kernel_triangles.cuh"
#include "renderer_cuda/cu_kernel_triangles_textured.cuh"
#include "renderer_cuda/cu_kernel_triangles_raytraced.cuh"
#include "renderer_cuda/cu_kernel_pathtracing.cuh"


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
