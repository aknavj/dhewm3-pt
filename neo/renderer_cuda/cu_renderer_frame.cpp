#ifdef HAVE_CUDA

#include "renderer/tr_local.h"
#include "renderer_cuda/cu_renderer.h"
#include <cuda_runtime.h>

// extern console variables
extern idCVar r_cuDebug;

/*
========================
idCudaRenderer::BeginFrame
========================
*/
void idCudaRenderer::BeginFrame() {
	h_vertices.Clear();
	h_triangles.Clear();
	h_lights.Clear();

	// lights are rebuilt each frame but deduplicated within a frame via lightHash
	lightHash.Free();
	h_lightPtrs.Clear();
	nextLightIndex = 0;

	// flush all textures and materials
	if (needTextureFlush) {
		common->Printf("idCudaRenderer::BeginFrame(): Flushing %d textures and %d materials (texture overflow)\n",
			h_textures.Num(), h_materials.Num());

		// free all GPU texture data
		for (int i = 0; i < h_textures.Num(); i++) {
			if (h_textures[i].data) {
				cudaFree(h_textures[i].data);
			}
		}

		h_textures.Clear();
		h_texnums.Clear();
		h_texturePtrs.Clear();
		textureHash.Free();

		h_materials.Clear();
		h_materialPtrs.Clear();
		materialHash.Free();
		materialEmission.Clear();
		nextMaterialIndex = 0;

		needTextureFlush = false;
	}

	// materials persist across frames 
	need_reload = 1;

	return;
}

/*
========================
idCudaRenderer::EndFrame
========================
*/
void idCudaRenderer::EndFrame() {
	num_triangles = h_triangles.Num();
	num_vertices = h_vertices.Num();

	// upload geometry to GPU first
	if (num_triangles > 0 && num_vertices > 0) {
		// reallocate triangle buffer if needed
		if (num_triangles > allocatedTriangles) {
			if (d_triangles) cudaFree(d_triangles);
			allocatedTriangles = num_triangles + 1000;
			cudaMalloc(&d_triangles, allocatedTriangles * sizeof(cudaTriangle_t));
		}

		CUDA_CHECK_VOID(cudaMemcpy(d_vertices, h_vertices.Ptr(),
			num_vertices * sizeof(cudaVertex_t), cudaMemcpyHostToDevice));
		CUDA_CHECK_VOID(cudaMemcpy(d_triangles, h_triangles.Ptr(),
			num_triangles * sizeof(cudaTriangle_t), cudaMemcpyHostToDevice));
	}

	// build LBVH on GPU
	FramePVStoBVH();

	// upload lights
	int numLightsToUpload = h_lights.Num();
	if (numLightsToUpload > MAX_LIGHTS) {
		common->Warning("idCudaRenderer::EndFrame(): Clamping %d lights to MAX_LIGHTS (%d)\n", numLightsToUpload, MAX_LIGHTS);
		numLightsToUpload = MAX_LIGHTS;
	}
	if (numLightsToUpload > 0) {
		CUDA_CHECK_VOID(cudaMemcpy(d_lights, h_lights.Ptr(), 
			numLightsToUpload * sizeof(cudaLight_t), cudaMemcpyHostToDevice));
	}

	// upload materials
	if (h_materials.Num() > 0) {
		if (r_cuDebug.GetBool()) {
			common->Printf("idCudaRenderer::EndFrame(): Uploading %d materials to GPU (device ptr: 0x%p)\n", h_materials.Num(), d_materials);
		}
		CUDA_CHECK_VOID(cudaMemcpy(d_materials, h_materials.Ptr(), 
			h_materials.Num() * sizeof(cudaMaterial_t), cudaMemcpyHostToDevice));
	}

	// upload textures
	if (h_textures.Num() > 0) {
		if (r_cuDebug.GetBool()) {
			common->Printf("idCudaRenderer::EndFrame():Uploading %d textures to GPU (device ptr: 0x%p)\n", h_textures.Num(), d_textures);
		}
		CUDA_CHECK_VOID(cudaMemcpy(d_textures, h_textures.Ptr(), 
			h_textures.Num() * sizeof(cudaTexture_t), cudaMemcpyHostToDevice));
	}

	CUDA_CHECK_VOID(cudaDeviceSynchronize());

	if (r_cuDebug.GetBool()) {
		common->Printf("idCudaRenderer::EndFrame():\n");
		common->Printf("  Triangles: %d\n", num_triangles);
		common->Printf("  Vertices: %d\n", num_vertices);
		common->Printf("  Materials: %d\n", h_materials.Num());
		common->Printf("  Lights: %d\n", h_lights.Num());
		common->Printf("  BVH Nodes: %d\n", num_bvh_nodes);
		common->Printf("  Kernel: %.2f ms (%.1f FPS)\n", kernel_ms, kernel_fps);
	}

	return;
}

#endif // HAVE_CUDA