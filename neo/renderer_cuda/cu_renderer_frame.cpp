#ifdef HAVE_CUDA

#include "renderer/tr_local.h"
#include "renderer_cuda/cu_renderer.h"
#include <cuda_runtime.h>

// extern console variables
extern idCVar r_cuDebug;
extern idCVar r_cuRenderMode;

/*
========================
idCudaRenderer::BeginFrame
========================
*/
void idCudaRenderer::BeginFrame() {
	h_vertices.Clear();
	h_triangles.Clear();
	h_materials.Clear();
	h_lights.Clear();

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

	BuildBVH();

	// upload lights
	if (h_lights.Num() > 0) {
		CUDA_CHECK_VOID(cudaMemcpy(d_lights, h_lights.Ptr(), 
			h_lights.Num() * sizeof(cudaLight_t), cudaMemcpyHostToDevice));
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
		common->Printf("  Kernel: %.2f ms (%.1f FPS)\n", kernel_ms, kernel_fps);
	}

	return;
}

#endif // HAVE_CUDA