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
		common->Printf("  Kernel: %.2f ms (%.1f FPS)\n", kernel_ms, kernel_fps);
	}

	return;
}

/*
========================
idCudaRenderer::AddTriangle
========================
*/
void idCudaRenderer::AddTriangle(const idDrawVert* verts, int numVerts, const int* indices, int numIndices, int materialIndex, const float* modelMatrix) {
	
	int vertexBase = h_vertices.Num();

	for (int i = 0; i < numVerts; i++) {
		cudaVertex_t v;
		if (modelMatrix) {
			float x = verts[i].xyz[0];
			float y = verts[i].xyz[1];
			float z = verts[i].xyz[2];
			v.position[0] = modelMatrix[0] * x + modelMatrix[4] * y + modelMatrix[8] * z + modelMatrix[12];
			v.position[1] = modelMatrix[1] * x + modelMatrix[5] * y + modelMatrix[9] * z + modelMatrix[13];
			v.position[2] = modelMatrix[2] * x + modelMatrix[6] * y + modelMatrix[10] * z + modelMatrix[14];
		} else {
			v.position[0] = verts[i].xyz[0];
			v.position[1] = verts[i].xyz[1];
			v.position[2] = verts[i].xyz[2];
		}

		v.texcoord[0] = verts[i].st[0];
		v.texcoord[1] = verts[i].st[1];
		
		h_vertices.Append(v);
	}

	for (int i = 0; i < numIndices; i += 3) {
		cudaTriangle_t tri;
		tri.vertexIndices[0] = vertexBase + indices[i + 0];
		tri.vertexIndices[1] = vertexBase + indices[i + 1];
		tri.vertexIndices[2] = vertexBase + indices[i + 2];
		tri.materialIndex = materialIndex;

		const cudaVertex_t& v0 = h_vertices[tri.vertexIndices[0]];
		const cudaVertex_t& v1 = h_vertices[tri.vertexIndices[1]];
		const cudaVertex_t& v2 = h_vertices[tri.vertexIndices[2]];
		
		tri.bounds[0] = Min(v0.position[0], Min(v1.position[0], v2.position[0]));
		tri.bounds[1] = Min(v0.position[1], Min(v1.position[1], v2.position[1]));
		tri.bounds[2] = Min(v0.position[2], Min(v1.position[2], v2.position[2]));
		tri.bounds[3] = Max(v0.position[0], Max(v1.position[0], v2.position[0]));
		tri.bounds[4] = Max(v0.position[1], Max(v1.position[1], v2.position[1]));
		tri.bounds[5] = Max(v0.position[2], Max(v1.position[2], v2.position[2]));

		h_triangles.Append(tri);
	}

	return;
}

#endif // HAVE_CUDA