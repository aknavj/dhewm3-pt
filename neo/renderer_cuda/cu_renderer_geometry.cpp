#ifdef HAVE_CUDA

#include "renderer/tr_local.h"
#include "renderer_cuda/cu_renderer.h"
#include <cuda_runtime.h>

// extern console variables
extern idCVar r_cuDebug;
extern idCVar r_cuRenderMode;

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

			// rotate normal by the 3x3 upper-left of modelMatrix (no translation)
			float nx = verts[i].normal[0];
			float ny = verts[i].normal[1];
			float nz = verts[i].normal[2];
			v.normal[0] = modelMatrix[0] * nx + modelMatrix[4] * ny + modelMatrix[8]  * nz;
			v.normal[1] = modelMatrix[1] * nx + modelMatrix[5] * ny + modelMatrix[9]  * nz;
			v.normal[2] = modelMatrix[2] * nx + modelMatrix[6] * ny + modelMatrix[10] * nz;

			// rotate tangent by the 3x3 upper-left of modelMatrix
			float tx = verts[i].tangents[0][0];
			float ty = verts[i].tangents[0][1];
			float tz = verts[i].tangents[0][2];
			v.tangent[0] = modelMatrix[0] * tx + modelMatrix[4] * ty + modelMatrix[8]  * tz;
			v.tangent[1] = modelMatrix[1] * tx + modelMatrix[5] * ty + modelMatrix[9]  * tz;
			v.tangent[2] = modelMatrix[2] * tx + modelMatrix[6] * ty + modelMatrix[10] * tz;

			// rotate bitangent by the 3x3 upper-left of modelMatrix
			float bx = verts[i].tangents[1][0];
			float by = verts[i].tangents[1][1];
			float bz = verts[i].tangents[1][2];
			v.bitangent[0] = modelMatrix[0] * bx + modelMatrix[4] * by + modelMatrix[8]  * bz;
			v.bitangent[1] = modelMatrix[1] * bx + modelMatrix[5] * by + modelMatrix[9]  * bz;
			v.bitangent[2] = modelMatrix[2] * bx + modelMatrix[6] * by + modelMatrix[10] * bz;
		} else {
			v.position[0] = verts[i].xyz[0];
			v.position[1] = verts[i].xyz[1];
			v.position[2] = verts[i].xyz[2];

			v.normal[0] = verts[i].normal[0];
			v.normal[1] = verts[i].normal[1];
			v.normal[2] = verts[i].normal[2];

			v.tangent[0] = verts[i].tangents[0][0];
			v.tangent[1] = verts[i].tangents[0][1];
			v.tangent[2] = verts[i].tangents[0][2];

			v.bitangent[0] = verts[i].tangents[1][0];
			v.bitangent[1] = verts[i].tangents[1][1];
			v.bitangent[2] = verts[i].tangents[1][2];
		}

		// normalize normal
		float nLen = sqrtf(v.normal[0] * v.normal[0] + v.normal[1] * v.normal[1] + v.normal[2] * v.normal[2]);
		if (nLen > 1e-6f) {
			v.normal[0] /= nLen;
			v.normal[1] /= nLen;
			v.normal[2] /= nLen;
		} else {
			v.normal[0] = 0.0f;
			v.normal[1] = 0.0f;
			v.normal[2] = 1.0f;
		}

		// normalize tangent
		float tLen = sqrtf(v.tangent[0] * v.tangent[0] + v.tangent[1] * v.tangent[1] + v.tangent[2] * v.tangent[2]);
		if (tLen > 1e-6f) {
			v.tangent[0] /= tLen;
			v.tangent[1] /= tLen;
			v.tangent[2] /= tLen;
		} else {
			v.tangent[0] = 1.0f;
			v.tangent[1] = 0.0f;
			v.tangent[2] = 0.0f;
		}

		// normalize bitangent
		float bLen = sqrtf(v.bitangent[0] * v.bitangent[0] + v.bitangent[1] * v.bitangent[1] + v.bitangent[2] * v.bitangent[2]);
		if (bLen > 1e-6f) {
			v.bitangent[0] /= bLen;
			v.bitangent[1] /= bLen;
			v.bitangent[2] /= bLen;
		} else {
			v.bitangent[0] = 0.0f;
			v.bitangent[1] = 1.0f;
			v.bitangent[2] = 0.0f;
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