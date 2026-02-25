#ifdef HAVE_CUDA

#include "renderer/tr_local.h"
#include "renderer_cuda/cu_renderer.h"

extern idCVar r_cuDebug;

/*
========================
idCudaRenderer::FramePVStoBVH
========================
*/
void idCudaRenderer::FramePVStoBVH() {
	BuildBVH();
}

/*
========================
idCudaRenderer::BuildBVH

GPU-accelerated Linear BVH construction.
========================
*/
void idCudaRenderer::BuildBVH() {

	int N = h_triangles.Num();
	if (N == 0) {
		num_bvh_nodes = 0;
		return;
	}

	int numNodes    = 2 * N - 1;
	int numInternal = N - 1;

	// reallocate BVH output buffers if needed
	if (numNodes > allocatedBVHNodes) {
		if (d_bvhNodes) cudaFree(d_bvhNodes);
		allocatedBVHNodes = numNodes + 1000;
		cudaMalloc(&d_bvhNodes, allocatedBVHNodes * sizeof(cudaBVHNode_t));
	}
	if (N > allocatedTriIndices) {
		if (d_triIndices) cudaFree(d_triIndices);
		allocatedTriIndices = N + 1000;
		cudaMalloc(&d_triIndices, allocatedTriIndices * sizeof(int));
	}

	// reallocate LBVH temporary buffers if needed
	if (N > allocatedLBVHSize) {
		if (d_mortonCodes)     cudaFree(d_mortonCodes);
		if (d_sortedIndices)   cudaFree(d_sortedIndices);
		if (d_parents)         cudaFree(d_parents);
		if (d_atomicCounters)  cudaFree(d_atomicCounters);

		allocatedLBVHSize = N + 1000;
		cudaMalloc(&d_mortonCodes,    allocatedLBVHSize * sizeof(unsigned int));
		cudaMalloc(&d_sortedIndices,  allocatedLBVHSize * sizeof(int));
		cudaMalloc(&d_parents,        (2 * allocatedLBVHSize) * sizeof(int));
		cudaMalloc(&d_atomicCounters, allocatedLBVHSize * sizeof(int));

		if (r_cuDebug.GetBool()) {
			common->Printf("LBVH: allocated temp buffers for %d triangles\n", allocatedLBVHSize);
		}
	}

	// compute scene bounds on CPU (fast for PVS-culled tri counts)
	float sceneBounds[6];
	sceneBounds[0] = sceneBounds[1] = sceneBounds[2] =  1e30f;  // min
	sceneBounds[3] = sceneBounds[4] = sceneBounds[5] = -1e30f;  // max

	for (int i = 0; i < N; i++) {
		const cudaTriangle_t& tri = h_triangles[i];
		for (int j = 0; j < 3; j++) {
			if (tri.bounds[j]     < sceneBounds[j])     sceneBounds[j]     = tri.bounds[j];
			if (tri.bounds[j + 3] > sceneBounds[j + 3]) sceneBounds[j + 3] = tri.bounds[j + 3];
		}
	}

	// small expansion to avoid degenerate extents
	for (int j = 0; j < 3; j++) {
		if (sceneBounds[j + 3] - sceneBounds[j] < 0.01f) {
			sceneBounds[j]     -= 0.5f;
			sceneBounds[j + 3] += 0.5f;
		}
	}

	// launch GPU LBVH build
	LaunchBuildLBVH(
		d_triangles,
		N,
		d_bvhNodes,
		d_triIndices,
		d_mortonCodes,
		d_sortedIndices,
		d_parents,
		d_atomicCounters,
		sceneBounds,
		stream
	);

	num_bvh_nodes = numNodes;
}

#endif // HAVE_CUDA