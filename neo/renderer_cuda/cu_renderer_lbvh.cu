#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "renderer_cuda/cu_renderer.h"

/*
================================================================================
Morton Code Utilities
================================================================================
*/

__device__ __forceinline__ unsigned int expandBits(unsigned int v) {
	// spread 10 bits across 30 bits with 2 gaps between each bit
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

__device__ __forceinline__ unsigned int morton3D(float x, float y, float z) {
	// quantize to 10-bit integers [0, 1023]
	unsigned int ix = (unsigned int)fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
	unsigned int iy = (unsigned int)fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
	unsigned int iz = (unsigned int)fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
	return (expandBits(iz) << 2) | (expandBits(iy) << 1) | expandBits(ix);
}

/*
================================================================================
Kernel 1: Compute Morton Codes
================================================================================
*/

__global__ void ComputeMortonCodesKernel(
	const cudaTriangle_t* triangles,
	int numTriangles,
	unsigned int* mortonCodes,
	int* indices,
	float sceneMinX, float sceneMinY, float sceneMinZ,
	float sceneInvExtX, float sceneInvExtY, float sceneInvExtZ
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numTriangles) return;

	const cudaTriangle_t& tri = triangles[idx];

	// Centroid from precomputed bounds
	float cx = (tri.bounds[0] + tri.bounds[3]) * 0.5f;
	float cy = (tri.bounds[1] + tri.bounds[4]) * 0.5f;
	float cz = (tri.bounds[2] + tri.bounds[5]) * 0.5f;

	// Normalize to [0, 1]
	float nx = (cx - sceneMinX) * sceneInvExtX;
	float ny = (cy - sceneMinY) * sceneInvExtY;
	float nz = (cz - sceneMinZ) * sceneInvExtZ;

	mortonCodes[idx] = morton3D(nx, ny, nz);
	indices[idx] = idx;
}

/*
================================================================================
Delta Function
================================================================================
*/

__device__ __forceinline__ int lbvhDelta(
	const unsigned int* sortedCodes,
	int numPrimitives,
	int i,
	int j
) {
	if (j < 0 || j >= numPrimitives) return -1;
	unsigned int ci = sortedCodes[i];
	unsigned int cj = sortedCodes[j];
	if (ci == cj) {
		// tiebreaker: use index XOR for duplicate Morton codes
		return 32 + __clz(i ^ j);
	}
	return __clz(ci ^ cj);
}

/*
================================================================================
Kernel 2: Build Radix Tree
================================================================================
*/

__global__ void BuildRadixTreeKernel(
	const unsigned int* sortedMortonCodes,
	int numPrimitives,
	cudaBVHNode_t* nodes,
	int* parents
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numPrimitives - 1) return;

	int N = numPrimitives;

	// determine direction of the range (+1 or -1)
	int deltaRight = lbvhDelta(sortedMortonCodes, N, i, i + 1);
	int deltaLeft  = lbvhDelta(sortedMortonCodes, N, i, i - 1);
	int d = (deltaRight >= deltaLeft) ? 1 : -1;

	// compute upper bound for range length
	int deltaMin = lbvhDelta(sortedMortonCodes, N, i, i - d);
	int lmax = 2;
	while (lbvhDelta(sortedMortonCodes, N, i, i + lmax * d) > deltaMin) {
		lmax <<= 1;
		if (lmax > N) { lmax = N; break; }  // Safety bound
	}

	// binary search for the actual range length
	int l = 0;
	for (int t = lmax >> 1; t >= 1; t >>= 1) {
		if (lbvhDelta(sortedMortonCodes, N, i, i + (l + t) * d) > deltaMin) {
			l += t;
		}
	}
	int j = i + l * d;  // other end of the range

	// find the split position
	int deltaNode = lbvhDelta(sortedMortonCodes, N, i, j);
	int s = 0;
	int maxLen = l;
	// binary search within the range for the highest split point
	for (int div = 2; ; div <<= 1) {
		int t = (maxLen + div - 1) / div;
		if (t <= 0) break;
		if (lbvhDelta(sortedMortonCodes, N, i, i + (s + t) * d) > deltaNode) {
			s += t;
		}
		if (t == 1) break;
	}
	int gamma = i + s * d + min(d, 0);

	// assign children
	int first = min(i, j);
	int last  = max(i, j);

	// left child: covers [first, gamma]
	int leftChild;
	if (first == gamma) {
		leftChild = N - 1 + gamma;       // single leaf -> leaf node
	} else {
		leftChild = gamma;               // multiple leaves -> internal node
	}

	// right child: covers [gamma+1, last]
	int rightChild;
	if (last == gamma + 1) {
		rightChild = N - 1 + gamma + 1;  // single leaf -> leaf node
	} else {
		rightChild = gamma + 1;          // multiple leaves -> internal node
	}

	// store internal node
	nodes[i].leftChild  = leftChild;
	nodes[i].rightChild = rightChild;
	nodes[i].primitiveCount = 0;  // internal node — no direct primitives
	nodes[i].firstPrimitive = 0;

	// store parent pointers (children -> this node)
	parents[leftChild]  = i;
	parents[rightChild] = i;
}

/*
================================================================================
Kernel 3: Initialize Leaf Nodes
================================================================================
*/

__global__ void InitLeafNodesKernel(
	const cudaTriangle_t* triangles,
	const int* sortedTriIndices,
	cudaBVHNode_t* nodes,
	int numPrimitives
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numPrimitives) return;

	int nodeIdx = numPrimitives - 1 + i;
	int triIdx  = sortedTriIndices[i];

	// mark as leaf
	nodes[nodeIdx].leftChild  = -1;
	nodes[nodeIdx].rightChild = -1;
	nodes[nodeIdx].firstPrimitive = i;       // offset into triIndices
	nodes[nodeIdx].primitiveCount = 1;

	// copy bounds with epsilon padding
	const float EPSILON = 0.001f;
	const cudaTriangle_t& tri = triangles[triIdx];
	nodes[nodeIdx].bounds[0] = tri.bounds[0] - EPSILON;
	nodes[nodeIdx].bounds[1] = tri.bounds[1] - EPSILON;
	nodes[nodeIdx].bounds[2] = tri.bounds[2] - EPSILON;
	nodes[nodeIdx].bounds[3] = tri.bounds[3] + EPSILON;
	nodes[nodeIdx].bounds[4] = tri.bounds[4] + EPSILON;
	nodes[nodeIdx].bounds[5] = tri.bounds[5] + EPSILON;
}

/*
================================================================================
Kernel 4: Bottom-Up AABB Propagation
================================================================================
*/

__global__ void ComputeNodeBoundsKernel(
	cudaBVHNode_t* nodes,
	const int* parents,
	int* atomicCounters,
	int numPrimitives
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numPrimitives) return;

	// start at leaf node
	int parent = parents[numPrimitives - 1 + i];

	while (parent >= 0) {
		// first child to arrive increments 0->1, exits.
		// second child increments 1->2, proceeds to merge.
		if (atomicAdd(&atomicCounters[parent], 1) == 0) {
			return;  // sibling not done yet — exit and let it finish
		}

		// both children ready — compute union AABB
		int left  = nodes[parent].leftChild;
		int right = nodes[parent].rightChild;

		nodes[parent].bounds[0] = fminf(nodes[left].bounds[0], nodes[right].bounds[0]);
		nodes[parent].bounds[1] = fminf(nodes[left].bounds[1], nodes[right].bounds[1]);
		nodes[parent].bounds[2] = fminf(nodes[left].bounds[2], nodes[right].bounds[2]);
		nodes[parent].bounds[3] = fmaxf(nodes[left].bounds[3], nodes[right].bounds[3]);
		nodes[parent].bounds[4] = fmaxf(nodes[left].bounds[4], nodes[right].bounds[4]);
		nodes[parent].bounds[5] = fmaxf(nodes[left].bounds[5], nodes[right].bounds[5]);

		// walk up to grandparent
		parent = parents[parent];
	}
}

/*
================================================================================
Kernel 5: Single-Triangle Fallback
================================================================================
*/

__global__ void InitSingleLeafKernel(
	const cudaTriangle_t* triangles,
	cudaBVHNode_t* nodes,
	int* triIndices
) {
	if (threadIdx.x != 0 || blockIdx.x != 0) return;

	triIndices[0] = 0;

	nodes[0].leftChild  = -1;
	nodes[0].rightChild = -1;
	nodes[0].firstPrimitive = 0;
	nodes[0].primitiveCount = 1;

	const float EPSILON = 0.001f;
	nodes[0].bounds[0] = triangles[0].bounds[0] - EPSILON;
	nodes[0].bounds[1] = triangles[0].bounds[1] - EPSILON;
	nodes[0].bounds[2] = triangles[0].bounds[2] - EPSILON;
	nodes[0].bounds[3] = triangles[0].bounds[3] + EPSILON;
	nodes[0].bounds[4] = triangles[0].bounds[4] + EPSILON;
	nodes[0].bounds[5] = triangles[0].bounds[5] + EPSILON;
}

/*
================================================================================
Launch Wrapper: LaunchBuildLBVH

All work stays on the GPU. The host only provides scene bounds
(computed from h_triangles on CPU, which is cheap for <100K tris).
================================================================================
*/

extern "C" void LaunchBuildLBVH(
	const cudaTriangle_t* d_triangles,
	int numTriangles,
	cudaBVHNode_t* d_bvhNodes,
	int* d_triIndices,
	unsigned int* d_mortonCodes,
	int* d_sortedIndices,
	int* d_parents,
	int* d_atomicCounters,
	const float* sceneBounds,
	cudaStream_t stream
) {
	if (numTriangles <= 0) return;

	// single triangle
	if (numTriangles == 1) {
		InitSingleLeafKernel<<<1, 1, 0, stream>>>(d_triangles, d_bvhNodes, d_triIndices);
		return;
	}

	// scene extent for normalization
	float sceneMin[3] = { sceneBounds[0], sceneBounds[1], sceneBounds[2] };
	float sceneMax[3] = { sceneBounds[3], sceneBounds[4], sceneBounds[5] };
	float invExt[3];
	for (int i = 0; i < 3; i++) {
		float ext = sceneMax[i] - sceneMin[i];
		invExt[i] = (ext > 1e-6f) ? (1.0f / ext) : 0.0f;
	}

	const int BLOCK = 256;
	int grid = (numTriangles + BLOCK - 1) / BLOCK;
	int numInternalNodes = numTriangles - 1;
	int numNodes = 2 * numTriangles - 1;

	// morton codes
	ComputeMortonCodesKernel<<<grid, BLOCK, 0, stream>>>(
		d_triangles, numTriangles,
		d_mortonCodes, d_sortedIndices,
		sceneMin[0], sceneMin[1], sceneMin[2],
		invExt[0], invExt[1], invExt[2]
	);

	// parallel radix sort by Morton code
	cudaStreamSynchronize(stream);

	thrust::device_ptr<unsigned int> keys(d_mortonCodes);
	thrust::device_ptr<int> vals(d_sortedIndices);
	thrust::sort_by_key(keys, keys + numTriangles, vals);

	// sync default stream before continuing on our stream
	cudaDeviceSynchronize();

	// copy sorted indices -> triIndices
	cudaMemcpyAsync(d_triIndices, d_sortedIndices,
		numTriangles * sizeof(int), cudaMemcpyDeviceToDevice, stream);

	// initialize parent array to -1
	cudaMemsetAsync(d_parents, 0xFF, numNodes * sizeof(int), stream);

	// initialize atomic counters to 0
	cudaMemsetAsync(d_atomicCounters, 0, numInternalNodes * sizeof(int), stream);

	// build radix tree topology
	int treeGrid = (numInternalNodes + BLOCK - 1) / BLOCK;
	BuildRadixTreeKernel<<<treeGrid, BLOCK, 0, stream>>>(
		d_mortonCodes, numTriangles, d_bvhNodes, d_parents
	);

	// initialize leaf nodes
	InitLeafNodesKernel<<<grid, BLOCK, 0, stream>>>(
		d_triangles, d_triIndices, d_bvhNodes, numTriangles
	);

	// bottom-up AABB propagation
	ComputeNodeBoundsKernel<<<grid, BLOCK, 0, stream>>>(
		d_bvhNodes, d_parents, d_atomicCounters, numTriangles
	);
}
