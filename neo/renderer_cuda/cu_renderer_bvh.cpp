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
 */
namespace {

    /*
     */
    struct BVHBuildEntry {
        int nodeIndex;		// index into h_bvhNodes to write
        int start;			// first triangle index in triIdx
        int count;			// number of triangles
    };

    /*
     */
    static void ComputeAABB(
        const idList<cudaTriangle_t>& tris,
        const int* indices,
        int start,
        int count,
        float outMin[3],
        float outMax[3]
    ) {
        outMin[0] = outMin[1] = outMin[2] =  1e30f;
        outMax[0] = outMax[1] = outMax[2] = -1e30f;

        for (int i = start; i < start + count; i++) {
            const cudaTriangle_t& t = tris[indices[i]];
            for (int a = 0; a < 3; a++) {
                if (t.bounds[a]     < outMin[a]) outMin[a] = t.bounds[a];
                if (t.bounds[3 + a] > outMax[a]) outMax[a] = t.bounds[3 + a];
            }
        }
    }

    /*
     */
    static float SurfaceArea(const float mn[3], const float mx[3]) {
        float dx = mx[0] - mn[0];
        float dy = mx[1] - mn[1];
        float dz = mx[2] - mn[2];
        return 2.0f * (dx * dy + dy * dz + dz * dx);
    }

    /*
     */
    static float TriCentroid(const cudaTriangle_t& t, int axis) {
        return 0.5f * (t.bounds[axis] + t.bounds[3 + axis]);
    }

}

/*
========================
idCudaRenderer::BuildBVH
========================
*/
void idCudaRenderer::BuildBVH() {

	int triCount = h_triangles.Num();
	if (triCount == 0) {
		num_bvh_nodes = 0;
		return;
	}

	idList<int> triIdx;
	triIdx.SetNum(triCount);
	for (int i = 0; i < triCount; i++) {
		triIdx[i] = i;
	}

	h_bvhNodes.SetNum(0);
	h_bvhNodes.SetGranularity(triCount * 2);

	int nodesUsed = 0;

	{
		cudaBVHNode_t root;
		memset(&root, 0, sizeof(root));
		h_bvhNodes.Append(root);
		nodesUsed = 1;
	}

	static const int MAX_STACK = 128;
	BVHBuildEntry stack[MAX_STACK];
	int stackPtr = 0;

	stack[stackPtr++] = { 0, 0, triCount };

	static const float COST_TRAVERSE  = 1.0f;
	static const float COST_INTERSECT = 1.0f;
	static const int   SAH_BINS = 12;
	static const int   LEAF_MAX = 4;

	while (stackPtr > 0) {
		BVHBuildEntry entry = stack[--stackPtr];

		int nodeIdx = entry.nodeIndex;
		int start   = entry.start;
		int count   = entry.count;

		float bMin[3], bMax[3];
		ComputeAABB(h_triangles, triIdx.Ptr(), start, count, bMin, bMax);

		cudaBVHNode_t& node = h_bvhNodes[nodeIdx];
		for (int a = 0; a < 3; a++) {
			node.bounds[a]     = bMin[a];
			node.bounds[3 + a] = bMax[a];
		}

		if (count <= LEAF_MAX) {
			node.first_primitive = start;
			node.primitive_count = count;
			node.l_child = -1;
			node.r_child = -1;
			continue;
		}

		float parentArea = SurfaceArea(bMin, bMax);
		if (parentArea < 1e-12f) {
			node.first_primitive = start;
			node.primitive_count = count;
			node.l_child = -1;
			node.r_child = -1;
			continue;
		}

		float bestCost  = 1e30f;
		int   bestAxis  = -1;
		int   bestBin   = -1;

		for (int axis = 0; axis < 3; axis++) {
			float cMin =  1e30f;
			float cMax = -1e30f;
			for (int i = start; i < start + count; i++) {
				float c = TriCentroid(h_triangles[triIdx[i]], axis);
				if (c < cMin) cMin = c;
				if (c > cMax) cMax = c;
			}

			if (cMax - cMin < 1e-7f) {
				continue;
			}

			struct Bin {
				float mn[3], mx[3];
				int   count;
			};
			Bin bins[SAH_BINS];
			for (int b = 0; b < SAH_BINS; b++) {
				bins[b].mn[0] = bins[b].mn[1] = bins[b].mn[2] =  1e30f;
				bins[b].mx[0] = bins[b].mx[1] = bins[b].mx[2] = -1e30f;
				bins[b].count = 0;
			}

			float scale = (float)SAH_BINS / (cMax - cMin);
			for (int i = start; i < start + count; i++) {
				const cudaTriangle_t& tri = h_triangles[triIdx[i]];
				int b = (int)((TriCentroid(tri, axis) - cMin) * scale);
				if (b >= SAH_BINS) b = SAH_BINS - 1;
				bins[b].count++;
				for (int a = 0; a < 3; a++) {
					if (tri.bounds[a]     < bins[b].mn[a]) bins[b].mn[a] = tri.bounds[a];
					if (tri.bounds[3 + a] > bins[b].mx[a]) bins[b].mx[a] = tri.bounds[3 + a];
				}
			}

			float leftArea[SAH_BINS - 1];
			int   leftCount[SAH_BINS - 1];
			{
				float mn[3] = { 1e30f,  1e30f,  1e30f};
				float mx[3] = {-1e30f, -1e30f, -1e30f};
				int   cnt = 0;
				for (int b = 0; b < SAH_BINS - 1; b++) {
					cnt += bins[b].count;
					for (int a = 0; a < 3; a++) {
						if (bins[b].mn[a] < mn[a]) mn[a] = bins[b].mn[a];
						if (bins[b].mx[a] > mx[a]) mx[a] = bins[b].mx[a];
					}
					leftArea[b]  = SurfaceArea(mn, mx);
					leftCount[b] = cnt;
				}
			}

			float rightArea[SAH_BINS - 1];
			int   rightCount[SAH_BINS - 1];
			{
				float mn[3] = { 1e30f,  1e30f,  1e30f};
				float mx[3] = {-1e30f, -1e30f, -1e30f};
				int   cnt = 0;
				for (int b = SAH_BINS - 1; b > 0; b--) {
					cnt += bins[b].count;
					for (int a = 0; a < 3; a++) {
						if (bins[b].mn[a] < mn[a]) mn[a] = bins[b].mn[a];
						if (bins[b].mx[a] > mx[a]) mx[a] = bins[b].mx[a];
					}
					rightArea[b - 1]  = SurfaceArea(mn, mx);
					rightCount[b - 1] = cnt;
				}
			}

			for (int b = 0; b < SAH_BINS - 1; b++) {
				float cost = COST_TRAVERSE
					+ COST_INTERSECT * (leftCount[b]  * leftArea[b]
					                  + rightCount[b] * rightArea[b]) / parentArea;
				if (cost < bestCost) {
					bestCost = cost;
					bestAxis = axis;
					bestBin  = b;
				}
			}
		}

		if (bestAxis < 0) {
			float dx = bMax[0] - bMin[0];
			float dy = bMax[1] - bMin[1];
			float dz = bMax[2] - bMin[2];
			bestAxis = (dx >= dy && dx >= dz) ? 0 : (dy >= dz) ? 1 : 2;
		}

		float leafCost = COST_INTERSECT * (float)count;
		if (bestCost >= leafCost && count <= LEAF_MAX * 4) {
			node.first_primitive = start;
			node.primitive_count = count;
			node.l_child = -1;
			node.r_child = -1;
			continue;
		}

		{
			float cMin =  1e30f;
			float cMax = -1e30f;
			for (int i = start; i < start + count; i++) {
				float c = TriCentroid(h_triangles[triIdx[i]], bestAxis);
				if (c < cMin) cMin = c;
				if (c > cMax) cMax = c;
			}

			float scale = (float)SAH_BINS / (cMax - cMin + 1e-30f);
			float pivot;
			if (bestBin >= 0) {
				pivot = cMin + ((float)(bestBin + 1)) / scale;
			} else {
				pivot = 0.5f * (cMin + cMax);
			}

			int lo = start;
			int hi = start + count - 1;
			int i = lo;
			while (i <= hi) {
				float c = TriCentroid(h_triangles[triIdx[i]], bestAxis);
				if (c < pivot) {
					int tmp = triIdx[lo]; triIdx[lo] = triIdx[i]; triIdx[i] = tmp;
					lo++;
					i++;
				} else if (c > pivot) {
					int tmp = triIdx[hi]; triIdx[hi] = triIdx[i]; triIdx[i] = tmp;
					hi--;
				} else {
					i++;
				}
			}

			int leftCount = lo - start;
			if (leftCount <= 0) leftCount = 1;
			if (leftCount >= count) leftCount = count - 1;

			int leftChild  = nodesUsed++;
			int rightChild = nodesUsed++;
			{
				cudaBVHNode_t emptyNode;
				memset(&emptyNode, 0, sizeof(emptyNode));
				while (h_bvhNodes.Num() <= rightChild) {
					h_bvhNodes.Append(emptyNode);
				}
			}
            
            h_bvhNodes[nodeIdx].l_child = leftChild;
			h_bvhNodes[nodeIdx].r_child = rightChild;
			h_bvhNodes[nodeIdx].first_primitive = 0;
			h_bvhNodes[nodeIdx].primitive_count = 0;

			if (stackPtr + 2 > MAX_STACK) {
				h_bvhNodes[nodeIdx].first_primitive = start;
				h_bvhNodes[nodeIdx].primitive_count = count;
				h_bvhNodes[nodeIdx].l_child = -1;
				h_bvhNodes[nodeIdx].r_child = -1;
				nodesUsed -= 2;
				continue;
			}

			stack[stackPtr++] = { rightChild, start + leftCount, count - leftCount };
			stack[stackPtr++] = { leftChild,  start,             leftCount };
		}
	}

	h_bvhNodes.SetNum(nodesUsed);
	num_bvh_nodes = nodesUsed;

	h_bvhTriIndices.SetNum(triCount);
	for (int i = 0; i < triCount; i++) {
		h_bvhTriIndices[i] = triIdx[i];
	}

	if (r_cuDebug.GetBool()) {
		common->Printf("  BVH: %d nodes for %d triangles\n", num_bvh_nodes, triCount);
	}
}

#endif // HAVE_CUDA