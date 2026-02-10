#ifndef __CU_RENDERER_MATH_CUH__
#define __CU_RENDERER_MATH_CUH__

#ifdef HAVE_CUDA

#include <cuda_runtime.h>


/*
========================
cupt_hash
========================
*/
__device__ __forceinline__ unsigned int cupt_hash(unsigned int seed) {
	seed = (seed ^ 61u) ^ (seed >> 16u);
	seed *= 9u;
	seed = seed ^ (seed >> 4u);
	seed *= 0x27d4eb2du;
	seed = seed ^ (seed >> 15u);
	return seed;
}

/*
========================
cupt_random
========================
*/
__device__ __forceinline__ float cupt_random(unsigned int seed) {
	return float(cupt_hash(seed)) / float(0xFFFFFFFFu);
}


/*
========================
clamp
========================
*/
__device__ inline float clamp(float x, float a, float b) {
	return fmaxf(a, fminf(b, x));
}

/*
========================
dot3
========================
*/
__device__ inline float dot3(const float* a, const float* b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/*
========================
cross3
========================
*/
__device__ inline void cross3(const float* a, const float* b, float* result) {
	result[0] = a[1] * b[2] - a[2] * b[1];
	result[1] = a[2] * b[0] - a[0] * b[2];
	result[2] = a[0] * b[1] - a[1] * b[0];
}

/*
========================
normalize3
========================
*/
__device__ inline void normalize3(float* v) {
	float len = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	if (len > 1e-6f) {
		v[0] /= len;
		v[1] /= len;
		v[2] /= len;
	}
}

/*
========================
length3
========================
*/
__device__ inline float length3(const float* v) {
	return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}


/*
========================
IntersectAABB
========================
*/
__device__ __forceinline__ bool IntersectAABB(
	float origX, float origY, float origZ,
	float invDirX, float invDirY, float invDirZ,
	const float* bounds,
	float tMax
) {
	float t1 = (bounds[0] - origX) * invDirX;
	float t2 = (bounds[3] - origX) * invDirX;
	float tmin = fminf(t1, t2);
	float tmax = fmaxf(t1, t2);

	t1 = (bounds[1] - origY) * invDirY;
	t2 = (bounds[4] - origY) * invDirY;
	tmin = fmaxf(tmin, fminf(t1, t2));
	tmax = fminf(tmax, fmaxf(t1, t2));

	t1 = (bounds[2] - origZ) * invDirZ;
	t2 = (bounds[5] - origZ) * invDirZ;
	tmin = fmaxf(tmin, fminf(t1, t2));
	tmax = fminf(tmax, fmaxf(t1, t2));

	return tmax >= fmaxf(tmin, 0.0f) && tmin < tMax;
}

/*
========================
IntersectTriangle
========================
*/
__device__ __forceinline__ float IntersectTriangle(
	float origX, float origY, float origZ,
	float dirX,  float dirY,  float dirZ,
	const cudaVertex_t* vertices,
	const cudaTriangle_t& tri
) {
	int i0 = tri.vertexIndices[0];
	int i1 = tri.vertexIndices[1];
	int i2 = tri.vertexIndices[2];

	float v0x = vertices[i0].position[0];
	float v0y = vertices[i0].position[1];
	float v0z = vertices[i0].position[2];

	float e1x = vertices[i1].position[0] - v0x;
	float e1y = vertices[i1].position[1] - v0y;
	float e1z = vertices[i1].position[2] - v0z;

	float e2x = vertices[i2].position[0] - v0x;
	float e2y = vertices[i2].position[1] - v0y;
	float e2z = vertices[i2].position[2] - v0z;

	float hx = dirY * e2z - dirZ * e2y;
	float hy = dirZ * e2x - dirX * e2z;
	float hz = dirX * e2y - dirY * e2x;

	float a = e1x * hx + e1y * hy + e1z * hz;
	if (fabsf(a) < 1e-7f) return -1.0f;

	float f = 1.0f / a;
	float sx = origX - v0x;
	float sy = origY - v0y;
	float sz = origZ - v0z;

	float ub = f * (sx * hx + sy * hy + sz * hz);
	if (ub < 0.0f || ub > 1.0f) return -1.0f;

	float qx = sy * e1z - sz * e1y;
	float qy = sz * e1x - sx * e1z;
	float qz = sx * e1y - sy * e1x;

	float vb = f * (dirX * qx + dirY * qy + dirZ * qz);
	if (vb < 0.0f || ub + vb > 1.0f) return -1.0f;

	float t = f * (e2x * qx + e2y * qy + e2z * qz);
	return (t > 0.001f) ? t : -1.0f;
}

/*
========================
IntersectTriangleUV
========================
*/
__device__ __forceinline__ float IntersectTriangleUV(
	float origX, float origY, float origZ,
	float dirX,  float dirY,  float dirZ,
	const cudaVertex_t* vertices,
	const cudaTriangle_t& tri,
	float& outU, float& outV
) {
	int i0 = tri.vertexIndices[0];
	int i1 = tri.vertexIndices[1];
	int i2 = tri.vertexIndices[2];

	float v0x = vertices[i0].position[0];
	float v0y = vertices[i0].position[1];
	float v0z = vertices[i0].position[2];

	float e1x = vertices[i1].position[0] - v0x;
	float e1y = vertices[i1].position[1] - v0y;
	float e1z = vertices[i1].position[2] - v0z;

	float e2x = vertices[i2].position[0] - v0x;
	float e2y = vertices[i2].position[1] - v0y;
	float e2z = vertices[i2].position[2] - v0z;

	float hx = dirY * e2z - dirZ * e2y;
	float hy = dirZ * e2x - dirX * e2z;
	float hz = dirX * e2y - dirY * e2x;

	float a = e1x * hx + e1y * hy + e1z * hz;
	if (fabsf(a) < 1e-7f) return -1.0f;

	float f = 1.0f / a;
	float sx = origX - v0x;
	float sy = origY - v0y;
	float sz = origZ - v0z;

	float ub = f * (sx * hx + sy * hy + sz * hz);
	if (ub < 0.0f || ub > 1.0f) return -1.0f;

	float qx = sy * e1z - sz * e1y;
	float qy = sz * e1x - sx * e1z;
	float qz = sx * e1y - sy * e1x;

	float vb = f * (dirX * qx + dirY * qy + dirZ * qz);
	if (vb < 0.0f || ub + vb > 1.0f) return -1.0f;

	float t = f * (e2x * qx + e2y * qy + e2z * qz);
	if (t > 0.001f) {
		outU = ub;
		outV = vb;
		return t;
	}
	return -1.0f;
}

/*
========================
SampleTexture
========================
*/
__device__ __forceinline__ void SampleTexture(
	const cudaTexture_t& tex,
	float u, float v,
	float& outR, float& outG, float& outB
) {
	if (!tex.data || tex.width <= 0 || tex.height <= 0) {
		outR = outG = outB = 1.0f;
		return;
	}

	u = u - floorf(u);
	v = v - floorf(v);

	int px = (int)(u * (float)tex.width) % tex.width;
	int py = (int)(v * (float)tex.height) % tex.height;
	if (px < 0) px += tex.width;
	if (py < 0) py += tex.height;

	int idx = (py * tex.width + px) * 4;
	outR = (float)tex.data[idx + 0] / 255.0f;
	outG = (float)tex.data[idx + 1] / 255.0f;
	outB = (float)tex.data[idx + 2] / 255.0f;
}

#endif // HAVE_CUDA
#endif // __CU_RENDERER_MATH_CUH__