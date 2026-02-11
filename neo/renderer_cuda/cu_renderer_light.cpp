#ifdef HAVE_CUDA

#include "renderer/tr_local.h"
#include "renderer_cuda/cu_renderer.h"
#include <cuda_runtime.h>

// extern console variables
extern idCVar r_cuDebug;
extern idCVar r_cuRenderMode;

/*
========================
idCudaRenderer::AddLight
========================
*/
void idCudaRenderer::AddLight(const idVec3& position, const idVec3& color, float intensity, float radius) {

	if (h_lights.Num() >= MAX_LIGHTS) {
		return;
	}

	cudaLight_t light;
	light.position[0] = position.x;
	light.position[1] = position.y;
	light.position[2] = position.z;
	light.color[0] = color.x;
	light.color[1] = color.y;
	light.color[2] = color.z;
	light.intensity = intensity;
	light.radius = radius;
	h_lights.Append(light);

	return;
}

#endif // HAVE_CUDA