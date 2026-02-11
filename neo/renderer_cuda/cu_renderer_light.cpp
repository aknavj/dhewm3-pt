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
void idCudaRenderer::AddLight(
    const idVec3& position, const idVec3& color, float intensity, 
    int type, float lightRadius, const float* lightProject, 
    int projTexIndex, const float* lightRadiusXYZ
) {

	if (h_lights.Num() >= MAX_LIGHTS) {
        common->Warning("idCudaRenderer::AddLight(): Maximum light count (%d) reached, cannot add more lights\n", MAX_LIGHTS);
		return;
	}

	cudaLight_t light;
	memset(&light, 0, sizeof(light));

	light.type = type;
    light.position[0] = position.x;
	light.position[1] = position.y;
	light.position[2] = position.z;
	light.color[0] = color.x;
	light.color[1] = color.y;
	light.color[2] = color.z;
	light.intensity = intensity;
	light.radius = lightRadius;

	// for directional lights, position stores the direction vector
	if (type == 1) {
		light.direction[0] = position.x;
		light.direction[1] = position.y;
		light.direction[2] = position.z;
		light.position[0] = light.position[1] = light.position[2] = 0.0f;
	}

	if (lightRadiusXYZ) {
		light.lightRadius3[0] = lightRadiusXYZ[0];
		light.lightRadius3[1] = lightRadiusXYZ[1];
		light.lightRadius3[2] = lightRadiusXYZ[2];
	} else {
		light.lightRadius3[0] = lightRadius;
		light.lightRadius3[1] = lightRadius;
		light.lightRadius3[2] = lightRadius;
	}

	if (lightProject) {
		memcpy(light.lightProject, lightProject, sizeof(light.lightProject));
	}

    light.textureIndex = -1;
	light.projectedTextureIndex = projTexIndex;

	h_lights.Append(light);

	return;
}

/*
========================
idCudaRenderer::AddSpotLight
========================
*/
void idCudaRenderer::AddSpotLight(
    const idVec3& position, const idVec3& direction, const idVec3& color, 
    float intensity, float coneAngle, float coneFalloff, float lightRadius,
    const float* lightProject, int projTexIndex, const float* rightAxis, 
    const float* upAxis
) {
    if (h_lights.Num() >= MAX_LIGHTS) {
        common->Warning("idCudaRenderer::AddSpotLight(): Maximum light count (%d) reached, cannot add more lights\n", MAX_LIGHTS);
        return;
	}

    cudaLight_t light;
	light.type = 0;
	light.position[0] = position.x;
	light.position[1] = position.y;
	light.position[2] = position.z;
	light.direction[0] = direction.x;
	light.direction[1] = direction.y;
	light.direction[2] = direction.z;
	light.color[0] = color.x;
	light.color[1] = color.y;
	light.color[2] = color.z;
	light.intensity = intensity;
	light.radius = lightRadius;
	light.area[0] = light.area[1] = light.area[2] = 0.0f;
	light.textureIndex = -1;
	light.coneAngle = coneAngle;
	light.coneFalloff = coneFalloff;

    // copy light projection planes
    if (lightProject) {
		memcpy(light.lightProject, lightProject, sizeof(light.lightProject));
	} else {
		memset(light.lightProject, 0, sizeof(light.lightProject));
	}

    // projected texture index fo spotlight projection
    light.projectedTextureIndex = projTexIndex;

    light.lightRadius3[0] = lightRadius * lightRadius * lightRadius;
	light.lightRadius3[1] = lightRadius * lightRadius * lightRadius;
	light.lightRadius3[2] = lightRadius * lightRadius * lightRadius;

    if (rightAxis) {
		light.right[0] = rightAxis[0];
		light.right[1] = rightAxis[1];
		light.right[2] = rightAxis[2];
	} else {
		light.right[0] = light.right[1] = light.right[2] = 0.0f;
	}

	if (upAxis) {
		light.up[0] = upAxis[0];
		light.up[1] = upAxis[1];
		light.up[2] = upAxis[2];
	} else {
		light.up[0] = light.up[1] = light.up[2] = 0.0f;
	}
	
	h_lights.Append(light);
}

#endif // HAVE_CUDA