#ifdef HAVE_CUDA

#include "renderer/tr_local.h"
#include "renderer_cuda/cu_renderer.h"

// extern console variables
extern idCVar r_cuDebug;

/*
========================
idCudaRenderer::AddMaterial
========================
*/
int idCudaRenderer::AddMaterial(const idMaterial* material) {

    if (!material) {
        return 0;
    }

    int index = h_materials.Num();
    if (index >= MAX_MATERIALS) {
		common->Warning("idCudaRenderer::AddMaterial(): Maximum material count (%d) reached, cannot add '%s'\n", 
			MAX_MATERIALS, material->GetName());
		return 0;
	}

    if (r_cuDebug.GetBool()) {
		common->Printf("idCudaRenderer::AddMaterial(): Material %d: %s\n", index, material->GetName());
	}

    // grow the list first
    cudaMaterial_t mat;
    mat.albedo[0] = 1.0f;
    mat.albedo[1] = 1.0f;
    mat.albedo[2] = 1.0f;
    mat.specular[0] = 0.0f;
    mat.specular[1] = 0.0f;
    mat.specular[2] = 0.0f;
    mat.emission[0] = 0.0f;
    mat.emission[1] = 0.0f;
    mat.emission[2] = 0.0f;
    mat.albedoTexture = -1;
    mat.normalTexture = -1;
    mat.specularTexture = -1;
    h_materials.Append(mat);

    SetMaterial(index, material);

    return index;
}

/*
========================
idCudaRenderer::SetMaterial
========================
*/
void idCudaRenderer::SetMaterial(int index, const idMaterial* material) {

    cudaMaterial_t mat;
    mat.albedo[0] = 1.0f;
    mat.albedo[1] = 1.0f;
    mat.albedo[2] = 1.0f;
    mat.specular[0] = 1.0f;
    mat.specular[1] = 1.0f;
    mat.specular[2] = 1.0f;
    mat.albedoTexture = -1;
    mat.normalTexture = -1;
    mat.specularTexture = -1;

    if (!material) {
        h_materials[index] = mat;
        return;
    }

    // extract textures
    const shaderStage_t *diffuseStage = NULL;
    const shaderStage_t *normalStage = NULL;
    const shaderStage_t *specularStage = NULL;

    for (int i = 0; i < material->GetNumStages(); i++) {
        const shaderStage_t* stage = material->GetStage(i);
		if (!stage) continue;

        switch (stage->lighting) {
            case SL_AMBIENT:
            case SL_BUMP:
                normalStage = stage;
                break;
            case SL_DIFFUSE:
                diffuseStage = stage;
                break;
            case SL_SPECULAR:
                specularStage = stage;
                break;
            default:
                break;
        }
    }

    bool hasDiffuse = (diffuseStage != NULL);
    if (hasDiffuse) {
        mat.albedoTexture = AddTexture(diffuseStage->texture.image);
        if (r_cuDebug.GetBool()) {
            common->Printf("idCudaRenderer::SetMaterial(): Material %d: %s\n", index, material->GetName());
        }
    }

    bool hasNormal = (normalStage != NULL);
    if (hasNormal) {
        mat.normalTexture = AddTexture(normalStage->texture.image);
        if (r_cuDebug.GetBool()) {
            common->Printf("idCudaRenderer::SetMaterial(): Material %d: %s (normal map)\n", index, material->GetName());
        }
    }

    bool hasSpecular = (specularStage != NULL);
    if (hasSpecular) {
        mat.specularTexture = AddTexture(specularStage->texture.image);
        if (r_cuDebug.GetBool()) {
            common->Printf("idCudaRenderer::SetMaterial(): Material %d: %s (specular map)\n", index, material->GetName());
        }
    }   

    h_materials[index] = mat;
}

#endif // HAVE_CUDA