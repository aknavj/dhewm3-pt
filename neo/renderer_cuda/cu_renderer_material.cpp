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

    // check if material already cached by pointer
    int key = (int)(intptr_t)material;
    for (int i = materialHash.First(key); i >= 0; i = materialHash.Next(i)) {
        if (h_materialPtrs[i] == material) {
            return i;
        }
    }

    int index = nextMaterialIndex;
    if (index >= MAX_MATERIALS) {
		common->Warning("idCudaRenderer::AddMaterial(): Maximum material count (%d) reached, cannot add '%s'\n", 
			MAX_MATERIALS, material->GetName());
		return 0;
	}

    if (r_cuDebug.GetBool()) {
		common->Printf("idCudaRenderer::AddMaterial(): Material %d: %s\n", index, material->GetName());
	}

    // grow the lists
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
    materialEmission.Append(idVec3(0.0f, 0.0f, 0.0f));
    h_materialPtrs.Append(material);
    materialHash.Add(key, index);
    nextMaterialIndex++;

    SetMaterial(index, material);

    return index;
}

/*
========================
idCudaRenderer::SetMaterial
========================
*/
void idCudaRenderer::SetMaterial(int index, const idMaterial* material) {

    // set host data for material at index; actual GPU upload happens in EndFrame()
	int oldSize = h_materials.Num();
	if (index >= oldSize) {
		h_materials.SetNum(index + 1);
		materialEmission.SetNum(index + 1);
		for (int i = oldSize; i < index; i++) {
			h_materials[i].albedo[0] = 0.5f;
			h_materials[i].albedo[1] = 0.5f;
			h_materials[i].albedo[2] = 0.5f;
            h_materials[i].specular[0] = 1.0f;
            h_materials[i].specular[1] = 1.0f;
            h_materials[i].specular[2] = 1.0f;
			h_materials[i].emission[0] = 0.0f;
			h_materials[i].emission[1] = 0.0f;
			h_materials[i].emission[2] = 0.0f;
            h_materials[i].albedoTexture = -1;
            h_materials[i].normalTexture = -1;
            h_materials[i].specularTexture = -1;
			materialEmission[i].Set(0.0f, 0.0f, 0.0f);
		}
	}

    cudaMaterial_t mat;
    mat.albedo[0] = 0.5f;
    mat.albedo[1] = 0.5f;
    mat.albedo[2] = 0.5f;
    mat.albedo[3] = 1.0f;
    mat.specular[0] = 1.0f;
    mat.specular[1] = 1.0f;
    mat.specular[2] = 1.0f;
    mat.emission[0] = 0.0f;
    mat.emission[1] = 0.0f;
    mat.emission[2] = 0.0f;
    mat.albedoTexture = -1;
    mat.normalTexture = -1;
    mat.specularTexture = -1;

    if (!material) {
        h_materials[index] = mat;
        materialEmission[index] = idVec3(0.0f, 0.0f, 0.0f);
        return;
    }

    // extract textures
    const shaderStage_t *ambientStage = NULL;
    const shaderStage_t *diffuseStage = NULL;
    const shaderStage_t *normalStage = NULL;
    const shaderStage_t *specularStage = NULL;

    for (int i = 0; i < material->GetNumStages(); i++) {
        const shaderStage_t* stage = material->GetStage(i);
		if (!stage) continue;

        switch (stage->lighting) {
            case SL_AMBIENT:
            {
                int blendBits = stage->drawStateBits & (GLS_SRCBLEND_BITS | GLS_DSTBLEND_BITS);
				if (blendBits == (GLS_SRCBLEND_ZERO | GLS_DSTBLEND_ONE)) {
					break;
				}
                ambientStage = stage;
                break;
            }
            case SL_DIFFUSE:
            {
                int blendBits = stage->drawStateBits & (GLS_SRCBLEND_BITS | GLS_DSTBLEND_BITS);
				if (blendBits == (GLS_SRCBLEND_ZERO | GLS_DSTBLEND_ONE)) {
					break;
				}
                diffuseStage = stage;
                break;
            }
            case SL_BUMP:
            {
                normalStage = stage;
                break;
            }
            case SL_SPECULAR:
            {
                specularStage = stage;
                break;
            }
            default:
                break;
        }
    }

    bool hasDiffuse = (diffuseStage != NULL);
    if (hasDiffuse) {
        mat.albedoTexture = AddTexture(diffuseStage->texture.image);
        if (mat.albedoTexture >= 0) {
            float scale = 1.0f;
            mat.albedo[0] = diffuseStage->color.registers[0] * scale;
            mat.albedo[1] = diffuseStage->color.registers[1] * scale;
            mat.albedo[2] = diffuseStage->color.registers[2] * scale;
            mat.albedo[3] = 1.0f;
        }

        if (r_cuDebug.GetBool()) {
            common->Printf("idCudaRenderer::SetMaterial(): Material %d: %s\n", index, material->GetName());
        }
    }

    bool hasAmbient = (ambientStage != NULL);
    if (hasAmbient) {
        int ambSrc = ambientStage->drawStateBits & GLS_SRCBLEND_BITS;
		int ambDst = ambientStage->drawStateBits & GLS_DSTBLEND_BITS;

        // only treat as true emission for additive blend modes (SRC_ONE + DST_ONE)
        bool isAdditive = (ambDst == GLS_DSTBLEND_ONE);
        //bool isSrcAlpha = (ambSrc == GLS_SRCBLEND_SRC_ALPHA && ambDst == GLS_DSTBLEND_ONE);

        if (isAdditive /*|| isSrcAlpha*/) {
            // scale down - Doom 3 ambient colors are in [0,1] but represent subtle glow
            const float scale = 0.01f;
            mat.emission[0] = ambientStage->color.registers[0] * scale;
            mat.emission[1] = ambientStage->color.registers[1] * scale;
            mat.emission[2] = ambientStage->color.registers[2] * scale;
            materialEmission[index] = idVec3(mat.emission[0], mat.emission[1], mat.emission[2]);
        } 

        if (r_cuDebug.GetBool()) {
            common->Printf("idCudaRenderer::SetMaterial(): Ambient Material %d: %s (additive=%d)\n", index, material->GetName(), isAdditive /*|| isSrcAlpha*/);
        }
    }

    bool hasNormal = (normalStage != NULL);
    if (hasNormal) {
        mat.normalTexture = AddTexture(normalStage->texture.image);
        if (mat.normalTexture >= 0) {

        }

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