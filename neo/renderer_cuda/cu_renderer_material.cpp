#ifdef HAVE_CUDA

#include "renderer/tr_local.h"
#include "renderer_cuda/cu_renderer.h"

// extern console variables
extern idCVar r_cuDebug;

/*
========================
idCudaRenderer::GetOrSetMaterial
========================
*/
int idCudaRenderer::GetOrSetMaterial(const idMaterial* material, const float* shaderRegisters) {

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
		common->Warning("idCudaRenderer::GetOrSetMaterial(): Maximum material count (%d) reached, cannot add '%s'\n", 
			MAX_MATERIALS, material->GetName());
		return 0;
	}

    if (r_cuDebug.GetBool()) {
		common->Printf("idCudaRenderer::GetOrSetMaterial(): Material %d: %s\n", index, material->GetName());
	}

    // grow the lists
    cudaMaterial_t mat;
    memset(&mat, 0, sizeof(mat));
    mat.albedo[0] = 1.0f;
    mat.albedo[1] = 1.0f;
    mat.albedo[2] = 1.0f;
    mat.albedo[3] = 1.0f;
    mat.emission[0] = 0.0f;
    mat.emission[1] = 0.0f;
    mat.emission[2] = 0.0f;
    mat.metallic = 0.0f;
    mat.roughness = 0.7f;
    mat.ior = 1.5f;
    mat.transmission = 0.0f;
    mat.albedoTexture = -1;
    mat.normalTexture = -1;
    mat.specularTexture = -1;
    mat.alphaTest = 0.0f;
    mat.blendMode = 0;
    mat.useVertexColor = 0;
    mat.isAmbientOnly = 0;
    mat.noShadows = 0;
    mat.bumpScale = 1.0f;
    mat.texTransform[0] = 1.0f;
    mat.texTransform[1] = 1.0f;
    mat.texTransform[2] = 0.0f;
    mat.texTransform[3] = 0.0f;
    mat.texTransform[4] = 0.0f;
    mat.texTransform[5] = 0.0f;
    mat.polygonOffset = 0.0f;
    h_materials.Append(mat);
    materialEmission.Append(idVec3(0.0f, 0.0f, 0.0f));
    h_materialPtrs.Append(material);
    materialHash.Add(key, index);
    nextMaterialIndex++;

    SetMaterial(index, material, shaderRegisters);

    return index;
}

/*
========================
idCudaRenderer::SetMaterial
========================
*/
void idCudaRenderer::SetMaterial(int index, const idMaterial* material, const float* shaderRegisters) {

    // set host data for material at index; actual GPU upload happens in EndFrame()
	int oldSize = h_materials.Num();
	if (index >= oldSize) {
		h_materials.SetNum(index + 1);
		materialEmission.SetNum(index + 1);
		for (int i = oldSize; i < index; i++) {
			memset(&h_materials[i], 0, sizeof(cudaMaterial_t));
			h_materials[i].albedo[0] = 0.5f;
			h_materials[i].albedo[1] = 0.5f;
			h_materials[i].albedo[2] = 0.5f;
			h_materials[i].albedo[3] = 1.0f;
			h_materials[i].emission[0] = 0.0f;
			h_materials[i].emission[1] = 0.0f;
			h_materials[i].emission[2] = 0.0f;
			h_materials[i].metallic = 0.0f;
			h_materials[i].roughness = 0.7f;
			h_materials[i].ior = 1.5f;
			h_materials[i].transmission = 0.0f;
            h_materials[i].albedoTexture = -1;
            h_materials[i].normalTexture = -1;
            h_materials[i].specularTexture = -1;
            h_materials[i].alphaTest = 0.0f;
            h_materials[i].blendMode = 0;
			h_materials[i].useVertexColor = 0;
			h_materials[i].isAmbientOnly = 0;
			h_materials[i].noShadows = 0;
			h_materials[i].bumpScale = 1.0f;
			h_materials[i].texTransform[0] = 1.0f;
			h_materials[i].texTransform[1] = 1.0f;
			h_materials[i].polygonOffset = 0.0f;
			materialEmission[i].Set(0.0f, 0.0f, 0.0f);
		}
	}

    cudaMaterial_t mat;
    memset(&mat, 0, sizeof(mat));
    mat.albedo[0] = 0.5f;
    mat.albedo[1] = 0.5f;
    mat.albedo[2] = 0.5f;
    mat.albedo[3] = 1.0f;
    mat.emission[0] = 0.0f;
    mat.emission[1] = 0.0f;
    mat.emission[2] = 0.0f;
    mat.metallic = 0.0f;
    mat.roughness = 0.7f;
    mat.ior = 1.5f;
    mat.transmission = 0.0f;
    mat.albedoTexture = -1;
    mat.normalTexture = -1;
    mat.specularTexture = -1;
    mat.alphaTest = 0.0f;
    mat.blendMode = 0;
    mat.useVertexColor = 0;
    mat.isAmbientOnly = 0;
    mat.noShadows = 0;
    mat.bumpScale = 1.0f;
    mat.texTransform[0] = 1.0f;
    mat.texTransform[1] = 1.0f;
    mat.texTransform[2] = 0.0f;
    mat.texTransform[3] = 0.0f;
    mat.texTransform[4] = 0.0f;
    mat.texTransform[5] = 0.0f;
    mat.polygonOffset = 0.0f;

    if (!material) {
        h_materials[index] = mat;
        materialEmission[index] = idVec3(0.0f, 0.0f, 0.0f);
        return;
    }

    if (r_cuDebug.GetBool()) {
        common->Printf("SetMaterial[%d]: %s (stages=%d, regs=%d, constRegs=%s)\n",
            index, material->GetName(), material->GetNumStages(),
            material->GetNumRegisters(),
            material->ConstantRegisters() ? "yes" : "no");
    }

    // extract coverage type and alpha test threshold from engine material
    int coverage = 0; // local: 0=opaque, 1=perforated, 2=translucent
    materialCoverage_t cov = material->Coverage();
    if (cov == MC_PERFORATED) {
        coverage = 1;
        mat.alphaTest = 0.5f; // default threshold for perforated
    } else if (cov == MC_TRANSLUCENT) {
        coverage = 2;
        mat.alphaTest = 0.0f; // translucent uses continuous alpha
    }

    // extract noShadows flag from material (SurfaceCastsShadow checks MF_NOSHADOWS and MF_FORCESHADOWS)
    if (!material->SurfaceCastsShadow()) {
        mat.noShadows = 1;
    }

    // detect vertex color usage from stages (particles use per-vertex alpha for opacity)
    for (int i = 0; i < material->GetNumStages(); i++) {
        const shaderStage_t* stage = material->GetStage(i);
        if (stage && (stage->vertexColor == SVC_MODULATE || stage->vertexColor == SVC_INVERSE_MODULATE)) {
            mat.useVertexColor = 1;
            break;
        }
    }

    // check if material is ambient-only (no diffuse/bump/specular interaction stages)
    bool hasInteraction = false;
    for (int i = 0; i < material->GetNumStages(); i++) {
        const shaderStage_t* stage = material->GetStage(i);
        if (stage && (stage->lighting == SL_DIFFUSE || stage->lighting == SL_BUMP || stage->lighting == SL_SPECULAR)) {
            hasInteraction = true;
            break;
        }
    }
    if (!hasInteraction && material->GetNumStages() > 0) {
        mat.isAmbientOnly = 1;
    }

    // resolve constant registers for this material
    const float* constRegs = material->ConstantRegisters();
    static float evalRegs[4096]; // MAX_EXPRESSION_REGISTERS
    int numRegs = material->GetNumRegisters();
    if (!constRegs) {
        // non-constant materials have animated registers
        if (shaderRegisters && numRegs > 0 && numRegs <= 4096) {
            memcpy(evalRegs, shaderRegisters, numRegs * sizeof(float));
            constRegs = evalRegs;
        } else {
            int fillCount = (numRegs > 0 && numRegs <= 4096) ? numRegs : 4096;
            for (int i = 0; i < fillCount; i++) {
                evalRegs[i] = 1.0f;
            }
            constRegs = evalRegs;
        }
    }

    // safe register lookup helper macro
    #define SAFE_REG(regIdx) ((regIdx >= 0 && regIdx < numRegs) ? constRegs[regIdx] : 1.0f)

    // extract textures
    const shaderStage_t *ambientBaseStage = NULL;   // first non-additive ambient stage
    const shaderStage_t *ambientGlowStage = NULL;   // first additive ambient stage
    const shaderStage_t *ambientMaskStage = NULL;   // first maskcolor ambient stage (alpha-only write)
    const shaderStage_t *diffuseStage = NULL;
    const shaderStage_t *normalStage = NULL;
    const shaderStage_t *specularStage = NULL;

    for (int i = 0; i < material->GetNumStages(); i++) {
        const shaderStage_t* stage = material->GetStage(i);
		if (!stage) continue;

        // skip fragment program stages
        if (stage->newStage) {
            if (r_cuDebug.GetBool()) {
                common->Printf("  stage %d: SKIP (fragment program)\n", i);
            }
            continue;
        }

        // skip stages with non-standard texture coordinate generation
        if (stage->texture.texgen != TG_EXPLICIT) {
            if (r_cuDebug.GetBool()) {
                common->Printf("  stage %d: SKIP (texgen=%d)\n", i, (int)stage->texture.texgen);
            }
            continue;
        }

        switch (stage->lighting) {
            case SL_AMBIENT:
            {
                int blendBits = stage->drawStateBits & (GLS_SRCBLEND_BITS | GLS_DSTBLEND_BITS);
				if (blendBits == (GLS_SRCBLEND_ZERO | GLS_DSTBLEND_ONE)) {
					break;
				}

                // maskcolor stages only write alpha
                bool isMaskColor = ((stage->drawStateBits & GLS_COLORMASK) == GLS_COLORMASK);
                if (isMaskColor) {
                    if (!ambientMaskStage) ambientMaskStage = stage;
                    if (r_cuDebug.GetBool()) {
                        common->Printf("  stage %d: maskcolor (alpha mask)\n", i);
                    }
                    break;
                }

                bool stageIsAdd = (blendBits == (GLS_SRCBLEND_ONE | GLS_DSTBLEND_ONE));
                if (stageIsAdd) {
                    if (!ambientGlowStage) ambientGlowStage = stage;
                } else {
                    if (!ambientBaseStage) ambientBaseStage = stage;
                }
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

    // validate perforated coverage
    if (coverage == 1) {
        bool usableAlphaTest = false;
        if (diffuseStage && diffuseStage->hasAlphaTest) usableAlphaTest = true;
        if (ambientBaseStage && ambientBaseStage->hasAlphaTest) usableAlphaTest = true;
        if (ambientGlowStage && ambientGlowStage->hasAlphaTest) usableAlphaTest = true;
        if (!usableAlphaTest) {
            if (r_cuDebug.GetBool()) {
                common->Printf("  coverage=1 but no usable alpha-tested stage, downgrading to opaque\n");
            }
            coverage = 0;
            mat.alphaTest = 0.0f;
        }
    }

    if (r_cuDebug.GetBool()) {
        common->Printf("  stages: coverage=%d ambBase=%s ambGlow=%s ambMask=%s diff=%s norm=%s spec=%s\n",
            coverage,
            ambientBaseStage ? "yes" : "no",
            ambientGlowStage ? "yes" : "no",
            ambientMaskStage ? "yes" : "no",
            hasDiffuse ? "yes" : "no",
            normalStage ? "yes" : "no",
            specularStage ? "yes" : "no");
    }

    if (hasDiffuse) {
        mat.albedoTexture = AddTexture(diffuseStage->texture.image);
        if (mat.albedoTexture >= 0) {
            mat.albedo[0] = SAFE_REG(diffuseStage->color.registers[0]);
            mat.albedo[1] = SAFE_REG(diffuseStage->color.registers[1]);
            mat.albedo[2] = SAFE_REG(diffuseStage->color.registers[2]);
        }

        // refine alpha test threshold from stage draw state bits
        if (coverage == 1 || coverage == 2) {
            int atest = diffuseStage->drawStateBits & GLS_ATEST_BITS;
            if (atest & GLS_ATEST_GE_128) {
                mat.alphaTest = 128.0f / 255.0f; // ~0.502
            } else if (atest & GLS_ATEST_LT_128) {
                mat.alphaTest = 128.0f / 255.0f;
            } else if (atest & GLS_ATEST_EQ_255) {
                mat.alphaTest = 254.0f / 255.0f;
            }
        }

        if (r_cuDebug.GetBool()) {
            common->Printf("idCudaRenderer::SetMaterial(): Material %d: %s (coverage=%d, alphaTest=%.3f)\n",
                index, material->GetName(), coverage, mat.alphaTest);
        }
    }

    // process glow stage (emission)
    int emissionTextureIdx = -1;
    if (ambientGlowStage && ambientGlowStage->texture.image && !mat.isAmbientOnly) {
        int emTex = AddTexture(ambientGlowStage->texture.image);
        if (emTex >= 0) {
            emissionTextureIdx = emTex;
            float emR = SAFE_REG(ambientGlowStage->color.registers[0]);
            float emG = SAFE_REG(ambientGlowStage->color.registers[1]);
            float emB = SAFE_REG(ambientGlowStage->color.registers[2]);
            mat.emission[0] = emR;
            mat.emission[1] = emG;
            mat.emission[2] = emB;
            materialEmission[index] = idVec3(mat.emission[0], mat.emission[1], mat.emission[2]);
        }
    }

    // process maskcolor stage (not stored in struct, just used for debug)
    if (ambientMaskStage && ambientMaskStage->texture.image) {
        int maskTex = AddTexture(ambientMaskStage->texture.image);
        if (maskTex >= 0) {
            if (r_cuDebug.GetBool()) {
                common->Printf("  -> alphaMaskTexture=%d from maskcolor stage\n", maskTex);
            }
        }
    }

    // for materials with NO diffuse stage
    if (!hasDiffuse && mat.albedoTexture < 0) {
        const shaderStage_t *albedoSource = ambientBaseStage;
        if (!albedoSource) albedoSource = ambientMaskStage;
        if (!albedoSource) albedoSource = ambientGlowStage;

        if (albedoSource && albedoSource->texture.image) {
            int ambTex = AddTexture(albedoSource->texture.image);
            if (ambTex >= 0) {
                mat.albedoTexture = ambTex;
                mat.albedo[0] = SAFE_REG(albedoSource->color.registers[0]);
                mat.albedo[1] = SAFE_REG(albedoSource->color.registers[1]);
                mat.albedo[2] = SAFE_REG(albedoSource->color.registers[2]);
                mat.albedo[3] = idMath::ClampFloat(0.0f, 1.0f, SAFE_REG(albedoSource->color.registers[3]));

                // extract blend mode from ambient stage draw state bits
                int srcBlend = albedoSource->drawStateBits & GLS_SRCBLEND_BITS;
                int dstBlend = albedoSource->drawStateBits & GLS_DSTBLEND_BITS;

                if (srcBlend == GLS_SRCBLEND_ONE && dstBlend == GLS_DSTBLEND_ONE) {
                    mat.blendMode = 2;  // Additive (GL_ONE, GL_ONE)
                } else if (srcBlend == GLS_SRCBLEND_SRC_ALPHA && dstBlend == GLS_DSTBLEND_ONE_MINUS_SRC_ALPHA) {
                    mat.blendMode = 1;  // Alpha blend
                } else if ((srcBlend == GLS_SRCBLEND_DST_COLOR && dstBlend == GLS_DSTBLEND_ZERO)
                        || (srcBlend == GLS_SRCBLEND_ZERO && dstBlend == GLS_DSTBLEND_SRC_COLOR)) {
                    mat.blendMode = 3;  // Multiply/filter
                } else if (srcBlend != GLS_SRCBLEND_ONE || dstBlend != GLS_DSTBLEND_ZERO) {
                    // Non-opaque blend mode: check for additive variant (e.g. GL_SRC_ALPHA + GL_ONE)
                    if (dstBlend == GLS_DSTBLEND_ONE) {
                        mat.blendMode = 2;  // Additive variant
                    } else {
                        mat.blendMode = 1;  // Default to alpha blend for any other non-opaque
                    }
                }

                // ambient-only surfaces without a glow map get mild self-emission
                if (mat.blendMode == 0 && emissionTextureIdx < 0) {
                    mat.emission[0] = mat.albedo[0] * 0.3f;
                    mat.emission[1] = mat.albedo[1] * 0.3f;
                    mat.emission[2] = mat.albedo[2] * 0.3f;
                    materialEmission[index] = idVec3(mat.emission[0], mat.emission[1], mat.emission[2]);
                }
            }
        }
    }

    // classify blend modes for translucent materials
    {
        const shaderStage_t *blendStage = ambientBaseStage ? ambientBaseStage : ambientGlowStage;
        if (blendStage && coverage == 2) {
            int ambSrc = blendStage->drawStateBits & GLS_SRCBLEND_BITS;
            int ambDst = blendStage->drawStateBits & GLS_DSTBLEND_BITS;

            bool isAdditive = (ambSrc == GLS_SRCBLEND_ONE && ambDst == GLS_DSTBLEND_ONE);
            bool isAlphaBlend = (ambSrc == GLS_SRCBLEND_SRC_ALPHA && ambDst == GLS_DSTBLEND_ONE_MINUS_SRC_ALPHA);
            bool isPremultAlpha = (ambSrc == GLS_SRCBLEND_ONE && ambDst == GLS_DSTBLEND_ONE_MINUS_SRC_ALPHA);
            bool isFilter = (ambSrc == GLS_SRCBLEND_DST_COLOR && ambDst == GLS_DSTBLEND_ZERO)
                         || (ambSrc == GLS_SRCBLEND_ZERO && ambDst == GLS_DSTBLEND_SRC_COLOR);

            if (isAdditive) {
                mat.blendMode = 2; // additive
            } else if (isAlphaBlend || isPremultAlpha) {
                mat.blendMode = 1; // alpha blend
            } else if (isFilter) {
                mat.blendMode = 3; // multiply/filter
            }

            // for blended surfaces, use the blend stage texture as albedo if we don't have one
            if (mat.blendMode > 0 && blendStage->texture.image && mat.albedoTexture < 0) {
                mat.albedoTexture = AddTexture(blendStage->texture.image);
                mat.albedo[0] = SAFE_REG(blendStage->color.registers[0]);
                mat.albedo[1] = SAFE_REG(blendStage->color.registers[1]);
                mat.albedo[2] = SAFE_REG(blendStage->color.registers[2]);
                mat.albedo[3] = SAFE_REG(blendStage->color.registers[3]);
            }

            if (r_cuDebug.GetBool()) {
                common->Printf("idCudaRenderer::SetMaterial(): Blend Material %d: %s (blendMode=%d)\n",
                    index, material->GetName(), mat.blendMode);
            }
        }
    }

    // process normal map stage
    bool hasNormal = (normalStage != NULL);
    if (hasNormal) {
        mat.normalTexture = AddTexture(normalStage->texture.image);
        if (r_cuDebug.GetBool()) {
            common->Printf("idCudaRenderer::SetMaterial(): Material %d: %s (normal map)\n", index, material->GetName());
        }
    }

    // process specular map stage
    bool hasSpecular = (specularStage != NULL);
    if (hasSpecular) {
        mat.specularTexture = AddTexture(specularStage->texture.image);
        if (r_cuDebug.GetBool()) {
            common->Printf("idCudaRenderer::SetMaterial(): Material %d: %s (specular map)\n", index, material->GetName());
        }
    }   

    #undef SAFE_REG

    h_materials[index] = mat;
}

#endif // HAVE_CUDA