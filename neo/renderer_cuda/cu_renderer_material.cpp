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
int idCudaRenderer::AddMaterial(const idMaterial* material, const float* shaderRegisters) {

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
    mat.emissionTexture = -1;
    mat.blendTexture = -1;
    mat.alphaMaskTexture = -1;
    mat.blendColor[0] = 1.0f;
    mat.blendColor[1] = 1.0f;
    mat.blendColor[2] = 1.0f;
    mat.blendColor[3] = 1.0f;
    mat.alphaTest = 0.0f;
    mat.coverage = 0;
    mat.blendMode = 0;
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
            h_materials[i].emissionTexture = -1;
            h_materials[i].blendTexture = -1;
            h_materials[i].alphaMaskTexture = -1;
            h_materials[i].blendColor[0] = 1.0f;
            h_materials[i].blendColor[1] = 1.0f;
            h_materials[i].blendColor[2] = 1.0f;
            h_materials[i].blendColor[3] = 1.0f;
            h_materials[i].alphaTest = 0.0f;
            h_materials[i].coverage = 0;
            h_materials[i].blendMode = 0;
			materialEmission[i].Set(0.0f, 0.0f, 0.0f);
		}
	}

    cudaMaterial_t mat;
    mat.albedo[0] = 0.5f;
    mat.albedo[1] = 0.5f;
    mat.albedo[2] = 0.5f;
    mat.specular[0] = 1.0f;
    mat.specular[1] = 1.0f;
    mat.specular[2] = 1.0f;
    mat.emission[0] = 0.0f;
    mat.emission[1] = 0.0f;
    mat.emission[2] = 0.0f;
    mat.albedoTexture = -1;
    mat.normalTexture = -1;
    mat.specularTexture = -1;
    mat.emissionTexture = -1;
    mat.blendTexture = -1;
    mat.alphaMaskTexture = -1;
    mat.blendColor[0] = 1.0f;
    mat.blendColor[1] = 1.0f;
    mat.blendColor[2] = 1.0f;
    mat.blendColor[3] = 1.0f;
    mat.alphaTest = 0.0f;
    mat.coverage = 0;
    mat.blendMode = 0;

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
    materialCoverage_t cov = material->Coverage();
    if (cov == MC_PERFORATED) {
        mat.coverage = 1;
        mat.alphaTest = 0.5f; // default threshold for perforated
    } else if (cov == MC_TRANSLUCENT) {
        mat.coverage = 2;
        mat.alphaTest = 0.0f; // translucent uses continuous alpha
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
    if (mat.coverage == 1) {
        bool usableAlphaTest = false;
        if (diffuseStage && diffuseStage->hasAlphaTest) usableAlphaTest = true;
        if (ambientBaseStage && ambientBaseStage->hasAlphaTest) usableAlphaTest = true;
        if (ambientGlowStage && ambientGlowStage->hasAlphaTest) usableAlphaTest = true;
        if (!usableAlphaTest) {
            if (r_cuDebug.GetBool()) {
                common->Printf("  coverage=1 but no usable alpha-tested stage, downgrading to opaque\n");
            }
            mat.coverage = 0;
            mat.alphaTest = 0.0f;
        }
    }

    if (r_cuDebug.GetBool()) {
        common->Printf("  stages: coverage=%d ambBase=%s ambGlow=%s ambMask=%s diff=%s norm=%s spec=%s\n",
            mat.coverage,
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
        if (mat.coverage == 1 || mat.coverage == 2) {
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
                index, material->GetName(), mat.coverage, mat.alphaTest);
        }
    }

    // process glow stage
    if (ambientGlowStage && ambientGlowStage->texture.image) {
        int emTex = AddTexture(ambientGlowStage->texture.image);
        if (emTex >= 0) {
            mat.emissionTexture = emTex;
            float emR = SAFE_REG(ambientGlowStage->color.registers[0]);
            float emG = SAFE_REG(ambientGlowStage->color.registers[1]);
            float emB = SAFE_REG(ambientGlowStage->color.registers[2]);
            mat.emission[0] = emR;
            mat.emission[1] = emG;
            mat.emission[2] = emB;
            materialEmission[index] = idVec3(mat.emission[0], mat.emission[1], mat.emission[2]);
        }
    }

    // process maskcolor stage
    if (ambientMaskStage && ambientMaskStage->texture.image) {
        int maskTex = AddTexture(ambientMaskStage->texture.image);
        if (maskTex >= 0) {
            mat.alphaMaskTexture = maskTex;
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

                // ambient-only surfaces without a glow map get mild self-emission
                if (mat.coverage == 0 && mat.emissionTexture < 0) {
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
        if (blendStage && mat.coverage == 2) {
            int ambSrc = blendStage->drawStateBits & GLS_SRCBLEND_BITS;
            int ambDst = blendStage->drawStateBits & GLS_DSTBLEND_BITS;

            bool isAdditive = (ambSrc == GLS_SRCBLEND_ONE && ambDst == GLS_DSTBLEND_ONE);
            bool isAlphaBlend = (ambSrc == GLS_SRCBLEND_SRC_ALPHA && ambDst == GLS_DSTBLEND_ONE_MINUS_SRC_ALPHA);
            bool isPremultAlpha = (ambSrc == GLS_SRCBLEND_ONE && ambDst == GLS_DSTBLEND_ONE_MINUS_SRC_ALPHA);
            bool isFilter = (ambSrc == GLS_SRCBLEND_DST_COLOR && ambDst == GLS_DSTBLEND_ZERO)
                         || (ambSrc == GLS_SRCBLEND_ZERO && ambDst == GLS_DSTBLEND_SRC_COLOR);

            if (isAdditive) {
                mat.blendMode = 1;
            } else if (isAlphaBlend || isPremultAlpha) {
                mat.blendMode = 2;
            } else if (isFilter) {
                mat.blendMode = 3;
            }

            // grab blend stage texture for blended surfaces (particles, decals, etc.)
            if (mat.blendMode > 0 && blendStage->texture.image) {
                mat.blendTexture = AddTexture(blendStage->texture.image);
                mat.blendColor[0] = SAFE_REG(blendStage->color.registers[0]);
                mat.blendColor[1] = SAFE_REG(blendStage->color.registers[1]);
                mat.blendColor[2] = SAFE_REG(blendStage->color.registers[2]);
                mat.blendColor[3] = SAFE_REG(blendStage->color.registers[3]);
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

    if (r_cuDebug.GetBool()) {
        common->Printf("  -> Final: albTex=%d blendTex=%d emTex=%d maskTex=%d blendMode=%d coverage=%d albedo=(%.2f,%.2f,%.2f) blendCol=(%.2f,%.2f,%.2f,%.2f)\n",
            mat.albedoTexture, mat.blendTexture, mat.emissionTexture, mat.alphaMaskTexture, mat.blendMode, mat.coverage,
            mat.albedo[0], mat.albedo[1], mat.albedo[2],
            mat.blendColor[0], mat.blendColor[1], mat.blendColor[2], mat.blendColor[3]);
    }

    // warn about materials with no usable textures — these will render as flat white
    if (mat.albedoTexture < 0 && mat.blendTexture < 0 && mat.emissionTexture < 0) {
        if (material->GetNumStages() > 0) {
            common->Warning("SetMaterial[%d]: %s has %d stages but NO textures loaded (defaulted/not-loaded?)\n",
                index, material->GetName(), material->GetNumStages());
        }
    }

    h_materials[index] = mat;
}

#endif // HAVE_CUDA