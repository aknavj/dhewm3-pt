#include "sys/platform.h"

#ifdef HAVE_CUDA

#include "framework/Console.h"
#include "renderer/tr_local.h"
#include "renderer_cuda/cu_renderer.h"

// extern console variables
extern idCVar r_cuDraw;
extern idCVar r_frameStats;
extern void R_CUDA_Compare_f(const idCmdArgs& args);
extern void R_CUDA_Sequence_f(const idCmdArgs& args);
extern void R_CUDA_Abort_f(const idCmdArgs& args);

/*
========================
RB_CUDA_Init
========================
*/
void RB_CUDA_Init() {
	if (!r_cuDraw.GetBool()) {
		return;
	}

	if (!idCudaRenderer::IsAvailable()) {
		common->Warning("CUDA renderer requested but CUDA is not available\n");
		r_cuDraw.SetBool(false);
		return;
	}
	
	if (!g_cuRenderer) {
		g_cuRenderer = new idCudaRenderer();
		if (!g_cuRenderer->Init(glConfig.vidWidth, glConfig.vidHeight)) {
			delete g_cuRenderer;
			g_cuRenderer = NULL;
			r_cuDraw.SetBool(false);
			common->Warning("\n*** CUDA Renderer initialization FAILED ***\n\n");
		} else {
			common->Printf("\n*** CUDA Renderer is ACTIVE ***\n");
			common->Printf("Use 'r_cuDraw 0' to disable\n\n");
		}
	}

	cmdSystem->AddCommand("r_cuCompare", R_CUDA_Compare_f, CMD_FL_RENDERER, "Compare CUDA renderer output to GL image");
	cmdSystem->AddCommand("r_cuSequence", R_CUDA_Sequence_f, CMD_FL_RENDERER, "Render a sequence of frames with CUDA renderer for benchmarking");
	cmdSystem->AddCommand("r_cuAbort", R_CUDA_Abort_f, CMD_FL_RENDERER, "Abort CUDA renderer sequence benchmark");
	cmdSystem->AddCommand("r_glSequence", R_GLSequence_f, CMD_FL_RENDERER, "Render a sequence of OpenGL frames as screenshots");
	cmdSystem->AddCommand("r_glAbort", R_GLAbort_f, CMD_FL_RENDERER, "Abort OpenGL sequence render");

}

/*
========================
RB_CUDA_Shutdown
========================
*/
void RB_CUDA_Shutdown() {
	if (g_cuRenderer) {
		delete g_cuRenderer;
		g_cuRenderer = NULL;
	}
}

/*
========================
RB_CUDA_DrawView
========================
*/
void RB_CUDA_DrawView() {

	if (!g_cuRenderer || !backEnd.viewDef) {
		return;
	}

    // process engine view data and prepare for CUDA rendering
	g_cuRenderer->BeginFrame();
	
    // Add all draw surfaces to CUDA renderer
	for (int i = 0; i < backEnd.viewDef->numDrawSurfs; i++) {
		const drawSurf_t* surf = backEnd.viewDef->drawSurfs[i];
		if (!surf || !surf->geo) {
			continue;
		}

		const srfTriangles_t* tri = surf->geo;
		if (!tri->verts || !tri->indexes) {
			continue;
		}

        // get entity transform
		const float* modelMatrix = NULL;
		if (surf->space) {
			modelMatrix = surf->space->modelMatrix;
		}

		// get material index
        int materialIndex = 0;
        if (surf->material) {
            materialIndex = g_cuRenderer->GetOrSetMaterial(surf->material, surf->shaderRegisters);
        }

        // add triangle data to CUDA renderer
		g_cuRenderer->AddTriangle(
			tri->verts,
			tri->numVerts,
			tri->indexes,
			tri->numIndexes,
            materialIndex,
			modelMatrix
		);
	}

    // add lights from the scene
    for (viewLight_t* vLight = backEnd.viewDef->viewLights; vLight; vLight = vLight->next) {

		if (!vLight->lightDef) {
			continue;
		}
		
		const idRenderLightLocal* light = vLight->lightDef;
		const renderLight_t& parms = light->parms;
		const idMaterial* lightShader = vLight->lightShader;
		if (!lightShader) {
			lightShader = light->lightShader;
		}

		if (!lightShader) {
			continue; // skip lights without valid shader
		}
		
		// skip fog lights and ambient lights (they don't cast direct light)
		if (lightShader->IsFogLight() || lightShader->IsAmbientLight() || lightShader->IsBlendLight()) {
			continue;
		}
		
		idVec3 lightOrigin = parms.origin;
		
		// extract light color from shader parameters
		idVec3 lightColor(
			parms.shaderParms[SHADERPARM_RED],
			parms.shaderParms[SHADERPARM_GREEN],
			parms.shaderParms[SHADERPARM_BLUE]
		);
		
		// if color is not set via shader parms, use default white
		if (lightColor.LengthSqr() < 0.01f) {
			lightColor.Set(1.0f, 1.0f, 1.0f);
		}
		
		// get base intensity from SHADERPARM_ALPHA or default
		float baseIntensity = parms.shaderParms[SHADERPARM_ALPHA];
		if (baseIntensity < 0.01f) {
			baseIntensity = 1.0f;
		}
		
		float intensity = baseIntensity;
		int lightType = 0;
		
		float projPlanes[4][4];
		for (int p = 0; p < 4; p++) {
			projPlanes[p][0] = vLight->lightProject[p].Normal().x;
			projPlanes[p][1] = vLight->lightProject[p].Normal().y;
			projPlanes[p][2] = vLight->lightProject[p].Normal().z;
			projPlanes[p][3] = vLight->lightProject[p][3];
		}
		
		int projTexIndex = -1;
		
		if (parms.pointLight) {
			lightType = 0; // point light
			
			// doom 3 light radius defines the falloff boundary.
			float radius = (parms.lightRadius.x + parms.lightRadius.y + parms.lightRadius.z) / 3.0f;
			
			// compensate for PBR BRDF's 1/pi normalization.
			intensity = baseIntensity * 3.14159f;
			if (intensity < 0.5f) intensity = 0.5f;
			if (intensity > 100.0f) intensity = 100.0f;
			
			// apply lightCenter offset if specified
			if (parms.lightCenter.LengthSqr() > 0.01f) {
				lightOrigin += parms.lightCenter;
			}
			
			// pass light radius
			float lightRadiusXYZ[3] = {
				parms.lightRadius.x,
				parms.lightRadius.y,
				parms.lightRadius.z
			};
			
			g_cuRenderer->AddLight(lightOrigin, lightColor, intensity, lightType, radius,
									 &projPlanes[0][0], -1, lightRadiusXYZ);
			continue;
		} else if (parms.parallel) {
			lightType = 1; // directional light
			
			// parallel lights are bright and uniform
			intensity = baseIntensity * 10.0f;
			
			// for directional lights, lightCenter gives the direction
			lightOrigin = parms.lightCenter;
			if (lightOrigin.LengthSqr() < 0.01f) {
				lightOrigin = parms.target - parms.origin;
			}

			lightOrigin.Normalize();
		} else {
			// projected light area source
			idVec3 worldTarget = parms.axis * parms.target;
			idVec3 worldRight = parms.axis * parms.right;
			idVec3 worldUp = parms.axis * parms.up;
			
			// Extract direction from world-space target vector
			idVec3 lightDir = worldTarget;
			float targetLen = lightDir.Length();
			if (targetLen > 0.01f) {
				lightDir /= targetLen;
			} else {
				lightDir.Set(1.0f, 0.0f, 0.0f);
			}
			
			// calculate frustum dimensions for cone angle computation
			float projWidth = worldRight.Length();
			float projHeight = worldUp.Length();
			float projDepth = (parms.end - parms.start).Length();
			
			// Compensate for PBR 1/pi (same as point lights)
			intensity = baseIntensity * 3.14159f;
			if (intensity < 0.5f) intensity = 0.5f;
			if (intensity > 100.0f) intensity = 100.0f;
			
			// compute cone half-angle from frustum right/up vs target length
			float halfWidth = fmaxf(projWidth, projHeight);
			float coneAngle = atanf(halfWidth / fmaxf(targetLen, 1.0f));
			
			// soft edge falloff exponent (higher = sharper edge)
			float coneFalloff = 2.0f;
			
			// use projection depth as the light's effective range
			float lightRange = projDepth;
			if (lightRange < 10.0f) lightRange = 500.0f;  // fallback for malformed frustums
			
			// extract normalized right and up axis vectors in WORLD space
			idVec3 rightNorm = worldRight;
			rightNorm.Normalize();
			idVec3 upNorm = worldUp;
			upNorm.Normalize();
			float rightAxis[3] = { rightNorm.x, rightNorm.y, rightNorm.z };
			float upAxis[3] = { upNorm.x, upNorm.y, upNorm.z };
			
			// extract projected texture for this projected light only.
			if (lightShader->GetNumStages() > 0) {
				const shaderStage_t* lightStage = lightShader->GetStage(0);
				if (lightStage && lightStage->texture.image) {
					projTexIndex = g_cuRenderer->AddTexture(lightStage->texture.image);
				}
			}
			
			g_cuRenderer->AddSpotLight(lightOrigin, lightDir, lightColor, 
										intensity, coneAngle, coneFalloff, 
										lightRange, &projPlanes[0][0], projTexIndex, 
										rightAxis, upAxis);
			continue;
		}
		
		g_cuRenderer->AddLight(lightOrigin, lightColor, intensity, lightType);
	}

	g_cuRenderer->EndFrame();

	const renderView_t *renderView = &backEnd.viewDef->renderView;
	g_cuRenderer->RenderView(renderView);

    // copy CUDA result into host-side pixel buffer
	int width = glConfig.vidWidth;
	int height = glConfig.vidHeight;
	static unsigned char* pixels = NULL;
	static size_t pixelsSize = 0;
	size_t needed = (size_t)width * height * 4;
	if (!pixels || pixelsSize < needed) {
		if (pixels) delete[] pixels;
		pixels = new unsigned char[needed];
		pixelsSize = needed;
	}
	
	g_cuRenderer->CopyToBackbuffer(pixels, width, height);

    // upload CUDA result to GL texture
	qglClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	qglDisable(GL_DEPTH_TEST);
	qglDisable(GL_BLEND);
	
	qglMatrixMode(GL_PROJECTION);
	qglLoadIdentity();
	qglOrtho(0, 1, 0, 1, -1, 1);
	
	qglMatrixMode(GL_MODELVIEW);
	qglLoadIdentity();
	
	qglRasterPos2i(0, 0);
	qglDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
	
	qglEnable(GL_DEPTH_TEST);
}

#endif // HAVE_CUDA