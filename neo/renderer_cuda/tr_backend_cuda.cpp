#include "sys/platform.h"

#ifdef HAVE_CUDA

#include "framework/Console.h"
#include "renderer/tr_local.h"
#include "renderer_cuda/cu_renderer.h"

// extern console variables
extern idCVar r_cuDraw;

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

        int materialIndex = 0;
        if (surf->material) {
            materialIndex = g_cuRenderer->AddMaterial(surf->material);
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