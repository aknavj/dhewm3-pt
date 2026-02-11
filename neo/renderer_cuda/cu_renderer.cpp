#ifdef HAVE_CUDA

#include "renderer/tr_local.h"
#include "renderer_cuda/cu_renderer.h"
#include <cuda_runtime.h>

// console variables
idCVar r_cuDraw("r_cuDraw", "1", CVAR_RENDERER | CVAR_ARCHIVE, "Use CUDA renderer");
idCVar r_cuDebug("r_cuDebug", "0", CVAR_RENDERER, "Show CUDA renderer debug info");
idCVar r_cuRenderMode("r_cuRenderMode", "0", CVAR_RENDERER, "CUDA renderer mode");

idCudaRenderer *g_cuRenderer = NULL;

/*
========================
idCudaRenderer::idCudaRenderer
========================
*/
idCudaRenderer::idCudaRenderer() {

	d_vertices = NULL;
	d_triangles = NULL;
	d_triIndices = NULL;
	d_textures = NULL;
	d_materials = NULL;
	d_lights = NULL;
	d_bvhNodes = NULL;

	d_framebuffer = NULL;
	d_outputBuffer = NULL;
	h_outputPixels = NULL;
	h_outputPixelsSize = 0;
	cu_context_ok = 1;

	cam_pos[0] = cam_pos[1] = cam_pos[2] = 0.0f;
	cam_forward[0] = 1.0f; cam_forward[1] = cam_forward[2] = 0.0f;
	cam_right[0] = cam_right[2] = 0.0f; cam_right[1] = 1.0f;
	cam_up[0] = cam_up[1] = 0.0f; cam_up[2] = 1.0f;
	fov_x = 90.0f;
	fov_y = 90.0f;

	prev_cam_pos[0] = prev_cam_pos[1] = prev_cam_pos[2] = 0.0f;
	prev_cam_forward[0] = 1.0f; prev_cam_forward[1] = prev_cam_forward[2] = 0.0f;
	accum_frame = 0;

	num_triangles = 0;
	num_vertices = 0;
	num_bvh_nodes = 0;
	need_reload = 0;

	width = 0;
	height = 0;
	renderWidth = 0;
	renderHeight = 0;

	timer_start = NULL;
	timer_stop = NULL;
	kernel_ms = 0.0f;
	kernel_fps = 0.0f;
}

/*
========================
idCudaRenderer::~idCudaRenderer
========================
*/
idCudaRenderer::~idCudaRenderer() {

	Shutdown();

}

/*
========================
idCudaRenderer::IsAvailable
========================
*/
bool idCudaRenderer::IsAvailable() {

	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	
	if (err != cudaSuccess || deviceCount == 0) {
		return 0;
	}
	
	// check if device supports compute capability 3.0 or higher
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	
	if (prop.major < 3) {
		return 0;
	}
	
	return 1;
}

/*
========================
idCudaRenderer::PrintDeviceInfo
========================
*/
void idCudaRenderer::PrintDeviceInfo() {

	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	
	common->Printf("CUDA Device Information:\n");
	common->Printf("  Found %d CUDA device(s)\n", deviceCount);
	
	for (int i = 0; i < deviceCount; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		
		common->Printf("\n  Device %d: %s\n", i, prop.name);
		common->Printf("	Compute Capability: %d.%d\n", prop.major, prop.minor);
		common->Printf("	Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
		common->Printf("	Multiprocessors: %d\n", prop.multiProcessorCount);
		common->Printf("	Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
		common->Printf("	Warp Size: %d\n", prop.warpSize);
	}

	return;
}

/*
========================
idCudaRenderer::Init
========================
*/
bool idCudaRenderer::Init(int w, int h) {

	if (!IsAvailable()) {
		common->Warning("idCudaPathTracer::Init(): No compatible CUDA device found\n");
		return 0;
	}

	common->Printf("\n----- Init CUDA Path Tracer -----\n");

	width = w;
	height = h;

	CUDA_CHECK(cudaSetDevice(0));
	
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	
	common->Printf("CUDA Device: %s\n", prop.name);
	common->Printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);

	Alloc();

	cudaEventCreate(&timer_start);
	cudaEventCreate(&timer_stop);

	common->Printf("\n\n");

	return 1;
}

/*
========================
idCudaRenderer::Shutdown
========================
*/
void idCudaRenderer::Shutdown() {

	Free();

	cudaDeviceReset();
}

/*
========================
idCudaRenderer::Alloc
========================
*/
void idCudaRenderer::Alloc() {

	// alloc framebuffer HDR float4 format for accumulation
	size_t framebufferSize = width * height * 4 * sizeof(float);
	CUDA_CHECK_VOID(cudaMalloc(&d_framebuffer, framebufferSize));
	CUDA_CHECK_VOID(cudaMemset(d_framebuffer, 0, framebufferSize));
	
	// alloc output LDR RGBA8 buffer
	size_t outputSize = width * height * 4 * sizeof(unsigned char);
	CUDA_CHECK_VOID(cudaMalloc(&d_outputBuffer, outputSize));

	// alloc geometry data
	CUDA_CHECK_VOID(cudaMalloc(&d_vertices, MAX_TRIANGLES * 3 * sizeof(cudaVertex_t)));
	CUDA_CHECK_VOID(cudaMalloc(&d_triangles, MAX_TRIANGLES * sizeof(cudaTriangle_t)));
	CUDA_CHECK_VOID(cudaMalloc(&d_triIndices, MAX_TRIANGLES * sizeof(int)));
	CUDA_CHECK_VOID(cudaMalloc(&d_materials, MAX_MATERIALS * sizeof(cudaMaterial_t)));
	CUDA_CHECK_VOID(cudaMalloc(&d_textures, MAX_TEXTURES * sizeof(cudaTexture_t)));
	CUDA_CHECK_VOID(cudaMalloc(&d_lights, MAX_LIGHTS * sizeof(cudaLight_t)));
	CUDA_CHECK_VOID(cudaMalloc(&d_bvhNodes, MAX_BVH_NODES * sizeof(cudaBVHNode_t)));
	
	return;
}

/*
========================
idCudaRenderer::Free
========================
*/
void idCudaRenderer::Free() {

	if (d_framebuffer) {
		cudaFree(d_framebuffer);
		d_framebuffer = NULL;
	}

	if (d_outputBuffer) {
		cudaFree(d_outputBuffer);
		d_outputBuffer = NULL;
	}

	if (h_outputPixels) {
		delete[] h_outputPixels;
		h_outputPixels = NULL;
		h_outputPixelsSize = 0;
	}

	if (d_vertices) {
		cudaFree(d_vertices);
		d_vertices = NULL;
	}

	if (d_triangles) {
		cudaFree(d_triangles);
		d_triangles = NULL;
	}

	if (d_triIndices) {
		cudaFree(d_triIndices);
		d_triIndices = NULL;
	}

	if (d_bvhNodes) {
		cudaFree(d_bvhNodes);
		d_bvhNodes = NULL;
	}

	if (d_materials) {
		cudaFree(d_materials);
		d_materials = NULL;
	}

	if (d_textures) {
		// free individual texture data
		for (int i = 0; i < h_textures.Num(); i++) {
			if (h_textures[i].data) {
				cudaFree(h_textures[i].data);
			}
		}
		cudaFree(d_textures);
		d_textures = NULL;
		h_textures.Clear();
		h_texnums.Clear();
		textureHash.Free();
	}

	if (d_lights) {
		cudaFree(d_lights);
		d_lights = NULL;
	}

	if (timer_start) {
		cudaEventDestroy(timer_start);
		timer_start = NULL;
	}
	if (timer_stop) {
		cudaEventDestroy(timer_stop);
		timer_stop = NULL;
	}

	return;
}

/*
========================
idCudaRenderer::UpdateCamera
========================
*/
void idCudaRenderer::UpdateCamera(const renderView_t* renderView) {

	// extract camera parameters from renderView
	cam_pos[0] = renderView->vieworg.x;
	cam_pos[1] = renderView->vieworg.y;
	cam_pos[2] = renderView->vieworg.z;
	
	// compute camera basis vectors from view matrix
	idMat3 axis = renderView->viewaxis;
	
	cam_forward[0] = axis[0][0];
	cam_forward[1] = axis[0][1];
	cam_forward[2] = axis[0][2];
	
	cam_right[0] = -axis[1][0];
	cam_right[1] = -axis[1][1];
	cam_right[2] = -axis[1][2];
	
	cam_up[0] = axis[2][0];
	cam_up[1] = axis[2][1];
	cam_up[2] = axis[2][2];
	
	fov_x = renderView->fov_x;
	fov_y = renderView->fov_y;
}

/*
========================
idCudaRenderer::RenderView
========================
*/
void idCudaRenderer::RenderView(const renderView_t* renderView) {

	if (!d_framebuffer) {
		return;
	}

	// clear framebuffer
	if (h_triangles.Num() == 0 || h_vertices.Num() == 0) {
		CUDA_CHECK_VOID(cudaMemset(d_framebuffer, 0, width * height * 4 * sizeof(float)));
		CUDA_CHECK_VOID(cudaMemset(d_outputBuffer, 0, width * height * 4 * sizeof(unsigned char)));
		return;
	}

	// update camera
	UpdateCamera(renderView);

	// set render resolution
	renderWidth = width;
	renderHeight = height;

	// upload geometry to GPU if it changed this frame
	if (need_reload && num_triangles > 0 && num_vertices > 0) {
		CUDA_CHECK_VOID(cudaMemcpy(d_vertices, h_vertices.Ptr(),
			num_vertices * sizeof(cudaVertex_t), cudaMemcpyHostToDevice));
		CUDA_CHECK_VOID(cudaMemcpy(d_triangles, h_triangles.Ptr(),
			num_triangles * sizeof(cudaTriangle_t), cudaMemcpyHostToDevice));

		if (num_bvh_nodes > 0) {
			CUDA_CHECK_VOID(cudaMemcpy(d_bvhNodes, h_bvhNodes.Ptr(),
				num_bvh_nodes * sizeof(cudaBVHNode_t), cudaMemcpyHostToDevice));
			CUDA_CHECK_VOID(cudaMemcpy(d_triIndices, h_bvhTriIndices.Ptr(),
				num_triangles * sizeof(int), cudaMemcpyHostToDevice));
		}

		need_reload = 0;
	}

	// detect camera or scene changes to reset progressive accumulation
	bool cameraChanged = false;
	for (int i = 0; i < 3; i++) {
		if (fabsf(cam_pos[i] - prev_cam_pos[i]) > 1e-4f ||
			fabsf(cam_forward[i] - prev_cam_forward[i]) > 1e-4f) {
			cameraChanged = true;
			break;
		}
	}

	// scene is rebuilt every frame (BeginFrame clears geometry), so need_reload indicates a scene change
	if (cameraChanged || need_reload) {
		accum_frame = 0;
		CUDA_CHECK_VOID(cudaMemset(d_framebuffer, 0, width * height * 4 * sizeof(float)));
	}
	CUDA_CHECK_VOID(cudaMemset(d_outputBuffer, 0, width * height * 4 * sizeof(unsigned char)));

	// save current camera for next frame comparison
	for (int i = 0; i < 3; i++) {
		prev_cam_pos[i] = cam_pos[i];
		prev_cam_forward[i] = cam_forward[i];
	}

	// render the view using CUDA kernel
	cudaEventRecord(timer_start);

	CUDA_LaunchRenderView(
		d_vertices,
		d_triangles,
		d_triIndices,
		d_materials,
		d_textures,
		d_lights,
		h_lights.Num(),
		d_bvhNodes,
		num_bvh_nodes,
		d_framebuffer,
		d_outputBuffer,
		renderWidth,
		renderHeight,
		num_triangles,
		cam_pos,
		cam_forward,
		cam_right,
		cam_up,
		fov_x,
		fov_y,
		r_cuRenderMode.GetInteger(),
		accum_frame
	);

	accum_frame++;

	cudaEventRecord(timer_stop);

	// synchronize to ensure kernel is complete before readback
	cudaDeviceSynchronize();

	// compute kernel elapsed time
	cudaEventElapsedTime(&kernel_ms, timer_start, timer_stop);
	kernel_fps = (kernel_ms > 0.0f) ? (1000.0f / kernel_ms) : 0.0f;

	// check for kernel error
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		common->Warning("CUDA kernel error: %s\n", cudaGetErrorString(err));
		cu_context_ok = 0;
	}

	return;
}

/*
========================
idCudaRenderer::CopyToBackbuffer
========================
*/
void idCudaRenderer::CopyToBackbuffer(unsigned char* dest, int destWidth, int destHeight) {
	if (!d_outputBuffer || !dest) {
		return;
	}

	// ensure persistent host buffer is large enough
	size_t needed = (size_t)renderWidth * renderHeight * 4;
	if (!h_outputPixels || h_outputPixelsSize < needed) {
		if (h_outputPixels) {
			delete[] h_outputPixels;
		}
		h_outputPixels = new unsigned char[needed];
		h_outputPixelsSize = needed;
	}
	
	// copy from GPU 
	CUDA_CHECK_VOID(cudaMemcpy(h_outputPixels, d_outputBuffer, 
		needed * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	
	// simple upscaling if render resolution differs from display resolution
	if (renderWidth != destWidth || renderHeight != destHeight) {
		for (int y = 0; y < destHeight; y++) {
			for (int x = 0; x < destWidth; x++) {
				int srcX = (x * renderWidth) / destWidth;
				int srcY = (y * renderHeight) / destHeight;
				int srcIdx = (srcY * renderWidth + srcX) * 4;
				int dstIdx = (y * destWidth + x) * 4;
				dest[dstIdx + 0] = h_outputPixels[srcIdx + 0];
				dest[dstIdx + 1] = h_outputPixels[srcIdx + 1];
				dest[dstIdx + 2] = h_outputPixels[srcIdx + 2];
				dest[dstIdx + 3] = h_outputPixels[srcIdx + 3];
			}
		}
	} else {
		// direct copy
		memcpy(dest, h_outputPixels, needed);
	}

	return;
}

#endif // HAVE_CUDA