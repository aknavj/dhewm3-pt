#ifdef HAVE_CUDA

#include "renderer/tr_local.h"
#include "renderer_cuda/cu_renderer.h"
#include <cuda_runtime.h>

/*
========================

CUDA Renderer console vars

========================
*/
idCVar r_cuDraw("r_cuDraw", "1", CVAR_RENDERER | CVAR_ARCHIVE, "Use CUDA path tracer");

/*
========================

Global variables

========================
*/

idCudaRenderer *g_cuRenderer = NULL;


/*
========================
idCudaRenderer::idCudaRenderer
========================
*/
idCudaRenderer::idCudaRenderer() {

	d_framebuffer = NULL;
	d_outputBuffer = NULL;
	h_outputPixels = NULL;
	h_outputPixelsSize = 0;
	cudaContextHealthy = 1;

    width = 0;
    height = 0;
    renderWidth = 0;
    renderHeight = 0;
}

/*
========================
idCudaRenderer::idCudaRenderer
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
		return false;
	}
	
	// check if device supports compute capability 3.0 or higher
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	
	if (prop.major < 3) {
		return false;
	}
	
	return true;
}

/*
========================
idCudaRenderer::IsAvailable
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
		common->Printf("    Compute Capability: %d.%d\n", prop.major, prop.minor);
		common->Printf("    Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
		common->Printf("    Multiprocessors: %d\n", prop.multiProcessorCount);
		common->Printf("    Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
		common->Printf("    Warp Size: %d\n", prop.warpSize);
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
		return false;
	}

	common->Printf("\n----- Init CUDA Path Tracer -----\n");

	width = w;
	height = h;

    // initialize CUDA
	CUDA_CHECK(cudaSetDevice(0));
	
	// print device info
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	
	common->Printf("CUDA Device: %s\n", prop.name);
	common->Printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);

    // allocate needed buffers
	Alloc();

    common->Printf("\n\n");

    return true;
}

/*
========================
idCudaRenderer::Shutdown
========================
*/
void idCudaRenderer::Shutdown() {

    // free used buffers
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

    return;
}


/*
========================
idCudaRenderer::BeginFrame
========================
*/
void idCudaRenderer::BeginFrame() {

}

/*
========================
idCudaRenderer::EndFrame
========================
*/
void idCudaRenderer::EndFrame() {

}

/*
========================
idCudaRenderer::RenderView
========================
*/
void idCudaRenderer::RenderView(const renderView_t* renderView) {

    // clear framebuffer
	CUDA_CHECK_VOID(cudaMemset(d_framebuffer, 0, width * height * 4 * sizeof(float)));
	CUDA_CHECK_VOID(cudaMemset(d_outputBuffer, 0, width * height * 4 * sizeof(unsigned char)));
	
    // set render resolution
    renderWidth = width;
    renderHeight = height;

    // render the view using CUDA kernel
    CUDA_LaunchRenderView(
        d_framebuffer,
        d_outputBuffer,
        renderWidth,
        renderHeight
    );

    // synchronize to ensure kernel is complete before readback
    cudaDeviceSynchronize();

    // check for kernel error
    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		common->Warning("CUDA kernel error: %s\n", cudaGetErrorString(err));
		cudaContextHealthy = 0;
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

	// ensure persistent host buffer is large enough (realloc only when needed)
	size_t needed = (size_t)renderWidth * renderHeight * 4;
	if (!h_outputPixels || h_outputPixelsSize < needed) {
		if (h_outputPixels) {
			delete[] h_outputPixels;
		}
		h_outputPixels = new unsigned char[needed];
		h_outputPixelsSize = needed;
	}
	
	// copy from GPU (at render resolution)
	CUDA_CHECK_VOID(cudaMemcpy(h_outputPixels, d_outputBuffer, 
		needed * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	
	// simple upscaling if render resolution differs from display resolution
	if (renderWidth != destWidth || renderHeight != destHeight) {
		// nearest neighbor upscaling
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