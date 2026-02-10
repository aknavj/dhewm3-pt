#ifndef __CUDA_PATH_TRACER_H__
#define __CUDA_PATH_TRACER_H__

#include "sys/platform.h"

#ifdef HAVE_CUDA

// CUDA error checking macro
#define CUDA_CHECK(call) \
	do { \
		cudaError_t err = call; \
		if (err != cudaSuccess) { \
			common->Printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
			return false; \
		} \
	} while(0)

#define CUDA_CHECK_VOID(call) \
	do { \
		cudaError_t err = call; \
		if (err != cudaSuccess) { \
			common->Printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		} \
	} while(0)

/*
=================================================================================

CUDA Path Tracer

=================================================================================
*/

class idCudaRenderer {

public:
    idCudaRenderer();
    ~idCudaRenderer();

    bool			Init(int w, int h);
    void			Shutdown();

    static bool		IsAvailable();
    void			PrintDeviceInfo();

    void            BeginFrame();
    void            EndFrame();
    void			RenderView(const renderView_t* renderView);
    void			CopyToBackbuffer(unsigned char* dest, int destWidth, int destHeight);

private:
    void			Alloc();
    void			Free();

    float*			d_framebuffer;      // HDR buffer
	unsigned char*	d_outputBuffer;     // LDR buffer

	unsigned char*	h_outputPixels;     // host-side staging buffer
	size_t			h_outputPixelsSize; // size of host staging buffer

    int             width;              
    int             height;
    int				renderWidth;
	int				renderHeight;

	bool			cudaContextHealthy;
};

// global cuda path tracer instance
extern idCudaRenderer* g_cuRenderer;


/*
=================================================================================

CUDA Kernel Functions

=================================================================================
*/

#ifdef __cplusplus
extern "C" {
#endif

/*
 */
void CUDA_LaunchRenderView(
    float* framebuffer,
	unsigned char* outputBuffer,
	int width,
	int height
);

#ifdef __cplusplus
}
#endif

#endif // HAVE_CUDA

#endif // __CUDA_PATH_TRACER_H__