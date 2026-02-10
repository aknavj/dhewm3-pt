#ifndef __CUDA_PATH_TRACER_H__
#define __CUDA_PATH_TRACER_H__

#include "sys/platform.h"

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef __CUDACC__
	class idVec3;
	struct idDrawVert;
	class idMaterial;
	struct renderView_s;
	typedef struct renderView_s renderView_t;
#endif

#ifndef __CUDACC__

#define CUDA_CHECK(call) \
	do { \
		cudaError_t err = call; \
		if (err != cudaSuccess) { \
			common->Printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
			return 0; \
		} \
	} while(0)

#define CUDA_CHECK_VOID(call) \
	do { \
		cudaError_t err = call; \
		if (err != cudaSuccess) { \
			common->Printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		} \
	} while(0)

#endif // !__CUDACC__

#define TILE_SIZE 16
#define MAX_TRIANGLES 1000000
#define MAX_TEXTURES 1024
#define MAX_MATERIALS 1024
#define MAX_BVH_NODES (MAX_TRIANGLES * 2)

/*
 */
struct cudaVertex_t {
	float position[3];
	float normal[3];
	float texcoord[2];
};

/*
 */
struct cudaTriangle_t {
	int vertexIndices[3];
	int materialIndex;
	float bounds[6];
};

/*
 */
struct cudaTexture_t {
	int width;
	int height;
	unsigned char* data;  // RGBA8 data on GPU
};

/*
 */
struct cudaMaterial_t {
    float albedo[3];
    int albedoTexture;
};

/*
 */
struct cudaBVHNode_t {
	float bounds[6];
	int l_child;
	int r_child;
	int first_primitive;
	int primitive_count;
};

/*
 */
struct cudaRay_t {
	float origin[3];
	float direction[3];
};

/*
 */
struct cudaHitInfo_t {
	float position[3];
	float normal[3];
};

#ifndef __CUDACC__

#include "../idlib/containers/List.h"
#include "../idlib/containers/HashIndex.h"

class idCudaRenderer {

public:
	idCudaRenderer();
	~idCudaRenderer();

	bool			Init(int w, int h);
	void			Shutdown();

	static bool		IsAvailable();
	void			PrintDeviceInfo();

    // texture management
    int             AddMaterial(const idMaterial* material);
    void            SetMaterial(int index, const idMaterial* material);
    int             AddTexture(const idImage* image);

    // frame management
	void			BeginFrame();
	void			EndFrame();
	void			AddTriangle(const idDrawVert* verts, int numVerts, const int* indices,
                                int numIndices, int materialIndex, const float* modelMatrix = NULL);
    void            FramePVStoBVH();

    // rendering
	void			RenderView(const renderView_t* renderView);
	void			CopyToBackbuffer(unsigned char* dest, int destWidth, int destHeight);

private:
	void			Alloc();
	void			Free();
	void			BuildBVH();

	void			UpdateCamera(const renderView_t* renderView);

    // CUDA memory
	cudaVertex_t*	d_vertices;
	cudaTriangle_t*	d_triangles;
    int*			d_triIndices;
	cudaBVHNode_t*	d_bvhNodes;
    cudaMaterial_t*	d_materials;
	float*			d_framebuffer;
	unsigned char*	d_outputBuffer;

    // host memory
	idList<cudaVertex_t>	h_vertices;
	idList<cudaTriangle_t>	h_triangles;
    idList<cudaMaterial_t>	h_materials;
	idList<cudaBVHNode_t>	h_bvhNodes;
	idList<int>				h_bvhTriIndices;
	unsigned char*	h_outputPixels;
	size_t			h_outputPixelsSize;

    // texture data    
    idHashIndex             textureHash;    // texnum -> h_textures index
    idList<cudaTexture_t>   h_textures;     // host-side texture data
    idList<int>             h_texnums;      // GL texnum per h_textures entry
    cudaTexture_t*          d_textures;     // GPU texture array

    // material data


    // camera parameters
	float			cam_pos[3];
	float			cam_forward[3];
	float			cam_right[3];
	float			cam_up[3];
	float			fov_x;
	float			fov_y;

    // geometry counts
	int				num_triangles;
	int				num_vertices;
	int				num_bvh_nodes;
	int				need_reload;

    // render dimensions
	int			    width;
	int			    height;
	int				renderWidth;
	int				renderHeight;

	bool			cu_context_ok;

    // kernel timing
	cudaEvent_t		timer_start;
	cudaEvent_t		timer_stop;
	float			kernel_ms;
	float			kernel_fps;
};

// global cuda renderer instance
extern idCudaRenderer* g_cuRenderer;

#endif // !__CUDACC__

#ifdef __cplusplus
extern "C" {
#endif

void CUDA_LaunchRenderView(
	const cudaVertex_t* vertices,
	const cudaTriangle_t* triangles,
	const int *triIndices,
    const cudaMaterial_t* materials,
    const cudaTexture_t* textures,
	const cudaBVHNode_t* bvhNodes,
	int numBVHNodes,
	float* framebuffer,
	unsigned char* outputBuffer,
	int width,
	int height,
	int numTriangles,
	const float* cam_pos,
	const float* cam_forward,
	const float* cam_right,
	const float* cam_up,
	float fov_x,
	float fov_y,
	int renderMode
);

#ifdef __cplusplus
}
#endif

#endif // HAVE_CUDA

#endif // __CUDA_PATH_TRACER_H__