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
#define MAX_LIGHTS 128
#define MAX_BVH_NODES (MAX_TRIANGLES * 2)

/*
 */
struct cudaVertex_t {
	float position[3];
	float normal[3];
	float tangent[3];
	float bitangent[3];
	float texcoord[2];
};

/*
 */
struct cudaTriangle_t {
	int vertexIndices[3];
	float bounds[6];
	int materialIndex;
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
	float specular[3];
	float emission[3];
	float blendColor[4];
    int albedoTexture;
	int normalTexture;
	int specularTexture;
	int emissionTexture;
	int blendTexture;
	int alphaMaskTexture;
	float alphaTest;
	int coverage; // 0 = opaque, 1 = perforated (alpha tested), 2 = translucent
	int blendMode; // 0 = opaque, 1 = additive (ONE,ONE), 2 = alpha blend (SRC_ALPHA,ONE_MINUS_SRC_ALPHA), 3 = filter/modulate (DST_COLOR,ZERO)
};

/*
 */
struct cudaLight_t {
	int type;
    float position[3];
	float direction[3]; 
    float color[3];
    float intensity;
    float radius;
	float area[3];
	int textureIndex;
	float coneAngle;
	float coneFalloff;

	// for point and projected lights
	float lightRadius3[3];

	// projected light orientation
	float right[3];
	float up[3];

	// projected light frustum planes (4 planes × 4 components)
	float lightProject[4][4];
	int projectedTextureIndex;
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

/*
========================
idCudaRenderer
========================
*/
class idCudaRenderer {

public:
	idCudaRenderer();
	~idCudaRenderer();

	bool			Init(int w, int h);
	void			Shutdown();

	static bool		IsAvailable();
	void			PrintDeviceInfo();

    // texture management
    int             AddMaterial(const idMaterial* material, const float* shaderRegisters = NULL);
    void            SetMaterial(int index, const idMaterial* material, const float* shaderRegisters = NULL);
    int             AddTexture(const idImage* image);

    // frame management
	void			BeginFrame();
	void			EndFrame();
	void            FramePVStoBVH();

	// scene data submission
	void			AddTriangle(const idDrawVert* verts, int numVerts, const int* indices,
                                int numIndices, int materialIndex, const float* modelMatrix = NULL);
    void            AddLight(const idVec3& position, const idVec3& color, float intensity, 
							int type, float lightRadius = 300.0f, const float* lightProject = NULL, 
							int projTexIndex = -1, const float* lightRadiusXYZ = NULL);
    void			AddSpotLight(const idVec3& position, const idVec3& direction, const idVec3& color, 
								float intensity, float coneAngle, float coneFalloff, 
								float lightRadius = 500.0f, const float* lightProject = NULL, int projTexIndex = -1, 
								const float* rightAxis = NULL, const float* upAxis = NULL);

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
    cudaLight_t*	d_lights;
	float*			d_framebuffer;
	unsigned char*	d_outputBuffer;

    // host memory
	idList<cudaVertex_t>	h_vertices;
	idList<cudaTriangle_t>	h_triangles;
    idList<cudaMaterial_t>	h_materials;
    idList<cudaLight_t>		h_lights;
	idList<cudaBVHNode_t>	h_bvhNodes;
	idList<int>				h_bvhTriIndices;
	unsigned char*          h_outputPixels;
	size_t                  h_outputPixelsSize;

    // texture data    
    idHashIndex             textureHash;    // texnum -> h_textures index
    idList<cudaTexture_t>   h_textures;     // host-side texture data
	idList<const idImage*>	h_texturePtrs;  // host-side texture pointers for hash lookup
    idList<int>             h_texnums;      // GL texnum per h_textures entry
    cudaTexture_t*          d_textures;     // GPU texture array

	// light data
	idList<idVec3>			materialEmission;
	idHashIndex				lightHash;			// light ptr hash -> h_lights index
	idList<intptr_t>		h_lightPtrs;		// host-side light pointer keys
	int						nextLightIndex;		// next available light slot

    // material data
    idHashIndex             materialHash;       // material ptr hash -> h_materials index
    idList<const idMaterial*> h_materialPtrs;   // host-side material pointers
    int                     nextMaterialIndex;  // next available material slot

    // overflow flag: set when texture limit is hit, flush happens next BeginFrame
    bool                    needTextureFlush;

    // camera parameters
	float			cam_pos[3];
	float			cam_forward[3];
	float			cam_right[3];
	float			cam_up[3];
	float			fov_x;
	float			fov_y;

	// previous camera state for accumulation invalidation
	float			prev_cam_pos[3];
	float			prev_cam_forward[3];
	unsigned int	accum_frame;

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
    const cudaLight_t* lights,
    int numLights,
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
	int renderMode,
	unsigned int frameNumber
);

#ifdef __cplusplus
}
#endif

#endif // HAVE_CUDA

#endif // __CUDA_PATH_TRACER_H__