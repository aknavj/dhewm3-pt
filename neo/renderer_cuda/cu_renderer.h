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
#define MAX_TEXTURES 2048
#define MAX_MATERIALS 1024
#define MAX_LIGHTS 512
#define MAX_BVH_NODES (MAX_TRIANGLES * 2)

// volumetric rendering
#define VOLUMETRIC_STEPS 4
#define VOLUMETRIC_DENSITY 0.01f

/*
 */
struct cudaVertex_t {
	float position[3];
	float normal[3];
	float texcoord[2];
	float tangent[3];
	float binormal[3];
	float color[4];            // Vertex color RGBA
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
	float albedo[4];           // base color + alpha
	float emission[3];         // emissive color (for glowing materials)
	float metallic;            // metallic factor
	float roughness;           // roughness factor
	float ior;                 // index of refraction
	float transmission;        // transmission factor (glass, etc)
	float alphaTest;           // alpha test threshold (0 = no test, >0 = test value)
	int blendMode;             // 0=opaque, 1=alpha_blend, 2=additive, 3=multiply
	int useVertexColor;        // use vertex colors for tinting
	int isAmbientOnly;         // 1=ambient-only material (no interaction stages), render as unlit overlay
	int noShadows;             // 1=material does not cast shadows (noshadows keyword)
	int albedoTexture;         // diffuse/base color texture index (-1 if none)
	int normalTexture;         // normal/bump map texture index (-1 if none)
	int specularTexture;       // specular map texture index (-1 if none)
	float bumpScale;           // normal map intensity
	float texTransform[6];     // 2x3 matrix: scale.x, scale.y, rotate, translate.x, translate.y, scroll_speed
	float polygonOffset;       // depth bias for decals (negative = toward camera)
};

/*
 */
struct cudaLight_t {
	int type;                  // 0=point, 1=directional, 2=area
	float position[3];
	float direction[3];        // normalized forward direction (for spotlights and directional)
	float color[3];
	float intensity;
	float radius;              // Doom 3 light radius (falloff boundary)
	float area[3];             // area light dimensions (for emissive triangle lights)
	int textureIndex;          // light texture (-1 if none)
	float volumetric;          // volumetric scattering intensity (0=off)
	float coneAngle;           // spotlight half-angle in radians (0=omnidirectional)
	float coneFalloff;         // spotlight edge softness exponent (higher=sharper edge)

	// Doom 3 light projection planes (global space)
	float lightProject[4][4];  // 4 planes, each (normal.xyz, distance)
	int projectedTextureIndex;

	// light extents for area light sampling (soft shadows)
	float lightRadius3[3];     // xyz extents for area sampling

	// axis vectors for projected light orientation
	float right[3];            // light's right axis (normalized)
	float up[3];               // light's up axis (normalized)
};

/*
 */
struct cudaBVHNode_t {
	float bounds[6];  // min/max xyz
	int leftChild;    // -1 if leaf
	int rightChild;   // -1 if leaf
	int firstPrimitive;  // for leaf nodes
	int primitiveCount;  // for leaf nodes
};

/*
 */
struct cudaRay_t {
	float origin[3];
	float direction[3];
	float tMin;
	float tMax;
};

/*
 */
struct cudaHitInfo_t {
	bool hit;
	float t;
	float position[3];
	float normal[3];
	float tangent[3];
	float binormal[3];
	float texcoord[2];
	float barycentrics[3];  // barycentric coordinates (w, u, v) for vertex interpolation
	int materialIndex;
	int triangleIndex;
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
    int             GetOrSetMaterial(const idMaterial* material, const float* shaderRegisters = NULL);
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

	// accumulation
	void			ResetAccumulation();
	int				GetAccumulatedFrames() const { return accum_frame; }

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

	// LBVH GPU temporaries
	unsigned int*	d_mortonCodes;
	int*			d_sortedIndices;
	int*			d_parents;           // parent pointer per node (2N-1)
	int*			d_atomicCounters;    // one per internal node (N-1)
	int				allocatedLBVHSize;   // number of triangles the temp buffers can handle
	int				allocatedTriangles;  // allocated GPU triangle buffer size
	int				allocatedBVHNodes;   // allocated GPU BVH buffer size
	int				allocatedTriIndices; // allocated GPU triIndices buffer size
    // host memory
	idList<cudaVertex_t>	h_vertices;
	idList<cudaTriangle_t>	h_triangles;
    idList<cudaMaterial_t>	h_materials;
    idList<cudaLight_t>		h_lights;
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
	int				prevRenderWidth;
	int				prevRenderHeight;

	bool			cu_context_ok;

	// CUDA stream
	cudaStream_t	stream;

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

// primary ray generation and path tracing kernel
void LaunchPathTracingKernel(
	const cudaVertex_t* vertices,
	const cudaTriangle_t* triangles,
	const cudaBVHNode_t* bvhNodes,
	const int* triIndices,
	const cudaMaterial_t* materials,
	const cudaTexture_t* textures,
	const cudaLight_t* lights,
	int numLights,
	int renderMode,
	float* framebuffer,
	int width,
	int height,
	const float* cameraPos,
	const float* cameraForward,
	const float* cameraRight,
	const float* cameraUp,
	float fov,
	float fovY,
	int samplesPerPixel,
	int maxDepth,
	int maxLightSamples,
	int frameIndex,
	float emissionBoost,
	float indirectProb,
	int rrEnabled,
	int rrMinBounces,
	float rrSurvivalMin,
	float earlyTermThreshold,
	float fireflyClamp,
	float throughputClamp,
	float rayOffset,
	float specularBoost,
	float skyIntensity,
	const float* skyColorZenith,
	const float* skyColorHorizon,
	const float* skyColorGround,
	float volumetricDensity,
	int volumetricSteps,
	float volumetricAnisotropy,
	float volFalloff,
	float volMaxDist,
	float softShadowScale,
	cudaStream_t stream
);

// tone mapping and output conversion kernel
void LaunchToneMappingKernel(
	const float* hdrBuffer,
	unsigned char* ldrBuffer,
	int width,
	int height,
	float exposure,
	float gamma,
	int toneMapMode,
	int frameCount,
	cudaStream_t stream
);

// GPU LBVH construction
void LaunchBuildLBVH(
	const cudaTriangle_t* d_triangles,
	int numTriangles,
	cudaBVHNode_t* d_bvhNodes,
	int* d_triIndices,
	unsigned int* d_mortonCodes,
	int* d_sortedIndices,
	int* d_parents,
	int* d_atomicCounters,
	const float* sceneBounds,
	cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // HAVE_CUDA

#endif // __CUDA_PATH_TRACER_H__