#ifdef HAVE_CUDA

#include "renderer/tr_local.h"
#include "renderer_cuda/cu_renderer.h"
#include <cuda_runtime.h>

// console variables
idCVar r_cuDraw("r_cuDraw", "1", CVAR_RENDERER | CVAR_ARCHIVE, "Use CUDA renderer");
idCVar r_cuMode("r_cuMode", "0", CVAR_RENDERER | CVAR_ARCHIVE, "CUDA render mode: 0=path tracing, 1=BVH debug, 2=textured combined, 3=base color, 4=albedo texture, 5=normal texture, 6=specular texture");
idCVar r_cuDebug("r_cuDebug", "0", CVAR_RENDERER, "Show CUDA renderer debug info");

// Resolution & Sampling
idCVar r_cuRenderScale("r_cuRenderScale", "0.5", CVAR_RENDERER | CVAR_ARCHIVE, "Render scale (0.25-1.0, lower = faster)");
idCVar r_cuSamplesPerPixel("r_cuSamplesPerPixel", "4", CVAR_RENDERER | CVAR_ARCHIVE, "Samples per pixel per frame (1-64)");

// Ray Tracing Depth
idCVar r_cuMaxDepth("r_cuMaxDepth", "3", CVAR_RENDERER | CVAR_ARCHIVE, "Maximum ray bounces (1-16)");
idCVar r_cuMaxLightSamples("r_cuMaxLightSamples", "2", CVAR_RENDERER | CVAR_ARCHIVE, "Direct light samples per hit (1-8)");
idCVar r_cuIndirectProb("r_cuIndirectProb", "0.15", CVAR_RENDERER | CVAR_ARCHIVE, "Indirect lighting probability (0.0-1.0)");

// Russian Roulette Path Termination
idCVar r_cuRussianRoulette("r_cuRussianRoulette", "1", CVAR_RENDERER | CVAR_ARCHIVE, "Enable Russian roulette path termination");
idCVar r_cuRRMinBounces("r_cuRRMinBounces", "2", CVAR_RENDERER | CVAR_ARCHIVE, "Min bounces before Russian roulette (1-8)");
idCVar r_cuRRSurvivalMin("r_cuRRSurvivalMin", "0.1", CVAR_RENDERER | CVAR_ARCHIVE, "Minimum survival probability (0.05-0.5)");
idCVar r_cuEarlyTermThreshold("r_cuEarlyTermThreshold", "0.05", CVAR_RENDERER | CVAR_ARCHIVE, "Throughput threshold for early termination (0.0-0.5)");

// Firefly & Clamping
idCVar r_cuFireflyClamp("r_cuFireflyClamp", "5.0", CVAR_RENDERER | CVAR_ARCHIVE, "Clamp max sample brightness (0=off)");
idCVar r_cuThroughputClamp("r_cuThroughputClamp", "0.25", CVAR_RENDERER | CVAR_ARCHIVE, "Early path termination on low throughput (0.0-1.0)");

// Temporal Accumulation
idCVar r_cuAccumulation("r_cuAccumulation", "1", CVAR_RENDERER | CVAR_ARCHIVE, "Temporal accumulation (0=off, 1=on)");

// Ray Offsets & Bias
idCVar r_cuRayOffset("r_cuRayOffset", "0.001", CVAR_RENDERER | CVAR_ARCHIVE, "Ray origin offset to prevent self-intersection");

// Material & Lighting
idCVar r_cuEmissionBoost("r_cuEmissionBoost", "1.0", CVAR_RENDERER | CVAR_ARCHIVE, "Emission multiplier (0.1-10.0)");
idCVar r_cuSpecularBoost("r_cuSpecularBoost", "2.0", CVAR_RENDERER | CVAR_ARCHIVE, "Specular BRDF multiplier (Doom3 uses 2x)");
idCVar r_cuSoftShadowScale("r_cuSoftShadowScale", "0.2", CVAR_RENDERER | CVAR_ARCHIVE, "Soft shadow jitter scale (0=hard, 0.05=subtle, 0.2=soft)");

// Tone Mapping & Display
idCVar r_cuExposure("r_cuExposure", "1.0", CVAR_RENDERER | CVAR_ARCHIVE, "Exposure multiplier (0.1-10.0)");
idCVar r_cuGamma("r_cuGamma", "2.2", CVAR_RENDERER | CVAR_ARCHIVE, "Gamma correction (1.8-2.6)");
idCVar r_cuToneMapMode("r_cuToneMapMode", "1", CVAR_RENDERER | CVAR_ARCHIVE, "Tone map: 0=Reinhard, 1=ACES, 2=Uncharted2");

// Sky & Ambient Lighting
idCVar r_cuSkyIntensity("r_cuSkyIntensity", "0.0", CVAR_RENDERER | CVAR_ARCHIVE, "Sky dome ambient intensity (0.0-2.0)");
idCVar r_cuSkyColorZenithR("r_cuSkyColorZenithR", "0.3", CVAR_RENDERER, "Sky zenith R");
idCVar r_cuSkyColorZenithG("r_cuSkyColorZenithG", "0.35", CVAR_RENDERER, "Sky zenith G");
idCVar r_cuSkyColorZenithB("r_cuSkyColorZenithB", "0.5", CVAR_RENDERER, "Sky zenith B");
idCVar r_cuSkyColorHorizonR("r_cuSkyColorHorizonR", "0.4", CVAR_RENDERER, "Sky horizon R");
idCVar r_cuSkyColorHorizonG("r_cuSkyColorHorizonG", "0.35", CVAR_RENDERER, "Sky horizon G");
idCVar r_cuSkyColorHorizonB("r_cuSkyColorHorizonB", "0.3", CVAR_RENDERER, "Sky horizon B");
idCVar r_cuSkyColorGroundR("r_cuSkyColorGroundR", "0.1", CVAR_RENDERER, "Sky ground R");
idCVar r_cuSkyColorGroundG("r_cuSkyColorGroundG", "0.08", CVAR_RENDERER, "Sky ground G");
idCVar r_cuSkyColorGroundB("r_cuSkyColorGroundB", "0.05", CVAR_RENDERER, "Sky ground B");

// Volumetric Effects
idCVar r_cuVolumetric("r_cuVolumetric", "0.0", CVAR_RENDERER | CVAR_ARCHIVE, "Volumetric light scattering intensity (0=off, 1=normal, expensive!)");
idCVar r_cuVolumetricDensity("r_cuVolumetricDensity", "0.008", CVAR_RENDERER | CVAR_ARCHIVE, "Volumetric fog density (0.001-0.1)");
idCVar r_cuVolumetricSteps("r_cuVolumetricSteps", "16", CVAR_RENDERER | CVAR_ARCHIVE, "Volumetric ray march steps (4-64)");
idCVar r_cuVolumetricAnisotropy("r_cuVolumetricAnisotropy", "0.6", CVAR_RENDERER | CVAR_ARCHIVE, "Volumetric scattering anisotropy (-1 to 1)");
idCVar r_cuVolFalloff("r_cuVolFalloff", "2.0", CVAR_RENDERER | CVAR_ARCHIVE, "Volumetric distance falloff exponent");
idCVar r_cuVolMaxDist("r_cuVolMaxDist", "500.0", CVAR_RENDERER | CVAR_ARCHIVE, "Volumetric max sampling distance");

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

	// LBVH temporaries
	d_mortonCodes = NULL;
	d_sortedIndices = NULL;
	d_parents = NULL;
	d_atomicCounters = NULL;
	allocatedLBVHSize = 0;
	allocatedTriangles = 0;
	allocatedBVHNodes = 0;
	allocatedTriIndices = 0;

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
	nextMaterialIndex = 0;
	nextLightIndex = 0;
	needTextureFlush = false;

	width = 0;
	height = 0;
	renderWidth = 0;
	renderHeight = 0;
	prevRenderWidth = 0;
	prevRenderHeight = 0;

	timer_start = NULL;
	timer_stop = NULL;
	kernel_ms = 0.0f;
	kernel_fps = 0.0f;
	stream = 0;
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
		common->Warning("idCudaRenderer::Init(): No compatible CUDA device found\n");
		return 0;
	}

	common->Printf("\n----- Init CUDA Renderer -----\n");

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
	allocatedTriangles = MAX_TRIANGLES;

	// BVH nodes and triIndices are dynamically allocated in BuildBVH()
	d_bvhNodes = NULL;
	d_triIndices = NULL;
	allocatedBVHNodes = 0;
	allocatedTriIndices = 0;

	// LBVH temp buffers are dynamically allocated in BuildBVH()
	d_mortonCodes = NULL;
	d_sortedIndices = NULL;
	d_parents = NULL;
	d_atomicCounters = NULL;
	allocatedLBVHSize = 0;

	CUDA_CHECK_VOID(cudaMalloc(&d_materials, MAX_MATERIALS * sizeof(cudaMaterial_t)));
	CUDA_CHECK_VOID(cudaMalloc(&d_textures, MAX_TEXTURES * sizeof(cudaTexture_t)));
	CUDA_CHECK_VOID(cudaMalloc(&d_lights, MAX_LIGHTS * sizeof(cudaLight_t)));

	// create CUDA stream
	cudaStreamCreate(&stream);
	
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

	// free LBVH temporaries
	if (d_mortonCodes) {
		cudaFree(d_mortonCodes);
		d_mortonCodes = NULL;
	}
	if (d_sortedIndices) {
		cudaFree(d_sortedIndices);
		d_sortedIndices = NULL;
	}
	if (d_parents) {
		cudaFree(d_parents);
		d_parents = NULL;
	}
	if (d_atomicCounters) {
		cudaFree(d_atomicCounters);
		d_atomicCounters = NULL;
	}
	allocatedLBVHSize = 0;
	allocatedTriangles = 0;
	allocatedBVHNodes = 0;
	allocatedTriIndices = 0;

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
		h_texturePtrs.Clear();
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

	if (stream) {
		cudaStreamDestroy(stream);
		stream = 0;
	}

	// clear the rest
	h_materials.Clear();
	h_lights.Clear();
	h_vertices.Clear();
	h_triangles.Clear();
	materialEmission.Clear();
	materialHash.Free();
	h_materialPtrs.Clear();
	nextMaterialIndex = 0;
	lightHash.Free();
	h_lightPtrs.Clear();
	nextLightIndex = 0;

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

	// render scale
	float targetScale = idMath::ClampFloat(0.25f, 1.0f, r_cuRenderScale.GetFloat());
	bool useAccumulation = r_cuAccumulation.GetBool();

	// detect camera/scene changes
	bool cameraChanged = false;
	for (int i = 0; i < 3; i++) {
		if (fabsf(cam_pos[i] - prev_cam_pos[i]) > 1e-4f ||
			fabsf(cam_forward[i] - prev_cam_forward[i]) > 1e-4f) {
			cameraChanged = true;
			break;
		}
	}

	// consume need_reload flag (geometry uploaded in EndFrame)
	if (need_reload && num_triangles > 0 && num_vertices > 0) {
		need_reload = 0;
	}

	// save current camera for next frame comparison
	for (int i = 0; i < 3; i++) {
		prev_cam_pos[i] = cam_pos[i];
		prev_cam_forward[i] = cam_forward[i];
	}

	// read user CVar targets with runtime clamping
	int targetSPP = idMath::ClampInt(1, 64, r_cuSamplesPerPixel.GetInteger());
	int targetDepth = idMath::ClampInt(1, 16, r_cuMaxDepth.GetInteger());
	int targetLightSamples = idMath::ClampInt(1, 8, r_cuMaxLightSamples.GetInteger());
	float targetIndirectProb = idMath::ClampFloat(0.0f, 1.0f, r_cuIndirectProb.GetFloat());

	// progressive quality refinement
	int frame = accum_frame;
	int activeSPP;
	int activeDepth;
	int activeLightSamples;
	float activeIndirectProb;

	if (cameraChanged) {
		// minimal quality for responsiveness
		renderWidth = (int)(width * targetScale);
		renderHeight = (int)(height * targetScale);
		activeSPP = 1;
		activeDepth = 2;
		activeLightSamples = 1;
		activeIndirectProb = 0.0f;
	} else if (frame < 2) {
		// low quality, target resolution
		renderWidth = (int)(width * targetScale);
		renderHeight = (int)(height * targetScale);
		activeSPP = 1;
		activeDepth = 2;
		activeLightSamples = 1;
		activeIndirectProb = 0.0f;
	} else if (frame < 4) {
		// medium quality
		renderWidth = (int)(width * targetScale);
		renderHeight = (int)(height * targetScale);
		activeSPP = idMath::ClampInt(1, targetSPP, 2);
		activeDepth = idMath::ClampInt(1, targetDepth, 2);
		activeLightSamples = 1;
		activeIndirectProb = targetIndirectProb * 0.25f;
	} else if (frame < 8) {
		// ramping up
		renderWidth = (int)(width * targetScale);
		renderHeight = (int)(height * targetScale);
		activeSPP = targetSPP;
		activeDepth = targetDepth;
		activeLightSamples = idMath::ClampInt(1, targetLightSamples, targetLightSamples / 2 + 1);
		activeIndirectProb = targetIndirectProb * 0.5f;
	} else {
		// full quality
		renderWidth = (int)(width * targetScale);
		renderHeight = (int)(height * targetScale);
		activeSPP = targetSPP;
		activeDepth = targetDepth;
		activeLightSamples = targetLightSamples;
		activeIndirectProb = targetIndirectProb;
	}

	// ensure minimum render dimensions
	if (renderWidth < 64) renderWidth = 64;
	if (renderHeight < 64) renderHeight = 64;

	// resolution change detection
	if (renderWidth != prevRenderWidth || renderHeight != prevRenderHeight) {
		accum_frame = 0;
		frame = 0;
		CUDA_CHECK_VOID(cudaMemset(d_framebuffer, 0, width * height * 4 * sizeof(float)));
	}
	prevRenderWidth = renderWidth;
	prevRenderHeight = renderHeight;

	// accumulation reset on camera/scene change
	if (cameraChanged || !useAccumulation) {
		accum_frame = 0;
		CUDA_CHECK_VOID(cudaMemset(d_framebuffer, 0, width * height * 4 * sizeof(float)));
	}
	CUDA_CHECK_VOID(cudaMemset(d_outputBuffer, 0, width * height * 4 * sizeof(unsigned char)));

	//////////////////////////////
	// renderer
	//////////////////////////////
	cudaEventRecord(timer_start, stream);

	// sky color from CVars
	float skyZenith[3] = {
		r_cuSkyColorZenithR.GetFloat(),
		r_cuSkyColorZenithG.GetFloat(),
		r_cuSkyColorZenithB.GetFloat()
	};
	float skyHorizon[3] = {
		r_cuSkyColorHorizonR.GetFloat(),
		r_cuSkyColorHorizonG.GetFloat(),
		r_cuSkyColorHorizonB.GetFloat()
	};
	float skyGround[3] = {
		r_cuSkyColorGroundR.GetFloat(),
		r_cuSkyColorGroundG.GetFloat(),
		r_cuSkyColorGroundB.GetFloat()
	};

	// clamp light count to prevent reading past the GPU buffer
	int numLightsForKernel = h_lights.Num();
	if (numLightsForKernel > MAX_LIGHTS) numLightsForKernel = MAX_LIGHTS;

	LaunchPathTracingKernel(
		d_vertices,
		d_triangles,
		d_bvhNodes,
		d_triIndices,
		d_materials,
		d_textures,
		d_lights,
		numLightsForKernel,
		r_cuMode.GetInteger(),
		d_framebuffer,
		renderWidth,
		renderHeight,
		cam_pos,
		cam_forward,
		cam_right,
		cam_up,
		fov_x,
		fov_y,
		activeSPP,
		activeDepth,
		activeLightSamples,
		accum_frame,
		fmaxf(0.0f, r_cuEmissionBoost.GetFloat()),
		activeIndirectProb,
		r_cuRussianRoulette.GetInteger(),
		idMath::ClampInt(0, 16, r_cuRRMinBounces.GetInteger()),
		idMath::ClampFloat(0.01f, 0.95f, r_cuRRSurvivalMin.GetFloat()),
		fmaxf(0.0f, r_cuEarlyTermThreshold.GetFloat()),
		fmaxf(0.0f, r_cuFireflyClamp.GetFloat()),
		fmaxf(0.0f, r_cuThroughputClamp.GetFloat()),
		idMath::ClampFloat(0.0001f, 0.1f, r_cuRayOffset.GetFloat()),
		fmaxf(0.0f, r_cuSpecularBoost.GetFloat()),
		fmaxf(0.0f, r_cuSkyIntensity.GetFloat()),
		skyZenith,
		skyHorizon,
		skyGround,
		fmaxf(0.0f, r_cuVolumetricDensity.GetFloat()),
		idMath::ClampInt(1, 128, r_cuVolumetricSteps.GetInteger()),
		idMath::ClampFloat(-0.999f, 0.999f, r_cuVolumetricAnisotropy.GetFloat()),
		fmaxf(0.0f, r_cuVolFalloff.GetFloat()),
		fmaxf(0.0f, r_cuVolMaxDist.GetFloat()),
		fmaxf(0.0f, r_cuSoftShadowScale.GetFloat()),
		stream
	);

	accum_frame++;

	// tone mapping pass: HDR accumulation -> LDR output
	LaunchToneMappingKernel(
		d_framebuffer,
		d_outputBuffer,
		renderWidth,
		renderHeight,
		fmaxf(0.001f, r_cuExposure.GetFloat()),
		idMath::ClampFloat(0.1f, 10.0f, r_cuGamma.GetFloat()),
		idMath::ClampInt(0, 2, r_cuToneMapMode.GetInteger()),
		accum_frame,
		stream
	);

	cudaEventRecord(timer_stop, stream);

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

/*
========================
idCudaRenderer::ResetAccumulation
========================
*/
void idCudaRenderer::ResetAccumulation() {
	accum_frame = 0;
	if (d_framebuffer) {
		cudaMemsetAsync(d_framebuffer, 0, width * height * 4 * sizeof(float), stream);
	}
}

#endif // HAVE_CUDA