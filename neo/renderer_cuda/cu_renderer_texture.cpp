#ifdef HAVE_CUDA

#include "renderer/tr_local.h"
#include "renderer_cuda/cu_renderer.h"
#include <cuda_runtime.h>

extern idCVar r_cuDebug;

/*
========================
idCudaRenderer::AddTexture
========================
*/
int idCudaRenderer::AddTexture(const idImage* image) {
    if (!image || image->defaulted) {
		return -1;
	}

    if (image->texnum == idImage::TEXTURE_NOT_LOADED) {
		return -1;
	}

    // check if texture already cached by GL texnum
    int key = (int)image->texnum;
    for (int i = textureHash.First(key); i >= 0; i = textureHash.Next(i)) {
        if (h_texnums[i] == key) {
            return i;
        }
    }

    if (h_textures.Num() >= MAX_TEXTURES) {
		common->Warning("idCudaRenderer::AddTexture(): Maximum texture count (%d) reached\n", MAX_TEXTURES);
		return -1;
	}

    // read texture data from GL
	int width = image->uploadWidth;
	int height = image->uploadHeight;
	
	if (width <= 0 || height <= 0 || width > 4096 || height > 4096) {
		return -1;
	}
	
	// bind the texture and read data from GL
	GLint oldTexture;
	qglGetIntegerv(GL_TEXTURE_BINDING_2D, &oldTexture);
	qglBindTexture(GL_TEXTURE_2D, image->texnum);
	
	// allocate temporary host buffer for texture data (RGBA8)
	size_t dataSize = width * height * 4;
	unsigned char* hostData = new unsigned char[dataSize];
	
	// read texture from GL
	qglGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, hostData);
	
	// restore previous texture binding
	qglBindTexture(GL_TEXTURE_2D, oldTexture);
	
	// allocate CUDA memory and upload
	unsigned char* deviceData = NULL;
	cudaError_t err = cudaMalloc(&deviceData, dataSize);
	if (err != cudaSuccess) {
		common->Warning("idCudaPathTracer::GetOrCreateTexture(): Failed to allocate texture memory: %s\n", cudaGetErrorString(err));
		delete[] hostData;
		return -1;
	}
	
	err = cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);
	delete[] hostData;
	
	if (err != cudaSuccess) {
		common->Warning("idCudaPathTracer::GetOrCreateTexture(): Failed to upload texture data: %s\n", cudaGetErrorString(err));
		cudaFree(deviceData);
		return -1;
	}
	
	// create texture descriptor
	cudaTexture_t tex;
	tex.data = deviceData;
	tex.width = width;
	tex.height = height;
	
	// add to list and register in cache
	int index = h_textures.Num();
	h_textures.Append(tex);
	h_texnums.Append(key);
	textureHash.Add(key, index);

    if (r_cuDebug.GetBool()) {
        common->Printf("idCudaRenderer::AddTexture(): Texture %d: %dx%d\n", index, width, height);
    }

    return index;
}

#endif // HAVE_CUDA