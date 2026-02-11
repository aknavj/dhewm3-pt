#ifdef HAVE_CUDA

#include "renderer/tr_local.h"
#include "renderer_cuda/cu_renderer.h"
#include <cuda_runtime.h>
#include <new>

// extern console variables
extern idCVar r_cuDebug;

/*
========================
idCudaRenderer::AddTexture
========================
*/
int idCudaRenderer::AddTexture(const idImage* image) {
    if (!image || image->defaulted) {
        if (image && image->defaulted && r_cuDebug.GetBool()) {
            common->Printf("AddTexture: SKIPPED defaulted image '%s'\n", image->imgName.c_str());
        }
		return -1;
	}

    if (image->texnum == idImage::TEXTURE_NOT_LOADED) {
        if (r_cuDebug.GetBool()) {
            common->Printf("AddTexture: SKIPPED not-loaded image '%s'\n", image->imgName.c_str());
        }
		return -1;
	}

	// skip non-2D textures (cubemaps, 3D textures, etc.) — they can't be read via GL_TEXTURE_2D
	if (image->type != TT_2D) {
		return -1;
	}

	// check if texture already cached by pointer
	int key = (int)(intptr_t)image;
	for (int i = textureHash.First(key); i >= 0; i = textureHash.Next(i)) {
		if (h_texturePtrs[i] == image) {
			return i;
		}
	}

	if (h_textures.Num() >= MAX_TEXTURES) {
		common->Warning("idCudaRenderer::AddTexture(): Maximum texture count (%d) reached, will flush next frame\n", MAX_TEXTURES);
		needTextureFlush = true;
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

	// verify actual GL texture dimensions match what we expect
	GLint glWidth = 0, glHeight = 0;
	qglGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &glWidth);
	qglGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &glHeight);
	if (glWidth <= 0 || glHeight <= 0) {
		qglBindTexture(GL_TEXTURE_2D, oldTexture);
		return -1;
	}
	// use actual GL dimensions — they may differ from uploadWidth/Height after driver downscaling
	if (glWidth != width || glHeight != height) {
		width = glWidth;
		height = glHeight;
		if (width > 4096 || height > 4096) {
			qglBindTexture(GL_TEXTURE_2D, oldTexture);
			return -1;
		}
	}

	// allocate temporary host buffer for texture data (RGBA8)
	size_t dataSize = (size_t)width * (size_t)height * 4;
	unsigned char* hostData = new (std::nothrow) unsigned char[dataSize];
	if (!hostData) {
		qglBindTexture(GL_TEXTURE_2D, oldTexture);
		return -1;
	}
	
	// read texture from GL
	qglGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, hostData);
	
	// restore previous texture binding
	qglBindTexture(GL_TEXTURE_2D, oldTexture);
	
	// allocate CUDA memory and upload
	unsigned char* deviceData = NULL;
	cudaError_t err = cudaMalloc(&deviceData, dataSize);
	if (err != cudaSuccess) {
		common->Warning("idCudaRenderer::AddTexture(): Failed to allocate texture memory: %s\n", cudaGetErrorString(err));
		delete[] hostData;
		return -1;
	}
	
	err = cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);
	delete[] hostData;
	
	if (err != cudaSuccess) {
		common->Warning("idCudaRenderer::AddTexture(): Failed to upload texture data: %s\n", cudaGetErrorString(err));
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
	h_texnums.Append((int)image->texnum);
	h_texturePtrs.Append(image);
	textureHash.Add(key, index);

    if (r_cuDebug.GetBool()) {
        common->Printf("idCudaRenderer::AddTexture(): Texture %d: %s %dx%d\n", index, image->imgName.c_str(), width, height);
    }

    return index;
}

#endif // HAVE_CUDA