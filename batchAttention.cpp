/*
* Calling convention:
* From Java, pass the handle, the scratch array, the pointer to the right device buffer, the offset in elements, and the count.
* The native side computes the destination pointer and does an async copy into the stream set on the cuBLAS handle.
* Due to use on release, the JVM doesn’t copy the scratch array back, it’s strictly one‑way staging.
* NOTE: We deal with 2 handles here, the cublasHandle and the handle created for the AttnCtx
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "com_neocoretechs_cublas_Gemm.h"
struct AttnCtx {
    cublasHandle_t handle;
    cudaStream_t stream;
    float* dQ;
    float* dK;
    float* dV;
    float* dS;
    float* dO;
    int maxB, maxH, maxTq, maxTk, d;
};
/*
* Upload an array of floats from Java to GPU device memory we allocated in Attn_init.
* ctxHandle is returned from Attn_init.
* hostBuf is the array of float from Java to recieve data from GPU.
* devicePtr is the source from GPU we allocated in init and lives in the attn Context dq, dk, dv etc.
* offset is the offset into the devicePtr pointing to source GPU strides.
* count is number of floats to transfer from offset.
* return 0 for success else error code -1 if ctxHandle is null, -2 if hostBuf is null or elemets return is bad
* or cudaError_t code from cudaMemcpyAsynch
*/
JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Attn_uploadSlice(JNIEnv* env, jclass clazz, jlong ctxHandle, jfloatArray hostBuf, jlong devicePtr, jlong offset, jint count) {
    auto* ctx = reinterpret_cast<AttnCtx*>(ctxHandle);
    if (!ctx) return -1;

    jfloat* hPtr = env->GetFloatArrayElements(hostBuf, nullptr);
    if (!hPtr) return -2;

    float* dBase = reinterpret_cast<float*>(devicePtr);
    float* dDst = dBase + offset;

    cudaError_t err = cudaMemcpyAsync(
        dDst,
        hPtr,
        count * sizeof(float),
        cudaMemcpyHostToDevice,
        ctx->stream
    );

    env->ReleaseFloatArrayElements(hostBuf, hPtr, JNI_ABORT);

    return (err == cudaSuccess) ? 0 : (int)err;
}
/*
* Allocate device memory en masse.
* maxB (batch size).
* maxH (number of heads).
* d (head dimension) (headSize in code).
* maxTq (query length) query index within the sequence.
* maxTk (key length) the context length so far. (the horizon of keys you can attend to). T in code.
* Flow:
* maxB set ctx->MaxB, same for maxH maxTq maxTk and d sizing metrics for strides.
* ctx dQ float32 sized to maxB * maxH * maxTq * d
* ctx dK float32 sized to maxB * maxH * maxTk * d
* ctx dV same as dK ?
* ctx dS float32 sized to maxB * maxH * maxTq * maxTk
* ctx dO float32 sized to maxB * maxH * maxTq * d
* return handle to AttnCtx struct
*/
JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Attn_init(JNIEnv* env, jclass clazz, jlong handle, jint maxB, jint maxH, jint maxTq, jint maxTk, jint d) {
    auto* ctx = new AttnCtx;
    ctx->maxB = maxB; 
    ctx->maxH = maxH;
    ctx->maxTq = maxTq; 
    ctx->maxTk = maxTk; 
    ctx->d = d;
    ctx->handle = (cublasHandle_t)handle;

    if(cudaStreamCreate(&ctx->stream) != cudaSuccess) {
        delete ctx;
        return -201;
    }
    if (cublasSetStream(ctx->handle, ctx->stream) != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(ctx->stream);
        delete ctx;
        return -202;
    }

    size_t qSize = (size_t)maxB * maxH * maxTq * d;
    size_t kSize = (size_t)maxB * maxH * maxTk * d;
    size_t vSize = (size_t)maxB * maxH * maxTk * d; // same as kSize?
    size_t sSize = (size_t)maxB * maxH * maxTq * maxTk;
    size_t oSize = (size_t)maxB * maxH * maxTq * d;

    if(cudaMalloc(&ctx->dQ, qSize * sizeof(float)) != cudaSuccess) return -200;
    if(cudaMalloc(&ctx->dK, kSize * sizeof(float)) != cudaSuccess) return -200;
    if(cudaMalloc(&ctx->dV, vSize * sizeof(float)) != cudaSuccess) return -200;
    if(cudaMalloc(&ctx->dS, sSize * sizeof(float)) != cudaSuccess) return -200;
    if(cudaMalloc(&ctx->dO, oSize * sizeof(float)) != cudaSuccess) return -200;

    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Attn_getDQ(JNIEnv* env, jclass clazz, jlong ctxHandle) {
    auto* ctx = reinterpret_cast<AttnCtx*>(ctxHandle);
    return reinterpret_cast<jlong>(ctx->dQ);
}

JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Attn_getDK(JNIEnv* env, jclass clazz, jlong ctxHandle) {
    auto* ctx = reinterpret_cast<AttnCtx*>(ctxHandle);
    return reinterpret_cast<jlong>(ctx->dK);
}

JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Attn_getDV(JNIEnv*, jclass clazz, jlong ctxHandle) {
    auto* ctx = reinterpret_cast<AttnCtx*>(ctxHandle);
    return reinterpret_cast<jlong>(ctx->dV);
}

JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Attn_getDS(JNIEnv* env, jclass clazz, jlong ctxHandle) {
    auto* ctx = reinterpret_cast<AttnCtx*>(ctxHandle);
    return reinterpret_cast<jlong>(ctx->dS);
}

JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Attn_getDO(JNIEnv* env, jclass clazz, jlong ctxHandle) {
    auto* ctx = reinterpret_cast<AttnCtx*>(ctxHandle);
    return reinterpret_cast<jlong>(ctx->dO);
}
/*
* Free the allocated device memory and the stream, dont free handle.
* handle is the handle to AttnCtx we got in Attn_init
*/
JNIEXPORT void JNICALL Java_com_neocoretechs_cublas_Attn_destroy(JNIEnv* env, jclass clazz, jlong handle) {
    auto* ctx = reinterpret_cast<AttnCtx*>(handle);
    if (!ctx) return;

    cudaFree(ctx->dQ);
    cudaFree(ctx->dK);
    cudaFree(ctx->dV);
    cudaFree(ctx->dS);
    cudaFree(ctx->dO);

    cudaStreamDestroy(ctx->stream);
    delete ctx;
}
/*
 * Download an array of floats from GPU device memory into a Java float[].
 * ctxHandle is returned from Attn_init.
 * hostBuf is the Java float[] to receive data.
 * devicePtr is the source pointer in device memory (e.g. ctx->dO).
 * offset is the offset in elements into devicePtr.
 * count is the number of floats to transfer.
 * Returns 0 for success, -1 if ctxHandle is null, -2 if hostBuf is null,
 * or a cudaError_t code if cudaMemcpyAsync fails.
 */
JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Attn_downloadSlice
(JNIEnv* env, jclass clazz, jlong ctxHandle, jfloatArray hostBuf,
    jlong devicePtr, jlong offset, jint count) {

    auto* ctx = reinterpret_cast<AttnCtx*>(ctxHandle);
    if (!ctx) return -1;

    jfloat* hPtr = env->GetFloatArrayElements(hostBuf, nullptr);
    if (!hPtr) return -2;

    float* dBase = reinterpret_cast<float*>(devicePtr);
    float* dSrc = dBase + offset;

    cudaError_t err = cudaMemcpyAsync(
        hPtr,
        dSrc,
        count * sizeof(float),
        cudaMemcpyDeviceToHost,
        ctx->stream
    );

    // Commit results back to Java array
    env->ReleaseFloatArrayElements(hostBuf, hPtr, 0);

    return (err == cudaSuccess) ? 0 : (int)err;
}