/*
* Calling convention:
* From Java, pass the handle, the scratch array, the pointer to the right device buffer, the offset in elements, and the count.
* The native side computes the destination pointer and does an async copy into the stream set on the cuBLAS handle.
* Due to use on release, the JVM doesn’t copy the scratch array back, it’s strictly one‑way staging.
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
JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_Attn_uploadSlice(JNIEnv * env, jclass, jlong ctxHandle, jfloatArray hostBuf, jlong devicePtr, jlong offset, jint count) {
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
JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Gemm_Attn_init(JNIEnv*, jclass, jint maxB, jint maxH, jint maxTq, jint maxTk, jint d) {
    auto* ctx = new AttnCtx;
    ctx->maxB = maxB; 
    ctx->maxH = maxH;
    ctx->maxTq = maxTq; 
    ctx->maxTk = maxTk; 
    ctx->d = d;

    cudaStreamCreate(&ctx->stream);
    cublasCreate(&ctx->handle);
    cublasSetStream(ctx->handle, ctx->stream);

    size_t qSize = (size_t)maxB * maxH * maxTq * d;
    size_t kSize = (size_t)maxB * maxH * maxTk * d;
    size_t vSize = (size_t)maxB * maxH * maxTk * d; // same as kSize?
    size_t sSize = (size_t)maxB * maxH * maxTq * maxTk;
    size_t oSize = (size_t)maxB * maxH * maxTq * d;

    cudaMalloc(&ctx->dQ, qSize * sizeof(float));
    cudaMalloc(&ctx->dK, kSize * sizeof(float));
    cudaMalloc(&ctx->dV, vSize * sizeof(float));
    cudaMalloc(&ctx->dS, sSize * sizeof(float));
    cudaMalloc(&ctx->dO, oSize * sizeof(float));

    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Gemm_Attn_getDQ(JNIEnv*, jclass, jlong ctxHandle) {
    auto* ctx = reinterpret_cast<AttnCtx*>(ctxHandle);
    return reinterpret_cast<jlong>(ctx->dQ);
}

JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Gemm_Attn_getDK(JNIEnv*, jclass, jlong ctxHandle) {
    auto* ctx = reinterpret_cast<AttnCtx*>(ctxHandle);
    return reinterpret_cast<jlong>(ctx->dK);
}

JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Gemm_Attn_getDV(JNIEnv*, jclass, jlong ctxHandle) {
    auto* ctx = reinterpret_cast<AttnCtx*>(ctxHandle);
    return reinterpret_cast<jlong>(ctx->dV);
}

JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Gemm_Attn_getDS(JNIEnv*, jclass, jlong ctxHandle) {
    auto* ctx = reinterpret_cast<AttnCtx*>(ctxHandle);
    return reinterpret_cast<jlong>(ctx->dS);
}

JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Gemm_Attn_getDO(JNIEnv*, jclass, jlong ctxHandle) {
    auto* ctx = reinterpret_cast<AttnCtx*>(ctxHandle);
    return reinterpret_cast<jlong>(ctx->dO);
}
/*
* Free the allocated device memory.
* handle is the handle to AttnCtx we got in Attn_init
*/
JNIEXPORT void JNICALL Java_com_neocoretechs_cublas_Gemm_Attn_destroy(JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<AttnCtx*>(handle);
    if (!ctx) return;

    cudaFree(ctx->dQ);
    cudaFree(ctx->dK);
    cudaFree(ctx->dV);
    cudaFree(ctx->dS);
    cudaFree(ctx->dO);

    cublasDestroy(ctx->handle);
    cudaStreamDestroy(ctx->stream);

    delete ctx;
}