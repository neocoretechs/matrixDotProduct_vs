
// helpers.h
#pragma once
#include <jni.h>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <vector>

void convertFloatToHalfCPU(const float* src, __half* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = __float2half(src[i]);
    }
}
inline jint jni_error(cudaError_t e) { return e == cudaSuccess ? 0 : (jint)e; }
inline jint jni_cublas_error(cublasStatus_t s) { return s == CUBLAS_STATUS_SUCCESS ? 0 : (jint)s; }

// RAII for device buffer
template<typename T>
struct DevBuf {
    T* p = nullptr;
    size_t bytes = 0;
    ~DevBuf() { if (p) cudaFree(p); }
    cudaError_t alloc(size_t n) { bytes = n * sizeof(T); return cudaMalloc((void**)&p, bytes); }
};


// Pull float[] from Java
inline float* jget_float_array(JNIEnv* env, jfloatArray arr, jboolean* isCopy) {
    return env->GetFloatArrayElements(arr, isCopy);
}
inline void jrelease_float_array(JNIEnv* env, jfloatArray arr, float* ptr) {
    env->ReleaseFloatArrayElements(arr, ptr, JNI_ABORT);
}

// Extract ArrayList<float[]> elements as jfloatArray vector
inline std::vector<jfloatArray> jarraylist_to_jfloat_arrays(JNIEnv* env, jobject aList) {
    std::vector<jfloatArray> out;
    jclass clsList = env->FindClass("java/util/ArrayList");
    jmethodID midSize = env->GetMethodID(clsList, "size", "()I");
    jmethodID midGet = env->GetMethodID(clsList, "get", "(I)Ljava/lang/Object;");
    jint size = env->CallIntMethod(aList, midSize);
    out.reserve(size);
    for (int i = 0; i < size; ++i) {
        jobject el = env->CallObjectMethod(aList, midGet, i);
        out.push_back(reinterpret_cast<jfloatArray>(el));
    }
    return out;
}
