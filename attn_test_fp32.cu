// attention_test_fp32.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include "com_neocoretechs_cublas_Gemm.h"
// ---------- Error helpers ----------
#define CHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); return -1; } } while(0)

#define CHECK_CUBLAS(x) do { cublasStatus_t st = (x); if (st != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st, __FILE__, __LINE__); return -2; } } while(0)

// ---------- Row-wise softmax ----------
__global__ void row_softmax_fp32(const float* __restrict__ S, float* __restrict__ A,
    int rows, int cols, int ldS, int ldA) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    const float* srow = S + r * ldS;
    float* arow = A + r * ldA;

    // 1) max
    float m = -1e30f;
    for (int c = 0; c < cols; ++c) m = fmaxf(m, srow[c]);

    // 2) exp and sum
    float sum = 0.f;
    for (int c = 0; c < cols; ++c) {
        float e = __expf(srow[c] - m);
        arow[c] = e;
        sum += e;
    }

    // 3) normalize
    float inv = 1.0f / sum;
    for (int c = 0; c < cols; ++c) arow[c] *= inv;
}

static int softmax_rows_fp32(const float* d_S, float* d_A, int rows, int cols, int ldS, int ldA, cudaStream_t stream) {
    int threads = 128;
    int blocks = (rows + threads - 1) / threads;
    row_softmax_fp32 << <blocks, threads, 0, stream >> > (d_S, d_A, rows, cols, ldS, ldA);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -3;
}

// ---------- CPU baseline (optional, small sizes) ----------
static void cpu_attention_fp32(const float* hQ, const float* hK, const float* hV,
    float* hO, int Tq, int Tk, int d) {
    std::vector<float> S(Tq * Tk), A(Tq * Tk);

    // S = (Q K^T) * 1/sqrt(d), row-major
    float alpha = 1.0f / std::sqrt((float)d);
    for (int i = 0; i < Tq; ++i) {
        for (int j = 0; j < Tk; ++j) {
            float acc = 0.f;
            for (int k = 0; k < d; ++k) acc += hQ[i * d + k] * hK[j * d + k];
            S[i * Tk + j] = acc * alpha;
        }
    }

    // row-wise softmax
    for (int i = 0; i < Tq; ++i) {
        float m = -1e30f;
        for (int j = 0; j < Tk; ++j) m = std::max(m, S[i * Tk + j]);
        float sum = 0.f;
        for (int j = 0; j < Tk; ++j) { float e = std::exp(S[i * Tk + j] - m); A[i * Tk + j] = e; sum += e; }
        float inv = 1.f / sum;
        for (int j = 0; j < Tk; ++j) A[i * Tk + j] *= inv;
    }

    // O = A V
    for (int i = 0; i < Tq; ++i) {
        for (int c = 0; c < d; ++c) {
            float acc = 0.f;
            for (int j = 0; j < Tk; ++j) acc += A[i * Tk + j] * hV[j * d + c];
            hO[i * d + c] = acc;
        }
    }
}

// ---------- Main test function ----------
extern "C"
int attention_test_fp32(cublasHandle_t handle,
    // Sizes
    int Tq, int Tk, int d,
    // Device pointers (row-major)
    const float* d_Q, int ldQ,      // [Tq x d], ldQ = d
    const float* d_K, int ldK,      // [Tk x d], ldK = d
    const float* d_V, int ldV,      // [Tk x d], ldV = d
    float* d_O, int ldO,            // [Tq x d], ldO = d
    // Workspaces
    float* d_S, int ldS,            // [Tq x Tk], ldS = Tk
    float* d_A, int ldA,            // [Tq x Tk], ldA = Tk
    // Options
    int enable_tf32,                // 1 to enable TF32 (Ampere+)
    int do_cpu_check,               // 1 to validate for small sizes
    // Output timings (ms)
    float* out_ms_qkt,
    float* out_ms_softmax,
    float* out_ms_av
) {
    // Basic asserts
    if (Tq <= 0 || Tk <= 0 || d <= 0) { fprintf(stderr, "Invalid sizes\n"); return -10; }
    if (ldQ != d || ldK != d || ldV != d || ldO != d || ldS != Tk || ldA != Tk) {
        fprintf(stderr, "Leading dims mismatch (expect row-major)\n"); return -11;
    }
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    if (enable_tf32) {
        // Best-effort: enable TF32 tensor ops mode (ignored on non-Ampere/Hopper)
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    }
    // Events for timing
    cudaEvent_t e0, e1, e2, e3, e4;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));
    CHECK_CUDA(cudaEventCreate(&e2));
    CHECK_CUDA(cudaEventCreate(&e3));
    CHECK_CUDA(cudaEventCreate(&e4));

    // ---- QK^T scaled: S = (Q K^T) * alpha
    float alpha = 1.0f / std::sqrt((float)d);
    float beta = 0.0f;
    // m = Tq, n = Tk, k = d
    cublasOperation_t opQ = CUBLAS_OP_N;
    cublasOperation_t opK = CUBLAS_OP_T;

    CHECK_CUDA(cudaEventRecord(e0, stream));
    CHECK_CUBLAS(cublasSgemm(
        handle, opQ, opK,
        Tq, Tk, d,
        &alpha,
        d_Q, ldQ,
        d_K, ldK,
        &beta,
        d_S, ldS
    ));
    CHECK_CUDA(cudaEventRecord(e1, stream));

    // ---- Softmax rows: A = softmax(S)
    int rc = softmax_rows_fp32(d_S, d_A, Tq, Tk, ldS, ldA, stream);
    if (rc) { fprintf(stderr, "Softmax kernel failed\n"); return rc; }
    CHECK_CUDA(cudaEventRecord(e2, stream));

    // ---- AV: O = A V
    CHECK_CUBLAS(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        Tq, d, Tk,
        &alpha /* or 1.0f if you don't want extra scale */,
        d_A, ldA,
        d_V, ldV,
        &beta,
        d_O, ldO
    ));
    CHECK_CUDA(cudaEventRecord(e3, stream));
    // ---- Sync and timings
    CHECK_CUDA(cudaEventRecord(e4, stream));
    CHECK_CUDA(cudaEventSynchronize(e4));
    float ms_qkt = 0, ms_softmax = 0, ms_av = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_qkt, e0, e1));
    CHECK_CUDA(cudaEventElapsedTime(&ms_softmax, e1, e2));
    CHECK_CUDA(cudaEventElapsedTime(&ms_av, e2, e3));
    if (out_ms_qkt) *out_ms_qkt = ms_qkt;
    if (out_ms_softmax) *out_ms_softmax = ms_softmax;
    if (out_ms_av) *out_ms_av = ms_av;

    // ---- Optional CPU baseline check (small sizes only)
    if (do_cpu_check && Tq <= 2 && Tk <= 256 && d <= 128) {
        std::vector<float> hQ(Tq * d), hK(Tk * d), hV(Tk * d), hO(Tq * d), hO_cpu(Tq * d);
        CHECK_CUDA(cudaMemcpy(hQ.data(), d_Q, sizeof(float) * Tq * d, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hK.data(), d_K, sizeof(float) * Tk * d, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hV.data(), d_V, sizeof(float) * Tk * d, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hO.data(), d_O, sizeof(float) * Tq * d, cudaMemcpyDeviceToHost));
        cpu_attention_fp32(hQ.data(), hK.data(), hV.data(), hO_cpu.data(), Tq, Tk, d);
        // Compare a few elements
        int mismatches = 0;
        for (int i = 0; i < std::min(16, Tq * d); ++i) {
            float a = hO[i], b = hO_cpu[i];
            float diff = std::abs(a - b);
            if (diff > 1e-3f) ++mismatches;
        }
        if (mismatches) {
            fprintf(stderr, "CPU vs GPU mismatch count=%d (tolerance 1e-3)\n", mismatches);
            // Not failing hard; you can tighten if needed
        }
    }
    // Cleanup
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3); cudaEventDestroy(e4);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    // Report
    fprintf(stdout, "QK^T: %.3f ms | Softmax: %.3f ms | AV: %.3f ms | Tq=%d Tk=%d d=%d\n",
        ms_qkt, ms_softmax, ms_av, Tq, Tk, d);
    return 0;
}

JNIEXPORT jfloatArray JNICALL Java_com_neocoretechs_cublas_Attn_softMax
(JNIEnv* env, jclass clazz, jfloatArray jInput, jint rows, jint cols) {

    jsize len = env->GetArrayLength(jInput);
    std::vector<float> hInput(len);
    env->GetFloatArrayRegion(jInput, 0, len, hInput.data());

    std::vector<float> hOutput(len);

    float* dS, * dA;
    size_t bytes = len * sizeof(float);
    cudaMalloc(&dS, bytes);
    cudaMalloc(&dA, bytes);
    cudaMemcpy(dS, hInput.data(), bytes, cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (rows + threads - 1) / threads;
    row_softmax_fp32 << <blocks, threads >> > (dS, dA, rows, cols, cols, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(hOutput.data(), dA, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dS);
    cudaFree(dA);

    jfloatArray jOut = env->NewFloatArray(len);
    env->SetFloatArrayRegion(jOut, 0, len, hOutput.data());
    return jOut;
}

struct Ctx {
    cublasHandle_t handle;
    int Tq, Tk, d;
    // optional: persistent device buffers
    float* dQ, * dK, * dV, * dO, * dS, * dA;
};

JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Attn_initContext(
    JNIEnv* env, jclass clazz, jlong cublasHandle, jint Tq, jint Tk, jint d, jint enableTf32) {
    auto* ctx = new Ctx{ (cublasHandle_t)cublasHandle, Tq, Tk, d };
    // Allocate persistent device buffers once
    size_t bQ = size_t(Tq) * d * sizeof(float);
    size_t bK = size_t(Tk) * d * sizeof(float);
    size_t bV = size_t(Tk) * d * sizeof(float);
    size_t bO = size_t(Tq) * d * sizeof(float);
    size_t bS = size_t(Tq) * Tk * sizeof(float);
    size_t bA = size_t(Tq) * Tk * sizeof(float);
    cudaMalloc(&ctx->dQ, bQ);
    cudaMalloc(&ctx->dK, bK);
    cudaMalloc(&ctx->dV, bV);
    cudaMalloc(&ctx->dO, bO);
    cudaMalloc(&ctx->dS, bS);
    cudaMalloc(&ctx->dA, bA);
    // Optionally set TF32 math mode via cuBLAS, etc.
    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT void JNICALL Java_com_neocoretechs_cublas_Attn_freeContext(JNIEnv* env, jclass clazz, jlong h) {
    auto* ctx = reinterpret_cast<Ctx*>(h);
    if (!ctx) return;
    cudaFree(ctx->dQ); cudaFree(ctx->dK); cudaFree(ctx->dV);
    cudaFree(ctx->dO); cudaFree(ctx->dS); cudaFree(ctx->dA);
    delete ctx;
}

static float* addr(JNIEnv* env, jobject buf) {
    return static_cast<float*>(env->GetDirectBufferAddress(buf));
}

JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Attn_attentionFp32(JNIEnv * env, jclass clazz, jlong h,
    jobject jQ, jint ldQ, jobject jK, jint ldK, jobject jV, jint ldV,
    jobject jO, jint ldO, jobject jS, jint ldS, jobject jA, jint ldA,
    jint doCpuCheck, jobject jMsQKT, jobject jMsSM, jobject jMsAV) {

    auto* ctx = reinterpret_cast<Ctx*>(h);
    float* hQ = addr(env, jQ);
    float* hK = addr(env, jK);
    float* hV = addr(env, jV);
    float* hO = addr(env, jO);
    float* hS = addr(env, jS);
    float* hA = addr(env, jA);

    float* msQKT = addr(env, jMsQKT);
    float* msSM = addr(env, jMsSM);
    float* msAV = addr(env, jMsAV);

    size_t bQ = size_t(ctx->Tq) * ctx->d * sizeof(float);
    size_t bK = size_t(ctx->Tk) * ctx->d * sizeof(float);
    size_t bV = size_t(ctx->Tk) * ctx->d * sizeof(float);
    size_t bO = size_t(ctx->Tq) * ctx->d * sizeof(float);
    size_t bS = size_t(ctx->Tq) * ctx->Tk * sizeof(float);
    size_t bA = size_t(ctx->Tq) * ctx->Tk * sizeof(float);

    // Host→Device (persistent device buffers)
    cudaMemcpy(ctx->dQ, hQ, bQ, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->dK, hK, bK, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->dV, hV, bV, cudaMemcpyHostToDevice);

    int ret = attention_test_fp32(ctx->handle, ctx->Tq, ctx->Tk, ctx->d,
        ctx->dQ, ldQ, ctx->dK, ldK, ctx->dV, ldV,
        ctx->dO, ldO, ctx->dS, ldS, ctx->dA, ldA,
        /*enable_tf32=*/1, doCpuCheck,
        msQKT, msSM, msAV);

    // Device→Host
    cudaMemcpy(hO, ctx->dO, bO, cudaMemcpyDeviceToHost);
    cudaMemcpy(hS, ctx->dS, bS, cudaMemcpyDeviceToHost);
    cudaMemcpy(hA, ctx->dA, bA, cudaMemcpyDeviceToHost);

    return ret;
}