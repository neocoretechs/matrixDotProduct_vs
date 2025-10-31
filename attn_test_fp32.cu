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

__global__ void convertQ8ToFloat(uint8_t* input, float* output,int blockSize, int typeSize, int headerBytes) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int blockIndex = index / blockSize;
    int withinBlock = index % blockSize;
    int blockOffset = blockIndex * typeSize;
    // Decode scale from header (assume FP16 at offset 0)
    uint16_t scaleBits = (uint16_t)input[blockOffset] |
        ((uint16_t)input[blockOffset + 1] << 8);
    __half hscale = *reinterpret_cast<__half*>(&scaleBits);
    float scale = __half2float(hscale);
    // Load quantized value (signed 8‑bit here)
    int8_t q = *(int8_t*)&input[blockOffset + headerBytes + withinBlock];
    output[index] = (float)q * scale;
}

__global__ void convertQ4ToFloat(uint8_t* input, float* output, int blockSize, int typeSize, int headerBytes) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int blockIndex = index / blockSize;
    int blockOffset = blockIndex * typeSize;
    // Decode scale from header (assume FP16 at offset 0)
    uint16_t scaleBits = (uint16_t)input[blockOffset] |
        ((uint16_t)input[blockOffset + 1] << 8);
    __half hscale = *reinterpret_cast<__half*>(&scaleBits);
    float scale = __half2float(hscale);
    int modIndex = index % blockSize;
    int8_t q;
    if (modIndex < blockSize / 2)
        // Load quantized value (signed 8‑bit here)
        q = *(int8_t*)&input[blockOffset + headerBytes + modIndex] & 0x0F;
    else
        q = (int8_t)((*(int8_t*)&input[blockOffset + headerBytes + modIndex - blockSize/2] >> 4) & 0x0F);
    output[index] = (float)q * scale;
}

static int softmax_rows_fp32(const float* d_S, float* d_A, int rows, int cols, int ldS, int ldA, cudaStream_t stream) {
    int threads = 128;
    int blocks = (rows + threads - 1) / threads;
    row_softmax_fp32 << <blocks, threads, 0, stream >> > (d_S, d_A, rows, cols, ldS, ldA);
    CHECK_CUDA(cudaGetLastError());
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
int attention_test_fp32(cublasHandle_t handle, cudaStream_t stream,
    int Tq, int Tk, int d, const float* d_Q, int ldQ, const float* d_K, int ldK, const float* d_V, int ldV,
    float* d_O, int ldO, float* d_S, int ldS, float* d_A, int ldA,
    int do_cpu_check, float* out_ms_qkt, float* out_ms_softmax, float* out_ms_av) {

    if (Tq <= 0 || Tk <= 0 || d <= 0) { fprintf(stderr, "Invalid sizes\n"); return -10; }
    if (ldQ != d || ldK != d || ldV != d || ldO != d || ldS != Tk || ldA != Tk) {
        fprintf(stderr, "Leading dims mismatch (expect row-major)\n"); return -11;
    }

    cudaEvent_t e0, e1, e2, e3, e4;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));
    CHECK_CUDA(cudaEventCreate(&e2));
    CHECK_CUDA(cudaEventCreate(&e3));
    CHECK_CUDA(cudaEventCreate(&e4));

    // QK^T scaled → S
    float alpha_scores = 1.0f / std::sqrt((float)d);
    float beta0 = 0.0f;

    CHECK_CUDA(cudaEventRecord(e0, stream));
    CHECK_CUBLAS(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_T,
        Tq, Tk, d,
        &alpha_scores,
        d_Q, ldQ,
        d_K, ldK,
        &beta0,
        d_S, ldS
    ));
    CHECK_CUDA(cudaEventRecord(e1, stream));

    // Softmax rows
    int rc = softmax_rows_fp32(d_S, d_A, Tq, Tk, ldS, ldA, stream);
    if (rc) { fprintf(stderr, "Softmax kernel failed\n"); return rc; }
    CHECK_CUDA(cudaEventRecord(e2, stream));

    // AV → O (no extra scale)
    float alpha_av = 1.0f, beta_av = 0.0f;
    CHECK_CUBLAS(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        Tq, d, Tk,
        &alpha_av,
        d_A, ldA,
        d_V, ldV,
        &beta_av,
        d_O, ldO
    ));
    CHECK_CUDA(cudaEventRecord(e3, stream));

    CHECK_CUDA(cudaEventRecord(e4, stream));
    CHECK_CUDA(cudaEventSynchronize(e4));

    float ms_qkt = 0, ms_sm = 0, ms_av = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_qkt, e0, e1));
    CHECK_CUDA(cudaEventElapsedTime(&ms_sm, e1, e2));
    CHECK_CUDA(cudaEventElapsedTime(&ms_av, e2, e3));

    if (out_ms_qkt)     *out_ms_qkt = ms_qkt;
    if (out_ms_softmax) *out_ms_softmax = ms_sm;
    if (out_ms_av)      *out_ms_av = ms_av;

    if (do_cpu_check && Tq <= 2 && Tk <= 256 && d <= 128) {
        std::vector<float> hQ(Tq * d), hK(Tk * d), hV(Tk * d), hO(Tq * d), hO_cpu(Tq * d);
        CHECK_CUDA(cudaMemcpy(hQ.data(), d_Q, sizeof(float) * Tq * d, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hK.data(), d_K, sizeof(float) * Tk * d, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hV.data(), d_V, sizeof(float) * Tk * d, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hO.data(), d_O, sizeof(float) * Tq * d, cudaMemcpyDeviceToHost));
        cpu_attention_fp32(hQ.data(), hK.data(), hV.data(), hO_cpu.data(), Tq, Tk, d);
        int mismatches = 0;
        for (int i = 0; i < std::min(16, Tq * d); ++i) {
            float a = hO[i], b = hO_cpu[i];
            float diff = std::abs(a - b);
            float rel = diff / std::max(1e-6f, std::abs(b));
            if (rel > 1e-3f) ++mismatches;
        }
        if (mismatches) fprintf(stderr, "CPU vs GPU mismatch count=%d (rel tol 1e-3)\n", mismatches);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(e2); cudaEventDestroy(e3); cudaEventDestroy(e4);
    return 0;
}
extern "C" int attention_batched_heads(
    cublasHandle_t handle, cudaStream_t stream,
    int Tq, int Tk, int d, int H,
    const float* d_Q_all, int ldQ,    // ldQ = H*d
    const float* d_K_all, int ldK,    // ldK = H*d
    const float* d_V_all, int ldV,    // ldV = H*d
    float* d_O_all, int ldO,          // ldO = H*d
    float* d_S_head, int ldS,         // workspace [Tq x Tk] per head (ldS = Tk)
    float* d_A_head, int ldA,         // workspace [Tq x Tk] per head (ldA = Tk)
    float* out_ms_qkt, float* out_ms_sm, float* out_ms_av)
{
    // Example preflight
    if (d <= 0 || H <= 0 || Tq <= 0 || Tk <= 0) return -91;            // invalid dims
    if (ldQ != H * d || ldK != H * d || ldV != H * d || ldO != H * d) return -92; // wrong leading dims
    if (ldS != Tk || ldA != Tk) return -93;                             // wrong S/A stride
    if (!handle || !stream) return -94;                                 // bad ctx
    float alpha_scores = 1.0f / std::sqrt((float)d), beta0 = 0.0f;
    float alpha_av = 1.0f, beta_av = 0.0f;

    // Optional: events for aggregated timing
    float ms_qkt_total = 0, ms_sm_total = 0, ms_av_total = 0;

    for (int h = 0; h < H; ++h) {
        const float* d_Q = d_Q_all + h * d;          // column offset in packed row-major
        const float* d_K = d_K_all + h * d;
        const float* d_V = d_V_all + h * d;
        float* d_O = d_O_all + h * d;

        // Q [Tq x d] is stored with ldQ_all = H*d; to address head h:
        // use pointer arithmetic plus appropriate lda/ldb with column-major mapping
        // GEMM: S = Q K^T
        CHECK_CUBLAS(cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_T,
            Tq, Tk, d,
            &alpha_scores,
            d_Q, ldQ,   // ldQ = H*d
            d_K, ldK,   // ldK = H*d
            &beta0,
            d_S_head, ldS
        ));

        // softmax
        int rc = softmax_rows_fp32(d_S_head, d_A_head, Tq, Tk, ldS, ldA, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        if (rc) return rc;

        // O = A V
        CHECK_CUBLAS(cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            Tq, d, Tk,
            &alpha_av,
            d_A_head, ldA,
            d_V, ldV,   // ldV = H*d
            &beta_av,
            d_O, ldO    // ldO = H*d
        ));
    }

    if (out_ms_qkt) *out_ms_qkt = ms_qkt_total;
    if (out_ms_sm)  *out_ms_sm = ms_sm_total;
    if (out_ms_av)  *out_ms_av = ms_av_total;
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
    cudaStream_t stream;
    int Tq, Tk, d, H;
    // optional: persistent device buffers
    float* dQ, * dK, * dV, * dO, * dS, * dA;
};

JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Attn_initContext(
    JNIEnv* env, jclass clazz, jlong handle, jint Tq, jint Tk, jint d, jint H) {
    // Best-effort: enable TF32 tensor ops mode (ignored on non-Ampere/Hopper)
    CHECK_CUBLAS(cublasSetMathMode((cublasHandle_t)handle, CUBLAS_TF32_TENSOR_OP_MATH));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream((cublasHandle_t)handle, stream));
    auto* ctx = new Ctx{ (cublasHandle_t)handle, stream, Tq, Tk, d, H };
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
    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT void JNICALL Java_com_neocoretechs_cublas_Attn_freeContext(JNIEnv* env, jclass clazz, jlong h) {
    auto* ctx = reinterpret_cast<Ctx*>(h);
    if (!ctx) return;
    cudaFree(ctx->dQ); cudaFree(ctx->dK); cudaFree(ctx->dV);
    cudaFree(ctx->dO); cudaFree(ctx->dS); cudaFree(ctx->dA);
    cudaStreamDestroy(ctx->stream);
    delete ctx;
}

static float* addr(JNIEnv* env, jobject buf) {
    return static_cast<float*>(env->GetDirectBufferAddress(buf));
}

JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Attn_attentionFp32(JNIEnv * env, jclass clazz, jlong h,
    jobject jQ, jint ldQ, jobject jK, jint ldK, jobject jV, jint ldV,
    jobject jO, jint ldO, 
    jobject jMsQKT, jobject jMsSM, jobject jMsAV) {

    auto* ctx = reinterpret_cast<Ctx*>(h);
    float* hQ = addr(env, jQ);
    float* hK = addr(env, jK);
    float* hV = addr(env, jV);
    float* hO = addr(env, jO);
    //float* hS = addr(env, jS);
    //float* hA = addr(env, jA);

    float* msQKT = addr(env, jMsQKT);
    float* msSM = addr(env, jMsSM);
    float* msAV = addr(env, jMsAV);

    size_t bQ = size_t(ctx->Tq) * ctx->d * sizeof(float);
    size_t bK = size_t(ctx->Tk) * ctx->d * sizeof(float);
    size_t bV = size_t(ctx->Tk) * ctx->d * sizeof(float);
    size_t bO = size_t(ctx->Tq) * ctx->d * sizeof(float);
    //size_t bS = size_t(ctx->Tq) * ctx->Tk * sizeof(float);
    //size_t bA = size_t(ctx->Tq) * ctx->Tk * sizeof(float);

    // Host→Device (persistent device buffers)
    cudaMemcpy(ctx->dQ, hQ, bQ, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->dK, hK, bK, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->dV, hV, bV, cudaMemcpyHostToDevice);

    int ret = attention_batched_heads(ctx->handle, ctx->stream, 
        ctx->Tq, ctx->Tk, ctx->d, ctx->H,
        ctx->dQ, ldQ, ctx->dK, ldK, ctx->dV, ldV,
        ctx->dO, ldO, ctx->dS, ctx->Tk, ctx->dA, ctx->Tk,
        msQKT, msSM, msAV);

    // Device→Host
    CHECK_CUDA(cudaMemcpy(hO, ctx->dO, bO, cudaMemcpyDeviceToHost));
    //cudaMemcpy(hS, ctx->dS, bS, cudaMemcpyDeviceToHost);
    //cudaMemcpy(hA, ctx->dA, bA, cudaMemcpyDeviceToHost);

    return ret;
}
/*
* Function to convert GGUF quantized types to float using CUDA
* 0 - Q4_0
* 1 - Q8_0
* 2 - F16
* 3 - B16
*/
JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Attn_convertBufferToFloat(JNIEnv* env, jobject obj, jobject byteBuffer, jint blockSize, jint typeSize, jint headerBytes, jint format) {
    // Get direct buffer address
    uint8_t* buffer = static_cast<uint8_t*>(env->GetDirectBufferAddress(byteBuffer));
    if(buffer == NULL) {
        // Handle error: buffer is not direct
        fprintf(stderr, "convertBufferToFloat -> ByteBuffer is not direct!\n");
        return -100;
    }
    size_t length = env->GetDirectBufferCapacity(byteBuffer);
    if (length % typeSize != 0) {
        fprintf(stderr, "length of buffer not a multiple of typeSize!\n");
        return -101;
    }
    float* d_output; // Device pointer
    size_t numFloats = length / typeSize;
    // Allocate memory on the GPU
    CHECK_CUDA(cudaMalloc((void**)&d_output, numFloats * sizeof(float)));
    int numBlocks = numFloats / blockSize; // how many quant blocks
    int totalElems = numBlocks * blockSize;
    dim3 threads(256);
    dim3 grid((totalElems + threads.x - 1) / threads.x);
    switch(format) {
        case 0: // Q4_0
            // Conversion logic for q4_0 to float on the GPU
            convertQ4ToFloat << <grid, threads >> > (buffer, d_output, blockSize, typeSize, headerBytes);
            break;
        case 1: // Q8_0
            convertQ8ToFloat << <grid, threads >> > (buffer, d_output, blockSize, typeSize, headerBytes);
            break;
        case 2: // F16
            // Launch kernel for F16 conversion
            break;
        case 3: // BF16
            // Launch kernel for BF16 conversion
            break;
        default:
            // Handle unsupported format
            break;
    }
    // Copy the results from device to host (assuming you need results back on CPU)
    //float* h_output = (float*)malloc(numFloats * sizeof(float)); // Host pointer
    //CHECK_CUDA(cudaMemcpy(h_output, d_output, numFloats * sizeof(float), cudaMemcpyDeviceToHost));
    // Use h_output as needed...
    // Clean up
    //free(h_output);
    //cudaFree(d_output); // Free device memory
    return (jlong)d_output;
}
