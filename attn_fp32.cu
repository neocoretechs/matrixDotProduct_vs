#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "com_neocoretechs_cublas_Gemm.h"
#include <crt/device_functions.h>

// ---------- Error helpers ----------
#define CHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); return -1; } } while(0)
#define NCHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); } } while(0)
#define CHECK_CUBLAS(x) do { cublasStatus_t st = (x); if (st != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS error %s at %s:%d\n", cublasGetStatusString(st), __FILE__, __LINE__); return -2; } } while(0)
#define PCHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); return nullptr; } } while(0)
#define GOCHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); goto fail; } } while(0)
#define PCHECK_CUBLAS(x) do { cublasStatus_t st = (x); if (st != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS error %s at %s:%d\n", cublasGetStatusString(st), __FILE__, __LINE__); return nullptr; } } while(0)
#define GOCHECK_CUBLAS(x) do { cublasStatus_t st = (x); if (st != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS error %s at %s:%d\n", cublasGetStatusString(st), __FILE__, __LINE__); goto fail; } } while(0)
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
/*
__global__ void convertQ8ToFloat(uint8_t* input, float* output, int blockSize, int typeSize, int headerBytes) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int blockIndex = index / blockSize;
    int withinBlock = index % blockSize;
    int blockOffset = blockIndex * typeSize;
    //if (index < 8) {
        printf("idx=%d b=%d i=%d base=%d off=%d  typeSize=%d hdr=%d\n",
            index, blockIndex, withinBlock, blockOffset, blockOffset + headerBytes + withinBlock, typeSize, headerBytes);
   //}
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
// FP16 (IEEE half) → float
__global__ void convertF16ToFloat(const uint8_t* __restrict__ input, float* __restrict__ output, int numElems, int elemStrideBytes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElems) return;
    const uint8_t* src = input + idx * elemStrideBytes;
    uint16_t hbits = (uint16_t)src[0] | ((uint16_t)src[1] << 8);
    __half h = *reinterpret_cast<const __half*>(&hbits);
    output[idx] = __half2float(h);
}

// BF16 → float (high 16 bits of f32)
__global__ void convertBF16ToFloat(const uint8_t* __restrict__ input, float* __restrict__ output, int numElems, int elemStrideBytes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElems) return;
    const uint8_t* src = input + idx * elemStrideBytes;
    uint16_t bf16 = (uint16_t)src[0] | ((uint16_t)src[1] << 8);
    uint32_t bits = ((uint32_t)bf16) << 16;
    output[idx] = static_cast<float>(bits);
}
*/
static inline float halfToFloat(uint16_t h) {
    uint16_t h_exp = (h & 0x7C00u);
    uint16_t h_sig = (h & 0x03FFu);
    uint32_t f_sgn = ((uint32_t)h & 0x8000u) << 16;
    uint32_t f_exp, f_sig;
    if (h_exp == 0) {
        if (h_sig == 0) {
            f_exp = 0;
            f_sig = 0;
        } else {
            // subnormal
            f_exp = (127 - 15 + 1) << 23;
            while ((h_sig & 0x0400u) == 0) {
                h_sig <<= 1;
                f_exp -= 1 << 23;
            }
            h_sig &= 0x03FFu;
            f_sig = (uint32_t)h_sig << 13;
        }
    } else if (h_exp == 0x7C00u) {
        f_exp = 0xFFu << 23;
        f_sig = (uint32_t)h_sig << 13;
    } else {
        f_exp = ((h_exp >> 10) + (127 - 15)) << 23;
        f_sig = (uint32_t)h_sig << 13;
    }
    uint32_t f_bits = f_sgn | f_exp | f_sig;
    float f;
    memcpy(&f, &f_bits, sizeof(f));
    return f;
}

// CPU conversion for Q8_0
std::vector<float> convertQ8ToFloat(const uint8_t* input, size_t len, int blockSize = 32, int headerBytes = 2) {
    int typeSize = blockSize + headerBytes;
    size_t blocks = len / typeSize;
    std::vector<float> out(blocks * blockSize);
    size_t pos = 0;
    for (size_t b = 0; b < blocks; ++b) {
        size_t base = b * typeSize;
        uint16_t scaleBits = (uint16_t)input[base]
            | ((uint16_t)input[base + 1] << 8);
        float scale = halfToFloat(scaleBits);

        for (int i = 0; i < blockSize; ++i) {
            int8_t q = *(int8_t*)&input[base + headerBytes + i];
            out[pos++] = (float)q * scale;
        }
    }
    return out;
}
// Convert and push Q8 buffer directly to device, return pointer to device buffer
float* toDeviceFloatQ8(const uint8_t* h_src, size_t len, int blockSize, int typeSize, int headerBytes) {
    int blocks = int(len / typeSize);
    size_t totalElems = size_t(blocks) * size_t(blockSize);
    size_t outBytes = totalElems * sizeof(float);
    // Try device alloc
    float* d_out = nullptr;
    PCHECK_CUDA(cudaMalloc((void**)&d_out, outBytes));
    // CPU decode into pinned host staging
    float* h_stage = nullptr;
    cudaError_t ce = cudaHostAlloc((void**)&h_stage, outBytes, cudaHostAllocDefault);
    if (ce != cudaSuccess) {
        std::cerr << "Pinned host alloc failed: " << cudaGetErrorString(ce) << "\n";
        h_stage = (float*)malloc(outBytes); // fallback
        if (!h_stage) {
            std::cerr << "Fallback malloc failed\n";
            cudaFree(d_out);
            exit(1);
        }
    }
    size_t pos = 0;
    for (int b = 0; b < blocks; ++b) {
        size_t base = size_t(b) * size_t(typeSize);
        uint16_t bits = (uint16_t)h_src[base] | ((uint16_t)h_src[base + 1] << 8);
        // half->float (use a robust helper)
        float scale = halfToFloat(bits);
        for (int i = 0; i < blockSize; ++i) {
            int8_t q = (int8_t)h_src[base + headerBytes + i];
            h_stage[pos++] = (float)q * scale;
        }
    }
    // Upload once
    PCHECK_CUDA(cudaMemcpy(d_out, h_stage, outBytes, cudaMemcpyHostToDevice));
    PCHECK_CUDA(cudaFreeHost(h_stage));
    return d_out;
}
// Convert and push Q4 buffer directly to device, return pointer to device buffer
float* toDeviceFloatQ4(const uint8_t * h_src, size_t len, int blockSize, int typeSize, int headerBytes) {
    int blocks = int(len / typeSize);
    size_t totalElems = size_t(blocks) * size_t(blockSize);
    size_t outBytes = totalElems * sizeof(float);
    // Try device alloc
    float* d_out = nullptr;
    PCHECK_CUDA(cudaMalloc((void**)&d_out, outBytes));
    // CPU decode into pinned host staging
    float* h_stage = nullptr;
    cudaError_t ce = cudaHostAlloc((void**)&h_stage, outBytes, cudaHostAllocDefault);
    if (ce != cudaSuccess) {
        std::cerr << "Pinned host alloc failed: " << cudaGetErrorString(ce) << "\n";
        h_stage = (float*)malloc(outBytes); // fallback
        if (!h_stage) {
            std::cerr << "Fallback malloc failed\n";
            cudaFree(d_out);
            exit(1);
        }
    }
    size_t pos = 0;
    for (int b = 0; b < blocks; ++b) {
        size_t base = size_t(b) * size_t(typeSize);
        uint16_t bits = (uint16_t)h_src[base] | ((uint16_t)h_src[base + 1] << 8);
        // half->float (use a robust helper)
        float scale = halfToFloat(bits);
        for (int i = 0; i < blockSize; ++i) {
            int modIndex = i % blockSize;
            int8_t q;
            if (modIndex < blockSize / 2)
                // Load quantized value (signed 8‑bit here)
                q = *(int8_t*)&h_src[base + headerBytes + modIndex] & 0x0F;
            else
                q = (int8_t)((*(int8_t*)&h_src[base + headerBytes + modIndex - blockSize / 2] >> 4) & 0x0F);
            h_stage[pos++] = (float)q * scale;
        }
    }
    // Upload once
    PCHECK_CUDA(cudaMemcpy(d_out, h_stage, outBytes, cudaMemcpyHostToDevice));
    PCHECK_CUDA(cudaFreeHost(h_stage));
    return d_out;
}
// Convert and push F16 buffer directly to device, return pointer to device buffer
float* toDeviceFloatF16(const uint8_t* h_src, size_t len, int typeSize) {
    int blocks = int(len / typeSize);
    size_t totalElems = size_t(blocks);
    size_t outBytes = totalElems * sizeof(float);
    // Try device alloc
    float* d_out = nullptr;
    PCHECK_CUDA(cudaMalloc((void**)&d_out, outBytes));
    // CPU decode into pinned host staging
    float* h_stage = nullptr;
    cudaError_t ce = cudaHostAlloc((void**)&h_stage, outBytes, cudaHostAllocDefault);
    if (ce != cudaSuccess) {
        std::cerr << "Pinned host alloc failed: " << cudaGetErrorString(ce) << "\n";
        h_stage = (float*)malloc(outBytes); // fallback
        if (!h_stage) {
            std::cerr << "Fallback malloc failed\n";
            cudaFree(d_out);
            exit(1);
        }
    }
    size_t pos = 0;
    for (int b = 0; b < blocks; ++b) {
        size_t base = size_t(b) * size_t(typeSize);
        uint16_t bits = (uint16_t)h_src[base] | ((uint16_t)h_src[base + 1] << 8);
        // half->float (use a robust helper)
        h_stage[pos++] = halfToFloat(bits);
    }
    // Upload once
    PCHECK_CUDA(cudaMemcpy(d_out, h_stage, outBytes, cudaMemcpyHostToDevice));
    PCHECK_CUDA(cudaFreeHost(h_stage));
    return d_out;
}
// Convert and push BF16 buffer directly to device, return pointer to device buffer
float* toDeviceFloatBF16(const uint8_t* h_src, size_t len, int typeSize) {
    int blocks = int(len / typeSize);
    size_t totalElems = size_t(blocks);
    size_t outBytes = totalElems * sizeof(float);
    // Try device alloc
    float* d_out = nullptr;
    PCHECK_CUDA(cudaMalloc((void**)&d_out, outBytes));
    // CPU decode into pinned host staging
    float* h_stage = nullptr;
    cudaError_t ce = cudaHostAlloc((void**)&h_stage, outBytes, cudaHostAllocDefault);
    if (ce != cudaSuccess) {
        std::cerr << "Pinned host alloc failed: " << cudaGetErrorString(ce) << "\n";
        h_stage = (float*)malloc(outBytes); // fallback
        if (!h_stage) {
            std::cerr << "Fallback malloc failed\n";
            cudaFree(d_out);
            exit(1);
        }
    }
    size_t pos = 0;
    for (int b = 0; b < blocks; ++b) {
        size_t base = size_t(b) * size_t(typeSize);
        uint16_t bits = (uint16_t)h_src[base] | ((uint16_t)h_src[base + 1] << 8);
        // half->float (use a robust helper)
        h_stage[pos++] = ((uint32_t)bits) << 16;
    }
    // Upload once
    PCHECK_CUDA(cudaMemcpy(d_out, h_stage, outBytes, cudaMemcpyHostToDevice));
    PCHECK_CUDA(cudaFreeHost(h_stage));
    return d_out;
}
/*
* Call out to SoftMax kernel
*/
static int softmax_rows_fp32(const float* d_S, float* d_A, int rows, int cols, int ldS, int ldA, cudaStream_t stream) {
    int threads = 128;
    int blocks = (rows + threads - 1) / threads;
    row_softmax_fp32 << <blocks, threads, 0, stream >> > (d_S, d_A, rows, cols, ldS, ldA);
    CHECK_CUDA(cudaGetLastError());
    return 0;
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
JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Attn_convertBufferToFloat(JNIEnv* env, jclass clazz, jobject byteBuffer, jint blockSize, jint typeSize, jint headerBytes, jint format) {
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
    // Allocate memory on the GPU
    int blocks = (int)((size_t)length / (size_t)typeSize);
    int totalElems = blocks * blockSize;
    size_t outBytes = (size_t)totalElems * sizeof(float);
    // Before launch (native side)
    fprintf(stderr, "***Allocating buffer:%lld format %d, %d floats (bytes=%zu) blocks=%d blockSize=%d len=%lld\n",
        buffer, format, totalElems, outBytes, blocks, blockSize, (long long)length);
    if (headerBytes != 2) { fprintf(stderr, "Bad headerBytes\n"); return -1; }
 
    if ((size_t)length % (size_t)typeSize != 0) {
        fprintf(stderr, "length %% typeSize != 0\n"); return -1;
    }
    CHECK_CUDA(cudaMalloc((void**)&d_output, outBytes));
    //dim3 threads(256);
    //dim3 grid((totalElems + threads.x - 1) / threads.x);
    //convertQ4ToFloat << <grid, threads >> > (buffer, d_output, blockSize, typeSize, headerBytes);
    std::vector<float> dout;
    switch(format) {
        case 0: // Q4_0
            // Conversion logic for q4_0 to float on the GPU
            if (typeSize != headerBytes + blockSize) {
                fprintf(stderr, "typeSize mismatch exp=%d got=%d\n", headerBytes + blockSize, typeSize);
                return -1;
            }
            return (jlong)(uintptr_t)toDeviceFloatQ4(buffer, outBytes, blockSize, typeSize, headerBytes);
        case 1: // Q8_0
            if (typeSize != headerBytes + blockSize) {
                fprintf(stderr, "typeSize mismatch exp=%d got=%d\n", headerBytes + blockSize, typeSize);
                return -1;
            }
           // convertQ8ToFloat << <grid, threads >> > (buffer, d_output, blockSize, typeSize, headerBytes);
           return (jlong)(uintptr_t)toDeviceFloatQ8(buffer, outBytes, blockSize, typeSize, headerBytes);
           //dout = convertQ8ToFloat(buffer, blockSize, typeSize, headerBytes);
           //break;
        case 2: // F16
            return (jlong)(uintptr_t)toDeviceFloatF16(buffer, outBytes, typeSize);
        case 3: // BF16
            return (jlong)(uintptr_t)toDeviceFloatBF16(buffer, outBytes, typeSize);
        default:
           // Handle F32
           // length is in bytes, so number of floats is length / sizeof(float)
           size_t numFloats = length / sizeof(float);
           size_t bytes = numFloats * sizeof(float);
           float* d_tensor = nullptr;
           cudaError_t err = cudaMalloc((void**)&d_tensor, bytes);
           if (err != cudaSuccess) {
                fprintf(stderr, "cudaMalloc F32 failed: %s\n", cudaGetErrorString(err));
                return 0;
           }
           // Copy raw floats from host ByteBuffer to device
           err = cudaMemcpy(d_tensor, buffer, bytes, cudaMemcpyHostToDevice);
           if (err != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy F32 failed: %s\n", cudaGetErrorString(err));
                cudaFree(d_tensor);
                return 0;
           }
           return (jlong)(uintptr_t)d_tensor;
           break;
    }
    // Copy the results from device to host (assuming you need results back on CPU)
    //float* h_output = (float*)malloc(numFloats * sizeof(float)); // Host pointer
    //CHECK_CUDA(cudaMemcpy(h_output, d_output, numFloats * sizeof(float), cudaMemcpyDeviceToHost));
    // Use h_output as needed...
    // Clean up
    //free(h_output);
    //cudaFree(d_output); // Free device memory
    float* d_tensor = nullptr;
    size_t bytes = dout.size() * sizeof(float);
    cudaMalloc((void**)&d_tensor, bytes);
    cudaMemcpy(d_tensor, dout.data(), bytes, cudaMemcpyHostToDevice);
    return (jlong)(uintptr_t)d_tensor;
}
__global__ void sdotKernel(const float* __restrict__ qBase, const float* __restrict__ kBase, int headSize, float* __restrict__ partial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.0f;
    if (idx < headSize) {
        float q = qBase[idx];
        float k = kBase[idx];
        v = q * k;
    }
    partial[idx] = v;
 }
   
__global__ void reduceKernel(const float* __restrict__ in, int n, float* out) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float x = (idx < n) ? in[idx] : 0.0f;
    s[tid] = x;
    __syncthreads();
    // tree reduction in shared memory
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

JNIEXPORT jfloat JNICALL Java_com_neocoretechs_cublas_Gemm_sdotSlice(JNIEnv* env, jclass clazz, jobject qBuf, jint qOffsetFloats, jobject kBuf, jint kOffsetFloats, jint headSize) {
    // Get base host pointers from direct ByteBuffers
    void* qHostBase = env->GetDirectBufferAddress(qBuf);
    void* kHostBase = env->GetDirectBufferAddress(kBuf);
    if (!qHostBase || !kHostBase) {
        // Not direct or null
        return NAN;
    }
    const float* qHost = reinterpret_cast<const float*>(qHostBase) + qOffsetFloats;
    const float* kHost = reinterpret_cast<const float*>(kHostBase) + kOffsetFloats;
    // Device buffers
    float* dQ = nullptr, * dK = nullptr, * dPartial = nullptr, * dOut = nullptr;
    size_t bytes = static_cast<size_t>(headSize) * sizeof(float);
    cudaError_t ce;
    ce = cudaMalloc((void**)&dQ, bytes); GOCHECK_CUDA(ce);
    ce = cudaMalloc((void**)&dK, bytes); GOCHECK_CUDA(ce);
    ce = cudaMalloc((void**)&dPartial, bytes); GOCHECK_CUDA(ce);
    ce = cudaMalloc((void**)&dOut, sizeof(float)); GOCHECK_CUDA(ce);
    // Initialize output accumulator to 0
    float zero = 0.0f;
    ce = cudaMemcpy(dOut, &zero, sizeof(float), cudaMemcpyHostToDevice); GOCHECK_CUDA(ce);
    // Copy only the slice needed
    ce = cudaMemcpy(dQ, qHost, bytes, cudaMemcpyHostToDevice); GOCHECK_CUDA(ce);
    ce = cudaMemcpy(dK, kHost, bytes, cudaMemcpyHostToDevice); GOCHECK_CUDA(ce);
    // Launch dot per element, then reduce
    int threads = 256;
    int blocks = (headSize + threads - 1) / threads;
    sdotKernel << <blocks, threads >> > (dQ, dK, headSize, dPartial);
    ce = cudaGetLastError(); GOCHECK_CUDA(ce);
    // Shared mem sized to threads
    reduceKernel << <blocks, threads, threads * sizeof(float) >> > (dPartial, headSize, dOut);
    ce = cudaGetLastError(); GOCHECK_CUDA(ce);
    // Final scalar result
    float result = NAN;
    ce = cudaMemcpy(&result, dOut, sizeof(float), cudaMemcpyDeviceToHost); GOCHECK_CUDA(ce);
    // Cleanup
    cudaFree(dQ); cudaFree(dK); cudaFree(dPartial); cudaFree(dOut);
    return result;
    fail:
        if (dQ) cudaFree(dQ);
        if (dK) cudaFree(dK);
        if (dPartial) cudaFree(dPartial);
        if (dOut) cudaFree(dOut);
       return NAN;
 } 
#ifdef __cplusplus
extern "C" {
#endif
EXPORT float sdotSlice(uint64_t handle, const float* q, const float* k, int headSize) {
    float* dQ = NULL, * dK = NULL, * dPartial = NULL, * dOut = NULL;
    size_t bytes = (size_t)headSize * sizeof(float);
    cudaError_t ce;
    ce = cudaMalloc((void**)&dQ, bytes); GOCHECK_CUDA(ce);
    ce = cudaMalloc((void**)&dK, bytes); GOCHECK_CUDA(ce);
    float zero = 0.0f;
    ce = cudaMemcpy(dQ, q, bytes, cudaMemcpyHostToDevice); GOCHECK_CUDA(ce);
    ce = cudaMemcpy(dK, k, bytes, cudaMemcpyHostToDevice); GOCHECK_CUDA(ce);
    /*
    int threads = 256;
    int blocks = (headSize + threads - 1) / threads;
    sdotKernel << <blocks, threads >> > (dQ, dK, headSize, dPartial);
    ce = cudaGetLastError(); if (ce) goto fail;
    reduceKernel << <blocks, threads, threads * sizeof(float) >> > (dPartial, headSize, dOut);
    */
    float result = -12345.0f;
    cublasSetPointerMode((cublasHandle_t)handle, CUBLAS_POINTER_MODE_HOST);
    //printf("sdotSlice size=%d q[0]=%f k[0]=%f\n", headSize, q[0], k[0]);
    GOCHECK_CUBLAS(cublasSdot((cublasHandle_t)handle, headSize, (const float*)dQ, 1, (const float*)dK, 1, &result));
    ce = cudaGetLastError(); if (ce) goto fail;
    //GOCHECK_CUDA(cudaDeviceSynchronize());
    //float result = NAN;
    //ce = cudaMemcpy(&result, dOut, sizeof(float), cudaMemcpyDeviceToHost); GOCHECK_CUDA(ce);
    cudaFree(dQ); cudaFree(dK);
    return result;
fail:
    if (dQ) cudaFree(dQ);
    if (dK) cudaFree(dK);
    return NAN;
}
EXPORT float sdotSliceQ8(uint64_t handle, const uint8_t* q, const float* k, int headSize, int blockSize, int index, int typeSize, int headerBytes) {
    float* dQ = NULL, * dK = NULL;
    size_t bytes = (size_t)headSize * sizeof(float);
    cudaError_t ce;
    ce = cudaMalloc((void**)&dQ, bytes); GOCHECK_CUDA(ce);
    ce = cudaMalloc((void**)&dK, bytes); GOCHECK_CUDA(ce);
    float zero = 0.0f;
    //ce = cudaMemcpy(dQ, q, bytes, cudaMemcpyHostToDevice); if (ce) goto fail;
    float* h_stage = NULL;
    ce = cudaMallocHost((void**)&h_stage, bytes); GOCHECK_CUDA(ce);
    //int blockIndex = index / GGMLType.Q8_0.getBlockSize();
    //int withinBlockIndex = index % GGMLType.Q8_0.getBlockSize();
    //int blockOffset = blockIndex * GGMLType.Q8_0.getTypeSize();
    //byte quant = readByte(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + withinBlockIndex);
    //float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
    //return quant * scale;
    size_t pos = 0;
    for (int b = index; b < (index+headSize); b++) {
        int blockIndex = b / blockSize;
        int withinBlockIndex = b % blockSize;
        int blockOffset = blockIndex * typeSize;
        int8_t quant = *(int8_t*)&q[blockOffset + headerBytes + withinBlockIndex];
        uint16_t bits = (uint16_t)q[blockOffset] | ((uint16_t)q[blockOffset + 1] << 8);
        float scale = halfToFloat(bits);
        //printf("b=%d blockIndex=%d quant=%d bits=%d scale=%.6f raw:%02x%02x bits:%04x\n",
        //   b, blockIndex, quant, bits, scale, q[blockOffset], q[blockOffset + 1], bits);
        h_stage[pos++] = (float)quant * scale;
    }
    ce = cudaMemcpy(dQ, h_stage, bytes, cudaMemcpyHostToDevice); GOCHECK_CUDA(ce);
    ce = cudaMemcpy(dK, k, bytes, cudaMemcpyHostToDevice); GOCHECK_CUDA(ce);
    /*
    int threads = 256;
    int threadBlocks = (headSize + threads - 1) / threads;
    sdotKernel << <threadBlocks, threads >> > (dQ, dK, headSize, dPartial);
    ce = cudaGetLastError(); GOCHECK_CUDA(ce);
    reduceKernel << <threadBlocks, threads, threads * sizeof(float) >> > (dPartial, headSize, dOut);
    */
    float result = -123435.0f;
    cublasSetPointerMode((cublasHandle_t)handle, CUBLAS_POINTER_MODE_HOST);
    //printf("sdotSliceQ8 size=%d q[0]=%f k[0]=%f\n", headSize, h_stage[0], k[0]);
    GOCHECK_CUBLAS(cublasSdot((cublasHandle_t)handle, headSize, (const float*)dQ, 1, (const float*)dK, 1, &result));
    ce = cudaGetLastError(); GOCHECK_CUDA(ce);
    //GOCHECK_CUDA(cudaDeviceSynchronize());
    //float result = NAN;
    //ce = cudaMemcpy(&result, dOut, sizeof(float), cudaMemcpyDeviceToHost); GOCHECK_CUDA(ce);
    cudaFree(dQ); cudaFree(dK); cudaFreeHost(h_stage);
    return result;
    fail:
        if (dQ) cudaFree(dQ);
        if (dK) cudaFree(dK);
        if(h_stage)cudaFreeHost(h_stage);
        return NAN;
 }
float sdotSliceQ4(uint64_t handle, const uint8_t* q, const float* k, int headSize, int blockSize, int index, int typeSize, int headerBytes) {
    float* dQ = NULL, * dK = NULL;
    size_t bytes = (size_t)headSize * sizeof(float);
    cudaError_t ce;
    ce = cudaMalloc((void**)&dQ, bytes); GOCHECK_CUDA(ce);
    ce = cudaMalloc((void**)&dK, bytes); GOCHECK_CUDA(ce);
    //ce = cudaMemcpy(dQ, q, bytes, cudaMemcpyHostToDevice); if (ce) goto fail;
    // from q to dQ device
    float* h_stage = NULL;
    ce = cudaMallocHost((void**)&h_stage, bytes); GOCHECK_CUDA(ce);
    //int blockIndex = index / GGMLType.Q4_0.getBlockSize();
    //int blockOffset = blockIndex * GGMLType.Q4_0.getTypeSize();
    //float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
    //byte quant;
    //int modIndex = index % GGMLType.Q4_0.getBlockSize();
    //if (modIndex < GGMLType.Q4_0.getBlockSize() / 2) {
    //    quant = (byte)(readByte(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + modIndex) & 0x0F);
    //}
    //else {
    //    quant = (byte)((readByte(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + modIndex - GGMLType.Q4_0.getBlockSize() / 2) >>> 4) & 0x0F);
    //}
    //quant -= 8;
    //return quant * scale;
    size_t pos = 0;
    for (int b = index; b < (index + headSize); b++) {
        int blockIndex = b / blockSize;
        int blockOffset = blockIndex * typeSize;
        uint16_t bits = (uint16_t)q[blockOffset] | ((uint16_t)q[blockOffset + 1] << 8);
        float scale = halfToFloat(bits);
        int8_t quant;
        int modIndex = b % blockSize;
        if (modIndex < blockSize / 2)
            quant = *(int8_t*)&q[blockOffset + headerBytes + modIndex] & 0x0F;
        else
            quant = (*(int8_t*)&q[blockOffset + headerBytes + modIndex - blockSize / 2] >> 4) & 0x0F;
        quant -= 8;
        h_stage[pos++] = (float)quant * scale;
    }
    ce = cudaMemcpy(dQ, h_stage, bytes, cudaMemcpyHostToDevice); GOCHECK_CUDA(ce);
    ce = cudaMemcpy(dK, k, bytes, cudaMemcpyHostToDevice); GOCHECK_CUDA(ce);
    /*
    int threads = 256;
    int threadBlocks = (headSize + threads - 1) / threads;
    sdotKernel << <threadBlocks, threads >> > (dQ, dK, headSize, dPartial);
    ce = cudaGetLastError(); GOCHECK_CUDA(ce);
    reduceKernel << <threadBlocks, threads, threads * sizeof(float) >> > (dPartial, headSize, dOut);
    ce = cudaGetLastError(); GOCHECK_CUDA(ce);
    */
    float result = -123435.0f;
    cublasSetPointerMode((cublasHandle_t)handle, CUBLAS_POINTER_MODE_HOST);
    //printf("sdotSliceQ4 size=%d q[0]=%f k[0]=%f\n", headSize, h_stage[0], k[0]);
    GOCHECK_CUBLAS(cublasSdot((cublasHandle_t)handle, headSize, (const float*)dQ, 1, (const float*)dK, 1, &result));
    ce = cudaGetLastError(); GOCHECK_CUDA(ce);
    cudaFree(dQ); cudaFree(dK); cudaFreeHost(h_stage);
    return result;
fail:
    if (dQ) cudaFree(dQ);
    if (dK) cudaFree(dK);
    if (h_stage)cudaFreeHost(h_stage);
    return NAN;
}
float sdotSliceF16(uint64_t handle, const uint8_t* q, const float* k, int headSize, int index, int typeSize) {
    float* dQ = NULL, * dK = NULL;
    size_t bytes = (size_t)headSize * sizeof(float);
    cudaError_t ce;
    ce = cudaMalloc((void**)&dQ, bytes); GOCHECK_CUDA(ce);
    ce = cudaMalloc((void**)&dK, bytes); GOCHECK_CUDA(ce);
    //ce = cudaMemcpy(dQ, q, bytes, cudaMemcpyHostToDevice); if (ce) goto fail;
    // from q to dQ device
    float* h_stage = NULL;
    ce = cudaMallocHost((void**)&h_stage, bytes); GOCHECK_CUDA(ce);
    size_t pos = 0;
    for (int b = 0; b < (index+headSize); ++b) {
        int blockOffset = b * typeSize;
        uint16_t bits = (uint16_t)q[blockOffset] | ((uint16_t)q[blockOffset + 1] << 8);
        h_stage[pos++] = halfToFloat(bits);
    }
    ce = cudaMemcpy(dQ, h_stage, bytes, cudaMemcpyHostToDevice); GOCHECK_CUDA(ce);
    ce = cudaMemcpy(dK, k, bytes, cudaMemcpyHostToDevice); GOCHECK_CUDA(ce);
    /*
    int threads = 256;
    int threadBlocks = (headSize + threads - 1) / threads;
    sdotKernel << <threadBlocks, threads >> > (dQ, dK, headSize, dPartial);
    ce = cudaGetLastError(); GOCHECK_CUDA(ce);
    reduceKernel << <threadBlocks, threads, threads * sizeof(float) >> > (dPartial, headSize, dOut);
    ce = cudaGetLastError(); GOCHECK_CUDA(ce);
    */
    float result = -123435.0f;
    cublasSetPointerMode((cublasHandle_t)handle, CUBLAS_POINTER_MODE_HOST);
    //printf("sdotSliceF16 size=%d q[0]=%f k[0]=%f\n", headSize, h_stage[0], k[0]);
    GOCHECK_CUBLAS(cublasSdot((cublasHandle_t)handle, headSize, (const float*)dQ, 1, (const float*)dK, 1, &result));
    ce = cudaGetLastError(); GOCHECK_CUDA(ce);
    cudaFree(dQ); cudaFree(dK); cudaFreeHost(h_stage);
    return result;
fail:
    if (dQ) cudaFree(dQ);
    if (dK) cudaFree(dK);
    if (h_stage)cudaFreeHost(h_stage);
    return NAN;
}
float sdotSliceBF16(uint64_t handle, const uint8_t* q, const float* k, int headSize, int index, int typeSize) {
    float* dQ = NULL, * dK = NULL;
    size_t bytes = (size_t)headSize * sizeof(float);
    cudaError_t ce;
    ce = cudaMalloc((void**)&dQ, bytes); GOCHECK_CUDA(ce);
    ce = cudaMalloc((void**)&dK, bytes); GOCHECK_CUDA(ce);
    //ce = cudaMemcpy(dQ, q, bytes, cudaMemcpyHostToDevice); if (ce) goto fail;
    // from q to dQ device
    float* h_stage = NULL;
    ce = cudaMallocHost((void**)&h_stage, bytes); GOCHECK_CUDA(ce);
    size_t pos = 0;
    for (int b = 0; b < (index + headSize); ++b) {
        int blockOffset = b * typeSize;
        uint16_t bits = (uint16_t)q[blockOffset] | ((uint16_t)q[blockOffset + 1] << 8);
        h_stage[pos++] = halfToFloat(bits);
    }
    ce = cudaMemcpy(dQ, h_stage, bytes, cudaMemcpyHostToDevice); GOCHECK_CUDA(ce);
    ce = cudaMemcpy(dK, k, bytes, cudaMemcpyHostToDevice); GOCHECK_CUDA(ce);
    /*
    int threads = 256;
    int threadBlocks = (headSize + threads - 1) / threads;
    sdotKernel << <threadBlocks, threads >> > (dQ, dK, headSize, dPartial);
    ce = cudaGetLastError(); GOCHECK_CUDA(ce);
    reduceKernel << <threadBlocks, threads, threads * sizeof(float) >> > (dPartial, headSize, dOut);
    ce = cudaGetLastError(); GOCHECK_CUDA(ce);
    */
    float result = -123435.0f;
    cublasSetPointerMode((cublasHandle_t)handle, CUBLAS_POINTER_MODE_HOST);
    //printf("sdotSliceBF16 size=%d q[0]=%f k[0]=%f\n", headSize, h_stage[0], k[0]);
    GOCHECK_CUBLAS(cublasSdot((cublasHandle_t)handle, headSize, (const float*)dQ, 1, (const float*)dK, 1, &result));
    ce = cudaGetLastError(); GOCHECK_CUDA(ce);
    cudaFree(dQ); cudaFree(dK); cudaFreeHost(h_stage);
    return result;
fail:
    if (dQ) cudaFree(dQ);
    if (dK) cudaFree(dK);
    if (h_stage)cudaFreeHost(h_stage);
    return NAN;
}
uint64_t cublasHandle() {
    cublasStatus_t status;
    cublasHandle_t handle = NULL;
    /* Initialize CUBLAS */
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasCreate CUBLAS initialization error %s\n", cublasGetStatusString(status));
        return NULL;
    }
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream(handle, stream);
    //cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    // once per handle
    // cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    // For FP16 GEMMs, use CUDA_R_16F inputs + tensor ops kernels
    // HOST mode means cuBLAS will synchronize the stream, write the value back into host memory, and only then return.
    // cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    // Device mode in this context means:
    // Keep data resident on the GPU as much as possible.
    // Don’t malloc / free / copy every call — allocate once, reuse buffers, and only transfer when you must.
    // Dequantize on device instead of staging on host.
    // A device kernel runs in parallel, writing directly into.That way you skip the pinned host buffer and the extra PCIe copy.
    // Fuse operations where possible.
    // For example, a custom kernel could dequantize Q8 and accumulate the dot product against K in one pass, avoiding even the intermediate .
    // Use cuBLAS / cuDNN primitives when they fit.
    // already device mode — it never brings data back to host until you copy the scalar result.
    //The kernel writes into that device buffer asynchronously, and you can copy it back later 
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    printf("Created CUBLAS handle %x\n",handle);
    return (uint64_t)handle;
}
void cublasHandleDestroy(uint64_t handle) {
    cublasDestroy((cublasHandle_t) handle);
}
int cudaGetMemInfo(size_t* free, size_t* total) {
    CHECK_CUDA(cudaMemGetInfo(free, total));
    return 0;
}

extern "C" __global__
void rmsnorm_fp32_rowmajor(const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int size, float eps) {
    // Block-wide reduction over x^2
    float acc = 0.f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float v = x[i];
        acc += v * v;
    }
    // Warp reduce
    for (int d = 16; d > 0; d >>= 1) acc += __shfl_down_sync(0xffffffff, acc, d);

    // Shared reduction across warps
    __shared__ float warpSum[32]; // supports up to 1024 threads
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) warpSum[wid] = acc;
    __syncthreads();

    float sumsq = 0.f;
    if (threadIdx.x < (blockDim.x >> 5)) sumsq = warpSum[threadIdx.x];
    __syncthreads();

    // Final fold by thread 0
    if (threadIdx.x == 0) {
        for (int w = 1; w < (blockDim.x >> 5); ++w) sumsq += warpSum[w];
        float inv = rsqrtf(sumsq / size + eps);
        // Write inv to shared for broadcast
        warpSum[0] = inv;
    }
    __syncthreads();
    float inv = warpSum[0];

    // Normalize and scale
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        out[i] = weight[i] * (inv * x[i]);
    }
}
extern "C" __global__
void qk_scores_fp32_rowmajor(
    const float* __restrict__ Q,       // [nHeads * headSize] for this token
    const float* __restrict__ Kcache,  // global K cache, row-major
    float* __restrict__ S,             // [nHeads * contextLength] scores for this token
    int nHeads,
    int headSize,
    int contextLength,        // stride between rows (attOffset logic)
    int kvDim,                // stride per timestep in K/V cache
    int kvMul,                // head group divisor
    int tMaxInclusive,        // position + token
    float invSqrtHeadSize,
    size_t layerBaseOffset    // base offset for curLayer in Kcache
) {
    int h = blockIdx.x;                   // head index
    int t = blockIdx.y * blockDim.y + threadIdx.y; // timestep index
    if (h >= nHeads || t > tMaxInclusive) return;

    const float* q_h = Q + h * headSize;
    const float* k_th = Kcache + layerBaseOffset + t * kvDim + (h / kvMul) * headSize;
    float* s_row = S + h * contextLength;

    // Parallel dot reduction over headSize
    float acc = 0.f;
    for (int i = threadIdx.x; i < headSize; i += blockDim.x) {
        acc += q_h[i] * k_th[i];
    }
    // Warp reduce
    for (int delta = 16; delta > 0; delta >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, delta);
    }
    // Shared reduce (one partial per warp)
    __shared__ float warpSum[32]; // up to 32 warps per block
    int warpId = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) warpSum[warpId] = acc;
    __syncthreads();

    if (threadIdx.x < (blockDim.x >> 5)) {
        float dot = warpSum[threadIdx.x];
        // fold remaining warps
        for (int w = threadIdx.x + 1; w < (blockDim.x >> 5); ++w) dot += warpSum[w];
        if (threadIdx.x == 0) s_row[t] = dot * invSqrtHeadSize;
    }
}
extern "C" __global__
void av_weighted_sum_fp32_rowmajor(
    const float* __restrict__ A,       // [nHeads * contextLength] attention weights
    const float* __restrict__ Vcache,  // global V cache, row-major
    float* __restrict__ Xb,            // [nHeads * headSize] output for this token
    int nHeads,
    int headSize,
    int contextLength,
    int kvDim,
    int kvMul,
    int tMaxInclusive,
    size_t layerBaseOffset
) {
    int h = blockIdx.x;                          // head
    int i = blockIdx.y * blockDim.x + threadIdx.x; // element in head vector
    if (h >= nHeads || i >= headSize) return;

    const float* a_h = A + h * contextLength;
    float* xb_h = Xb + h * headSize;

    float acc = 0.f;
#pragma unroll 4
    for (int t = 0; t <= tMaxInclusive; ++t) {
        const float* v_th = Vcache + layerBaseOffset + t * kvDim + (h / kvMul) * headSize;
        acc += a_h[t] * v_th[i];
    }
    xb_h[i] = acc;
}
extern "C" void launch_rmsnorm_fp32_rowmajor(const float* x, const float* weight, float* out, int size, float eps) {
    // One block is enough for vector sizes up to a few thousand; tune if needed
    int threads = 256;
    rmsnorm_fp32_rowmajor << <1, threads >> > (x, weight, out, size, eps);
    NCHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();
}
extern "C" void launch_av_weighted_sum_fp32_rowmajor(const float* Q, const float* K, float* S, int nHeads, int headSize, int contextLength, int kvDim, int kvMul, int tMaxInclusive, size_t layerBaseOffset) {
    int threads = 256;
    av_weighted_sum_fp32_rowmajor << <1, threads >> > (Q, K, S, nHeads, headSize, contextLength, kvDim, kvMul, tMaxInclusive, layerBaseOffset);
    cudaDeviceSynchronize();
}
extern "C" void launch_qk_scores_fp32_rowmajor(const float* Q, const float* K, float* S, int nHeads, int headSize, int contextLength, int kvDim, int kvMul, int tMaxInclusive, float invSqrtHeadSize, size_t layerBaseOffset) {
    int threads = 256;
    qk_scores_fp32_rowmajor << <1, threads >> > (Q, K, S, nHeads, headSize, contextLength, kvDim, kvMul, tMaxInclusive, invSqrtHeadSize, layerBaseOffset);
    cudaDeviceSynchronize();
}
extern "C" void launch_row_softmax_fp32(const float* S, float* A, int rows, int cols, int ldS, int ldA) {
    int threads = 256;
    row_softmax_fp32 << <1, threads >> > (S, A, rows, cols, ldS, ldA);
    cudaDeviceSynchronize();
}
#ifdef __cplusplus
}
#endif