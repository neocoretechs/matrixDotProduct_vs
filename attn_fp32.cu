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

// Q8 block: header has FP16 scale at offset 0, optional zp at offset 2
__device__ inline float loadQ8(const uint8_t* base, int blockSize, int typeSize, int headerBytes, int index) {
    int blockIndex = index / blockSize;
    int withinBlock = index % blockSize;
    int blockOffset = blockIndex * typeSize;
    // scale from header
    const uint8_t* hdr = base + blockOffset;
    uint16_t scaleBits = (uint16_t)hdr[0] | ((uint16_t)hdr[1] << 8);
    __half hscale = __ushort_as_half(scaleBits);   // reinterpret raw bits
    float scale = __half2float(hscale);
    // quantized value
    const uint8_t* payload = base + blockOffset + headerBytes;
    int8_t q = reinterpret_cast<const int8_t*>(payload)[withinBlock];
    return (float)(q) * scale;
}

// Q4 block: header has FP16 scale at offset 0
__device__ inline float loadQ4(const uint8_t* base, int blockSize, int typeSize, int headerBytes, int index) {
    int blockIndex = index / blockSize;
    int withinBlock = index % blockSize;
    int blockOffset = blockIndex * typeSize;
    // scale from header
    const uint8_t* hdr = base + blockOffset;
    uint16_t scaleBits = (uint16_t)hdr[0] | ((uint16_t)hdr[1] << 8);
    __half hscale = __ushort_as_half(scaleBits);   // reinterpret raw bits
    float scale = __half2float(hscale);
    // payload
    const uint8_t* payload = base + blockOffset + headerBytes;
    int byteIdx = withinBlock >> 1;
    uint8_t byteVal = payload[byteIdx];
    uint8_t nibble = (withinBlock & 1) ? (byteVal >> 4) & 0x0F : byteVal & 0x0F;
    int signed4 = (int)nibble - 8; // map 0..15 → -8..7
    return (float)signed4 * scale;
}

// FP16 element at stride
__device__ inline float loadF16(const uint8_t* base, int idx, int strideBytes) {
    const uint8_t* src = base + idx * strideBytes;
    uint16_t hbits = (uint16_t)src[0] | ((uint16_t)src[1] << 8);
    __half h = __ushort_as_half(hbits);   // reinterpret raw bits
    return __half2float(h);
}

// BF16 element at stride
__device__ inline float loadBF16(const uint8_t* base, int idx, int strideBytes) {
    const uint8_t* src = base + idx * strideBytes;
    uint16_t bf16 = (uint16_t)src[0] | ((uint16_t)src[1] << 8);
    uint32_t fbits = ((uint32_t)bf16) << 16;
    float f;
    memcpy(&f, &fbits, sizeof(float));
    return f;
}

/*__global__ void dotProduct(float* A, float* B, float* result, int N) {
    // Shared memory for accumulating partial results
    __shared__ float temp[256];
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;

    // Compute the dot product for each thread
    for (int i = threadId; i < N; i += gridDim.x * blockDim.x) {
        sum += A[i] * B[i];
    }

    // Store the result in shared memory
    temp[threadIdx.x] = sum;

    // Synchronize threads
    __syncthreads();

    // Reduce the partial results to a single value
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            temp[threadIdx.x] += temp[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write the block result to the global result array
    if (threadIdx.x == 0) {
        atomicAdd(result, temp[0]);
    }*/
extern "C" __global__ void dotProduct(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ result, int N) {
        // Use dynamic shared memory sized to blockDim.x
        extern __shared__ float smem[];
        int tid = threadIdx.x;
        int gid = blockIdx.x * blockDim.x + tid;
        int stride = blockDim.x * gridDim.x;

        float sum = 0.f;
        for (int i = gid; i < N; i += stride) {
            sum += A[i] * B[i];
        }
        smem[tid] = sum;
        __syncthreads();
        // Reduce within block
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (tid < s) smem[tid] += smem[tid + s];
            __syncthreads();
        }
        // One atomicAdd per block
        if (tid == 0) atomicAdd(result, smem[0]);
}

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
    for (int b = 0; b < blocks; b++) {
        int blockIndex = b / blockSize;
        int withinBlockIndex = b % blockSize;
        int blockOffset = blockIndex * typeSize;
        int8_t quant = *(int8_t*)&input[blockOffset + headerBytes + withinBlockIndex];
        uint16_t bits = (uint16_t)input[blockOffset] | ((uint16_t)input[blockOffset + 1] << 8);
        float scale = halfToFloat(bits);
        //printf("b=%d blockIndex=%d quant=%d bits=%d scale=%.6f raw:%02x%02x bits:%04x\n",
        //   b, blockIndex, quant, bits, scale, q[blockOffset], q[blockOffset + 1], bits);
        out.push_back((float)quant * scale);
    }
    return out;
}
// Convert and push Q8 buffer directly to device, return pointer to device buffer
float* toDeviceFloatQ8(const uint8_t* q, size_t headSize, int index, int blockSize, int typeSize, int headerBytes) {
    float* dQ = NULL;
    size_t bytes = (size_t)headSize * sizeof(float);
    cudaError_t ce;
    ce = cudaMalloc((void**)&dQ, bytes); GOCHECK_CUDA(ce);

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
    for (int b = index; b < (index + headSize); b++) {
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
    GOCHECK_CUDA(cudaFree(dQ)); GOCHECK_CUDA(cudaFreeHost(h_stage));
    return dQ;
fail:
    if (dQ) cudaFree(dQ);
    return NULL;
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

#ifdef __cplusplus
extern "C" {
#endif
cudaError_t launchDotProductKernel(float* d_A, float* d_B, float* result, int N) {
        float* d_result;
        cudaError_t ce = cudaMalloc((void**)&d_result, sizeof(float));
        if (ce) return ce;
        ce = cudaMemset(d_result, 0, sizeof(float));
        if (ce) {
            cudaFree(d_result);
            return ce;
        }
        // Launch kernel with an appropriate grid and block size
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        dotProduct << <numBlocks, blockSize >> > (d_A, d_B, d_result, N);
        ce = cudaGetLastError();
        if (ce) {
            cudaFree(d_result);
            return ce;
        }
        ce = cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        return ce;
}
EXPORT float sdotSliceCuBLAS(uint64_t handle, const float* q, const float* k, int headSize) {
    float* dQ = NULL, * dK = NULL;
    size_t bytes = (size_t)headSize * sizeof(float);
    cudaError_t ce;
    ce = cudaMalloc((void**)&dQ, bytes); GOCHECK_CUDA(ce);
    ce = cudaMalloc((void**)&dK, bytes); GOCHECK_CUDA(ce);

    ce = cudaMemcpy(dQ, q, bytes, cudaMemcpyHostToDevice); GOCHECK_CUDA(ce);
    ce = cudaMemcpy(dK, k, bytes, cudaMemcpyHostToDevice); GOCHECK_CUDA(ce);

    float result = -12345.0f;
    cublasSetPointerMode((cublasHandle_t)handle, CUBLAS_POINTER_MODE_HOST);
    //printf("sdotSlice size=%d q[0]=%f k[0]=%f\n", headSize, q[0], k[0]);
    GOCHECK_CUBLAS(cublasSdot((cublasHandle_t)handle, headSize, (const float*)dQ, 1, (const float*)dK, 1, &result));
    cudaFree(dQ); cudaFree(dK);
    return result;
fail:
    if (dQ) cudaFree(dQ);
    if (dK) cudaFree(dK);
    return NAN;
}
EXPORT float sdotSlice(const float* q, const float* k, int headSize) {
    float* dQ = NULL, * dK = NULL;
    size_t bytes = (size_t)headSize * sizeof(float);
    cudaError_t ce;
    ce = cudaMalloc((void**)&dQ, bytes); GOCHECK_CUDA(ce);
    ce = cudaMalloc((void**)&dK, bytes); GOCHECK_CUDA(ce);
 
    ce = cudaMemcpy(dQ, q, bytes, cudaMemcpyHostToDevice); GOCHECK_CUDA(ce);
    ce = cudaMemcpy(dK, k, bytes, cudaMemcpyHostToDevice); GOCHECK_CUDA(ce);
    
    float result = -12345.0f;
    //cublasSetPointerMode((cublasHandle_t)handle, CUBLAS_POINTER_MODE_HOST);
    //printf("sdotSlice size=%d q[0]=%f k[0]=%f\n", headSize, q[0], k[0]);
    //GOCHECK_CUBLAS(cublasSdot((cublasHandle_t)handle, headSize, (const float*)dQ, 1, (const float*)dK, 1, &result));
    GOCHECK_CUDA(launchDotProductKernel((float*)dQ, (float*)dK, &result, headSize));
    cudaFree(dQ); cudaFree(dK);
    return result;
fail:
    if (dQ) cudaFree(dQ);
    if (dK) cudaFree(dK);
    return NAN;
}
EXPORT float sdotSliceQ8(const uint8_t* q, const float* k, int headSize, int blockSize, int index, int typeSize, int headerBytes) {
    float* dQ = NULL, * dK = NULL;
    size_t bytes = (size_t)headSize * sizeof(float);
    cudaError_t ce;
    ce = cudaMalloc((void**)&dQ, bytes); GOCHECK_CUDA(ce);
    ce = cudaMalloc((void**)&dK, bytes); GOCHECK_CUDA(ce);

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

    float result = -123435.0f;
    //cublasSetPointerMode((cublasHandle_t)handle, CUBLAS_POINTER_MODE_HOST);
    //printf("sdotSliceQ8 size=%d q[0]=%f k[0]=%f\n", headSize, h_stage[0], k[0]);
    //GOCHECK_CUBLAS(cublasSdot((cublasHandle_t)handle, headSize, (const float*)dQ, 1, (const float*)dK, 1, &result));
    GOCHECK_CUDA(launchDotProductKernel((float*)dQ, (float*)dK, &result, headSize));
    cudaFree(dQ); cudaFree(dK);  cudaFreeHost(h_stage);
    return result;
    fail:
        if (dQ) cudaFree(dQ);
        if (dK) cudaFree(dK);
        if(h_stage)cudaFreeHost(h_stage);
        return NAN;
 }
float sdotSliceQ4(const uint8_t* q, const float* k, int headSize, int blockSize, int index, int typeSize, int headerBytes) {
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
    //cublasSetPointerMode((cublasHandle_t)handle, CUBLAS_POINTER_MODE_HOST);
    //printf("sdotSliceQ4 size=%d q[0]=%f k[0]=%f\n", headSize, h_stage[0], k[0]);
    //GOCHECK_CUBLAS(cublasSdot((cublasHandle_t)handle, headSize, (const float*)dQ, 1, (const float*)dK, 1, &result));
    GOCHECK_CUDA(launchDotProductKernel((float*)dQ, (float*)dK, &result, headSize));
    ce = cudaGetLastError(); GOCHECK_CUDA(ce);
    cudaFree(dQ); cudaFree(dK); cudaFreeHost(h_stage);
    return result;
fail:
    if (dQ) cudaFree(dQ);
    if (dK) cudaFree(dK);
    if (h_stage)cudaFreeHost(h_stage);
    return NAN;
}
float sdotSliceF16(const uint8_t* q, const float* k, int headSize, int index, int typeSize) {
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
    //cublasSetPointerMode((cublasHandle_t)handle, CUBLAS_POINTER_MODE_HOST);
    //printf("sdotSliceF16 size=%d q[0]=%f k[0]=%f\n", headSize, h_stage[0], k[0]);
    //GOCHECK_CUBLAS(cublasSdot((cublasHandle_t)handle, headSize, (const float*)dQ, 1, (const float*)dK, 1, &result));
    GOCHECK_CUDA(launchDotProductKernel((float*)dQ, (float*)dK, &result, headSize));
    ce = cudaGetLastError(); GOCHECK_CUDA(ce);
    cudaFree(dQ); cudaFree(dK); cudaFreeHost(h_stage);
    return result;
fail:
    if (dQ) cudaFree(dQ);
    if (dK) cudaFree(dK);
    if (h_stage)cudaFreeHost(h_stage);
    return NAN;
}
float sdotSliceBF16(const uint8_t* q, const float* k, int headSize, int index, int typeSize) {
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
    //cublasSetPointerMode((cublasHandle_t)handle, CUBLAS_POINTER_MODE_HOST);
    //printf("sdotSliceBF16 size=%d q[0]=%f k[0]=%f\n", headSize, h_stage[0], k[0]);
    //GOCHECK_CUBLAS(cublasSdot((cublasHandle_t)handle, headSize, (const float*)dQ, 1, (const float*)dK, 1, &result));
    GOCHECK_CUDA(launchDotProductKernel((float*)dQ, (float*)dK, &result, headSize));
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
extern "C" __device__ float dquant(const uint8_t* q, int index, int format, int blockSize, int typeSize, int headerBytes) {
    switch (format) {
    case 1: // Q80
        return loadQ8(q,index,blockSize,typeSize, headerBytes);
    case 2: // Q4
        return loadQ4(q, index, blockSize, typeSize, headerBytes);
    case 3: // F16
        return loadF16(q, index, typeSize);
    case 4:
        return loadBF16(q, index, typeSize);
    default:
        return *reinterpret_cast<const float*>(q + index * blockSize);
    }
}
extern "C" __global__
void qk_scores(const float* __restrict__ Q,      // [nHeads * headSize]
    const uint8_t* __restrict__ Kraw, // packed keys
    float* __restrict__ Att,          // [nHeads * contextLen]
    float* __restrict__ buf,          // length of Q dequant buffer
    int nHeads, int headSize, int contextLen,
    int kvDim, int kvMul, int tMax, int tensorSize, float sqrtHeadSize,
    int format, int blockSize, int typeSize, int headerBytes)
{
    // Use dynamic shared memory sized to blockDim.x
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.f;
    int h = blockIdx.x;
    int attOffset = h * contextLen;
    float res = 0.0f;
    for (int t = 0; t <= tMax; t++) {
        int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
        // stride per KV head region is (kvMul * headSize), not headSize
        int cnt = 0;
        for (int i = 0; i < tensorSize; i++) {
            float k_i = dquant(Kraw, keyCacheOffset + i, format, blockSize, typeSize, headerBytes);
            buf[cnt++] = k_i;
        }
        //dotProductDevice(Q, buf, &res, tensorSize);
        //extern "C" __device__ void dotProductDevice(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ result, int N) { 
        for (int i = gid; i < tensorSize; i += stride) {
            sum += Q[i] * buf[i];
        }
        smem[tid] = sum;
        __syncthreads();
        // Reduce within block
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (tid < s) smem[tid] += smem[tid + s];
            __syncthreads();
        }
        // One atomicAdd per block
        if (tid == 0) atomicAdd(&res, smem[0]);
        res /= sqrtHeadSize;
        Att[attOffset + t] = res;
        __syncthreads();
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
    // layerBaseOffset must be in “elements” (not bytes). 
    // If it’s bytes, convert: reinterpret_cast<const float*>((const char*)Vcache + layerBaseOffset).
#pragma unroll 4
    for (int t = 0; t <= tMaxInclusive; ++t) {
        const float* v_th = Vcache + layerBaseOffset + t * kvDim + (h / kvMul) * headSize;
        acc += a_h[t] * v_th[i];
    }
    xb_h[i] = acc;
}
extern "C" __global__
void attention_av_weighted_sum(const float* __restrict__ attTok,       // [nHeads*contextLen]
    const uint8_t * __restrict__ vCacheRaw,  // quantized/raw bytes for V: [contextLen*kvTypeSizeTotal]
    float* __restrict__ xbTok,              // [nHeads*headSize]
    int nHeads, int headSize, int kvDim, int kvMul,int contextLen, int tMax,
    // Quantization params for V (match your FloatVector format):
    int vBlockSize,      // elements per quant block
    int vTypeSize,       // bytes per block (header + payload)
    int vHeaderBytes,    // bytes before payload
    int vIsQ8         // 1 if q8, 0 otherwise
) {
    int h = blockIdx.x;
    if (h >= nHeads) return;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    const int kvHead = h / kvMul;
    const int attBase = h * contextLen;
    float* xb = xbTok + h * headSize;
    for (int i = tid; i < headSize; i += stride) {
        float acc = 0.f;
        // Loop over timesteps and accumulate weighted V[i]
        for (int t = 0; t <= tMax; ++t) {
            float w = attTok[attBase + t];
            // Index within the kvDim slice
            int vIndexWithinKv = kvHead * headSize + i;
            int globalElemIndex = t * kvDim + vIndexWithinKv;
            float v_i = dquant(vCacheRaw, globalElemIndex, vIsQ8, vBlockSize, vTypeSize, vHeaderBytes);
            acc += w * v_i;
        }
        xb[i] = acc;
    }
}
extern "C" void launch_rmsnorm_fp32_rowmajor(const float* x, const float* weight, float* out, int size, float eps) {
    // One block is enough for vector sizes up to a few thousand; tune if needed
    int threads = 256;
    rmsnorm_fp32_rowmajor << <1, threads >> > (x, weight, out, size, eps);
    NCHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();
}
/*
* Java loop
*for (t) {
*  float a = att[h,t];
*  for (i) xb[h,i] += a * v[t,h,i];
*}
* CUDA kernel
*for (i in parallel) {
*  float acc = 0;
*  for (t) acc += att[h,t] * v[t,h,i];
*  xb[h,i] = acc;
*}
*/
extern "C" void launch_attention_av_weighted_sum(
    const float* d_attTok,       // device pointer [nHeads*contextLen]
    const uint8_t* d_vCacheRaw,  // device pointer [contextLen*kvTypeSizeTotal]
    float* d_xbTok,              // device pointer [nHeads*headSize]
    int nHeads, int headSize, int kvDim, int kvMul, int contextLen,int tMax, int vBlockSize,int vTypeSize, int vHeaderBytes,
    int vFormat,                 // 1=Q8, 2=Q4, 3=F16, 4=BF16, 5=F32
    int threadsPerBlock = 128    // default launch config
) {
    dim3 grid(nHeads);
    dim3 block(threadsPerBlock);
    attention_av_weighted_sum << <grid, block >> > ( d_attTok, d_vCacheRaw, d_xbTok, nHeads, headSize, kvDim, kvMul, contextLen, tMax, vBlockSize, vTypeSize, vHeaderBytes, vFormat);
    cudaDeviceSynchronize();
}

extern "C" void launch_qk_scores_fp32_rowmajor(
    const float* Q, const uint8_t* K, float* S,
    int ht, int nHeads, int headSize, int contextLength,
    int kvDim, int kvMul, int tMaxInclusive, int tensorSize, float sqrtHeadSize,
    int format, int blockSize, int typeSize, int headerBytes)
{
    int threads = 256; // safe starting point; raise after correctness
    int blockSizeGrid = 256;
    int numBlocks = (tensorSize + blockSizeGrid - 1) / blockSizeGrid;
    float* buf = NULL;
    printf("trying malloc:%d\n", tensorSize);
    NCHECK_CUDA(cudaMalloc(&buf, (size_t)(tensorSize * sizeof(float))));
    int token = (int)(ht / nHeads);
    int h = (int)(ht % nHeads);
    // get the query vector for this head
    // float* q = s.q + h * headSize;
    int qOffset = h * headSize;
    // attention scores for this head
    // float* att = s.att + h * config.seq_len;
    int attOffset = h * contextLength;
    qk_scores << <numBlocks, blockSizeGrid >> > (
        Q, K, S, buf,
        nHeads, headSize, contextLength,
        kvDim, kvMul, tMaxInclusive, tensorSize, sqrtHeadSize,
        format, blockSize, typeSize, headerBytes);
    NCHECK_CUDA(cudaDeviceSynchronize());
    NCHECK_CUDA(cudaGetLastError());
    NCHECK_CUDA(cudaFree(buf));
}
extern "C" void launch_row_softmax_fp32(const float* S, float* A, int rows, int cols, int ldS, int ldA) {
    int threads = 256;
    row_softmax_fp32 << <1, threads >> > (S, A, rows, cols, ldS, ldA);
    cudaDeviceSynchronize();
}
extern "C" void copyHostToDevice(uint8_t* tensor, uint64_t d_tensor, uint64_t bytes) {
    NCHECK_CUDA(cudaMemcpy((uint8_t*)d_tensor, tensor, bytes, cudaMemcpyHostToDevice));
}
extern "C" void copyDeviceToHost(uint64_t d_tensor, uint8_t* tensor, uint64_t bytes) {
    NCHECK_CUDA(cudaMemcpy((uint8_t*)d_tensor, tensor, bytes, cudaMemcpyDeviceToHost));
}
extern "C" uint64_t allocDevicePtr(uint64_t bytes) {
    uint8_t* d_tensor = nullptr;
    GOCHECK_CUDA(cudaMalloc((void**)&d_tensor, bytes));
    return (uint64_t)d_tensor;
fail:
    if (d_tensor) cudaFree(d_tensor);
    return NULL;
}
extern "C" void freeDevicePtr(uint64_t d_tensor) {
    if (d_tensor) cudaFree((uint8_t*)d_tensor);
}
extern "C" void cudaInit() {
    NCHECK_CUDA(cudaSetDevice(0));
    NCHECK_CUDA(cudaFree(0)); // init
    size_t freeB = 0, totalB = 0;
    NCHECK_CUDA(cudaMemGetInfo(&freeB, &totalB));
    printf("GPU mem free=%zu total=%zu\n", freeB, totalB);
}
#ifdef __cplusplus
}
#endif