#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <math.h>
#include <iostream>
#include "com_neocoretechs_cublas_Gemm.h"
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
#define CHECK_CUDA_ERROR(call) { cudaError_t err = call; if (err != cudaSuccess) {  fprintf(stderr, "CUDA error: %s, in file '%s', line %d\n",  cudaGetErrorString(err), __FILE__, __LINE__);  exit(EXIT_FAILURE); } }

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
* ldS is offset, not stride
*/
__global__ void row_softmax_inplace_fp32(float* __restrict__ S,
    int rows, int cols, int ldS) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float* srow = S + r + ldS;
    // 1) max
    float m = -1e30f;
    for (int c = 0; c < cols; ++c) m = fmaxf(m, srow[c]);
    // 2) exp and sum (overwrite in place)
    float sum = 0.f;
    for (int c = 0; c < cols; ++c) {
        float e = __expf(srow[c] - m);
        srow[c] = e;
        sum += e;
    }
    // 3) normalize
    float inv = 1.0f / sum;
    for (int c = 0; c < cols; ++c) srow[c] *= inv;
}


__device__ inline float getFloatDevice(const float* d_q, int index) {
    //const float* d_q = reinterpret_cast<const float*>(q);
    return *(d_q + index);
}
__device__ inline float getFloatQ8Device(const uint8_t* d_q, int index, int blockSize, int typeSize, int headerBytes) {
    // const uint8_t* d_q = reinterpret_cast<const uint8_t*>(q);
    uint8_t* h_q = nullptr;
    //uint16_t* s_q = nullptr;
    int blockIndex = index / blockSize;
    int withinBlockIndex = index % blockSize;
    int blockOffset = blockIndex * typeSize;
    // read byte
    //h_q = (d_q + blockOffset + headerBytes + withinBlockIndex);
    // read short
    //s_q = (uint16_t*)(d_q + blockOffset);
    //int8_t quant = (int8_t)*h_q;
    //float scale = __half2float(__ushort_as_half(*s_q));
    //printf("GPU index=%d quant=%d scale=%f\n",index, quant, scale);
    //return ((float)quant) * scale;
    // alternate load -- uint16_t scaleBits = (uint16_t)(d_q + blockOffset) | ((uint16_t)(d_q + blockOffset + 1) << 8);
    const uint16_t* s_q = reinterpret_cast<const uint16_t*>(d_q + blockOffset);
    float scale = __half2float(__ushort_as_half(*s_q));
    // Payload starts after headerBytes
    const int8_t* payload = reinterpret_cast<const int8_t*>(d_q + blockOffset + headerBytes);
    int8_t quant = payload[withinBlockIndex];
    return static_cast<float>(quant) * scale;
}
__device__ inline float getFloatQ4Device(const uint8_t* d_q, int index, int blockSize, int typeSize, int headerBytes) {
    // const uint8_t* d_q = reinterpret_cast<const uint8_t*>(q);
    //int blockIndex = index / GGMLType.Q4_0.getBlockSize();
    //int blockOffset = blockIndex * GGMLType.Q4_0.getTypeSize();
    //float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
    //byte quant;
    //int modIndex = index % GGMLType.Q4_0.getBlockSize();
    //if (modIndex < GGMLType.Q4_0.getBlockSize() / 2) {
    //   quant = (byte)(readByte(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + modIndex) & 0x0F);
    //}
    //else {
    //    quant = (byte)((readByte(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + modIndex - GGMLType.Q4_0.getBlockSize() / 2) >> > 4) & 0x0F);
    //}
    //quant -= 8;
    //return quant * scale;
    //uint16_t* s_q = nullptr;
    int blockIndex = index / blockSize;
    int modIndex = index % blockSize;
    int blockOffset = blockIndex * typeSize;
    const uint16_t* s_q = reinterpret_cast<const uint16_t*>(d_q + blockOffset);
    float scale = __half2float(__ushort_as_half(*s_q));
    const int8_t* quant;
    int8_t iquant;
    float rquant;
    if (modIndex < blockSize / 2) {
        quant = reinterpret_cast<const int8_t*>((d_q + blockOffset + headerBytes + modIndex) );
        iquant = quant[0];
    } else {
        quant = reinterpret_cast<const int8_t*>((d_q + blockOffset + headerBytes + modIndex - blockSize / 2) );
        iquant = quant[0];
        iquant = iquant >> 4;
    }
    // Payload starts after headerBytes
    iquant = iquant & 0x0F;
    iquant -= 8;
    return static_cast<float>(iquant);
}

// FP16 element at index
__device__ inline float getFloatF16Device(const uint8_t* d_q, int index, int headerBytes) {
    //return Float.float16ToFloat(readShort(memorySegment, index * GGMLType.FLOAT16_BYTES));
    const uint16_t* s_q = reinterpret_cast<const uint16_t*>(d_q + (index * headerBytes));
    __half h = __ushort_as_half(*s_q);   // reinterpret raw bits
    return __half2float(h);
}

// FP16 element at index
__device__ inline float getFloatBF16Device(const uint8_t* d_q, int index, int headerBytes) {
    //return Float.intBitsToFloat(readShort(memorySegment, index * GGMLType.BFLOAT16_BYTES) << 16);
    const uint16_t* s_q = reinterpret_cast<const uint16_t*>(d_q + (index * headerBytes));
    __half h = __ushort_as_half(*s_q);   // reinterpret raw bits
    int x = (int)h << 16;
    return __half2float(h);
}

extern "C" __device__ float dquant(const uint8_t* q, int index, int format, int blockSize, int typeSize, int headerBytes) {
    switch (format) {
    case 1: return getFloatQ8Device(q, index, blockSize, typeSize, headerBytes);
    case 2: return getFloatQ4Device(q, index, blockSize, typeSize, headerBytes);
    case 3: return getFloatF16Device(q, index, typeSize);
    case 4: return getFloatBF16Device(q, index, typeSize);
    default: return reinterpret_cast<const float*>(q)[index];
    }
}

__device__ void dotProduct(const uint8_t* __restrict__ qA, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t* __restrict__ qB, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    float* result, int N) {
    // Shared memory for accumulating partial results
    __shared__ float temp[256];
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;
    // Compute the dot product for each thread
    for (int i = threadId; i < N; i += gridDim.x * blockDim.x) {
        sum += dquant(qA, indexA + (i * typeSizeA), formatA, blockSizeA, typeSizeA, headerBytesA) *
            dquant(qB, indexB + (i * typeSizeB), formatB, blockSizeB, typeSizeB, headerBytesB);
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
    }
}
extern "C" __global__ void matmul(const uint8_t* thiz, int dim0, int formatA, int blockSizeA, int typeSizeA, int headerBytesA, 
    const uint8_t* that, int dim1, int formatB, int blockSizeB, int typeSizeB, int headerBytesB, uint8_t* out, int size) {
    //Parallel.parallelFor(0, indexA, i->out.setFloat(i
    float result;
    for (int i = threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
        dotProduct(thiz, i * dim1, formatA, blockSizeA, typeSizeA, headerBytesA,
            that, 0, formatB, blockSizeB, typeSizeB, headerBytesB, &result, dim1);
        reinterpret_cast<float*>(out)[i] = result;
    }
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

extern "C" __global__
void rmsnorm_fp32_rowmajor(const uint8_t* x, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t* weight, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    uint8_t* out, int size, float eps) {
        // Shared buffer for reduction
        __shared__ float partial[256];  // assuming <=256 threads
        float acc = 0.f;
        // Each thread accumulates its slice
        for (int i = threadIdx.x; i < size; i += blockDim.x) {
            float v = dquant(x + (i*typeSizeA), indexA, formatA, blockSizeA, typeSizeA, headerBytesA);// x[i];//x[i];
            acc += v * v;
        }
        partial[threadIdx.x] = acc;
        __syncthreads();
        // Simple reduction by thread 0
        if (threadIdx.x == 0) {
            float sumsq = 0.f;
            for (int t = 0; t < blockDim.x; ++t) sumsq += partial[t];
            float inv = rsqrtf(sumsq / size + eps);
            partial[0] = inv; // broadcast
        }
        __syncthreads();
        float inv = partial[0];
        // Normalize and scale
        for (int i = threadIdx.x; i < size; i += blockDim.x) {
           // printf("i = %d ) weight=%f x=%f * inv=%f\n", i, dquant(weight+(i * typeSizeB), indexB, formatB, blockSizeB, typeSizeB, headerBytesB), 
                //dquant(x + (i * typeSizeA), indexA, formatA, blockSizeA, typeSizeA, headerBytesA), inv);
            reinterpret_cast<float*>(out)[i] = dquant(weight + (i*typeSizeB), indexB, formatB, blockSizeB, typeSizeB, headerBytesB) /*weight[i]*/ *
                (inv * dquant(x + (i*typeSizeA), indexA, formatA, blockSizeA, typeSizeA, headerBytesA));// (inv * x[i]);
        }
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

__global__ void dotProductSetup(const uint8_t* qA, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t* qB, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    float* result, int N) {
    dotProduct(qA, indexA, formatA, blockSizeA, typeSizeA, headerBytesA,
        qB, indexB, formatB, blockSizeB, typeSizeB, headerBytesB,
        result, N);
}
#ifdef __cplusplus
extern "C" {
#endif
cudaError_t launchDotProductKernel(const uint8_t* qA, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t* qB, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    float* result, int N) {
        float* d_result;
        cudaError_t ce = cudaMalloc((void**)&d_result, sizeof(float));
        if (ce) return ce;
        ce = cudaMemset(d_result, 0, sizeof(float));
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        dotProductSetup << <numBlocks, blockSize>> > (qA, indexA, formatA, blockSizeA, typeSizeA, headerBytesA,
            qB, indexB, formatB, blockSizeB, typeSizeB, headerBytesB,
            d_result, N);
        ce = cudaGetLastError();
        if (ce) {
            NCHECK_CUDA(ce);
            cudaFree(d_result);
            return ce;
        }
        ce = cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        return ce;
}
EXPORT void launchMatmulKernel(const uint8_t* qA, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t* qB, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    uint8_t* out, int N) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    matmul << <numBlocks, blockSize>> > (qA, indexA, formatA, blockSizeA, typeSizeA, headerBytesA,
        qB, indexB, formatB, blockSizeB, typeSizeB, headerBytesB,
        out, N);
    NCHECK_CUDA(cudaGetLastError());
}
// Each block computes the dot for a single timestep t using parallel dotProduct.
// All pointers are uint8_t*; offsets are BYTES.
__global__ void qk_scores_grid(
    const uint8_t* __restrict__ Q, int qOffsetBytes,
    int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t* __restrict__ keyCache,
    int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    uint8_t* __restrict__ Att, size_t attOffsetBytes,
    int h, int headSize, int kvDim, int kvMul, float sqrtHeadSize,
    int t0, int tCount // process timesteps [t0, t0 + tCount)
) {
    int t = t0 + blockIdx.x;
    if (t >= t0 + tCount) return;

    // Compute key index in ELEMENTS, then convert to BYTES with typeSizeB.
    int keyIndex = t * kvDim + (h / kvMul) * headSize;
    size_t keyCacheOffsetBytes = (size_t)keyIndex * (size_t)typeSizeB;

    // Device float accumulator for this block's result
    __shared__ float scoreBlock;

    // Let dotProduct write directly to scoreBlock (address visible to device)
    float* out = &scoreBlock;

    // Ensure blockDim.x == 256 to match your helper's shared memory expectations
    dotProduct(Q, qOffsetBytes, formatA, blockSizeA, typeSizeA, headerBytesA,
        keyCache, (int)keyCacheOffsetBytes, formatB, blockSizeB, typeSizeB, headerBytesB,
        out, headSize);

    // Use one thread to store the final score scaled into Att as bytes
    if (threadIdx.x == 0) {
        float score = scoreBlock / sqrtHeadSize;
        size_t writeOffset = attOffsetBytes + (size_t)(t - t0) * sizeof(float);
        *reinterpret_cast<float*>(Att + writeOffset) = score; // 4-byte alignment assumed
    }
}
EXPORT void launch_qk_scores_grid(
    const uint8_t* Q, int qOffsetBytes,
    int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t* keyCache,
    int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    uint8_t* Att, int attOffset,
    int h, int headSize, int kvDim, int kvMul,
    int token, int position, float sqrtHeadSize
) {
    int tCount = position + token + 1;              // number of timesteps to score
    dim3 grid(tCount);                              // one block per t
    dim3 block(256);                                // matches dotProduct helper

    qk_scores_grid << <grid, block >> > (
        Q, qOffsetBytes, formatA, blockSizeA, typeSizeA, headerBytesA,
        keyCache, formatB, blockSizeB, typeSizeB, headerBytesB,
        Att, (size_t)attOffset,
        h, headSize, kvDim, kvMul, sqrtHeadSize,
        /*t0*/ 0, tCount
        );
    NCHECK_CUDA(cudaDeviceSynchronize());
    NCHECK_CUDA(cudaGetLastError());
    // softmax the scores to get attention weights, from 0..position inclusively
    //state.att[token].softmaxInPlace(attOffset, position + token + 1);
    // weighted sum of the values, store back into xb
    int threads = 256;
    row_softmax_inplace_fp32 << <1, threads >> > ((float*)Att, (position + token + 1)* sizeof(float), 1, (attOffset * sizeof(float)));
    cudaDeviceSynchronize();
    NCHECK_CUDA(cudaGetLastError());
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
/*
* q and k are device addresses cast to float, we have to move them down
*/
static float scalarDot(const float* d_q, const float* d_k,
    size_t thisOffset, size_t thatOffset, size_t size) {
    float* h_q = nullptr;
    float* h_k = nullptr;
    float result = 0.0f;
    cudaError_t ce = cudaMallocHost(&h_q, size * sizeof(float));
    if (ce != cudaSuccess) return 0.0f;
    ce = cudaMallocHost(&h_k, size * sizeof(float));
    if (ce != cudaSuccess) { cudaFreeHost(h_q); return 0.0f; }
    cudaMemcpy(h_q, d_q + thisOffset, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_k, d_k + thatOffset, size * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t j = 0; j < size; ++j) {
        result += h_q[j] * h_k[j];
    }
    cudaFreeHost(h_q);
    cudaFreeHost(h_k);
    return result;
}

EXPORT float getFloat(const uint64_t q, int index, int blockSize, int typeSize, int headerBytes) {
    const float* d_q = reinterpret_cast<const float*>(q);
    float value = 0.0f;
    NCHECK_CUDA(cudaMemcpy(&value, d_q + index, sizeof(float), cudaMemcpyDeviceToHost));
    NCHECK_CUDA(cudaDeviceSynchronize());
    return value;
}
EXPORT float getFloatQ8(const uint64_t q, int index, int blockSize, int typeSize, int headerBytes) {
    const uint8_t* d_q = reinterpret_cast<const uint8_t*>(q);
    uint8_t* h_q = nullptr;
    uint16_t* s_q = nullptr;
    int blockIndex = index / blockSize;
    int withinBlockIndex = index % blockSize;
    int blockOffset = blockIndex * typeSize;
    NCHECK_CUDA(cudaMallocHost(&h_q, 1));
    NCHECK_CUDA(cudaMallocHost(&s_q, 2));
    // read byte
    NCHECK_CUDA(cudaMemcpy(h_q, (d_q + blockOffset + headerBytes + withinBlockIndex), 1, cudaMemcpyDeviceToHost));
    // read short
    NCHECK_CUDA(cudaMemcpy(s_q, (d_q + blockOffset), 2, cudaMemcpyDeviceToHost));
    int8_t quant = (int8_t)*h_q;
    float scale = halfToFloat(*s_q);
    NCHECK_CUDA(cudaFreeHost(h_q));
    NCHECK_CUDA(cudaFreeHost(s_q));
    //printf("GPU index=%d quant=%d scale=%f\n",index, quant, scale);
    return ((float)quant) * scale;
}
/*
* Scalar dot routines working on quantized device buffers
*/
EXPORT float sdotSliceDevice(const uint8_t* qA, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t* qB, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    int N) {
    float result;
    GOCHECK_CUDA(launchDotProductKernel(
        qA, indexA, formatA, blockSizeA, typeSizeA, headerBytesA,
        qB, indexB, formatB, blockSizeB, typeSizeB, headerBytesB,
        &result, N));
    GOCHECK_CUDA(cudaDeviceSynchronize());
    return result;
fail:
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
/*
* Launch Fused Head Attention kernels
*/
extern "C" void launch_rmsnorm_fp32_rowmajor(uint8_t* qA, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    uint8_t* qB, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    uint8_t* out, int size, float eps) {
    // One block is enough for vector sizes up to a few thousand; tune if needed
    //printf("rmsnorm %p %p %p %d %d %d %d %d %d %d %d %d %d\n", (void*)qA, (void*)qB, (void*)out, indexA, formatA, blockSizeA, typeSizeA, headerBytesA, indexB, formatB, blockSizeB, typeSizeB, headerBytesB);
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    rmsnorm_fp32_rowmajor << <blocks, threads >> > (qA, indexA, formatA, blockSizeA, typeSizeA, headerBytesA,
        qB, indexB, formatB, blockSizeB, typeSizeB, headerBytesB,
        out, size, eps);
    NCHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();
    // Host buffer to read back results
    /*float* host = (float*)malloc(size * sizeof(float));
    if (!host) { printf("malloc failed\n"); return; }
    // Copy device -> host
    NCHECK_CUDA( cudaMemcpy(host, reinterpret_cast<float*>(out), size * sizeof(float), cudaMemcpyDeviceToHost));
    // Print a few sentinels (avoid giant logs)
    //printf("out[0]=%f out[mid]=%f out[last]=%f\n", host[0], host[size / 2], host[size - 1]);
    for (int i = 0; i < size; ++i) {
        printf("i=%d val=%f\n", i, host[i]);
    }
    free(host);*/
}

extern "C" void launch_row_softmax_fp32(const float* S, float* A, int rows, int cols, int ldS, int ldA) {
    int threads = 256;
    row_softmax_fp32 << <1, threads >> > (S, A, rows, cols, ldS, ldA);
    cudaDeviceSynchronize();
}
extern "C" void launch_row_softmax_inplace_fp32(float* S, int rows, int cols, int offset) {
    int threads = 256;
    row_softmax_inplace_fp32 << <1, threads >> > (S, rows, cols, offset);
    cudaDeviceSynchronize();
}
extern "C" void copyHostToDevice(uint8_t* tensor, uint64_t d_tensor, uint64_t bytes) {
    NCHECK_CUDA(cudaMemcpy((uint8_t*)d_tensor, tensor, bytes, cudaMemcpyHostToDevice));
}
extern "C" void copyDeviceToHost(uint64_t d_tensor, uint8_t* tensor, uint64_t bytes) {
    // dst = host buffer (tensor), src = device buffer (d_tensor)
    NCHECK_CUDA(cudaMemcpy(tensor, (void*)d_tensor, bytes, cudaMemcpyDeviceToHost));
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
// Initialize random data for query, key, value matrices
void initializeRandomData(float* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = 2.0f * (float)rand() / RAND_MAX - 1.0f; // Values between -1 and 1
    }
}
// Utility function to print a tensor snippet
void printTensorSnippet(const char* name, float* tensor, int rows, int cols, int stride) {
    printf("%s (shape: %dx%d, showing top-left corner):\n", name, rows, cols);
    int printRows = rows < 5 ? rows : 5;
    int printCols = cols < 5 ? cols : 5;
    for (int i = 0; i < printRows; i++) {
        for (int j = 0; j < printCols; j++) {
            printf("%.4f ", tensor[i * stride + j]);
        }
        printf("...\n");
    }
    printf("...\n\n");
}
// CUDA kernel for matrix transpose: B = A^T
__global__ void transposeKernel(
    float* A, float* B, int rows, int cols, int strideA, int strideB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        B[col * strideB + row] = A[row * strideA + col];
    }
}
// CUDA kernel for scaling a matrix: A = A * scale
__global__ void scaleKernel(float* A, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] *= scale;
    }
}
// CUDA kernel for Softmax over the last dimension
__global__ void softmaxKernel(float* input, int rows, int cols, int stride) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        // Find max value for numerical stability
        float maxVal = -FLT_MAX;
        for (int col = 0; col < cols; col++) {
            maxVal = fmaxf(maxVal, input[row * stride + col]);
        }
        // Compute exponentials and sum
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            input[row * stride + col] = expf(input[row * stride + col] - maxVal);
            sum += input[row * stride + col];
        }
        // Normalize
        for (int col = 0; col < cols; col++) {
            input[row * stride + col] /= sum;
        }
    }
}

#ifdef __cplusplus
}
#endif