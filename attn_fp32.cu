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
    __shared__ float temp[64];
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
__global__ void simpleDotProduct(
    const uint8_t * __restrict__ qA, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t * __restrict__ qB, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    float* __restrict__ result, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        float Aval = dquant(qA, indexA + i, formatA, blockSizeA, typeSizeA, headerBytesA);
        float Bval = dquant(qB, indexB + i, formatB, blockSizeB, typeSizeB, headerBytesB);                                      
    }
    *result = sum;
}
// Kernel function for inner-product-based matrix multiplication
__global__ void innerProductKernel(const uint8_t* __restrict__ A, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t* __restrict__ B, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    float* C, int dim0, int dim1) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // rows use y
    printf("kernel row=%d dim0=%d\n", row, dim0);
    if (row < dim0) {
        float sum = 0.0f;
        for (int k = 0; k < dim1; ++k) {
            float Aval = dquant(A, indexA + row * dim1 + k, formatA, blockSizeA, typeSizeA, headerBytesA);
            float Bval = dquant(B, indexB + k, formatB, blockSizeB, typeSizeB, headerBytesB);
            sum += Aval * Bval;
        }
        C[row] = sum;
        printf("kernel row=%d sum=%f\n",row, C[row]);
    }
}

/*
* int threads = 256;
* int blocks = (size + threads - 1) / threads;
* Att = state.att[token], xb = state.xb[token], vCache = state.valueCache[curLayer]
* size = position + token
*/
extern "C" __global__ void weighted_sum(uint8_t* Att, uint8_t* xb, const uint8_t* vCache, int h, int headSize, int attOffset, int xbOffset, int kvDim, int kvMul, int size) {
    float* attF = reinterpret_cast<float*>(Att);
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < size) {
        int vOffset = t * kvDim + (h / kvMul) * headSize;
        // get the attention weight for this timestep
        float a = attF[attOffset + t];
        // accumulate the weighted value into xb
        printf("into saxpy:%d %d %p %d %p %d %d %f\n", t, size, xb, xbOffset, vCache, vOffset, headSize, a);
        float* thizF = reinterpret_cast<float*>(Att);
        const float* thatF = reinterpret_cast<const float*>(vCache);
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        printf("saxpy0: %p %p %d %d\n", thizF, thatF, attOffset, i);
        // saxpy in place
        if (i < size) {
            thizF[attOffset + i] =
                a * thatF[xbOffset + i] + thizF[attOffset + i];
            printf("saxpy: %f\n", thizF[attOffset + i]);
        }
    }
}
extern "C" __global__ void matmul(const uint8_t* thiz, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA, 
    const uint8_t* that, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB, uint8_t* out, int dim0, int dim1) {
    //void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1) {
    //    Parallel.parallelFor(0, dim0, i->out.setFloat(i, dot(i * dim1, that, 0, dim1)));
    //}
    float result;
    printf("ThreadIdx.x=%d dim0=%d gridDim.x=%d blockDim.x=%d gridDim.x*blockDim.x=%d\n", threadIdx.x, dim0, gridDim.x, blockDim.x, (gridDim.x * blockDim.x));
    for (int i = threadIdx.x; i < dim0; i += gridDim.x * blockDim.x) {
        printf("out=%p [%d]\n", out, i);
        dotProduct(thiz, i * dim1, formatA, blockSizeA, typeSizeA, headerBytesA,
            that, 0, formatB, blockSizeB, typeSizeB, headerBytesB, &result, dim1);
        printf("out=%p [%d] = %f\n", out, i, result);
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
        int blockSize = 64;
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
EXPORT void launch_Matmul(const uint8_t* qA, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t* qB, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    uint8_t* out, int dim0, int dim1) {
    //float* qC = reinterpret_cast<float*>(out); // device pointer to output
    // Define the number of threads per block and the number of blocks per grid
    // dim0 M rows dim1 K shared dimension N is columns, now = 1
    // 2-D launch, rows mapped to y
    int N = 1;
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(N, (dim0 + threadsPerBlock.y - 1) / threadsPerBlock.y); // N=1 => x=1

    // Launch the kernel
    innerProductKernel << <blocksPerGrid, threadsPerBlock >> > (qA, indexA, formatA, blockSizeA, typeSizeA, headerBytesA,
        qB, indexB, formatB, blockSizeB, typeSizeB, headerBytesB, (float*)out, dim0, dim1);
    // Check for errors in kernel launch
    NCHECK_CUDA(cudaGetLastError());
    // Copy the resultant matrix C from device to host
    float* h_stage = nullptr;
    cudaError_t ce = cudaHostAlloc((void**)&h_stage, dim0*sizeof(float), cudaHostAllocDefault);
    cudaMemcpy(h_stage, out, dim0*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < dim0; i++)
        printf("dim0=%d i=%d out=%f\n", dim0, i, h_stage[i]);
    cudaFreeHost(h_stage);
    //void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1) {
    //    Parallel.parallelFor(0, dim0, i->out.setFloat(i, dot(i * dim1, that, 0, dim1)));
    //}
    //for (int i = 0; i < dim0; i++) {
    //    float result;
     //   NCHECK_CUDA(launchDotProductKernel(qA, i * dim1, formatA, blockSizeA, typeSizeA, headerBytesA,
     //       qB, 0, formatB, blockSizeB, typeSizeB, headerBytesB, &result, dim1));
        //printf("Result=%f\n",result);
        // Write the host result into device out[i]
   //     cudaError_t ce = cudaMemcpy(d_out + i, &result, sizeof(float), cudaMemcpyHostToDevice);
     //   if (ce) { NCHECK_CUDA(ce); return; }
    //}
    //matmul << <numBlocks, blockSize>> >(qA, indexA, formatA, blockSizeA, typeSizeA, headerBytesA,
    //    qB, indexB, formatB, blockSizeB, typeSizeB, headerBytesB,
    //    out, dim0, dim1);
    
    //NCHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();
}

__global__ void qk_scores_grid(
    const uint8_t* __restrict__ Q, int qOffset,
    int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t* __restrict__ keyCache, int keyCacheOffset,
    int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    uint8_t* __restrict__ Att, int attOffset,
    int h, int headSize, int kvDim, int kvMul, int t, float sqrtHeadSize) {
    float out;
    //printf("qk_scores_grid0 keyCacheOffset=%d qOffset=%d  attOffset=%d for t=%d Q=%f Kc=%f Att=%f\n", keyCacheOffset, qOffset, attOffset, t,
    //dquant(Q, qOffset, formatA, blockSizeA, typeSizeA, headerBytesA),
    //    dquant(keyCache, keyCacheOffset, formatB, blockSizeB, typeSizeB, headerBytesB),
    //    dquant(Att, attOffset, formatB, blockSizeB, typeSizeB, headerBytesB));

    dotProduct(Q, qOffset, formatA, blockSizeA, typeSizeA, headerBytesA,
       keyCache, keyCacheOffset, formatB, blockSizeB, typeSizeB, headerBytesB,
        &out, headSize);
    printf("qk_scores_grid1=%f for t=%d\n", out, t);
    float score = out / sqrtHeadSize;
    reinterpret_cast<float*>(Att)[attOffset + t] = score; // 4-byte alignment assumed
    printf("qk_scores_grid2=%f for t=%d\n", reinterpret_cast<float*>(Att)[attOffset + t], t);
}

/*
* int threads = 256;
* int blocks = (size + threads - 1) / threads;
* Att = state.att[token], xb = state.xb[token], vCache = state.valueCache[curLayer]
* size = position + token
*/
EXPORT void launch_weighted_sum(uint8_t* Att, uint8_t* xb, const uint8_t* vCache, int h, int headSize, int attOffset, int xbOffset, int kvDim, int kvMul, int size) {
    //state.xb[token].fillInPlace(xbOffset, headSize, 0f);
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    //printf("%p %p %p %d %d %d %d %d %d %d\n", Att, xb, vCache, h, headSize, attOffset, xbOffset, kvDim, kvMul, size);
    weighted_sum << <blocks, threads >> > (Att, xb, vCache, h, headSize, attOffset, xbOffset, kvDim, kvMul, size);
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
* Does simple dquant and scalar dot, return result to testing/verification
* q and k are device addresses cast to float, we have to move them down
*/
EXPORT float launch_cpu_scalar_Dot(const uint8_t* d_q, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t* d_k, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB, int size) {
        float result_host = 0.0f;
        float* d_result = nullptr;
        NCHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
        NCHECK_CUDA(cudaMemset(d_result, 0, sizeof(float)));
        simpleDotProduct << <1, 1 >> > (
            d_q, indexA, formatA, blockSizeA, typeSizeA, headerBytesA,
            d_k, indexB, formatB, blockSizeB, typeSizeB, headerBytesB,
            d_result, size
            );
        NCHECK_CUDA(cudaGetLastError());
        NCHECK_CUDA(cudaDeviceSynchronize());
        NCHECK_CUDA(cudaMemcpy(&result_host, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        NCHECK_CUDA(cudaFree(d_result));
        //printf("simpleDot d_q=%p d_k=%p result=%f for %d elements.\n", d_q, d_k, result_host, size);
        return result_host;
    //float* h_q = nullptr;
    //float* h_k = nullptr;
    //cudaError_t ce = cudaMallocHost(&h_q, size * sizeof(float));
    //if (ce != cudaSuccess) return 0.0f;
    //ce = cudaMallocHost(&h_k, size * sizeof(float));
    //if (ce != cudaSuccess) { cudaFreeHost(h_q); return 0.0f; }
    //cudaMemcpy(h_q, d_q + indexA, size * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_k, d_k + indexB, size * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaFreeHost(h_q);
    //cudaFreeHost(h_k);
}

EXPORT float getFloat(const uint64_t q, int index) {
    const float* d_q = reinterpret_cast<const float*>(q);
    float value = 0.0f;
    NCHECK_CUDA(cudaMemcpy(&value, d_q + index, sizeof(float), cudaMemcpyDeviceToHost));
    return value;
}
/*EXPORT float getFloatQ8(const uint64_t q, int index, int blockSize, int typeSize, int headerBytes) {
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
}*/
EXPORT float getFloatQ8(const uint64_t q, int index,
    int blockSize, int typeSize, int headerBytes) {
    const uint8_t* d_q = reinterpret_cast<const uint8_t*>(q);
    int blockIndex = index / blockSize;
    int withinBlockIndex = index % blockSize;
    int blockOffset = blockIndex * typeSize;
    uint8_t h_quant;
    uint16_t h_scale;
    // read quantized byte
    NCHECK_CUDA(cudaMemcpy(&h_quant,
        d_q + blockOffset + headerBytes + withinBlockIndex,
        sizeof(uint8_t), cudaMemcpyDeviceToHost));
    // read scale (stored as half at start of block)
    NCHECK_CUDA(cudaMemcpy(&h_scale,
        d_q + blockOffset,
        sizeof(uint16_t), cudaMemcpyDeviceToHost));
    int8_t quant = static_cast<int8_t>(h_quant);
    float scale = halfToFloat(h_scale);
    return static_cast<float>(quant) * scale;
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