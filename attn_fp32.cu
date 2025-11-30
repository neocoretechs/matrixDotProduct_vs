#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <math.h>
#include <float.h>
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

__device__ float atomicMaxFloat(float* address, float val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    // Keep going until successful
    do {
        if (__uint_as_float(old) >= val) return __uint_as_float(old);
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __float_as_uint(val));
    } while (assumed != old);
    return __uint_as_float(old);
}

__global__ void compute_exponentials_and_max(float* data, float* d_max, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float shared_data[];
    // Each thread stores its result in shared memory
    if (idx < size) {
        shared_data[threadIdx.x] = data[idx];
    }
    else {
        shared_data[threadIdx.x] = -FLT_MAX; // Ensure unused threads don't affect max
    }
    __syncthreads();
    // Perform a parallel reduction to find the max value
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] = fmax(shared_data[threadIdx.x], shared_data[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    // Write the max value for the block to the global max
    if (threadIdx.x == 0) {
        atomicMaxFloat(d_max, shared_data[0]);
    }
}

__global__ void compute_exponentials(float* data, float max, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Store the exponentials in-place, after subtracting max for numerical stability
        data[idx] = expf(data[idx] - max);
    }
}

__global__ void compute_total_exponentials(float* data, float* total, int size) {
    extern __shared__ float shared_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread loads one element into shared memory
    if (idx < size) {
        shared_data[threadIdx.x] = data[idx];
    }
    else {
        shared_data[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    // Perform reduction to calculate the total sum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    // Write the result for the block
    if (threadIdx.x == 0) {
        atomicAdd(total, shared_data[0]);
    }
}

__global__ void normalize_softmax(float* data, float total, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] /= total;
    }
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
__global__ void rope(const uint8_t* d_real, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t* d_imag, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    uint8_t* d_q, uint8_t* d_k, // state.q , state.k
    int nTokens, int dim, int position, int headSize, int kvDim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float* vec = NULL;
    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    // Parallel.parallelFor(0, nTokens, t -> { 
    for (int t = 0; t < nTokens; t++)
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % headSize;
            float fcr = dquant(d_real, indexA + (((position + t) * (headSize / 2) + (head_dim / 2)) * typeSizeA), formatA, blockSizeA, typeSizeA, headerBytesA);
            float fci = dquant(d_imag, indexB + (((position + t) * (headSize / 2) + (head_dim / 2)) * typeSizeB), formatB, blockSizeB, typeSizeB, headerBytesB);
            //float fcr = weights.freq_cis_real_dev.getFloat((position + t) * (headSize / 2) + (head_dim / 2));
            //float fci = weights.freq_cis_imag_dev.getFloat((position + t) * (headSize / 2) + (head_dim / 2));
            int rotn = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int vi = 0; vi < rotn; vi++) {
                //vec = (vi == 0 ? d_q[t] : d_k[t]); // the vector to rotate (query or key)
                // Load the actual device pointers for this token
                if (vi == 0) {
                    vec = reinterpret_cast<float*>(d_q[t]);
                }
                else {
                    vec = reinterpret_cast<float*>(d_k[t]);
                }
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
    //});  
}
/*
* 64 thread shared temp
*/
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
    //float result = 0f;
    //for (int j = 0; j < size; j++) {
    //    result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j);
    //}
    //return result;
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        float Aval = dquant(qA, indexA + i, formatA, blockSizeA, typeSizeA, headerBytesA);
        float Bval = dquant(qB, indexB + i, formatB, blockSizeB, typeSizeB, headerBytesB);  
        sum += Aval * Bval;
    }
    *result = sum;
}
// Kernel function for inner-product-based matrix multiplication
__global__ void innerProductKernel(const uint8_t* __restrict__ A, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t* __restrict__ B, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    float* C, int dim0, int dim1) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // rows use y
    //printf("kernel row=%d dim0=%d\n", row, dim0);
    if (row < dim0) {
        float sum = 0.0f;
        for (int k = 0; k < dim1; ++k) {
            float Aval = dquant(A, indexA + row * dim1 + k, formatA, blockSizeA, typeSizeA, headerBytesA);
            float Bval = dquant(B, indexB + k, formatB, blockSizeB, typeSizeB, headerBytesB);
            sum += Aval * Bval;
        }
        C[row] = sum;
        //printf("kernel row=%d sum=%f\n",row, C[row]);
    }
}

/*
* int threads = 256;
* int blocks = (size + threads - 1) / threads;
* Att = state.att[token], xb = state.xb[token], vCache = state.valueCache[curLayer]
* size = position + token
*/
extern "C" __global__ void weighted_sum(uint8_t* Att, uint8_t* xb, uint8_t* vCache, int h, int headSize, int attOffset, int xbOffset, int kvDim, int kvMul, int position, int token) {
    float* attF = reinterpret_cast<float*>(Att);
    float* xbF = reinterpret_cast<float*>(xb);
    float* vcacheF = reinterpret_cast<float*>(vCache);
    //int t = blockIdx.x * blockDim.x + threadIdx.x;
    for (int f = xbOffset; f < xbOffset+headSize; f++) {
        xbF[f] = 0.0f;
    }
    for (int t = 0; t <= position + token; t++) {
         int vOffset = t * kvDim + (h / kvMul) * headSize;
         // get the attention weight for this timestep
         float a = attF[attOffset + t];
         // accumulate the weighted value into xb
         //printf("into saxpy:%d %p %d %p %d %d %f\n", t, xb, xbOffset, vCache, vOffset, headSize, a);
         // get the attention weight for this timestep
         // float a = state.att[token].getFloat(attOffset + t);
         // accumulate the weighted value into xb
         // state.xb[token].saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
         // saxpy in place (s = a * x + y)
         for (int i = 0; i < headSize; ++i) {
             //setFloat(thisOffset + i, a * that.getFloat(thatOffset + i) + this.getFloat(thisOffset + i));
             xbF[xbOffset + i] = a * vcacheF[vOffset + i] + xbF[xbOffset + i];
         }
         //if (i < size) {
         //    thizF[attOffset + i] =
         //        a * thatF[xbOffset + i] + thizF[attOffset + i];
         //    printf("saxpy: %f\n", thizF[attOffset + i]);
         //}
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

__global__ void dotProductSetup(const uint8_t* qA, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t* qB, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    float* result, int N) {
    dotProduct(qA, indexA, formatA, blockSizeA, typeSizeA, headerBytesA,
        qB, indexB, formatB, blockSizeB, typeSizeB, headerBytesB,
        result, N);
    //printf("dotProduct res=%f\n",*result);
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
    //for(int i = 0; i < dim0; i++)
        //printf("dim0=%d i=%d out=%f\n", dim0, i, h_stage[i]);
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
/*
* 	for (int t = 0; t <= position + token; t++) {
*    	// get the key vector for this head and at this timestep
*    	// float* k = s.key_cache + loff + t * dim + h * headSize;
*    	int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
*    	// calculate the attention score as the dot product of q and k
*    	float score = state.q[token].dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
*    	score /= sqrtHeadSize;
*    	// save the score to the attention buffer
*    	state.att[token].setFloat(attOffset + t, score);
*   }
*/
__global__ void qk_scores_grid(
    const uint8_t* __restrict__ Q, int qOffset,
    int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t* __restrict__ keyCache,
    int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    uint8_t* __restrict__ Att, int attOffset, int position, int token, int h, int headSize, int kvDim, int kvMul) {
    float out;
    //printf("qk_scores_grid0 keyCacheOffset=%d qOffset=%d  attOffset=%d for t=%d Q=%f Kc=%f Att=%f\n", keyCacheOffset, qOffset, attOffset, t,
    //dquant(Q, qOffset, formatA, blockSizeA, typeSizeA, headerBytesA),
    //    dquant(keyCache, keyCacheOffset, formatB, blockSizeB, typeSizeB, headerBytesB),
    //    dquant(Att, attOffset, formatB, blockSizeB, typeSizeB, headerBytesB));
    float sqrtHeadSize = sqrtf(headSize);
    for (int t = 0; t <= position + token; t++) {
        int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
        //dotProduct(Q, qOffset, formatA, blockSizeA, typeSizeA, headerBytesA,
            //keyCache, keyCacheOffset, formatB, blockSizeB, typeSizeB, headerBytesB,
            //&out, headSize);
        //float result = 0f;
        //for (int j = 0; j < size; j++) {
        //    result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j);
        //}
        //return result;
        float sum = 0.0f;
        for (int i = 0; i < headSize; i++) {
            float Aval = dquant(Q, qOffset + i, formatA, blockSizeA, typeSizeA, headerBytesA);
            float Bval = dquant(keyCache, keyCacheOffset + i, formatB, blockSizeB, typeSizeB, headerBytesB);
            sum += Aval * Bval;
        }
        //printf("qk_scores_grid1=%f for t=%d\n", out, t);
        float score = sum / sqrtHeadSize;
        reinterpret_cast<float*>(Att)[attOffset + t] = score;
        //printf("qk_scores kernel attOffset = %d att[%d]=%f for t=%d\n", reinterpret_cast<float*>(Att)[attOffset + t], t);
    }
}

/*
* int threads = 256;
* int blocks = (size + threads - 1) / threads;
* Att = state.att[token], xb = state.xb[token], vCache = state.valueCache[curLayer]
* size = position + token
*/
EXPORT void launch_weighted_sum(uint8_t* Att, uint8_t* xb, uint8_t* vCache, int h, int headSize, int attOffset, int xbOffset, int kvDim, int kvMul, int position, int token) {
    int threads = 1;
    int blocks = 1;
    //printf("%p %p %p %d %d %d %d %d %d\n", Att, xb, vCache, h, headSize, attOffset, xbOffset, kvDim, kvMul);
    weighted_sum << <blocks, threads >> > (Att, xb, vCache, h, headSize, attOffset, xbOffset, kvDim, kvMul, position, token);
    cudaDeviceSynchronize();
    NCHECK_CUDA(cudaGetLastError());
}
/*
* qk scores followed by softmax
*/
EXPORT void launch_qkscores(uint8_t* q, int qOffset, int formatA, int blockSizeA, int typeSizeA, int headerBlockA,
    uint8_t* keyCache, int formatB, int blockSizeB, int typeSizeB, int headerBlockB, 
    uint8_t* Att, int attOffset, 
    int position, int token, int h, int headSize, int numHeads, int contextLength, int kvDim, int kvMul ) {
    int threads = 1;
    int blocks = 1;
    //printf("%p %p %p %d %d %d %d %d %d\n", Att, xb, vCache, h, headSize, attOffset, xbOffset, kvDim, kvMul);
    qk_scores_grid << <blocks, threads >> > (q, qOffset, formatA, blockSizeA, typeSizeA, headerBlockA,
        keyCache, formatB, blockSizeB, typeSizeB, headerBlockB,
        Att, attOffset, position, token, h, headSize, kvDim, kvMul);
    //state.att[token].softmaxInPlace(attOffset, position + token + 1);
    cudaDeviceSynchronize();
    NCHECK_CUDA(cudaGetLastError());
    float* data = reinterpret_cast<float*>(Att)+attOffset;
    float* d_max, * d_total; // Pointers for max and total on device
    int size = position + token + 1;
    NCHECK_CUDA(cudaMalloc((void**)&d_max, sizeof(float)));
    NCHECK_CUDA(cudaMalloc((void**)&d_total, sizeof(float)));
    // Initialize d_max to a very small value
    float init_max = -FLT_MAX;
    NCHECK_CUDA(cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice));
    // First kernel call: Compute exponentials and find max
    int threads_per_block = 64; // Adjust based on architecture
    blocks = (size + threads_per_block - 1) / threads_per_block;
    compute_exponentials_and_max << <blocks, threads_per_block, threads_per_block * sizeof(float) >> > (data, d_max, size);
    // Copy max back to host
    float h_max;
    cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    // Now compute the exponentials with the max value
    compute_exponentials << <blocks, threads_per_block >> > (data, h_max, size);
    // Compute total of exponentials
    float h_total = 0.0f;
    compute_total_exponentials << <blocks, threads_per_block, threads_per_block * sizeof(float) >> > (data, d_total, size);
    // Copy total back to host
    NCHECK_CUDA(cudaMemcpy(&h_total, d_total, sizeof(float), cudaMemcpyDeviceToHost));
    // Normalize to get softmax probabilities
    normalize_softmax << <blocks, threads_per_block >> > (data, h_total, size);
    NCHECK_CUDA(cudaDeviceSynchronize());
    //NCHECK_CUDA(cudaFree(d_max));
    //NCHECK_CUDA(cudaFree(d_total));
    NCHECK_CUDA(cudaGetLastError());
    //printf("position=%d token=%d attOffset=%d attRow len=%d \n", position, token, attOffset, (position + token + 1));
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

extern "C" void launch_row_softmax_inplace_fp32(uint8_t* q, int offset, int size) {
    float* data = reinterpret_cast<float*>(q) + offset;
    float* d_max, * d_total;
    NCHECK_CUDA(cudaMalloc((void**)&d_max, sizeof(float)));
    NCHECK_CUDA(cudaMalloc((void**)&d_total, sizeof(float)));
    // Initialize d_max to a very small value
    float init_max = -FLT_MAX;
    NCHECK_CUDA(cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice));
    // First kernel call: Compute exponentials and find max
    int threads_per_block = 64; // Adjust based on architecture
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    compute_exponentials_and_max << <blocks, threads_per_block, threads_per_block * sizeof(float) >> > (data, d_max, size);
    // Copy max back to host
    float h_max;
    cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    // Now compute the exponentials with the max value
    compute_exponentials << <blocks, threads_per_block >> > (data, h_max, size);
    // Compute total of exponentials
    float h_total = 0.0f;
    compute_total_exponentials << <blocks, threads_per_block, threads_per_block * sizeof(float) >> > (data, d_total, size);
    // Copy total back to host
    NCHECK_CUDA(cudaMemcpy(&h_total, d_total, sizeof(float), cudaMemcpyDeviceToHost));
    // Normalize to get softmax probabilities
    normalize_softmax << <blocks, threads_per_block >> > (data, h_total, size);
    //cudaFree(d_max);
    //cudaFree(d_total);
    cudaDeviceSynchronize();
    NCHECK_CUDA(cudaGetLastError());
}
extern "C" void launch_rope(const uint8_t* d_real, int indexA, int formatA, int blockSizeA, int typeSizeA, int headerBytesA,
    const uint8_t* d_imag, int indexB, int formatB, int blockSizeB, int typeSizeB, int headerBytesB,
    uint8_t* d_q, uint8_t* d_k, // state.q , state.k
    int nTokens, int dim, int position, int headSize, int kvDim) {
    int threads = 1;
    int blocks = 1;
    //printf("%p %p %p %d %d %d %d %d %d\n", Att, xb, vCache, h, headSize, attOffset, xbOffset, kvDim, kvMul);
    rope << <blocks, threads >> > (d_real, indexA, formatA, blockSizeA, typeSizeA, headerBytesA, 
        d_imag, indexB, formatB, blockSizeB, typeSizeB, headerBytesB, d_q, d_k, nTokens, dim, position, headSize, kvDim);
    cudaDeviceSynchronize();
    NCHECK_CUDA(cudaGetLastError());
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
    //printf("GPU mem free=%zu total=%zu\n", freeB, totalB);
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

#ifdef __cplusplus
}
#endif