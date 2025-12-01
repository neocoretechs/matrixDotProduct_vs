/*
* Single precision matrix dot product JNI implementation via CUDA CUBLAS v12.1
* author: Jonathan Groff Copyright (C) NeoCoreTechs 2025
* CUBLAS params:
* handle: handle to the cuBLAS library context.
* transa: operation op(A) that is non- or (conj.) transpose.
* transb :operation op(B) that is non- or (conj.) transpose.
* m: number of rows of matrix op(A) and C.
* n: number of columns of matrix op(B) and C.
* k: number of columns of op(A) and rows of op(B).
* alpha: <type> scalar used for multiplication.
* A <type> array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
* lda: leading dimension of two-dimensional array used to store the matrix A.
* B: <type> array of dimension ldb x n with ldb>=max(1,k) if transb == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.
* ldb: leading dimension of two-dimensional array used to store matrix B.
* beta: <type> scalar used for multiplication. If beta==0, C does not have to be a valid input.
* C: in/out <type> array of dimensions ldc x n with ldc>=max(1,m).
* ldc leading dimension of a two-dimensional array used to store the matrix C.
*
* The possible error values returned by this function and their meanings are listed below.
* Error Value 					Meaning
* CUBLAS_STATUS_SUCCESS			the operation completed successfully
* CUBLAS_STATUS_NOT_INITIALIZED	the library was not initialized
* CUBLAS_STATUS_INVALID_VALUE	If m, n, k < 0 or if transa, transb != CUBLAS_OP_N, CUBLAS_OP_C, CUBLAS_OP_T or
*								if lda < max(1, m) if transa == CUBLAS_OP_N and lda < max(1, k) otherwise or
*								if ldb < max(1, k) if transb == CUBLAS_OP_N and ldb < max(1, n) otherwise or if ldc < max(1, m) or
*								if alpha, beta == NULL or C == NULL if C needs to be scaled
* CUBLAS_STATUS_ARCH_MISMATCH	in the case of cublasHgemm the device does not support math in half precision.
* CUBLAS_STATUS_EXECUTION_FAILED	the function failed to launch on the GPU
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

#include "com_neocoretechs_cublas_Gemm.h"
#include <iostream>
#define GOCHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); goto fail; } } while(0)
#define PCHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); return nullptr; } } while(0)
#define NCHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); } } while(0)

float* toDeviceFloatQ8(const uint8_t*, size_t, int, int, int, int);
float* toDeviceFloatQ4(const uint8_t*, size_t, int, int, int);
float* toDeviceFloatF16(const uint8_t*, size_t, int);
float* toDeviceFloatBF16(const uint8_t*, size_t, int);

#define CHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); return -1; } } while(0)

#define CHECK_CUBLAS(x) do { cublasStatus_t st = (x); if (st != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st, __FILE__, __LINE__); return -2; } } while(0)

timespec stop, start;

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
    printf("Created CUBLAS handle %x\n", handle);
    return (uint64_t)handle;
}
void cublasHandleDestroy(uint64_t handle) {
    cublasDestroy((cublasHandle_t)handle);
}
/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int rows1, int cols1, int rows2, int cols2, const float* A, const float* B, float* C) {
    int i;
    int j;
    int k;

    for (i = 0; i < rows1; ++i) {
        for (j = 0; j < cols2; ++j) {
            float prod = 0;
            for (k = 0; k < cols1; ++k) {
                prod += A[k * rows1 + i] * B[j * rows2 + k];
            }
            C[j * rows1 + i] = prod;// alpha* prod + beta * C[j * rows1 + i];
        }
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
        }
        else {
            // subnormal
            f_exp = (127 - 15 + 1) << 23;
            while ((h_sig & 0x0400u) == 0) {
                h_sig <<= 1;
                f_exp -= 1 << 23;
            }
            h_sig &= 0x03FFu;
            f_sig = (uint32_t)h_sig << 13;
        }
    }
    else if (h_exp == 0x7C00u) {
        f_exp = 0xFFu << 23;
        f_sig = (uint32_t)h_sig << 13;
    }
    else {
        f_exp = ((h_exp >> 10) + (127 - 15)) << 23;
        f_sig = (uint32_t)h_sig << 13;
    }
    uint32_t f_bits = f_sgn | f_exp | f_sig;
    float f;
    memcpy(&f, &f_bits, sizeof(f));
    return f;
}
static inline float htof(half h) {
    uint16_t* p = (uint16_t*)(&h);
    uint32_t fval = 0;
    fval |= (*p & 0x8000) << 16;				// sign
    uint32_t mant = *p & 0x03ff;
    uint32_t exp = (*p & 0x7c00) >> 10;			// exponential
    if (exp == 0x1f) {							// NaN or Infinity
        fval |= mant ? 0x7fc00000 : 0x7f800000;
    }
    else
        if (exp > 0) {							// normalized
            fval |= (exp + 0x70) << 23;
            if (mant != 0) {
                fval |= mant << 13;
            }
        }
        else
            if (mant != 0) {						// denormarlized
                for (int i = 9; i >= 0; i--) {
                    if (mant & (1 << i)) {
                        fval |= ((0x67 + i) << 23) | ((mant << (23 - i)) & 0x7fffff);
                        break;
                    }
                }
            }
    return *((float*)&fval);
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
float* toDeviceFloatF16(const uint8_t * h_src, size_t len, int typeSize) {
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
float* toDeviceFloatBF16(const uint8_t * h_src, size_t len, int typeSize) {
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

JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Gemm_cublasHandle(JNIEnv* env, jclass clazz) {
    cublasStatus_t status;
    cublasHandle_t handle = NULL;

    /* Initialize CUBLAS */
    printf("CUBLAS creating handle...\n");
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasCreate CUBLAS initialization error %s\n", cublasGetStatusString(status));
        return (jlong)JNI_ERR;
    }
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream(handle, stream);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    // JNI-side once per handle
    //cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    // For FP16 GEMMs, use CUDA_R_16F inputs + tensor ops kernels
    //HOST mode means cuBLAS will synchronize the stream, write the value back into host memory, and only then return.
    //cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    //The kernel writes into that device buffer asynchronously, and you can copy it back later 
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    printf("CUBLAS handle created...\n");
    return (jlong)handle;
}

JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_cublasHandleDestroy(JNIEnv* env, jclass clazz, jlong handle) {
    cublasStatus_t status;
    /* Shutdown */
    status = cublasDestroy((cublasHandle_t)handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasDestroy shutdown error (A) %s\n", cublasGetStatusString(status));
        return JNI_ERR;
    }
    return JNI_OK;
}
JNIEXPORT jfloat JNICALL Java_com_neocoretechs_cublas_Gemm_sdotSlice(JNIEnv* env, jclass clazz, jlong handle, jobject qBuf, jint qOffsetFloats, jobject kBuf, jint kOffsetFloats, jint headSize) {
    // Get base host pointers from direct ByteBuffers
    void* qHostBase = env->GetDirectBufferAddress(qBuf);
    void* kHostBase = env->GetDirectBufferAddress(kBuf);
    if (!qHostBase || !kHostBase) {
        // Not direct or null
        return NAN;
    }
    const float* qHost = reinterpret_cast<const float*>(qHostBase) + qOffsetFloats;
    const float* kHost = reinterpret_cast<const float*>(kHostBase) + kOffsetFloats;
    return sdotSliceCuBLAS((uint64_t)handle, qHost, kHost, headSize);
}
/*
 * Class:     com_neocoretechs_cublas_Gemm
 * Method:    matrixDotProductF
 * Signature: (LII[DII[D[D)I
 */
JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_matrixDotProductF(JNIEnv* env, jclass clazz, jlong handle, jint rows1, jint columns1, jfloatArray m1, jint rows2, jint columns2, jfloatArray m2, jfloatArray mr) {
    cublasStatus_t status;
    cudaError_t cudaErr;

    float* h_A = 0;
    float* h_B = 0;
    float* h_C = 0;
    //float *h_C_ref = 0;
    float* d_A = 0;
    float* d_B = 0;
    float* d_C = 0;

    float alpha = 1.0f;
    float beta = 0.0f;

    const int n2 = rows2 * columns2;
    const int n1 = rows1 * columns1;
    const int nc = rows1 * columns2;
    /* for test vs CPU
    float error_norm;
    float ref_norm;
    float diff;
    */

    /* Allocate host memory for the matrices */
    /*h_A = (float*)(malloc(n1 * sizeof(h_A[0])));

    if (h_A == 0) {
      fprintf(stderr, "!!!! host memory allocation error (A)\n");
      return NULL;
    }
    */
    /*h_B = (float*)(malloc(n2 * sizeof(h_B[0])));

    if (h_B == 0) {
      fprintf(stderr, "!!!! host memory allocation error (B)\n");
      return NULL;
    }
    */
    //printf("Get Float h_A...\n");
    h_A = env->GetFloatArrayElements(m1, NULL);
    //printf("Get Float h_B...\n");
    h_B = env->GetFloatArrayElements(m2, NULL);
    //printf("Get Float h_C...\n");
    h_C = env->GetFloatArrayElements(mr, NULL);
    /*
    h_C = (float *)(malloc(nc * sizeof(h_C[0])));

    if (h_C == 0) {
      printf("!!!! host memory allocation error (C)\n");
      return JNI_ERR;
    }
    */
    /* Allocate device memory for the matrices */

    //_timespec64 start;
    //_timespec64 stop;

    //_timespec64_get(&start, TIME_UTC);
    //printf("CUDA malloc d_A...\n");
    cudaErr = cudaMalloc((void**)(&d_A), n1 * sizeof(d_A[0]));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasSgemm device memory allocation error (allocate A) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA malloc d_A/B...%d\n",(stop.tv_nsec - start.tv_nsec));
    //_timespec64_get(&start, TIME_UTC);
    cudaErr = cudaMalloc((void**)(&d_B), n2 * sizeof(d_B[0]));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasSgemm device memory allocation error (allocate B) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA malloc d_B/C...%d\n", (stop.tv_nsec - start.tv_nsec));
    //_timespec64_get(&start, TIME_UTC);
    cudaErr = cudaMalloc((void**)(&d_C), nc * sizeof(d_C[0]));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasSgemm device memory allocation error (allocate C) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA malloc d_C...%d\n", (stop.tv_nsec - start.tv_nsec));
    /* Initialize the device matrices with the host matrices */
    //_timespec64_get(&start, TIME_UTC);
    status = cublasSetVector(n1, sizeof(float), h_A, 1, d_A, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasSgemm device access error (write A) %s\n", cublasGetStatusString(status));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA setVector d_A/B...%d\n", (stop.tv_nsec - start.tv_nsec));
    //_timespec64_get(&start, TIME_UTC);
    status = cublasSetVector(n2, sizeof(float), h_B, 1, d_B, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasSgemm device access error (write B) %s\n", cublasGetStatusString(status));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA setVector d_B/C...%d\n", (stop.tv_nsec - start.tv_nsec));
    //_timespec64_get(&start, TIME_UTC);
    status = cublasSetVector(nc, sizeof(float), h_C, 1, d_C, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasSgemm device access error (write C) %s\n", cublasGetStatusString(status));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA setVector C...%d\n", (stop.tv_nsec - start.tv_nsec));
    /* Performs operation using plain C code */
   // simple_sgemm(N, alpha, h_A, h_B, beta, h_C);
   // h_C_ref = h_C;
   // printf("cublasDgemm...\n");
    /* Performs operation using cublas */
    //_timespec64_get(&start, TIME_UTC);
    status = cublasSgemm((cublasHandle_t)handle, CUBLAS_OP_N, CUBLAS_OP_N, rows1, columns2, columns1, &alpha, d_A, rows1, d_B, rows2, &beta, d_C, rows1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasSgemm kernel execution error %s\n", cublasGetStatusString(status));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA cublasSgemm...%d\n", (stop.tv_nsec - start.tv_nsec));
    /* Allocate host memory for reading back the result from device memory
    h_C = (float *)(malloc(nc * sizeof(h_C[0])));

    if (h_C == 0) {
      printf("!!!! host memory allocation error (C)\n");
      return JNI_ERR;
    }
    */
    /* Read the result back */
   //printf("cublasGetVector d_C...\n");
    //_timespec64_get(&start, TIME_UTC);
    status = cublasGetVector(nc, sizeof(h_C[0]), d_C, 1, h_C, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasSgemm device access error (read C) %s\n", cublasGetStatusString(status));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA getVector d_C...%d\n", (stop.tv_nsec - start.tv_nsec));
    /* Check result against reference
    error_norm = 0;
    ref_norm = 0;

    for (i = 0; i < n2; ++i) {
      diff = h_C_ref[i] - h_C[i];
      error_norm += diff * diff;
      ref_norm += h_C_ref[i] * h_C_ref[i];
    }

    error_norm = static_cast<float>(sqrt(static_cast<float>(error_norm)));
    ref_norm = static_cast<float>(sqrt(static_cast<float>(ref_norm)));

    if (fabs(ref_norm) < 1e-7) {
      fprintf(stderr, "!!!! reference norm is 0\n");
      return NULL;
    }
    */
    //_timespec64_get(&start, TIME_UTC);
    //printf("set h_C/mr float array region...\n");
    env->SetFloatArrayRegion(mr, 0, nc, h_C);
    //printf("release h_A/m1 array region...\n");
    env->ReleaseFloatArrayElements(m1, h_A, JNI_ABORT);
    //printf("release h_B/m2 array region...\n");
    env->ReleaseFloatArrayElements(m2, h_B, JNI_ABORT);
    //printf("release h_C/mr array region...\n");
    env->ReleaseFloatArrayElements(mr, h_C, JNI_ABORT);
    /* Memory clean up */
    //free(h_A);
    //free(h_B);
    //free(h_C);
    //free(h_C_ref);
    cudaErr = cudaFree(d_A);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasSgemm memory free error (A) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaFree(d_B);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasSgemm memory free error (B) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaFree(d_C);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasSgemm memory free error (C) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA FREE ALL...%d\n", (stop.tv_nsec - start.tv_nsec));
    return JNI_OK;

    /*
    if (error_norm / ref_norm < 1e-6f) {
      printf("cublasSgemm test passed.\n");
      exit(EXIT_SUCCESS);
    } else {
      printf("cublasSgemm test failed.\n");
      exit(EXIT_FAILURE);
    }
    */
}
/*
 * Class:     com_neocoretechs_cublas_Gemm
 * Method:    matrixDotProductFBatch
 * Signature: (JIILjava/util/ArrayList;IILjava/util/ArrayList;Ljava/util/ArrayList;I)I
 */
JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_matrixDotProductFBatch
(JNIEnv* env, jclass clazz, jlong handle, jint rows1, jint columns1, jobject m1_AList, jint rows2, jint columns2, jobject m2_AList, jobject mr_AList, jint batchSize) {
    cublasStatus_t status;
    cudaError_t cudaErr;

    // Allocate host storage for batch_count A,B,C matrices
    float** A, ** B, ** C;

    float** h_A = 0;
    float** h_B = 0;
    float** h_C = 0;

    float** d_A = 0;
    float** d_B = 0;
    float** d_C = 0;

    jobject* m1s;
    jobject* m2s;
    jobject* mrs;

    float alpha = 1.0f;
    float beta = 0.0f;

    const int n2 = rows2 * columns2;
    const int n1 = rows1 * columns1;
    const int nc = rows1 * columns2;
    int i;

    //_timespec64 start;
    //_timespec64 stop;

    //_timespec64_get(&start, TIME_UTC);

    h_A = (float**)malloc(batchSize * sizeof(float*));
    h_B = (float**)malloc(batchSize * sizeof(float*));
    h_C = (float**)malloc(batchSize * sizeof(float*));
    //d_A = (float**)malloc(batchSize * sizeof(float*));
    //d_B = (float**)malloc(batchSize * sizeof(float*));
    //d_C = (float**)malloc(batchSize * sizeof(float*));

    m1s = (jobject*)malloc(batchSize * sizeof(jobject));
    m2s = (jobject*)malloc(batchSize * sizeof(jobject));
    mrs = (jobject*)malloc(batchSize * sizeof(jobject));

    A = (float**)malloc(batchSize * sizeof(float*));
    B = (float**)malloc(batchSize * sizeof(float*));
    C = (float**)malloc(batchSize * sizeof(float*));

    jclass aListClass = env->GetObjectClass(m1_AList);
    jmethodID alGetId = env->GetMethodID(aListClass, "get", "(I)Ljava/lang/Object;");

    for (i = 0; i < batchSize; i++) {
        cudaErr = cudaMalloc((void**)(&h_A[i]), n1 * sizeof(float));
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasSgemmBatched device memory allocation error (allocate A) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
        cudaErr = cudaMalloc((void**)(&h_B[i]), n2 * sizeof(float));
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasSgemmBatched device memory allocation error (allocate B) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
        cudaErr = cudaMalloc((void**)(&h_C[i]), nc * sizeof(float));
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasSgemmBatched device memory allocation error (allocate C) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
    }
    //printf("cublasSgemmBatched cudaMalloc1\n");
    // Copy the host array of device pointers to the device
    cudaErr = cudaMalloc((void**)&d_A, batchSize * sizeof(float*));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasSgemmBatched device memory allocation error (allocate d_A) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaMalloc((void**)&d_B, batchSize * sizeof(float*));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasSgemmBatched device memory allocation error (allocate d_B) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaMalloc((void**)&d_C, batchSize * sizeof(float*));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasSgemmBatched device memory allocation error (allocate d_C) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //printf("cublasSgemmBatched cudaMalloc2\n");
    cudaErr = cudaMemcpy(d_A, h_A, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasSgemmBatched device memory copy error (copy h_A to d_A) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaMemcpy(d_B, h_B, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasSgemmBatched device memory copy error (copy h_B to d_B) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaMemcpy(d_C, h_C, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasSgemmBatched device memory copy error (copy h_C to d_C) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //printf("cublasSgemmBatched cudaMemcpy1\n");
    // move JNI ArrayList data to allocated memory
    for (i = 0; i < batchSize; i++) {
        m1s[i] = env->CallObjectMethod(m1_AList, alGetId, i);
        A[i] = env->GetFloatArrayElements((jfloatArray)m1s[i], NULL);
        m2s[i] = env->CallObjectMethod(m2_AList, alGetId, i);
        B[i] = env->GetFloatArrayElements((jfloatArray)m2s[i], NULL);
        mrs[i] = env->CallObjectMethod(mr_AList, alGetId, i);
        C[i] = env->GetFloatArrayElements((jfloatArray)mrs[i], NULL);
        //printf("cublasSgemmBatched JNI get %d\n",i);
        status = cublasSetMatrix(rows1, columns1, sizeof(float), A[i], rows1, h_A[i], rows1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasSgemmBatched device access error (write A) %s for batch # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        //printf("cublasSgemmBatched setMatrix 1 %d\n", i);
        status = cublasSetMatrix(rows2, columns2, sizeof(float), B[i], rows2, h_B[i], rows2);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasSgemmBatched device access error (write B) %s for batch # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        //printf("cublasSgemmBatched setMatrix 2 %d\n", i);
        status = cublasSetMatrix(rows1, columns2, sizeof(float), C[i], rows1, h_C[i], rows1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasSgemmBatched device access error (write C) %s for batch # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        //printf("cublasSgemmBatched setMatrix 3 %d\n", i);
    }
    // perform the matrix ops
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA cublasDgemmBatched finished setup\n");
    /*
    _timespec64_get(&start, TIME_UTC);
    //printf("CUDA malloc d_A...\n");
    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA malloc d_A/B...%d\n", (stop.tv_nsec - start.tv_nsec));
    _timespec64_get(&start, TIME_UTC);
    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA malloc d_B/C...%d\n", (stop.tv_nsec - start.tv_nsec));
    _timespec64_get(&start, TIME_UTC);
    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA malloc d_C...%d\n", (stop.tv_nsec - start.tv_nsec));
    */
    /*
    _timespec64_get(&start, TIME_UTC);
    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA setVector d_A/B...%d\n", (stop.tv_nsec - start.tv_nsec));
    _timespec64_get(&start, TIME_UTC);

    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA setVector d_B/C...%d\n", (stop.tv_nsec - start.tv_nsec));
    _timespec64_get(&start, TIME_UTC);
    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA setVector C...%d\n", (stop.tv_nsec - start.tv_nsec));
    */
    // printf("cublasSgemm...\n");
     /* Performs operation using cublas */
     //_timespec64_get(&start, TIME_UTC);
    status = cublasSgemmBatched((cublasHandle_t)handle, CUBLAS_OP_N, CUBLAS_OP_N, rows1, columns2, columns1, &alpha, d_A, rows1, d_B, rows2, &beta, d_C, rows1, batchSize);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasSgemmBatched kernel execution error %s\n", cublasGetStatusString(status));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA cublasSgemmBatched...%d\n", (stop.tv_nsec - start.tv_nsec));

   //printf("cublasGetVector d_C...\n");
   // _timespec64_get(&start, TIME_UTC);
    for (i = 0; i < batchSize; i++) {
        //printf("CUDA cublasSgemmBatched getVector...%d\n", i);
        status = cublasGetMatrix(rows1, columns2, sizeof(float), h_C[i], rows1, C[i], rows1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasSgemmBatched device access error (read C) %s for batch # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        // _timespec64_get(&stop, TIME_UTC);
        // printf("CUDA getVector d_C...%d\n", (stop.tv_nsec - start.tv_nsec));
         //_timespec64_get(&start, TIME_UTC);
         //printf("set h_C/mr float array region...\n");
        env->SetFloatArrayRegion((jfloatArray)mrs[i], 0, nc, C[i]);
        //printf("release A/m1 array region...\n");
        env->ReleaseFloatArrayElements((jfloatArray)m1s[i], A[i], JNI_ABORT);
        //printf("release B/m2 array region...\n");
        env->ReleaseFloatArrayElements((jfloatArray)m2s[i], B[i], JNI_ABORT);
        //printf("release C/mr array region...\n");
        env->ReleaseFloatArrayElements((jfloatArray)mrs[i], C[i], JNI_ABORT);

        cudaErr = cudaFree(h_A[i]);
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasSgemmBatched memory free error (A) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }

        cudaErr = cudaFree(h_B[i]);
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasSgemmBatched memory free error (B) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }

        cudaErr = cudaFree(h_C[i]);
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasSgemmBatched memory free error (C) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }

    }
    /* JNI cleanup */
    //printf("delete local class ref...\n");
    env->DeleteLocalRef(aListClass);
    /* Pointer Memory clean up */
    //printf("free pointers A B C...\n");
    free(A);
    free(B);
    free(C);
    //printf("free pointers h_A h_B h_C...\n");
    free(h_A);
    free(h_B);
    free(h_C);
    //printf("free pointers d_A d_B d_C...\n");
    cudaErr = cudaFree(d_A);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasSgemmBatched memory free error (d_A) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaFree(d_B);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasSgemmBatched memory free error (d_B) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaFree(d_C);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasSgemmBatched memory free error (d_C) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //printf("free pointers m1s m2s mrs...\n");
    free(m1s);
    free(m2s);
    free(mrs);
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA cublasDgemmBatched getVector and FREE ALL...%d\n", (stop.tv_nsec - start.tv_nsec));
    return JNI_OK;
}

/*
 * Class:     com_neocoretechs_cublas_Gemm
 * Method:    matrixDotProductFStream
 * Signature: (JIILjava/util/ArrayList;IILjava/util/ArrayList;Ljava/util/ArrayList;I)I
 */
JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_matrixDotProductFStream
(JNIEnv* env, jclass clazz, jlong handle, jint rows1, jint columns1, jobject m1_AList, jint rows2, jint columns2, jobject m2_AList, jobject mr_AList, jint batchSize) {
    cublasStatus_t status;
    cudaError_t cudaErr;

    float** h_A = 0;
    float** h_B = 0;
    float** h_C = 0;

    float** d_A = 0;
    float** d_B = 0;
    float** d_C = 0;

    jobject* m1s;
    jobject* m2s;
    jobject* mrs;

    float alpha = 1.0f;
    float beta = 0.0f;

    const int n2 = rows2 * columns2;
    const int n1 = rows1 * columns1;
    const int nc = rows1 * columns2;
    int i;

    //_timespec64 start;
    //_timespec64 stop;

    //_timespec64_get(&start, TIME_UTC);

    h_A = (float**)malloc(batchSize * sizeof(float*));
    h_B = (float**)malloc(batchSize * sizeof(float*));
    h_C = (float**)malloc(batchSize * sizeof(float*));
    d_A = (float**)malloc(batchSize * sizeof(float*));
    d_B = (float**)malloc(batchSize * sizeof(float*));
    d_C = (float**)malloc(batchSize * sizeof(float*));

    m1s = (jobject*)malloc(batchSize * sizeof(jobject));
    m2s = (jobject*)malloc(batchSize * sizeof(jobject));
    mrs = (jobject*)malloc(batchSize * sizeof(jobject));

    jclass aListClass = env->GetObjectClass(m1_AList);
    jmethodID alGetId = env->GetMethodID(aListClass, "get", "(I)Ljava/lang/Object;");
    for (i = 0; i < batchSize; i++) {
        m1s[i] = env->CallObjectMethod(m1_AList, alGetId, i);
        h_A[i] = env->GetFloatArrayElements((jfloatArray)m1s[i], NULL);
        m2s[i] = env->CallObjectMethod(m2_AList, alGetId, i);
        h_B[i] = env->GetFloatArrayElements((jfloatArray)m2s[i], NULL);
        mrs[i] = env->CallObjectMethod(mr_AList, alGetId, i);
        h_C[i] = env->GetFloatArrayElements((jfloatArray)mrs[i], NULL);
        cudaErr = cudaMalloc((void**)(&d_A[i]), n1 * sizeof(d_A[0]));
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasSgemmStream device memory allocation error (allocate A) %s for stream # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
        cudaErr = cudaMalloc((void**)(&d_B[i]), n2 * sizeof(d_B[0]));
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasSgemmStream device memory allocation error (allocate B) %s for stream # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
        cudaErr = cudaMalloc((void**)(&d_C[i]), nc * sizeof(d_C[0]));
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasSgemmStream device memory allocation error (allocate C) %s for stream # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
        status = cublasSetVector(n1, sizeof(float), h_A[i], 1, d_A[i], 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasSgemmStream device access error (write A) %s for stream # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        status = cublasSetVector(n2, sizeof(float), h_B[i], 1, d_B[i], 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasSgemmStream device access error (write B) %s for stream # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        status = cublasSetVector(nc, sizeof(float), h_C[i], 1, d_C[i], 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasSgemmStream device access error (write C) %s for stream # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
    }

    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA cublasDgemmStream total setup time...%d\n", (stop.tv_nsec - start.tv_nsec));
    /*
    _timespec64_get(&start, TIME_UTC);
    //printf("CUDA malloc d_A...\n");
    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA malloc d_A/B...%d\n", (stop.tv_nsec - start.tv_nsec));
    _timespec64_get(&start, TIME_UTC);
    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA malloc d_B/C...%d\n", (stop.tv_nsec - start.tv_nsec));
    _timespec64_get(&start, TIME_UTC);
    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA malloc d_C...%d\n", (stop.tv_nsec - start.tv_nsec));
    */
    /*
    _timespec64_get(&start, TIME_UTC);
    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA setVector d_A/B...%d\n", (stop.tv_nsec - start.tv_nsec));
    _timespec64_get(&start, TIME_UTC);

    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA setVector d_B/C...%d\n", (stop.tv_nsec - start.tv_nsec));
    _timespec64_get(&start, TIME_UTC);
    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA setVector C...%d\n", (stop.tv_nsec - start.tv_nsec));
    */
    // Create a stream for every DGEMM operation
    cudaStream_t* streams = (cudaStream_t*)malloc(batchSize * sizeof(cudaStream_t));
    for (i = 0; i < batchSize; i++) {
        cudaErr = cudaStreamCreate(&streams[i]);
        if (cudaErr != cudaSuccess) {
            printf("!!!!GPU kernel execution error for cudaStreamCreate: %s for stream # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
    }

    // printf("cublasSgemm...\n");
     /* Performs operation using cublas */
    //_timespec64_get(&start, TIME_UTC);
    // Launch each SGEMM operation in own CUDA stream
    for (i = 0; i < batchSize; i++) {
        // Set CUDA stream
        status = cublasSetStream((cublasHandle_t)handle, streams[i]);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasSgemmStream set stream execution error %s for stream # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        status = cublasSgemm((cublasHandle_t)handle, CUBLAS_OP_N, CUBLAS_OP_N, rows1, columns2, columns1, &alpha, d_A[i], rows1, d_B[i], rows2, &beta, d_C[i], rows1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasSgemmStream kernel execution error %s for stream # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA cublasSgemmStream...%d\n", (stop.tv_nsec - start.tv_nsec));

    //printf("cublasGetVector d_C...\n");
    //_timespec64_get(&start, TIME_UTC);
    for (i = 0; i < batchSize; i++) {
        status = cublasGetVector(nc, sizeof(float), d_C[i], 1, h_C[i], 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasSgemmStream device access error (read C)  %s for stream # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        // _timespec64_get(&stop, TIME_UTC);
        // printf("CUDA getVector d_C...%d\n", (stop.tv_nsec - start.tv_nsec));
         //_timespec64_get(&start, TIME_UTC);
         //printf("set h_C/mr float array region...\n");
        env->SetFloatArrayRegion((jfloatArray)mrs[i], 0, nc, h_C[i]);
        //printf("release h_A/m1 array region...\n");
        env->ReleaseFloatArrayElements((jfloatArray)m1s[i], h_A[i], JNI_ABORT);
        //printf("release h_B/m2 array region...\n");
        env->ReleaseFloatArrayElements((jfloatArray)m2s[i], h_B[i], JNI_ABORT);
        //printf("release h_C/mr array region...\n");
        env->ReleaseFloatArrayElements((jfloatArray)mrs[i], h_C[i], JNI_ABORT);

        cudaErr = cudaFree(d_A[i]);
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasSgemmStream memory free error (A) %s for stream # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }

        cudaErr = cudaFree(d_B[i]);
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasSgemmStream memory free error (B) %s for stream # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }

        cudaErr = cudaFree(d_C[i]);
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasSgemmStream memory free error (C) %s for stream # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }

        cudaErr = cudaStreamDestroy(streams[i]);
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasSgemmStream memory free error (C) %s for stream # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
    }
    /* JNI cleanup */
    env->DeleteLocalRef(aListClass);
    /* Pointer Memory clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    free(d_A);
    free(d_B);
    free(d_C);
    free(m1s);
    free(m2s);
    free(mrs);
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA cublasDgemmStream getVector and FREE ALL...%d\n", (stop.tv_nsec - start.tv_nsec));
    return JNI_OK;
}

/*
 * Class:     com_neocoretechs_cublas_Gemm
 * Method:    matrixDotProductFCPU
 * Signature: (LII[DII[D[D)I
 */
JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_matrixDotProductFCPU(JNIEnv* env, jclass clazz, jint rows1, jint columns1, jfloatArray m1, jint rows2, jint columns2, jfloatArray m2, jfloatArray mr) {

    float* h_A = 0;
    float* h_B = 0;
    float* h_C = 0;
    //float *h_C_ref = 0;

    //float alpha = 1.0f;
    //float beta = 0.0f;

    const int n2 = rows2 * columns2;
    const int n1 = rows1 * columns1;
    const int nc = rows1 * columns2;
    /* for test vs CPU
    float error_norm;
    float ref_norm;
    float diff;
    */

    /* Allocate host memory for the matrices */
    /*h_A = (float*)(malloc(n1 * sizeof(h_A[0])));
    if (h_A == 0) {
      fprintf(stderr, "!!!! host memory allocation error (A)\n");
      return NULL;
    }
    */
    /*h_B = (float*)(malloc(n2 * sizeof(h_B[0])));
    if (h_B == 0) {
      fprintf(stderr, "!!!! host memory allocation error (B)\n");
      return NULL;
    }
    */
    //printf("Get Float h_A...\n");
    h_A = env->GetFloatArrayElements(m1, NULL);
    //printf("Get Float h_B...\n");
    h_B = env->GetFloatArrayElements(m2, NULL);
    //printf("Get Float h_C...\n");
    h_C = env->GetFloatArrayElements(mr, NULL);
    /*
    h_C = (float *)(malloc(nc * sizeof(h_C[0])));
    if (h_C == 0) {
      printf("!!!! host memory allocation error (C)\n");
      return JNI_ERR;
    }
    */
    /* Allocate device memory for the matrices */
    //_timespec64 start;
    //_timespec64 stop;
    //_timespec64_get(&start, TIME_UTC);

    //simple_sgemm(rows1, columns1, rows2, columns2, alpha, h_A, h_B, beta, h_C);
    simple_sgemm(rows1, columns1, rows2, columns2, h_A, h_B, h_C);

    //_timespec64_get(&stop, TIME_UTC);
    /* Allocate host memory for reading back the result from device memory
    h_C = (float *)(malloc(nc * sizeof(h_C[0])));
    if (h_C == 0) {
      printf("!!!! host memory allocation error (C)\n");
      return JNI_ERR;
    }
    */
    /* Check result against reference
    error_norm = 0;
    ref_norm = 0;
    for (i = 0; i < n2; ++i) {
      diff = h_C_ref[i] - h_C[i];
      error_norm += diff * diff;
      ref_norm += h_C_ref[i] * h_C_ref[i];
    }
    error_norm = static_cast<float>(sqrt(static_cast<float>(error_norm)));
    ref_norm = static_cast<float>(sqrt(static_cast<float>(ref_norm)));
    if (fabs(ref_norm) < 1e-7) {
      fprintf(stderr, "!!!! reference norm is 0\n");
      return NULL;
    }
    */
    //_timespec64_get(&start, TIME_UTC);
    //printf("set h_C/mr float array region...\n");
    env->SetFloatArrayRegion(mr, 0, nc, h_C);
    //printf("release h_A/m1 array region...\n");
    env->ReleaseFloatArrayElements(m1, h_A, JNI_ABORT);
    //printf("release h_B/m2 array region...\n");
    env->ReleaseFloatArrayElements(m2, h_B, JNI_ABORT);
    //printf("release h_C/mr array region...\n");
    env->ReleaseFloatArrayElements(mr, h_C, JNI_ABORT);
    /* Memory clean up */
    //free(h_A);
    //free(h_B);
    //free(h_C);
    //free(h_C_ref);

    //printf("FREE ALL...%d\n", (stop.tv_nsec - start.tv_nsec));
    /*
    if (error_norm / ref_norm < 1e-6f) {
      printf("cublasSgemm test passed.\n");
      exit(EXIT_SUCCESS);
    } else {
      printf("simpleSgemm test failed.\n");
      exit(EXIT_FAILURE);
    }
    */
    return JNI_OK;
}

/*
 * Class:     com_neocoretechs_cublas_Gemm
 * Method:    matrixDotProductFCPUBatch
 * Signature: (JIILjava/util/ArrayList;IILjava/util/ArrayList;Ljava/util/ArrayList;I)I
 */
JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_matrixDotProductFCPUBatch
(JNIEnv* env, jclass clazz, jint rows1, jint columns1, jobject m1_AList, jint rows2, jint columns2, jobject m2_AList, jobject mr_AList, jint batchSize) {

    float** h_A = 0;
    float** h_B = 0;
    float** h_C = 0;

    jobject* m1s;
    jobject* m2s;
    jobject* mrs;

    //float alpha = 1.0f;
    //float beta = 0.0f;

    const int n2 = rows2 * columns2;
    const int n1 = rows1 * columns1;
    const int nc = rows1 * columns2;
    int i;

    //_timespec64 start;
    //_timespec64 stop;

    //_timespec64_get(&start, TIME_UTC);

    h_A = (float**)malloc(batchSize * sizeof(float*));
    h_B = (float**)malloc(batchSize * sizeof(float*));
    h_C = (float**)malloc(batchSize * sizeof(float*));

    m1s = (jobject*)malloc(batchSize * sizeof(jobject));
    m2s = (jobject*)malloc(batchSize * sizeof(jobject));
    mrs = (jobject*)malloc(batchSize * sizeof(jobject));

    jclass aListClass = env->GetObjectClass(m1_AList);
    jmethodID alGetId = env->GetMethodID(aListClass, "get", "(I)Ljava/lang/Object;");
    for (i = 0; i < batchSize; i++) {
        m1s[i] = env->CallObjectMethod(m1_AList, alGetId, i);
        h_A[i] = env->GetFloatArrayElements((jfloatArray)m1s[i], NULL);
        m2s[i] = env->CallObjectMethod(m2_AList, alGetId, i);
        h_B[i] = env->GetFloatArrayElements((jfloatArray)m2s[i], NULL);
        mrs[i] = env->CallObjectMethod(mr_AList, alGetId, i);
        h_C[i] = env->GetFloatArrayElements((jfloatArray)mrs[i], NULL);
    }
    // Launch each SGEMM operation
    for (i = 0; i < batchSize; i++) {
        //simple_sgemm(rows1, columns1, rows2, columns2, alpha, h_A[i], h_B[i], beta, h_C[i]);
        simple_sgemm(rows1, columns1, rows2, columns2, h_A[i], h_B[i], h_C[i]);
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("simple_Dgemm...%d\n", (stop.tv_nsec - start.tv_nsec));

    //_timespec64_get(&start, TIME_UTC);
    for (i = 0; i < batchSize; i++) {
        // _timespec64_get(&stop, TIME_UTC);
         //_timespec64_get(&start, TIME_UTC);
         //printf("set h_C/mr float array region...\n");
        env->SetFloatArrayRegion((jfloatArray)mrs[i], 0, nc, h_C[i]);
        //printf("release h_A/m1 array region...\n");
        env->ReleaseFloatArrayElements((jfloatArray)m1s[i], h_A[i], JNI_ABORT);
        //printf("release h_B/m2 array region...\n");
        env->ReleaseFloatArrayElements((jfloatArray)m2s[i], h_B[i], JNI_ABORT);
        //printf("release h_C/mr array region...\n");
        env->ReleaseFloatArrayElements((jfloatArray)mrs[i], h_C[i], JNI_ABORT);
    }
    /* JNI cleanup */
    env->DeleteLocalRef(aListClass);
    /* Pointer Memory clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    free(m1s);
    free(m2s);
    free(mrs);
    //_timespec64_get(&stop, TIME_UTC);
    //printf("simple_sgemm FREE ALL...%d\n", (stop.tv_nsec - start.tv_nsec));
    return JNI_OK;
}
JNIEXPORT jlongArray JNICALL Java_com_neocoretechs_cublas_Gemm_cudaMemGetInfo(JNIEnv* env, jclass clazz) {
    size_t freeMem = 0, totalMem = 0;
    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    jlongArray result = env->NewLongArray(2);
    if (result == NULL) return NULL;

    jlong vals[2];
    if (err == cudaSuccess) {
        vals[0] = (jlong)freeMem;
        vals[1] = (jlong)totalMem;
    }
    else {
        // On error, return -1s so you can detect it in Java
        vals[0] = -1;
        vals[1] = -1;
    }
    env->SetLongArrayRegion(result, 0, 2, vals);
    return result;
}
/**
* Simple dot product single precision
*/
JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_sdot(JNIEnv* env, jclass clazz, jlong handle, jint n, jfloatArray x, jint incx, jfloatArray y, jint incy, jfloatArray result) {
    // Copy input arrays from JVM
    jfloat* hx = env->GetFloatArrayElements(x, nullptr);
    jfloat* hy = env->GetFloatArrayElements(y, nullptr);
    float* dx, * dy, * dres;
    cudaMalloc(&dx, n * sizeof(float));
    cudaMalloc(&dy, n * sizeof(float));
    cudaMalloc(&dres, sizeof(float));
    cudaMemcpy(dx, hx, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, n * sizeof(float), cudaMemcpyHostToDevice);
    float hostRes = 0.0f;
    cublasStatus_t stat = cublasSdot((cublasHandle_t)handle, n, dx, incx, dy, incy, &hostRes);
    // Copy result back
    env->SetFloatArrayRegion(result, 0, 1, &hostRes);
    // Cleanup
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dres);
    env->ReleaseFloatArrayElements(x, hx, JNI_ABORT);
    env->ReleaseFloatArrayElements(y, hy, JNI_ABORT);
    return (stat == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
}

// Device buffer helpers
JNIEXPORT jlong JNICALL Java_com_neocoretechs_cublas_Gemm_cudaMallocBytes(JNIEnv* env, jclass clazz, jlong bytes) {
    void* dptr = nullptr;
    CHECK_CUDA(cudaMalloc(&dptr, (size_t)bytes));
    return (jlong)dptr;
}

JNIEXPORT void JNICALL Java_com_neocoretechs_cublas_Gemm_cudaFreePtr(JNIEnv* env, jclass clazz, jlong dptr) {
    cudaFree((void*)dptr);
}

JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_cudaMemcpyHtoD(JNIEnv* env, jclass clazz, jlong dptr, jobject srcBuffer, jlong bytes) {
    void* hptr = env->GetDirectBufferAddress(srcBuffer);
    if (!hptr) {
        printf("!!!!cudaMemcpyHtoD GetDirectBufferAddress failed!\n");
        return -2;
    }
    CHECK_CUDA(cudaMemcpy((void*)dptr, hptr, (size_t)bytes, cudaMemcpyHostToDevice));
    return 0;
}

JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_cudaMemcpyDtoH(JNIEnv* env, jclass clazz, jobject dstBuffer, jlong dptr, jlong bytes) {
    void* hptr = env->GetDirectBufferAddress(dstBuffer);
    if (!hptr) {
        printf("!!!!cudaMemcpyDtoH GetDirectBufferAddress failed!\n");
        return -2;
    }
    CHECK_CUDA(cudaMemcpy(hptr, (void*)dptr, (size_t)bytes, cudaMemcpyDeviceToHost));
    return 0;
}

// Device-pointer dot (no array marshalling, no per-call mallocs)
JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_sdotDevice(JNIEnv* env, jclass clazz, jlong handle, jint n, jlong dX, jint incx, jlong dY, jint incy, jlong dResult) {
    CHECK_CUBLAS(cublasSdot((cublasHandle_t)handle, n, (const float*)dX, incx, (const float*)dY, incy, (float*)dResult));
    return 0;
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
    if (buffer == NULL) {
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
    switch (format) {
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
        return (jlong)(uintptr_t)toDeviceFloatQ8(buffer, outBytes, 0, blockSize, typeSize, headerBytes);
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
