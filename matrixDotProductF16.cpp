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
* cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH); once per handle to enable tensor cores
*
*cublasGemmEx(handle,
*    CUBLAS_OP_N, CUBLAS_OP_T,
*    nTokens, T, headSize,
*    &alpha,
*    dQ, CUDA_R_16F, nTokens,   // FP16 Q
*    dK, CUDA_R_16F, T,         // FP16 K
*    &beta,
*    dS, CUDA_R_32F, nTokens,   // FP32 output
*    CUDA_R_32F,                // compute type (accumulate in FP32)
*    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <concurrent_vector.h>
#include "com_neocoretechs_cublas_Gemm.h"
#include "helpers.h"

auto ck = [](cudaError_t e, const char* msg) { if (e != cudaSuccess) { printf("%s: %s\n", msg, cudaGetErrorString(e)); return true; } return false; };


/*
 * Class:     com_neocoretechs_cublas_Gemm
 * Method:    matrixDotProductF16
 * Signature: (LII[DII[D[D)I
 */
JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_matrixDotProductF16(JNIEnv* env, jclass clazz, jlong handle, jint rows1, jint columns1, jfloatArray m1, jint rows2, jint columns2, jfloatArray m2, jfloatArray mr) {
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
    int i;
   
    //printf("Get Float h_A...\n");
    h_A = env->GetFloatArrayElements(m1, NULL);
    //printf("Get Float h_B...\n");
    h_B = env->GetFloatArrayElements(m2, NULL);
    //printf("Get Float h_C...\n");
    h_C = env->GetFloatArrayElements(mr, NULL);
  
    //_timespec64 start;
    //_timespec64 stop;

    //_timespec64_get(&start, TIME_UTC);
    //printf("CUDA malloc d_A...\n");
    cudaErr = cudaMalloc((void**)(&d_A), n1 * sizeof(d_A[0]));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmEx device memory allocation error (allocate A) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA malloc d_A/B...%d\n",(stop.tv_nsec - start.tv_nsec));
    //_timespec64_get(&start, TIME_UTC);
    cudaErr = cudaMalloc((void**)(&d_B), n2 * sizeof(d_B[0]));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmEx device memory allocation error (allocate B) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA malloc d_B/C...%d\n", (stop.tv_nsec - start.tv_nsec));
    //_timespec64_get(&start, TIME_UTC);
    cudaErr = cudaMalloc((void**)(&d_C), nc * sizeof(d_C[0]));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmEx device memory allocation error (allocate C) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA malloc d_C...%d\n", (stop.tv_nsec - start.tv_nsec));
    /* Initialize the device matrices with the host matrices */
    //_timespec64_get(&start, TIME_UTC);
    status = cublasSetVector(n1, sizeof(float), h_A, 1, d_A, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasGemmEx device access error (write A) %s\n", cublasGetStatusString(status));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA setVector d_A/B...%d\n", (stop.tv_nsec - start.tv_nsec));
    //_timespec64_get(&start, TIME_UTC);
    status = cublasSetVector(n2, sizeof(float), h_B, 1, d_B, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasGemmEx device access error (write B) %s\n", cublasGetStatusString(status));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA setVector d_B/C...%d\n", (stop.tv_nsec - start.tv_nsec));
    //_timespec64_get(&start, TIME_UTC);
    status = cublasSetVector(nc, sizeof(float), h_C, 1, d_C, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasGemmEx device access error (write C) %s\n", cublasGetStatusString(status));
        return JNI_ERR;
    }

    status = cublasGemmEx((cublasHandle_t)handle, CUBLAS_OP_N, CUBLAS_OP_N, rows1, columns2, columns1, &alpha,
        d_A, CUDA_R_32F, rows1, 
        d_B, CUDA_R_32F, rows2, &beta,
        d_C, CUDA_R_32F, rows1, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasGemmEx kernel execution error %s\n", cublasGetStatusString(status));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    /* Read the result back */
   //printf("cublasGetVector d_C...\n");
    //_timespec64_get(&start, TIME_UTC);
    status = cublasGetVector(nc, sizeof(h_C[0]), d_C, 1, h_C, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasGemmEx device access error (read C) %s\n", cublasGetStatusString(status));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA getVector d_C...%d\n", (stop.tv_nsec - start.tv_nsec));
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
    cudaErr = cudaFree(d_A);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmEx memory free error (A) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaFree(d_B);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmEx memory free error (B) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaFree(d_C);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmEx memory free error (C) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA FREE ALL...%d\n", (stop.tv_nsec - start.tv_nsec));
    return JNI_OK;
}
/*
 * Class:     com_neocoretechs_cublas_Gemm
 * Method:    matrixDotProductF16Batch
 * Signature: (JIILjava/util/ArrayList;IILjava/util/ArrayList;Ljava/util/ArrayList;I)I
 */
JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_matrixDotProductF16Batch
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
            printf("!!!!cublasGemmBatchedEx device memory allocation error (allocate A) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
        cudaErr = cudaMalloc((void**)(&h_B[i]), n2 * sizeof(float));
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmBatchedEx device memory allocation error (allocate B) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
        cudaErr = cudaMalloc((void**)(&h_C[i]), nc * sizeof(float));
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmBatchedEx device memory allocation error (allocate C) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
    }
    //printf("cublasGemmBatchedEx cudaMalloc1\n");
    // Copy the host array of device pointers to the device
    cudaErr = cudaMalloc((void**)&d_A, batchSize * sizeof(float*));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmBatchedEx device memory allocation error (allocate d_A) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaMalloc((void**)&d_B, batchSize * sizeof(float*));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmBatchedEx device memory allocation error (allocate d_B) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaMalloc((void**)&d_C, batchSize * sizeof(float*));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmBatchedEx device memory allocation error (allocate d_C) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //printf("cublasGemmBatchedEx cudaMalloc2\n");
    cudaErr = cudaMemcpy(d_A, h_A, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmBatchedEx device memory copy error (copy h_A to d_A) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaMemcpy(d_B, h_B, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmBatchedEx device memory copy error (copy h_B to d_B) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaMemcpy(d_C, h_C, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmBatchedEx device memory copy error (copy h_C to d_C) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //printf("cublasgGemmBatchedEx cudaMemcpy1\n");
    // move JNI ArrayList data to allocated memory
    for (i = 0; i < batchSize; i++) {
        m1s[i] = env->CallObjectMethod(m1_AList, alGetId, i);
        A[i] = env->GetFloatArrayElements((jfloatArray)m1s[i], NULL);
        m2s[i] = env->CallObjectMethod(m2_AList, alGetId, i);
        B[i] = env->GetFloatArrayElements((jfloatArray)m2s[i], NULL);
        mrs[i] = env->CallObjectMethod(mr_AList, alGetId, i);
        C[i] = env->GetFloatArrayElements((jfloatArray)mrs[i], NULL);
        //printf("cublasGemmBatchedEx JNI get %d\n",i);
        status = cublasSetMatrix(rows1, columns1, sizeof(float), A[i], rows1, h_A[i], rows1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasGemmBatchedEx device access error (write A) %s for batch # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        //printf("cublasGemmBatchedEx setMatrix 1 %d\n", i);
        status = cublasSetMatrix(rows2, columns2, sizeof(float), B[i], rows2, h_B[i], rows2);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasGemmBatchedEx device access error (write B) %s for batch # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        //printf("cublasGemmBatchedEx setMatrix 2 %d\n", i);
        status = cublasSetMatrix(rows1, columns2, sizeof(float), C[i], rows1, h_C[i], rows1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasGemmBatchedEx device access error (write C) %s for batch # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        //printf("cublasGemmBatchedEx setMatrix 3 %d\n", i);
    }
  
    status = cublasGemmBatchedEx((cublasHandle_t)handle, CUBLAS_OP_N, CUBLAS_OP_N, rows1, columns2, 
        columns1, &alpha,
        (const void**)d_A, CUDA_R_32F, rows1,
        (const void**)d_B, CUDA_R_32F, rows2, &beta,
        (void**)d_C, CUDA_R_32F, rows1, batchSize, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT/*_TENSOR_OP*/);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasGemmBatched kernel execution error %s\n", cublasGetStatusString(status));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA cublasGemmBatchedEx...%d\n", (stop.tv_nsec - start.tv_nsec));

   //printf("cublasGetVector d_C...\n");
   // _timespec64_get(&start, TIME_UTC);
    for (i = 0; i < batchSize; i++) {
        //printf("CUDA cublasGemmBatchedEx getVector...%d\n", i);
        status = cublasGetMatrix(rows1, columns2, sizeof(float), h_C[i], rows1, C[i], rows1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasGemmBatchedEx device access error (read C) %s for batch # %d\n", cublasGetStatusString(status), i);
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
            printf("!!!!cublasGemmBatchedEx memory free error (A) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }

        cudaErr = cudaFree(h_B[i]);
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmBatchedEx memory free error (B) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }

        cudaErr = cudaFree(h_C[i]);
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmBatchedEx memory free error (C) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
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
        printf("!!!!cublasGemmBatchedEx memory free error (d_A) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaFree(d_B);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmBatchedEx memory free error (d_B) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaFree(d_C);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmBatchedEx memory free error (d_C) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //printf("free pointers m1s m2s mrs...\n");
    free(m1s);
    free(m2s);
    free(mrs);
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA cublasGemmBatchedEx getVector and FREE ALL...%d\n", (stop.tv_nsec - start.tv_nsec));
    return JNI_OK;
}
/*
* F16 Strided Batch
* Takes ArrayList of float 32 and coverts to F16 before matrix multiply
*/
JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_matrixDotProductF16StridedBatch
(JNIEnv* env, jclass clazz, jlong handle, jint rows1, jint columns1, jobject m1_AList, jint rows2, jint columns2, jobject m2_AList, jobject mr_AList, jint batchSize) {
 
    cublasStatus_t status;
    cudaError_t cudaErr;

    // Allocate host storage for batch_count A,B,C matrices
    float** A, ** B, ** C;

    float** h_A = 0;
    float** h_B = 0;
    float** h_C = 0;
    // device pointers
    float** d_A = 0;
    float** d_B = 0;
    float** d_C = 0;

    jobject* m1s;
    jobject* m2s;
    jobject* mrs;

    float alpha = 1.0f;
    float beta = 0.0f;

    const int M = rows1; // Number of rows in A and C
    const int N = columns2; // Number of columns in B and C
    const int K = rows2; // Number of columns in A and rows in B

    const int n2 = rows2 * columns2;
    const int n1 = rows1 * columns1;
    const int nc = rows1 * columns2;
    int i;

    // Device pointers
    cudaMalloc((void**)&d_A, batchSize * sizeof(float*));
    cudaMalloc((void**)&d_B, batchSize * sizeof(float*));
    cudaMalloc((void**)&d_C, batchSize * sizeof(float*));
    // Copy matrices to device
    for (int i = 0; i < batchSize; i++) {
        cudaMalloc((void**)&d_A[i], M * K * sizeof(float));
        cudaMemcpy(d_A[i], h_A[i], M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_B[i], K * N * sizeof(float));
        cudaMemcpy(d_B[i], h_B[i], K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_C[i], M * N * sizeof(float));
    }

    jclass aListClass = env->GetObjectClass(m1_AList);
    jmethodID alGetId = env->GetMethodID(aListClass, "get", "(I)Ljava/lang/Object;");

    for (i = 0; i < batchSize; i++) {
        cudaErr = cudaMalloc((void**)(&h_A[i]), n1 * sizeof(float));
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmBatchedEx device memory allocation error (allocate A) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
        cudaErr = cudaMalloc((void**)(&h_B[i]), n2 * sizeof(float));
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmBatchedEx device memory allocation error (allocate B) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
        cudaErr = cudaMalloc((void**)(&h_C[i]), nc * sizeof(float));
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmBatchedEx device memory allocation error (allocate C) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
    }
    //printf("cublasGemmBatchedEx cudaMalloc1\n");
    // Copy the host array of device pointers to the device
    cudaErr = cudaMalloc((void**)&d_A, batchSize * sizeof(float*));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmBatchedEx device memory allocation error (allocate d_A) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaMalloc((void**)&d_B, batchSize * sizeof(float*));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmBatchedEx device memory allocation error (allocate d_B) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaMalloc((void**)&d_C, batchSize * sizeof(float*));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmBatchedEx device memory allocation error (allocate d_C) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //printf("cublasGemmBatchedEx cudaMalloc2\n");
    cudaErr = cudaMemcpy(d_A, h_A, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmBatchedEx device memory copy error (copy h_A to d_A) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaMemcpy(d_B, h_B, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmBatchedEx device memory copy error (copy h_B to d_B) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaMemcpy(d_C, h_C, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmBatchedEx device memory copy error (copy h_C to d_C) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //printf("cublasgGemmBatchedEx cudaMemcpy1\n");
    // move JNI ArrayList data to allocated memory
    for (i = 0; i < batchSize; i++) {
        m1s[i] = env->CallObjectMethod(m1_AList, alGetId, i);
        A[i] = env->GetFloatArrayElements((jfloatArray)m1s[i], NULL);
        m2s[i] = env->CallObjectMethod(m2_AList, alGetId, i);
        B[i] = env->GetFloatArrayElements((jfloatArray)m2s[i], NULL);
        mrs[i] = env->CallObjectMethod(mr_AList, alGetId, i);
        C[i] = env->GetFloatArrayElements((jfloatArray)mrs[i], NULL);
        //printf("cublasGemmBatchedEx JNI get %d\n",i);
        status = cublasSetMatrix(rows1, columns1, sizeof(float), A[i], rows1, h_A[i], rows1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasGemmBatchedEx device access error (write A) %s for batch # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        //printf("cublasGemmBatchedEx setMatrix 1 %d\n", i);
        status = cublasSetMatrix(rows2, columns2, sizeof(float), B[i], rows2, h_B[i], rows2);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasGemmBatchedEx device access error (write B) %s for batch # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        //printf("cublasGemmBatchedEx setMatrix 2 %d\n", i);
        status = cublasSetMatrix(rows1, columns2, sizeof(float), C[i], rows1, h_C[i], rows1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasGemmBatchedEx device access error (write C) %s for batch # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        //printf("cublasGemmBatchedEx setMatrix 3 %d\n", i);
    }
    /*
    cublasStatus_t cublasGemmStridedBatchedEx(
        cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m, int n, int k,
        const void* alpha,
        const void* A, cudaDataType Atype, int lda, long long int strideA,
        const void* B, cudaDataType Btype, int ldb, long long int strideB,
        const void* beta,
        void* C, cudaDataType Ctype, int ldc, long long int strideC,
        int batchCount,
        cublasComputeType_t computeType,
        cudaDataType scaleType
    );
    */
  
    long long strideA = (long long)(columns1 * columns2);   // V
    long long strideB = (long long)(rows1 * columns1);  // S
    long long strideC = (long long)(rows1 * columns2);   // O

    int lda = columns2;    // A=V, opA=N → lda ≥ m=d
    int ldb = columns1;   // B=S, opB=T → ldb ≥ k=Tk
    int ldc = columns2;    // C=O,       → ldc ≥ m=d

    status = cublasGemmStridedBatchedEx((cublasHandle_t)handle, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K, &alpha, 
        (const void**)d_A,
        CUDA_R_32F, lda, strideA, 
        (const void**)d_B,
        CUDA_R_32F, ldb, strideB, &beta,
        (void**)d_C,
        CUDA_R_32F, ldc, strideC, batchSize, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasGemmBatched kernel execution error %s\n", cublasGetStatusString(status));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA cublasGemmBatchedEx...%d\n", (stop.tv_nsec - start.tv_nsec));
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("Post-GEMM sync error: %s\n", cudaGetErrorString(e));
        return -206;
    }
   //printf("cublasGetVector d_C...\n");
   // _timespec64_get(&start, TIME_UTC);
    for (i = 0; i < batchSize; i++) {
        //printf("CUDA cublasGemmBatchedEx getVector...%d\n", i);
        status = cublasGetMatrix(rows1, columns2, sizeof(float), h_C[i], rows1, C[i], rows1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasGemmBatchedEx device access error (read C) %s for batch # %d\n", cublasGetStatusString(status), i);
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
            printf("!!!!cublasGemmBatchedEx memory free error (A) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }

        cudaErr = cudaFree(h_B[i]);
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmBatchedEx memory free error (B) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }

        cudaErr = cudaFree(h_C[i]);
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmBatchedEx memory free error (C) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
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
        printf("!!!!cublasGemmBatchedEx memory free error (d_A) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaFree(d_B);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmBatchedEx memory free error (d_B) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaFree(d_C);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmBatchedEx memory free error (d_C) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //printf("free pointers m1s m2s mrs...\n");
    free(m1s);
    free(m2s);
    free(mrs);
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA cublasGemmBatchedEx getVector and FREE ALL...%d\n", (stop.tv_nsec - start.tv_nsec));
    return JNI_OK;
}
/**
* Start of flat strided batch 2 and 1
* Takes arrays of native float 32 and converts them to F16 before matrix multiply
*/
JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_matrixDotProductF16StridedBatchFlat2
(JNIEnv* env, jclass clazz, jlong jHandle,
    jint rowsA, jint colsA, jfloatArray jA,
    jint rowsB, jint colsB, jfloatArray jB,
    jfloatArray jC, jint batchSize) {

    cublasHandle_t handle = reinterpret_cast<cublasHandle_t>(jHandle);
    cublasStatus_t cs = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        printf("SetMathMode tensor failed: %d, falling back to DEFAULT\n", (int)cs);
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
    cs = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        printf("SetPointerMode failed: %d\n", (int)cs);
        return -203;
    }

    size_t oneA = (size_t)rowsA * colsA; // Tq*Tk
    size_t oneB = (size_t)rowsB * colsB; // Tk*d
    size_t oneC = (size_t)rowsA * colsB; // Tq*d

    printf("Flat2 sizes: oneA=%zu oneB=%zu oneC=%zu totalA=%zu totalB=%zu totalC=%zu\n",
        oneA, oneB, oneC,
        oneA * batchSize, oneB * batchSize, oneC * batchSize);

    if (env->GetArrayLength(jA) != oneA * batchSize ||
        env->GetArrayLength(jB) != oneB * batchSize ||
        env->GetArrayLength(jC) != oneC * batchSize) {
        return -100; // size mismatch
    }

    jfloat* fA = env->GetFloatArrayElements(jA, nullptr);
    jfloat* fB = env->GetFloatArrayElements(jB, nullptr);
    jfloat* fC = env->GetFloatArrayElements(jC, nullptr);

    // Convert to half with per-head offsets
    std::vector<__half> hA(oneA * batchSize);
    std::vector<__half> hB(oneB * batchSize);
    for (int h = 0; h < batchSize; ++h) {
        const jfloat* aBase = fA + h * oneA;
        const jfloat* bBase = fB + h * oneB;
        for (size_t i = 0; i < oneA; ++i) hA[h * oneA + i] = __float2half(aBase[i]);
        for (size_t i = 0; i < oneB; ++i) hB[h * oneB + i] = __float2half(bBase[i]);
    }

    __half* dA; __half* dB; float* dC;
    if(ck(cudaMalloc(&dA, hA.size() * sizeof(__half)), "cudaMalloc dA")) return -200;
    if(ck(cudaMalloc(&dB, hB.size() * sizeof(__half)), "cudaMalloc dB")) return -200;
    if(ck(cudaMalloc(&dC, oneC * batchSize * sizeof(float)), "cudaMalloc dC")) return -200;
    if(ck(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(__half), cudaMemcpyHostToDevice), "cudaMemcpy H2D A")) return -201;
    if(ck(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(__half), cudaMemcpyHostToDevice), "cudaMemcpy H2D B")) return -201;

    float alpha = 1.0f, beta = 0.0f;

    int d = colsB;   // 64
    int Tq = rowsA;  // e.g., 1
    int Tk = colsA;  // e.g., 250

    int m = d, n = Tq, k = Tk;

    long long strideA = (long long)(Tk * d);   // V
    long long strideB = (long long)(Tq * Tk);  // S
    long long strideC = (long long)(Tq * d);   // O

    int lda = d;    // A=V, opA=N → lda ≥ m=d
    int ldb = Tk;   // B=S, opB=T → ldb ≥ k=Tk
    int ldc = d;    // C=O,       → ldc ≥ m=d
    printf("cublasGemmStridedBatchedFlat2 m=%d n=%d k=%d strideA=%ld strideB=%ld strideC=%ld\n", m, n, k, strideA, strideB, strideC);
    cublasStatus_t stat = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,   // <-- changed
        m, n, k,
        &alpha,
        dB, CUDA_R_16F, lda, strideA,  // V
        dA, CUDA_R_16F, ldb, strideB,  // S
        &beta,
        dC, CUDA_R_32F, ldc, strideC,  // O
        batchSize,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasGemmStridedBatchedFlat2 kernel execution error %s\n", cublasGetStatusString(stat));
        return (int)stat;
    }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("Post-GEMM sync error: %s\n", cudaGetErrorString(e));
        return -206;
    }
    // Copy results back into flat C
    for (int h = 0; h < batchSize; ++h) {
        if(ck(cudaMemcpy(fC + h * oneC, dC + h * oneC, oneC * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H C")) return -202;
    }

    env->ReleaseFloatArrayElements(jA, fA, JNI_ABORT);
    env->ReleaseFloatArrayElements(jB, fB, JNI_ABORT);
    env->ReleaseFloatArrayElements(jC, fC, 0);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return (stat == CUBLAS_STATUS_SUCCESS) ? 0 : (int)stat;
}
/*
* Strided Batch Flat 1 processing F16 types after conversion from F32
*/
JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_matrixDotProductF16StridedBatchFlat
(JNIEnv* env, jclass clazz, jlong jHandle,
    jint rowsA, jint colsA, jfloatArray jA,
    jint rowsB, jint colsB, jfloatArray jB,
    jfloatArray jC, jint batchSize) {

    cublasHandle_t handle = reinterpret_cast<cublasHandle_t>(jHandle);
    cublasStatus_t cs = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        printf("SetMathMode tensor failed: %d, falling back to DEFAULT\n", (int)cs);
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
    cs = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        printf("SetPointerMode failed: %d\n", (int)cs);
        return -203;
    }

    size_t oneA = (size_t)rowsA * colsA; // Tq*d
    size_t oneB = (size_t)rowsB * colsB; // Tk*d
    size_t oneC = (size_t)rowsA * rowsB; // Tq*Tk
    if (env->GetArrayLength(jA) != oneA * batchSize ||
        env->GetArrayLength(jB) != oneB * batchSize ||
        env->GetArrayLength(jC) != oneC * batchSize) {
        printf("Flat size mismatch: jA:%d should equal (oneA:%d * batchSize:%d)and jB:%d should equal (oneB:%d * batchSize:%d) and jC:%d should equal (oneC:%d * batchSize:%d) \n",
            env->GetArrayLength(jA), oneA ,batchSize,
            env->GetArrayLength(jB), oneB , batchSize,
            env->GetArrayLength(jC), oneC , batchSize);
        return -100;
    }
    jfloat* fA = env->GetFloatArrayElements(jA, nullptr);
    jfloat* fB = env->GetFloatArrayElements(jB, nullptr);
    jfloat* fC = env->GetFloatArrayElements(jC, nullptr);

    // Convert to half with per-head offsets
    std::vector<__half> hA(oneA * batchSize);
    std::vector<__half> hB(oneB * batchSize);
    for (int h = 0; h < batchSize; ++h) {
        const jfloat* aBase = fA + h * oneA;
        const jfloat* bBase = fB + h * oneB;
        for (size_t i = 0; i < oneA; ++i) hA[h * oneA + i] = __float2half(aBase[i]);
        for (size_t i = 0; i < oneB; ++i) hB[h * oneB + i] = __float2half(bBase[i]);
    }

    __half* dA; __half* dB; float* dC;
    if(ck(cudaMalloc(&dA, hA.size() * sizeof(__half)), "cudaMalloc dA")) return -200;
    if(ck(cudaMalloc(&dB, hB.size() * sizeof(__half)), "cudaMalloc dB")) return -200;
    if(ck(cudaMalloc(&dC, oneC * batchSize * sizeof(float)), "cudaMalloc dC")) return -200;
    if(ck(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(__half), cudaMemcpyHostToDevice), "cudaMemcpy H2DA")) return -201;
    if(ck(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(__half), cudaMemcpyHostToDevice), "cudaMemcpy H2DB")) return -201;

    float alpha = 1.0f, beta = 0.0f;

    int Tq = rowsA; // queries
    int Tk = rowsB; // keys
    int d = colsA; // head size

    int m = Tk, n = Tq, k = d;

    long long strideA = (long long)(Tk * d);  // K
    long long strideB = (long long)(Tq * d);  // Q
    long long strideC = (long long)(Tq * Tk); // S

    int lda = d; // row length of K
    int ldb = d; // row length of Q
    int ldc = m; // Tk
    //printf("cublasGemmStridedBatchedFlat m=%d n=%d k=%d strideA=%ld strideB=%ld strideC=%ld\n", m, n, k, strideA, strideB, strideC);

    cublasStatus_t stat = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        dB, CUDA_R_16F, lda, strideB,
        dA, CUDA_R_16F, ldb, strideA,
        &beta,
        dC, CUDA_R_32F, ldc, strideC,
        batchSize,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasGemmStridedBatchedFlat kernel execution error %s\n", cublasGetStatusString(stat));
        return (int)stat;
    }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("Post-GEMM sync error: %s\n", cudaGetErrorString(e));
        return -206;
    }
    // Copy results back into flat C
    for (int h = 0; h < batchSize; ++h) {
        if(ck(cudaMemcpy(fC + h * oneC, dC + h * oneC, oneC * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2HC")) return -202;
    }

    env->ReleaseFloatArrayElements(jA, fA, JNI_ABORT);
    env->ReleaseFloatArrayElements(jB, fB, JNI_ABORT);
    env->ReleaseFloatArrayElements(jC, fC, 0);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return (stat == CUBLAS_STATUS_SUCCESS) ? 0 : (int)stat;
}
/*
 * Class:     com_neocoretechs_cublas_Gemm
 * Method:    matrixDotProductF32StridedBatch
 * Signature: (JIILjava/util/ArrayList;IILjava/util/ArrayList;Ljava/util/ArrayList;I)I
 */
JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_matrixDotProductF32StridedBatch
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
    const int n3 = columns2 * columns1;
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

    A = (float**)malloc(batchSize * sizeof(float*));
    B = (float**)malloc(batchSize * sizeof(float*));
    C = (float**)malloc(batchSize * sizeof(float*));

    jclass aListClass = env->GetObjectClass(m1_AList);
    jmethodID alGetId = env->GetMethodID(aListClass, "get", "(I)Ljava/lang/Object;");

    for (i = 0; i < batchSize; i++) {
        cudaErr = cudaMalloc((void**)(&h_A[i]), n1 * sizeof(float));
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmStridedBatchedEx device memory allocation error (allocate A) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
        cudaErr = cudaMalloc((void**)(&h_B[i]), n2 * sizeof(float));
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmStridedBatchedEx device memory allocation error (allocate B) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
        cudaErr = cudaMalloc((void**)(&h_C[i]), nc * sizeof(float));
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmStridedBatchedEx device memory allocation error (allocate C) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
    }
    //printf("cublasGemmStridedBatchedEx cudaMalloc1\n");
    // Copy the host array of device pointers to the device
    cudaErr = cudaMalloc((void**)&d_A, batchSize * sizeof(float*));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmStridedBatchedEx device memory allocation error (allocate d_A) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaMalloc((void**)&d_B, batchSize * sizeof(float*));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmStridedBatchedEx device memory allocation error (allocate d_B) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaMalloc((void**)&d_C, batchSize * sizeof(float*));
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmStridedBatchedEx device memory allocation error (allocate d_C) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //printf("cublasGemmStridedBatchedEx cudaMalloc2\n");
    cudaErr = cudaMemcpy(d_A, h_A, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmStridedBatchedEx device memory copy error (copy h_A to d_A) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaMemcpy(d_B, h_B, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmStridedBatchedEx device memory copy error (copy h_B to d_B) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaMemcpy(d_C, h_C, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmStridedBatchedEx device memory copy error (copy h_C to d_C) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //printf("cublasgGemmStridedBatchedEx cudaMemcpy1\n");
    // move JNI ArrayList data to allocated memory
    for (i = 0; i < batchSize; i++) {
        m1s[i] = env->CallObjectMethod(m1_AList, alGetId, i);
        A[i] = env->GetFloatArrayElements((jfloatArray)m1s[i], NULL);
        m2s[i] = env->CallObjectMethod(m2_AList, alGetId, i);
        B[i] = env->GetFloatArrayElements((jfloatArray)m2s[i], NULL);
        mrs[i] = env->CallObjectMethod(mr_AList, alGetId, i);
        C[i] = env->GetFloatArrayElements((jfloatArray)mrs[i], NULL);
        //printf("cublasGemmStridedBatchedEx JNI get %d\n",i);
        status = cublasSetMatrix(rows1, columns1, sizeof(float), A[i], rows1, h_A[i], rows1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasGemmStridedBatchedEx device access error (write A) %s for batch # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        //printf("cublasGemmStridedBatchedEx setMatrix 1 %d\n", i);
        status = cublasSetMatrix(rows2, columns2, sizeof(float), B[i], rows2, h_B[i], rows2);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasGemmStridedBatchedEx device access error (write B) %s for batch # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        //printf("cublasGemmBatchedEx setMatrix 2 %d\n", i);
        status = cublasSetMatrix(rows1, columns2, sizeof(float), C[i], rows1, h_C[i], rows1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasGemmStridedBatchedEx device access error (write C) %s for batch # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        //printf("cublasGemmStridedBatchedEx setMatrix 3 %d\n", i);
    }

    status = cublasGemmStridedBatchedEx((cublasHandle_t)handle, CUBLAS_OP_N, CUBLAS_OP_N, rows1, columns2,
        columns1, &alpha,
        (const void**)d_A, CUDA_R_32F, rows1, n1,
        (const void**)d_B, CUDA_R_32F, rows2, n3, &beta,
        (void**)d_C, CUDA_R_32F, rows1, nc, batchSize, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT/*_TENSOR_OP*/);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!cublasGemmStridedBatchedEx kernel execution error %s\n", cublasGetStatusString(status));
        return JNI_ERR;
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA cublasGemmStridedBatchedEx...%d\n", (stop.tv_nsec - start.tv_nsec));
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("Post-GEMM sync error: %s\n", cudaGetErrorString(e));
        return -206;
    }
   //printf("cublasGetVector d_C...\n");
   // _timespec64_get(&start, TIME_UTC);
    for (i = 0; i < batchSize; i++) {
        //printf("CUDA cublasGemmStridedBatchedEx getVector...%d\n", i);
        status = cublasGetMatrix(rows1, columns2, sizeof(float), h_C[i], rows1, C[i], rows1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasGemmStridedBatched device access error (read C) %s for batch # %d\n", cublasGetStatusString(status), i);
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
            printf("!!!!cublasGemmStridedBatchedEx memory free error (A) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }

        cudaErr = cudaFree(h_B[i]);
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmStridedBatchedEx memory free error (B) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }

        cudaErr = cudaFree(h_C[i]);
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmStridedBatchedEx memory free error (C) %s for batch # %d\n", cudaGetErrorString(cudaErr), i);
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
        printf("!!!!cublasGemmStridedBatchedEx memory free error (d_A) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaFree(d_B);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmStridedBatchedEx memory free error (d_B) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    cudaErr = cudaFree(d_C);
    if (cudaErr != cudaSuccess) {
        printf("!!!!cublasGemmStridedBatchedEx memory free error (d_C) %s\n", cudaGetErrorString(cudaErr));
        return JNI_ERR;
    }
    //printf("free pointers m1s m2s mrs...\n");
    free(m1s);
    free(m2s);
    free(mrs);
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA cublasGemmStridedBatchedEx getVector and FREE ALL...%d\n", (stop.tv_nsec - start.tv_nsec));
    return JNI_OK;
}
/*
 * Class:     com_neocoretechs_cublas_Gemm
 * Method:    matrixDotProductF16Stream
 * Signature: (JIILjava/util/ArrayList;IILjava/util/ArrayList;Ljava/util/ArrayList;I)I
 */
JNIEXPORT jint JNICALL Java_com_neocoretechs_cublas_Gemm_matrixDotProductF16Stream
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
            printf("!!!!cublasGemmExStream device memory allocation error (allocate A) %s for stream # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
        cudaErr = cudaMalloc((void**)(&d_B[i]), n2 * sizeof(d_B[0]));
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmExStream device memory allocation error (allocate B) %s for stream # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
        cudaErr = cudaMalloc((void**)(&d_C[i]), nc * sizeof(d_C[0]));
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmExStream device memory allocation error (allocate C) %s for stream # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }
        status = cublasSetVector(n1, sizeof(float), h_A[i], 1, d_A[i], 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasGemmExStream device access error (write A) %s for stream # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        status = cublasSetVector(n2, sizeof(float), h_B[i], 1, d_B[i], 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasGemmExStream device access error (write B) %s for stream # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        status = cublasSetVector(nc, sizeof(float), h_C[i], 1, d_C[i], 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasGemmExStream device access error (write C) %s for stream # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
    }

    // Create a stream for every GEMM operation
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
            printf("!!!!cublasGemmExStream set stream execution error %s for stream # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
        status = cublasGemmEx((cublasHandle_t)handle, CUBLAS_OP_N, CUBLAS_OP_T, rows1, columns2, columns1, &alpha,
            d_A, CUDA_R_16F, rows1,
            d_B, CUDA_R_16F, rows2, &beta,
            d_C, CUDA_R_32F, rows1, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasGemmExStream kernel execution error %s for stream # %d\n", cublasGetStatusString(status), i);
            return JNI_ERR;
        }
    }
    //_timespec64_get(&stop, TIME_UTC);
    //printf("CUDA cublasGemmExStream...%d\n", (stop.tv_nsec - start.tv_nsec));

    //printf("cublasGetVector d_C...\n");
    //_timespec64_get(&start, TIME_UTC);
    for (i = 0; i < batchSize; i++) {
        status = cublasGetVector(nc, sizeof(float), d_C[i], 1, h_C[i], 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!!cublasGemmExStream device access error (read C)  %s for stream # %d\n", cublasGetStatusString(status), i);
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
            printf("!!!!cublasGemmExStream memory free error (A) %s for stream # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }

        cudaErr = cudaFree(d_B[i]);
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmExStream memory free error (B) %s for stream # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }

        cudaErr = cudaFree(d_C[i]);
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmExStream memory free error (C) %s for stream # %d\n", cudaGetErrorString(cudaErr), i);
            return JNI_ERR;
        }

        cudaErr = cudaStreamDestroy(streams[i]);
        if (cudaErr != cudaSuccess) {
            printf("!!!!cublasGemmExStream memory free error (C) %s for stream # %d\n", cudaGetErrorString(cudaErr), i);
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
    //printf("CUDA cublasGemmExStream getVector and FREE ALL...%d\n", (stop.tv_nsec - start.tv_nsec));
    return JNI_OK;
}
