/*
* Double precision matrix dot product JNI implementation via CUDA CUBLAS v12.1
* author: Jonathan Groff Copyright (C) NeoCoreTechs 2023
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "com_neocoretechs_neurovolve_Matrix.h"

timespec stop, start;

/* Host implementation of a simple version of sgemm */
static void simple_dgemm(int rows1, int cols1, int rows2, int cols2, double alpha, const double *A, const double *B,
                         double beta, double *C) {
  int i;
  int j;
  int k;

  for (i = 0; i < rows1; ++i) {
    for (j = 0; j < cols2; ++j) {
      double prod = 0;
      for (k = 0; k < cols1; ++k) {
        prod += A[k * rows1 + i] * B[j * rows2 + k];
      }
      C[j * rows1 + i] = alpha * prod + beta * C[j * rows1 + i];
    }
  }
}

JNIEXPORT jlong JNICALL Java_com_neocoretechs_neurovolve_Matrix_cublasHandle(JNIEnv* env, jclass clazz) {
    cublasStatus_t status;
    cublasHandle_t handle = NULL;

    /* Initialize CUBLAS */
    printf("CUBLAS creating handle...\n");
    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!! CUBLAS initialization error\n");
        return (jlong)JNI_ERR;
    }
    printf("CUBLAS handle created...\n");
    return (jlong)handle;
}

JNIEXPORT jint JNICALL Java_com_neocoretechs_neurovolve_Matrix_cublasHandleDestroy(JNIEnv* env, jclass clazz, jlong handle) {
  cublasStatus_t status;
  /* Shutdown */
  status = cublasDestroy((cublasHandle_t)handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("!!!! shutdown error (A)\n");
    return JNI_ERR;
  }
  return JNI_OK;
}

/*
 * Class:     com_neocoretechs_neurovolve_MatrixCu
 * Method:    matrixDotProductD
 * Signature: (LII[DII[D[D)I
 */
JNIEXPORT jint JNICALL Java_com_neocoretechs_neurovolve_Matrix_matrixDotProductD(JNIEnv* env, jclass clazz, jlong handle, jint rows1, jint columns1, jdoubleArray m1, jint rows2, jint columns2, jdoubleArray m2, jdoubleArray mr) {
  cublasStatus_t status;

  double *h_A = 0;
  double *h_B = 0;
  double *h_C = 0;
  //double *h_C_ref = 0;
  double *d_A = 0;
  double *d_B = 0;
  double *d_C = 0;

  double alpha = 1.0f;
  double beta = 0.0f;

  const int n2 = rows2 * columns2;
  const int n1 = rows1 * columns1;
  const int nc = rows1 * columns2;
  int i;
  /* for test vs CPU
  float error_norm;
  float ref_norm;
  float diff;
  */
 
  /* Allocate host memory for the matrices */
  /*h_A = (double*)(malloc(n1 * sizeof(h_A[0])));

  if (h_A == 0) {
    fprintf(stderr, "!!!! host memory allocation error (A)\n");
    return NULL;
  }
  */
  /*h_B = (double*)(malloc(n2 * sizeof(h_B[0])));

  if (h_B == 0) {
    fprintf(stderr, "!!!! host memory allocation error (B)\n");
    return NULL;
  }
  */
  //printf("Get Double h_A...\n");
  h_A = env->GetDoubleArrayElements(m1, NULL);
  //printf("Get Double h_B...\n");
  h_B = env->GetDoubleArrayElements(m2, NULL);
  //printf("Get Double h_C...\n");
  h_C = env->GetDoubleArrayElements(mr, NULL);
  /*
  h_C = (double *)(malloc(nc * sizeof(h_C[0])));

  if (h_C == 0) {
    printf("!!!! host memory allocation error (C)\n");
    return JNI_ERR;
  }
  */
  /* Allocate device memory for the matrices */
  
  _timespec64 start;
  _timespec64 stop;
 
  _timespec64_get(&start, TIME_UTC);
  //printf("CUDA malloc d_A...\n");
  if (cudaMalloc((void **)(&d_A), n1 * sizeof(d_A[0])) != cudaSuccess) {
    printf("!!!! device memory allocation error (allocate A)\n");
    return JNI_ERR;
  }
  _timespec64_get(&stop, TIME_UTC);
  printf("CUDA malloc d_A/B...%d\n",(stop.tv_nsec - start.tv_nsec));
  _timespec64_get(&start, TIME_UTC);
  if (cudaMalloc((void **)(&d_B), n2 * sizeof(d_B[0])) != cudaSuccess) {
    printf("!!!! device memory allocation error (allocate B)\n");
    return JNI_ERR;
  }
  _timespec64_get(&stop, TIME_UTC);
  printf("CUDA malloc d_B/C...%d\n", (stop.tv_nsec - start.tv_nsec));
  _timespec64_get(&start, TIME_UTC);
  if (cudaMalloc((void **)(&d_C), nc * sizeof(d_C[0])) != cudaSuccess) {
    printf("!!!! device memory allocation error (allocate C)\n");
    return JNI_ERR;
  }
  _timespec64_get(&stop, TIME_UTC);
  printf("CUDA malloc d_C...%d\n", (stop.tv_nsec - start.tv_nsec));
  /* Initialize the device matrices with the host matrices */
  _timespec64_get(&start, TIME_UTC);
  status = cublasSetVector(n1, sizeof(h_A[0]), h_A, 1, d_A, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("!!!! device access error (write A)\n");
    return JNI_ERR;
  }
  _timespec64_get(&stop, TIME_UTC);
  printf("CUDA setVector d_A/B...%d\n", (stop.tv_nsec - start.tv_nsec));
  _timespec64_get(&start, TIME_UTC);
  status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("!!!! device access error (write B)\n");
    return JNI_ERR;
  }
  _timespec64_get(&stop, TIME_UTC);
  printf("CUDA setVector d_B/C...%d\n", (stop.tv_nsec - start.tv_nsec));
  _timespec64_get(&start, TIME_UTC);
  status = cublasSetVector(nc, sizeof(h_C[0]), h_C, 1, d_C, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("!!!! device access error (write C)\n");
    return JNI_ERR;
  }
  _timespec64_get(&stop, TIME_UTC);
  printf("CUDA setVector C...%d\n", (stop.tv_nsec - start.tv_nsec));
  /* Performs operation using plain C code */
 // simple_sgemm(N, alpha, h_A, h_B, beta, h_C);
 // h_C_ref = h_C;
 // printf("cublasDgemm...\n");
  /* Performs operation using cublas */
  _timespec64_get(&start, TIME_UTC);
  status = cublasDgemm((cublasHandle_t)handle, CUBLAS_OP_N, CUBLAS_OP_N, rows1, columns2, columns1, &alpha, d_A, rows1, d_B, rows2, &beta, d_C, rows1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("!!!! kernel execution error.\n");
    return JNI_ERR;
  }
  _timespec64_get(&stop, TIME_UTC);
  printf("CUDA cublasDgemm...%d\n", (stop.tv_nsec - start.tv_nsec));
  /* Allocate host memory for reading back the result from device memory
  h_C = (double *)(malloc(nc * sizeof(h_C[0])));

  if (h_C == 0) {
    printf("!!!! host memory allocation error (C)\n");
    return JNI_ERR;
  }
  */
  /* Read the result back */
 //printf("cublasGetVector d_C...\n");
  _timespec64_get(&start, TIME_UTC);
  status = cublasGetVector(nc, sizeof(h_C[0]), d_C, 1, h_C, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("!!!! device access error (read C)\n");
    return JNI_ERR;
  }
  _timespec64_get(&stop, TIME_UTC);
  printf("CUDA getVector d_C...%d\n", (stop.tv_nsec - start.tv_nsec));
  /* Check result against reference
  error_norm = 0;
  ref_norm = 0;

  for (i = 0; i < n2; ++i) {
    diff = h_C_ref[i] - h_C[i];
    error_norm += diff * diff;
    ref_norm += h_C_ref[i] * h_C_ref[i];
  }

  error_norm = static_cast<float>(sqrt(static_cast<double>(error_norm)));
  ref_norm = static_cast<float>(sqrt(static_cast<double>(ref_norm)));

  if (fabs(ref_norm) < 1e-7) {
    fprintf(stderr, "!!!! reference norm is 0\n");
    return NULL;
  }
  */
  _timespec64_get(&start, TIME_UTC);
  //printf("set h_C/mr double array region...\n");
  env->SetDoubleArrayRegion(mr, 0, nc, h_C);
  //printf("release h_A/m1 array region...\n");
  env->ReleaseDoubleArrayElements(m1, h_A, JNI_ABORT);
  //printf("release h_B/m2 array region...\n");
  env->ReleaseDoubleArrayElements(m2, h_B, JNI_ABORT);
  //printf("release h_C/mr array region...\n");
  env->ReleaseDoubleArrayElements(mr, h_C, JNI_ABORT);
  /* Memory clean up */
  //free(h_A);
  //free(h_B);
  //free(h_C);
  //free(h_C_ref);

  if (cudaFree(d_A) != cudaSuccess) {
    printf("!!!! memory free error (A)\n");
    return JNI_ERR;
  }

  if (cudaFree(d_B) != cudaSuccess) {
    printf("!!!! memory free error (B)\n");
    return JNI_ERR;
  }

  if (cudaFree(d_C) != cudaSuccess) {
    printf("!!!! memory free error (C)\n");
    return JNI_ERR;
  }
  _timespec64_get(&stop, TIME_UTC);
  printf("CUDA FREE ALL...%d\n", (stop.tv_nsec - start.tv_nsec));
  return JNI_OK;

  /*
  if (error_norm / ref_norm < 1e-6f) {
    printf("cublasDgemm test passed.\n");
    exit(EXIT_SUCCESS);
  } else {
    printf("cublasDgemm test failed.\n");
    exit(EXIT_FAILURE);
  }
  */
}
/*
 * Class:     com_neocoretechs_neurovolve_Matrix
 * Method:    matrixDotProductDBatch
 * Signature: (JIILjava/util/ArrayList;IILjava/util/ArrayList;Ljava/util/ArrayList;I)I
 */
JNIEXPORT jint JNICALL Java_com_neocoretechs_neurovolve_Matrix_matrixDotProductDBatch
(JNIEnv* env, jclass clazz, jlong handle, jint rows1, jint columns1, jobject m1_AList, jint rows2, jint columns2, jobject m2_AList, jobject mr_AList, jint batchSize) {
    cublasStatus_t status;

    double** h_A = 0;
    double** h_B = 0;
    double** h_C = 0;
  
    double** d_A = 0;
    double** d_B = 0;
    double** d_C = 0;

    jobject* m1s;
    jobject* m2s;
    jobject* mrs;

    double alpha = 1.0f;
    double beta = 0.0f;

    const int n2 = rows2 * columns2;
    const int n1 = rows1 * columns1;
    const int nc = rows1 * columns2;
    int i;

    _timespec64 start;
    _timespec64 stop;
    
    _timespec64_get(&start, TIME_UTC);

    h_A = (double**)malloc(batchSize * sizeof(double*));
    h_B = (double**)malloc(batchSize * sizeof(double*));
    h_C = (double**)malloc(batchSize * sizeof(double*));
    d_A = (double**)malloc(batchSize * sizeof(double*));
    d_B = (double**)malloc(batchSize * sizeof(double*));
    d_C = (double**)malloc(batchSize * sizeof(double*));
    
    m1s = (jobject*)malloc(batchSize * sizeof(jobject));
    m2s = (jobject*)malloc(batchSize * sizeof(jobject));
    mrs = (jobject*)malloc(batchSize * sizeof(jobject));

    jclass aListClass = env->GetObjectClass(m1_AList);
    jmethodID alGetId = env->GetMethodID(aListClass, "get", "(I)Ljava/lang/Object;");
    for (i = 0; i < batchSize; i++) {
        m1s[i] = env->CallObjectMethod(m1_AList, alGetId, i);
        h_A[i] = env->GetDoubleArrayElements((jdoubleArray) m1s[i], NULL);
        m2s[i] = env->CallObjectMethod(m2_AList, alGetId, i);
        h_B[i] = env->GetDoubleArrayElements((jdoubleArray) m2s[i], NULL);
        mrs[i] = env->CallObjectMethod(mr_AList, alGetId, i);
        h_C[i] = env->GetDoubleArrayElements((jdoubleArray) mrs[i], NULL);
        if (cudaMalloc((void**)(&d_A[i]), n1 * sizeof(d_A[0])) != cudaSuccess) {
            printf("!!!! device memory allocation error (allocate A)\n");
            return JNI_ERR;
        }
        if (cudaMalloc((void**)(&d_B[i]), n2 * sizeof(d_B[0])) != cudaSuccess) {
            printf("!!!! device memory allocation error (allocate B)\n");
            return JNI_ERR;
        }
        if (cudaMalloc((void**)(&d_C[i]), nc * sizeof(d_C[0])) != cudaSuccess) {
            printf("!!!! device memory allocation error (allocate C)\n");
            return JNI_ERR;
        }
        status = cublasSetVector(n1, sizeof(h_A[0]), h_A[i], 1, d_A[i], 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!! device access error (write A)\n");
            return JNI_ERR;
        }
        status = cublasSetVector(n2, sizeof(h_B[0]), h_B[i], 1, d_B[i], 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!! device access error (write B)\n");
            return JNI_ERR;
        }
        status = cublasSetVector(nc, sizeof(h_C[0]), h_C[i], 1, d_C[i], 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!! device access error (write C)\n");
            return JNI_ERR;
        }
    }

    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA cublasDgemmBatched total setup time...%d\n", (stop.tv_nsec - start.tv_nsec));
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
   // printf("cublasDgemm...\n");
    /* Performs operation using cublas */
    _timespec64_get(&start, TIME_UTC);
    status = cublasDgemmBatched((cublasHandle_t)handle, CUBLAS_OP_N, CUBLAS_OP_N, rows1, columns2, columns1, &alpha, d_A, rows1, d_B, rows2, &beta, d_C, rows1, batchSize);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!! kernel execution error.\n");
        return JNI_ERR;
    }
    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA cublasDgemmBatched...%d\n", (stop.tv_nsec - start.tv_nsec));
  
   //printf("cublasGetVector d_C...\n");
    _timespec64_get(&start, TIME_UTC);
    for (i = 0; i < batchSize; i++) {
        status = cublasGetVector(nc, sizeof(h_C[0]), d_C[i], 1, h_C[i], 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!! device access error (read C)\n");
            return JNI_ERR;
        }
       // _timespec64_get(&stop, TIME_UTC);
       // printf("CUDA getVector d_C...%d\n", (stop.tv_nsec - start.tv_nsec));
        //_timespec64_get(&start, TIME_UTC);
        //printf("set h_C/mr double array region...\n");
        env->SetDoubleArrayRegion((jdoubleArray)mrs[i], 0, nc, h_C[i]);
        //printf("release h_A/m1 array region...\n");
        env->ReleaseDoubleArrayElements((jdoubleArray)m1s[i], h_A[i], JNI_ABORT);
        //printf("release h_B/m2 array region...\n");
        env->ReleaseDoubleArrayElements((jdoubleArray)m2s[i], h_B[i], JNI_ABORT);
        //printf("release h_C/mr array region...\n");
        env->ReleaseDoubleArrayElements((jdoubleArray)mrs[i], h_C[i], JNI_ABORT);

        if (cudaFree(d_A[i]) != cudaSuccess) {
            printf("!!!! memory free error (A)\n");
            return JNI_ERR;
        }

        if (cudaFree(d_B[i]) != cudaSuccess) {
            printf("!!!! memory free error (B)\n");
            return JNI_ERR;
        }

        if (cudaFree(d_C[i]) != cudaSuccess) {
            printf("!!!! memory free error (C)\n");
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
    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA cublasDgemmBatched getVector and FREE ALL...%d\n", (stop.tv_nsec - start.tv_nsec));
    return JNI_OK;
}

/*
 * Class:     com_neocoretechs_neurovolve_Matrix
 * Method:    matrixDotProductDStream
 * Signature: (JIILjava/util/ArrayList;IILjava/util/ArrayList;Ljava/util/ArrayList;I)I
 */
JNIEXPORT jint JNICALL Java_com_neocoretechs_neurovolve_Matrix_matrixDotProductDStream
(JNIEnv* env, jclass clazz, jlong handle, jint rows1, jint columns1, jobject m1_AList, jint rows2, jint columns2, jobject m2_AList, jobject mr_AList, jint batchSize) {
    cublasStatus_t status;

    double** h_A = 0;
    double** h_B = 0;
    double** h_C = 0;

    double** d_A = 0;
    double** d_B = 0;
    double** d_C = 0;

    jobject* m1s;
    jobject* m2s;
    jobject* mrs;

    double alpha = 1.0f;
    double beta = 0.0f;

    const int n2 = rows2 * columns2;
    const int n1 = rows1 * columns1;
    const int nc = rows1 * columns2;
    int i;

    _timespec64 start;
    _timespec64 stop;

    _timespec64_get(&start, TIME_UTC);

    h_A = (double**)malloc(batchSize * sizeof(double*));
    h_B = (double**)malloc(batchSize * sizeof(double*));
    h_C = (double**)malloc(batchSize * sizeof(double*));
    d_A = (double**)malloc(batchSize * sizeof(double*));
    d_B = (double**)malloc(batchSize * sizeof(double*));
    d_C = (double**)malloc(batchSize * sizeof(double*));

    m1s = (jobject*)malloc(batchSize * sizeof(jobject));
    m2s = (jobject*)malloc(batchSize * sizeof(jobject));
    mrs = (jobject*)malloc(batchSize * sizeof(jobject));

    jclass aListClass = env->GetObjectClass(m1_AList);
    jmethodID alGetId = env->GetMethodID(aListClass, "get", "(I)Ljava/lang/Object;");
    for (i = 0; i < batchSize; i++) {
        m1s[i] = env->CallObjectMethod(m1_AList, alGetId, i);
        h_A[i] = env->GetDoubleArrayElements((jdoubleArray)m1s[i], NULL);
        m2s[i] = env->CallObjectMethod(m2_AList, alGetId, i);
        h_B[i] = env->GetDoubleArrayElements((jdoubleArray)m2s[i], NULL);
        mrs[i] = env->CallObjectMethod(mr_AList, alGetId, i);
        h_C[i] = env->GetDoubleArrayElements((jdoubleArray)mrs[i], NULL);
        if (cudaMalloc((void**)(&d_A[i]), n1 * sizeof(d_A[0])) != cudaSuccess) {
            printf("!!!! device memory allocation error (allocate A)\n");
            return JNI_ERR;
        }
        if (cudaMalloc((void**)(&d_B[i]), n2 * sizeof(d_B[0])) != cudaSuccess) {
            printf("!!!! device memory allocation error (allocate B)\n");
            return JNI_ERR;
        }
        if (cudaMalloc((void**)(&d_C[i]), nc * sizeof(d_C[0])) != cudaSuccess) {
            printf("!!!! device memory allocation error (allocate C)\n");
            return JNI_ERR;
        }
        status = cublasSetVector(n1, sizeof(h_A[0]), h_A[i], 1, d_A[i], 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!! device access error (write A)\n");
            return JNI_ERR;
        }
        status = cublasSetVector(n2, sizeof(h_B[0]), h_B[i], 1, d_B[i], 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!! device access error (write B)\n");
            return JNI_ERR;
        }
        status = cublasSetVector(nc, sizeof(h_C[0]), h_C[i], 1, d_C[i], 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!! device access error (write C)\n");
            return JNI_ERR;
        }
    }

    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA cublasDgemmBatched total setup time...%d\n", (stop.tv_nsec - start.tv_nsec));
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
    for (i = 0; i < batchSize; i++)
        cudaStreamCreate(&streams[i]);

    // printf("cublasDgemm...\n");
     /* Performs operation using cublas */
    _timespec64_get(&start, TIME_UTC);
    // Launch each DGEMM operation in own CUDA stream
    for (i = 0; i < batchSize; i++) {
        // Set CUDA stream
        cublasSetStream((cublasHandle_t)handle, streams[i]);
        status = cublasDgemm((cublasHandle_t)handle, CUBLAS_OP_N, CUBLAS_OP_N, rows1, columns2, columns1, &alpha, d_A[i], rows1, d_B[i], rows2, &beta, d_C[i], rows1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!! kernel execution error.\n");
            return JNI_ERR;
        }
    }
    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA cublasDgemm Streams...%d\n", (stop.tv_nsec - start.tv_nsec));

    //printf("cublasGetVector d_C...\n");
    _timespec64_get(&start, TIME_UTC);
    for (i = 0; i < batchSize; i++) {
        status = cublasGetVector(nc, sizeof(h_C[0]), d_C[i], 1, h_C[i], 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("!!!! device access error (read C)\n");
            return JNI_ERR;
        }
        // _timespec64_get(&stop, TIME_UTC);
        // printf("CUDA getVector d_C...%d\n", (stop.tv_nsec - start.tv_nsec));
         //_timespec64_get(&start, TIME_UTC);
         //printf("set h_C/mr double array region...\n");
        env->SetDoubleArrayRegion((jdoubleArray)mrs[i], 0, nc, h_C[i]);
        //printf("release h_A/m1 array region...\n");
        env->ReleaseDoubleArrayElements((jdoubleArray)m1s[i], h_A[i], JNI_ABORT);
        //printf("release h_B/m2 array region...\n");
        env->ReleaseDoubleArrayElements((jdoubleArray)m2s[i], h_B[i], JNI_ABORT);
        //printf("release h_C/mr array region...\n");
        env->ReleaseDoubleArrayElements((jdoubleArray)mrs[i], h_C[i], JNI_ABORT);

        if (cudaFree(d_A[i]) != cudaSuccess) {
            printf("!!!! memory free error (A)\n");
            return JNI_ERR;
        }

        if (cudaFree(d_B[i]) != cudaSuccess) {
            printf("!!!! memory free error (B)\n");
            return JNI_ERR;
        }

        if (cudaFree(d_C[i]) != cudaSuccess) {
            printf("!!!! memory free error (C)\n");
            return JNI_ERR;
        }
        cudaStreamDestroy(streams[i]);
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
    _timespec64_get(&stop, TIME_UTC);
    printf("CUDA cublasDgemmBatched getVector and FREE ALL...%d\n", (stop.tv_nsec - start.tv_nsec));
    return JNI_OK;
}
