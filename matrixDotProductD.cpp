/*
* Double precision matrix dot product JNI implementation via CUDA CUBLAS v12.1
* author: Jonathan Groff Copyright (C) NeoCoreTechs 2023
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "com_neocoretechs_neurovolve_MatrixCu.h"

static cublasHandle_t handle = NULL;
static double alpha = 1.0f;
static double beta = 0.0f;

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

/*
 * Class:     com_neocoretechs_neurovolve_MatrixCu
 * Method:    matrixDotProductD
 * Signature: (II[D[D)[D
 */
JNIEXPORT jint JNICALL Java_com_neocoretechs_neurovolve_MatrixCu_matrixDotProductD(JNIEnv* env, jclass clazz, jint rows1, jint columns1, jdoubleArray m1, jint rows2, jint columns2, jdoubleArray m2, jdoubleArray mr) {
  cublasStatus_t status;
  double *h_A = 0;
  double *h_B = 0;
  double *h_C = 0;
  //double *h_C_ref = 0;
  double *d_A = 0;
  double *d_B = 0;
  double *d_C = 0;

  const int n2 = rows2 * columns2;
  const int n1 = rows1 * columns1;
  const int nc = rows1 * columns2;
  int i;
  /* for test vs CPU
  float error_norm;
  float ref_norm;
  float diff;
  */
  
  /* Initialize CUBLAS */
  if (handle == NULL) {
      status = cublasCreate(&handle);

      if (status != CUBLAS_STATUS_SUCCESS) {
          fprintf(stderr, "!!!! CUBLAS initialization error\n");
          return JNI_ERR;
      }
  }
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
  h_A = env->GetDoubleArrayElements(m1, NULL);
  h_B = env->GetDoubleArrayElements(m2, NULL);
  h_C = env->GetDoubleArrayElements(mr, NULL);
  /*
  h_C = (double *)(malloc(nc * sizeof(h_C[0])));

  if (h_C == 0) {
    fprintf(stderr, "!!!! host memory allocation error (C)\n");
    return JNI_ERR;
  }
  */
  /* Allocate device memory for the matrices */
  if (cudaMalloc((void **)(&d_A), n1 * sizeof(d_A[0])) != cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
    return JNI_ERR;
  }

  if (cudaMalloc((void **)(&d_B), n2 * sizeof(d_B[0])) != cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
    return JNI_ERR;
  }

  if (cudaMalloc((void **)(&d_C), nc * sizeof(d_C[0])) != cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
    return JNI_ERR;
  }

  /* Initialize the device matrices with the host matrices */
  status = cublasSetVector(n1, sizeof(h_A[0]), h_A, 1, d_A, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write A)\n");
    return JNI_ERR;
  }

  status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write B)\n");
    return JNI_ERR;
  }

  status = cublasSetVector(nc, sizeof(h_C[0]), h_C, 1, d_C, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write C)\n");
    return JNI_ERR;
  }

  /* Performs operation using plain C code */
 // simple_sgemm(N, alpha, h_A, h_B, beta, h_C);
 // h_C_ref = h_C;

  /* Performs operation using cublas */
  status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows1, columns2, columns1, &alpha, d_A, rows1, d_B, rows2, &beta, d_C, rows1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! kernel execution error.\n");
    return JNI_ERR;
  }

  /* Allocate host memory for reading back the result from device memory */
  h_C = (double *)(malloc(nc * sizeof(h_C[0])));

  if (h_C == 0) {
    fprintf(stderr, "!!!! host memory allocation error (C)\n");
    return JNI_ERR;
  }

  /* Read the result back */
  status = cublasGetVector(nc, sizeof(h_C[0]), d_C, 1, h_C, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (read C)\n");
    return JNI_ERR;
  }

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
  env->SetDoubleArrayRegion(mr, 0, nc, h_C);

  env->ReleaseDoubleArrayElements(m1, h_A, JNI_ABORT);
  env->ReleaseDoubleArrayElements(m2, h_B, JNI_ABORT);
  env->ReleaseDoubleArrayElements(mr, h_C, JNI_ABORT);
  /* Memory clean up */
  free(h_A);
  free(h_B);
  free(h_C);
  //free(h_C_ref);

  if (cudaFree(d_A) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (A)\n");
    return JNI_ERR;
  }

  if (cudaFree(d_B) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (B)\n");
    return JNI_ERR;
  }

  if (cudaFree(d_C) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (C)\n");
    return JNI_ERR;
  }

  /* Shutdown 
  status = cublasDestroy(handle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! shutdown error (A)\n");
    return NULL;
  }
  */
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
