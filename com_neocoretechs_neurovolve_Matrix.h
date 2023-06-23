/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_neocoretechs_neurovolve_Matrix */

#ifndef _Included_com_neocoretechs_neurovolve_Matrix
#define _Included_com_neocoretechs_neurovolve_Matrix
#ifdef __cplusplus
extern "C" {
#endif
#undef com_neocoretechs_neurovolve_Matrix_serialVersionUID
#define com_neocoretechs_neurovolve_Matrix_serialVersionUID -5112564161732998513i64
/*
 * Class:     com_neocoretechs_neurovolve_Matrix
 * Method:    matrixDotProductD
 * Signature: (JII[DII[D[D)I
 */
JNIEXPORT jint JNICALL Java_com_neocoretechs_neurovolve_Matrix_matrixDotProductD
  (JNIEnv *, jclass, jlong, jint, jint, jdoubleArray, jint, jint, jdoubleArray, jdoubleArray);

/*
 * Class:     com_neocoretechs_neurovolve_Matrix
 * Method:    matrixDotProductDBatch
 * Signature: (JIILjava/util/ArrayList;IILjava/util/ArrayList;Ljava/util/ArrayList;I)I
 */
JNIEXPORT jint JNICALL Java_com_neocoretechs_neurovolve_Matrix_matrixDotProductDBatch
  (JNIEnv *, jclass, jlong, jint, jint, jobject, jint, jint, jobject, jobject, jint);

/*
 * Class:     com_neocoretechs_neurovolve_Matrix
 * Method:    cublasHandle
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_neocoretechs_neurovolve_Matrix_cublasHandle
  (JNIEnv *, jclass);

/*
 * Class:     com_neocoretechs_neurovolve_Matrix
 * Method:    cublasHandleDestroy
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_neocoretechs_neurovolve_Matrix_cublasHandleDestroy
  (JNIEnv *, jclass, jlong);

#ifdef __cplusplus
}
#endif
#endif