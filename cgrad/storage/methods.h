#ifndef METHODS_H
#define METHODS_H

#include "Float_tensor.h"

//helper functions
void cal_stride(FloatTensor *tensor);
void broadcast_stride(FloatTensor* tensor1, int* r_stride1, int max_dim);
int broadcast_shape(FloatTensor* tensor1, FloatTensor* tensor2, int *ans);
void display_tensor(FloatTensor *tensor);
int matmul_broadcast_shape(int dim1, int dim2, int* shape1, int* shape2, int* shape3, int max_dim);
void matmul2d(float* data1, float* data2, float* ans_data, int I_shape, int J_shape, int K_shape);

//initalization
FloatTensor* init_tensor(float *data, int *shape, int dim);

//oprations
FloatTensor* add_tensor(FloatTensor* tensor1, FloatTensor* tensor2);
FloatTensor* mul_ele_tensor(FloatTensor* tensor1, FloatTensor* tensor2);
FloatTensor* pow_tensor(FloatTensor* tenosr1, float num);
FloatTensor* pow_two_tensor(FloatTensor* tensor1, FloatTensor* tenso2);
FloatTensor* matmulNd(FloatTensor* tensor1, FloatTensor* tensor2);

//random number methods
FloatTensor* random_tensor(int *shape, int ndim,int min, int max);
FloatTensor* random_tensor_n(int *shape, int ndim);
#endif