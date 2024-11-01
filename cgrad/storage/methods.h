#ifndef METHODS_H
#define METHODS_H

#include "Float_tensor.h"

//helper functions
void cal_stride(int* shape, int* stride, int dim);
void broadcast_stride(int* shape, int* stride, int* r_stride1, int dim, int max_dim);
int broadcast_shape(int* shape1, int dim1, int* shape2, int dim2, int *ans);
void display_tensor(FloatTensor *tensor);
int matmul_broadcast_shape(int dim1, int dim2, int* shape1, int* shape2, int* shape3);
void matmul2d(float* data1, float* data2, float* ans_data, int I_shape, int J_shape, int K_shape);
void transpose2d(float* src_matrix, float* dst_matrix, int rows, int cols);

//initalization
FloatTensor* init_tensor(float *data, int *shape, int dim);

//oprations
void add_tensor(float *data1, float *data2, float *r_data, 
                int *shape1, int* shape2, int* r_shape, 
                int* stride1, int* stride2, 
                int dim1, int dim2, int r_dim, int max_dim);

void mul_ele_tensor(float *data1, float *data2, float *r_data, 
                    int *shape1, int *shape2, int *r_shape, 
                    int *stride1, int *stride2, 
                    int dim1, int dim2, int r_dim, int max_dim);

void pow_tensor(float *data, float *r_data, int size, float num);

void pow_two_tensor(float *data1, float *data2, float *r_data, 
                    int *shape1, int *shape2, int *r_shape, 
                    int *stride1, int *stride2, 
                    int dim1, int dim2, int r_dim, int max_dim);

void sum_tensor(float* data1, int* shape1, int dim1, 
                float* r_data, int* r_shape, int axis, int keepdims);

// matrix
void matmulNd(float* data1, int* shape1, int* stride1, int dim1,
              float* data2, int* shape2, int* stride2, int dim2,
              float* result_data, int* result_shape, int* result_size, int* result_dim);
              
void transposeNd(float* input_data, int* input_shape, int input_dim,
                 float* transposed_data, int* new_shape, int* new_size);

//random number methods
void random_tensor(int *shape, int ndim, int min, int max, int seed, float* data);
void random_tensor_n(int *shape, int ndim, int seed, float* data);

//ones and zeros
void zeros_tensor(int* shape, int ndim, float* data);
void ones_tensor(int *shape, int ndim, float* data);
#endif