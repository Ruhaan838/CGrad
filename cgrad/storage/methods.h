#ifndef METHODS_H
#define METHODS_H

#include "Float_tensor.h"

void cal_stride(FloatTensor *tensor);
void broadcast_stride(FloatTensor* tensor1, int* r_stride1, int max_dim);
int broadcast_shape(FloatTensor* tensor1, FloatTensor* tensor2, int *ans);
FloatTensor* init_tensor(float *data, int *shape, int dim);
void display_tensor(FloatTensor *tensor);
FloatTensor* add_tensor(FloatTensor* tensor1, FloatTensor* tensor2);
FloatTensor* mul_ele_tensor(FloatTensor* tensor1, FloatTensor* tensor2);
FloatTensor* pow_tensor(FloatTensor* tenosr1, float num);
FloatTensor* pow_two_tensor(FloatTensor* tensor1, FloatTensor* tenso2);

#endif