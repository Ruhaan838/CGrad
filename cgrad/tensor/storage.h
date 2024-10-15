#ifndef STORAGE_H
#define STORAGE_H

typedef struct{
    float *data;
    int *shape;
    int *stride;
    int dim;
    int size;
} CTensor;

void cal_stride(CTensor *tensor);
CTensor* init_tensor(float *data, int *shape, int dim);
void display_tensor(CTensor *tensor);
CTensor* add_tensor(CTensor* tensor1, CTensor* tensor2);
CTensor* mul_ele_tensor(CTensor* tensor1, CTensor* tensor2);
CTensor* pow_tensor(CTensor* tenosr1, float num);
CTensor* pow_two_tensor(CTensor* tensor1, CTensor* tenso2);
#endif