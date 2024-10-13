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
CTensor* add_tensor(float *data1, float *data2, int *shape, int ndim, int size);
#endif