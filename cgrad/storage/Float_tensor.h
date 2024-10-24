#ifndef FLOAT_TENSOR_H
#define FLOAT_TENSOR_H

typedef struct{
    float *data;
    int *shape;
    int *stride;
    int dim;
    int size;
} FloatTensor;

#endif