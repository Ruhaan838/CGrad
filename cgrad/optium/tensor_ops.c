#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "../storage/Float_tensor.h"
#include "../tensor/tensor.c"
#include "philox_random.c" // random number generator

FloatTensor* random_tensor(int *shape, int ndim, int min, int max, int seed) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }

    float* data = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) {
        float num =  philox4x32_float(seed, i, min, max);
        data[i] = num;
    }

    FloatTensor* new_tensor = init_tensor(data, shape, ndim);
    return new_tensor;
}

FloatTensor* random_tensor_n(int *shape, int ndim, int seed) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }

    float* data = (float*)malloc(size * sizeof(float));
    
    for (int i = 0; i < size; i++) {
        float num = philox4x32_float_n(seed, i);
        data[i] = num;
    }

    FloatTensor* new_tensor = init_tensor(data, shape, ndim);
    return new_tensor;
}

FloatTensor* zeros_tensor(int* shape, int ndim){
    int size = 1;
    for (int i = 0; i < ndim; i++){
        size *= shape[i];
    }

    float* data = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++)
        data[i] = 0.0;
    
    FloatTensor* new_tensor = init_tensor(data, shape, ndim);

    return new_tensor;
}

FloatTensor* ones_tensor(int *shape, int ndim){
    int size = 1;
    for (int i = 0; i < ndim; i++){
        size *= shape[i];
    }

    float* data = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++)
        data[i] = 1.0;
    
    FloatTensor* new_tensor = init_tensor(data, shape, ndim);

    return new_tensor;
}