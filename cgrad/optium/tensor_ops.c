#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "../storage/Float_tensor.h"
#include "../tensor/tensor.c"
#include "philox_random.c" // random number generator

void random_tensor(int *shape, int ndim, int min, int max, int seed, float* data) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }

    for (int i = 0; i < size; i++) {
        data[i] = philox4x32_float(seed, i, min, max);
    }
}

void random_tensor_n(int *shape, int ndim, int seed, float* data) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }

    for (int i = 0; i < size; i++) {
        data[i] = philox4x32_float_n(seed, i);
    }
}

void zeros_tensor(int* shape, int ndim, float* data) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }

    for (int i = 0; i < size; i++) {
        data[i] = 0.0;
    }
}

void ones_tensor(int *shape, int ndim, float* data) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }

    for (int i = 0; i < size; i++) {
        data[i] = 1.0;
    }
}
