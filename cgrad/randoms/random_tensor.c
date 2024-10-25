#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "../storage/Float_tensor.h"
#include "../storage/methods.h"
#include "../tensor/tensor.c"

FloatTensor* random_tensor(int *shape, int ndim, int min, int max) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }

    float* data = (float*)malloc(size * sizeof(float));

    srand(time(NULL));

    for (int i = 0; i < size; i++) {
        float num = ((float)rand() / (float)RAND_MAX) * (max - min) + min; 
        data[i] = num;
    }

    FloatTensor* new_tensor = init_tensor(data, shape, ndim);
    return new_tensor;
}

FloatTensor* random_tensor_n(int *shape, int ndim) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }

    float* data = (float*)malloc(size * sizeof(float));

    srand(time(NULL));

    for (int i = 0; i < size; i++) {
        float num = (float)rand() / (float)RAND_MAX; 
        data[i] = num;
    }

    FloatTensor* new_tensor = init_tensor(data, shape, ndim);
    return new_tensor;
}

// int main(){
//     int shape[] = {2,3};
//     int dim = 2;
    
//     FloatTensor* random_ten = random_tensor_n(shape, dim);

//     display_tensor(random_ten);
// }