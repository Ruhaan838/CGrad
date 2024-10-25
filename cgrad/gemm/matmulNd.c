#include <stdio.h>
#include  <stdlib.h>
#include <assert.h>

#include "matmul2d.c"
#include "../storage/Float_tensor.h"
#include "../storage/methods.h"

FloatTensor* matmulNd(FloatTensor* tensor1, FloatTensor* tensor2){
    //check the matmul is possible 
    int max_dim = matmul_broadcast_shape(tensor1->dim, tensor2->dim, tensor1->shape, tensor2->shape, NULL);
    if (max_dim == -1){
        return NULL;
    }
    
    //stroage the new shape to the result shape 
    int *result_shape = (int*)malloc(max_dim * sizeof(int));

    matmul_broadcast_shape(tensor1->dim, tensor2->dim, tensor1->shape, tensor2->shape, result_shape);
    int result_size = 1;
    for (int i = 0; i < max_dim; i++) {
        result_size *= result_shape[i];
    }
    float* result_data = (float*)malloc(result_size * sizeof(float));
    
    int* stride1 = (int*)malloc(max_dim * sizeof(int));
    int* stride2 = (int*)malloc(max_dim * sizeof(int));
    broadcast_stride(tensor1, stride1, max_dim);
    broadcast_stride(tensor2, stride2, max_dim);

    int outer_size = 1;
    for (int i = 0; i < max_dim - 2; i++) {
        outer_size *= result_shape[i];
    }
    //call matmul 2d for all possible batch size
    for (int i = 0; i < outer_size; i++) {
        float* data1 = tensor1->data + i * stride1[0];
        float* data2 = tensor2->data + i * stride2[0];
        matmul2d(data1, data2, result_data + i * result_shape[max_dim - 2] * result_shape[max_dim - 1], 
                 result_shape[max_dim - 2], tensor1->shape[tensor1->dim - 1], result_shape[max_dim - 1]);
    }
    //init new tensor
    FloatTensor* result_tensor = init_tensor(result_data, result_shape, max_dim);
    free(stride1); // remove trash
    free(stride2);
    free(result_data);
    free(result_shape);

    return result_tensor;
}
