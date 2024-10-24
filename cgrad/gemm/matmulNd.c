#include <stdio.h>
#include  <stdlib.h>
#include <assert.h>

#include "matmul2d.c"
#include "../storage/Float_tensor.h"
#include "../storage/methods.h"

FloatTensor* matmulNd(FloatTensor* tensor1, FloatTensor* tensor2){

    int max_dim = tensor1->dim > tensor2->dim ? tensor1->dim : tensor2->dim;
    int *result_shape = (int*)malloc(max_dim * sizeof(int));

    int bool_dim = matmul_broadcast_shape(tensor1->dim, tensor2->dim, tensor1->shape, tensor2->shape, result_shape, max_dim);
    assert((bool_dim == -1) && "Shapes are not compatible for matmul");

    // Initialize result tensor
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

    for (int i = 0; i < outer_size; i++) {
        float* data1 = tensor1->data + i * stride1[0];
        float* data2 = tensor2->data + i * stride2[0];
        matmul2d(data1, data2, result_data + i * result_shape[max_dim - 2] * result_shape[max_dim - 1], 
                 result_shape[max_dim - 2], tensor1->shape[tensor1->dim - 1], result_shape[max_dim - 1]);
    }

    FloatTensor* result_tensor = init_tensor(result_data, result_shape, max_dim);
    free(stride1);
    free(stride2);
    free(result_data);
    free(result_shape);

    return result_tensor;
}
