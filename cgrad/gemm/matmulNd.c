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
    broadcast_stride(tensor1->shape, tensor1->stride, stride1, tensor1->dim, max_dim);
    broadcast_stride(tensor2->shape, tensor2->stride, stride1, tensor2->dim, max_dim);

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

FloatTensor* transposeNd(FloatTensor* input_tensor) {
    int dim = input_tensor->dim;
    if (dim < 2) return NULL;

    int* new_shape = (int*)malloc(dim * sizeof(int));
    for (int i = 0; i < dim - 2; i++) {
        new_shape[i] = input_tensor->shape[i];
    }
    new_shape[dim - 2] = input_tensor->shape[dim - 1];
    new_shape[dim - 1] = input_tensor->shape[dim - 2];

    int batch_size = 1;
    for (int i = 0; i < dim - 2; i++) {
        batch_size *= new_shape[i];
    }
    int rows = input_tensor->shape[dim - 2];
    int cols = input_tensor->shape[dim - 1];
    int new_size = batch_size * rows * cols;

    float* transposed_data = (float*)malloc(new_size * sizeof(float));

    for (int i = 0; i < batch_size; i++) {
        float* src_matrix = input_tensor->data + i * rows * cols;
        float* dst_matrix = transposed_data + i * rows * cols;
        transpose2d(src_matrix, dst_matrix, rows, cols);
    }

    FloatTensor* result_tensor = init_tensor(transposed_data, new_shape, dim);

    return result_tensor;
}