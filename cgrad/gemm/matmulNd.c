#include <stdio.h>
#include  <stdlib.h>
#include <assert.h>

#include "matmul2d.c"
#include "../storage/Float_tensor.h"
#include "../storage/methods.h"

void matmulNd(float* data1, int* shape1, int* stride1, int dim1,
              float* data2, int* shape2, int* stride2, int dim2,
              float* result_data, int* result_shape, int* result_size, int* result_dim) {
    // check if matmul is possible
    *result_dim = matmul_broadcast_shape(dim1, dim2, shape1, shape2, NULL);
    if (*result_dim == -1) {
        result_data = NULL;
        return;
    }


    for (int i = 0; i < *result_dim; i++) {
        result_shape[i] = shape1[i];
    }
    matmul_broadcast_shape(dim1, dim2, shape1, shape2, result_shape);
    *result_size = 1;
    for (int i = 0; i < *result_dim; i++) {
        *result_size *= result_shape[i];
    }

    // Broadcast strides for each tensor
    int* result_stride1 = (int*)malloc(*result_dim * sizeof(int));
    int* result_stride2 = (int*)malloc(*result_dim * sizeof(int));
    broadcast_stride(shape1, stride1, result_stride1, dim1, *result_dim);
    broadcast_stride(shape2, stride2, result_stride2, dim2, *result_dim);

    int outer_size = 1;
    for (int i = 0; i < *result_dim - 2; i++) {
        outer_size *= result_shape[i];
    }

    // Perform 2D matmul for each batch
    for (int i = 0; i < outer_size; i++) {
        float* batch_data1 = data1 + i * result_stride1[0];
        float* batch_data2 = data2 + i * result_stride2[0];
        matmul2d(batch_data1, batch_data2, result_data + i * result_shape[*result_dim - 2] * result_shape[*result_dim - 1],
                 result_shape[*result_dim - 2], shape1[dim1 - 1], result_shape[*result_dim - 1]);
    }

    free(result_stride1);
    free(result_stride2);
}

void transposeNd(float* input_data, int* input_shape, int input_dim,
                 float* transposed_data, int* new_shape, int* new_size) {
    if (input_dim < 2) {
        transposed_data = NULL;
        return;
    }

    for (int i = 0; i < input_dim - 2; i++) {
        new_shape[i] = input_shape[i];
    }
    new_shape[input_dim - 2] = input_shape[input_dim - 1];
    new_shape[input_dim - 1] = input_shape[input_dim - 2];

    int batch_size = 1;
    for (int i = 0; i < input_dim - 2; i++) {
        batch_size *= new_shape[i];
    }
    int rows = input_shape[input_dim - 2];
    int cols = input_shape[input_dim - 1];
    *new_size = batch_size * rows * cols;

    for (int i = 0; i < batch_size; i++) {
        float* src_matrix = input_data + i * rows * cols;
        float* dst_matrix = transposed_data + i * rows * cols;
        transpose2d(src_matrix, dst_matrix, rows, cols);
    }
}
