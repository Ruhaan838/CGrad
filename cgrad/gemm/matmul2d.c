#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>

#include "../storage/methods.h"

int matmul_broadcast_shape(int dim1, int dim2, int* shape1, int* shape2, int* shape3) {
    int max_dim = (dim1 > dim2) ? dim1 : dim2;

    if (dim2 == 1) {
        if (shape1[dim1 - 1] != shape2[0]) {
            return -1;
        }
    } else if (shape1[dim1 - 1] != shape2[dim2 - 2]) {
        return -1;
    }

    for (int i = 0; i < max_dim - 2; i++) {
        int new_dim_a = (i >= dim1 - 2) ? 1 : shape1[dim1 - 3 - i];
        int new_dim_b = (i >= dim2 - 2) ? 1 : shape2[dim2 - 3 - i];

        if (new_dim_a != 1 && new_dim_b != 1 && new_dim_a != new_dim_b) {
            return -1;
        }
        if (shape3 != NULL) {
            shape3[max_dim - 3 - i] = (new_dim_a > new_dim_b) ? new_dim_a : new_dim_b;
        }
    }

    if (shape3 != NULL) {
        shape3[max_dim - 2] = shape1[dim1 - 2]; 
        shape3[max_dim - 1] = (dim2 == 1) ? 1 : shape2[dim2 - 1]; 
    }

    return max_dim;
}

void matmul2d(float* data1, float* data2, float* ans_data, int I_shape, int K_shape, int J_shape) {
    int block_size = 256; 

    for (int i = 0; i < I_shape; i++) {
        for (int j = 0; j < J_shape; j++) {
            ans_data[i * J_shape + j] = 0.0;
        }
    }

    for (int ii = 0; ii < I_shape; ii += block_size) {
        for (int jj = 0; jj < J_shape; jj += block_size) {
            for (int kk = 0; kk < K_shape; kk += block_size) {
                
                for (int i = ii; i < ii + block_size && i < I_shape; i++) {
                    for (int k = kk; k < kk + block_size && k < K_shape; k++) {
                        float temp = data1[i * K_shape + k];
                        for (int j = jj; j < jj + block_size && j < J_shape; j++) {
                            ans_data[i * J_shape + j] += temp * data2[k * J_shape + j];
                        }
                    }
                }
            }
        }
    }
}

void transpose2d(float* src_matrix, float* dst_matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst_matrix[j * rows + i] = src_matrix[i * cols + j];
        }
    }
}