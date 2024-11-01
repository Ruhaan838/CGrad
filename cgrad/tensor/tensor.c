#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../storage/Float_tensor.h"
#include "../storage/methods.h"

#define len(arr) (sizeof(arr) / sizeof(arr[0]))

void cal_stride(int* shape, int* stride, int dim){
    stride[dim - 1] = 1; // the last dim is always 1
    for (int i = dim - 2; i >= 0; i--){
        stride[i] = stride[i + 1] * shape[i + 1];
    }
}
//caculate the stride for broadcast
void broadcast_stride(int* shape, int* stride, int* r_stride1, int dim, int max_dim){
    for (int i = 0; i<max_dim; i++){
        //basicly we are access the last dim of both tensor shape & stride and check if it's 1 or not and update the result stride so that help to get new stride.
        int dim_a = (i >= dim) ? 1: shape[dim - 1 - i];
        // now we are change the result stride if the dim is 1 so we make the stride to 0 and if it's anything else we make it 1.
        r_stride1[max_dim - 1 - i] = (dim_a == 1) ? 0 : stride[dim - 1 - i];
    }
}

int broadcast_shape(int *shape1, int dim1, int *shape2, int dim2, int *ans) {
    int max_dim = (dim1 > dim2) ? dim1 : dim2;
    
    if (ans != NULL) {
        for (int i = 0; i < max_dim; i++) {
            int dim_a = (i >= dim1) ? 1 : shape1[dim1 - 1 - i];
            int dim_b = (i >= dim2) ? 1 : shape2[dim2 - 1 - i];
            
            if (dim_a != 1 && dim_b != 1 && dim_a != dim_b) {
                return -1;  
            }

            ans[max_dim - 1 - i] = (dim_a > dim_b) ? dim_a : dim_b;
        }
    }
    return max_dim;
}


// tensor initialization
FloatTensor* init_tensor(float *data, int *shape, int ndim){
    FloatTensor* newtensor = (FloatTensor*)malloc(sizeof(FloatTensor)); // allocate memory for tensor struct
    if (newtensor == NULL){
        free(newtensor);
        perror("Failed to allocate memory for FloatTensor");
        return NULL;
    }

    newtensor->shape = (int*)malloc(ndim * sizeof(int));
    newtensor->stride = (int*)malloc(ndim * sizeof(int));
    newtensor->dim = ndim;

    // Define the size of the tensor first
    newtensor->size = 1;
    for (int i = 0; i < ndim; i++){
        newtensor->shape[i] = shape[i];
        newtensor->size *= shape[i]; // calculating the total size of the tensor
    }

    // Now allocate memory for tensor data based on the calculated size
    newtensor->data = (float*)malloc(newtensor->size * sizeof(float)); 

    if (newtensor->shape == NULL || newtensor->data == NULL){
        perror("Failed to allocate memory for shape or data");
        free(newtensor);
        return NULL;
    }

    cal_stride(newtensor->shape, newtensor->stride, newtensor->dim);

    // Insert the data
    for (int i = 0; i < newtensor->size; i++){
        newtensor->data[i] = data[i];
    }

    return newtensor;
}

void add_tensor(float *data1, float *data2, float *r_data, 
                int *shape1, int *shape2, int *r_shape, 
                int *stride1, int *stride2, 
                int dim1, int dim2, int r_dim, int max_dim) {

    int *result_stride1 = (int *)malloc(max_dim * sizeof(int));
    int *result_stride2 = (int *)malloc(max_dim * sizeof(int));
    broadcast_stride(shape1, stride1, result_stride1, dim1, max_dim);
    broadcast_stride(shape2, stride2, result_stride2, dim2, max_dim);

    int total_elements = 1;
    for (int i = 0; i < max_dim; i++) {
        total_elements *= r_shape[i];
    }

    int *r_stride = (int *)malloc(max_dim * sizeof(int));
    cal_stride(r_shape, r_stride, max_dim);

    for (int idx = 0; idx < total_elements; idx++) {
        int offset1 = 0, offset2 = 0;
        int n_idx = idx;

        for (int i = 0; i < max_dim; i++) {
            int stride_idx = n_idx / r_stride[i];
            n_idx %= r_stride[i];

            offset1 += stride_idx * result_stride1[i];
            offset2 += stride_idx * result_stride2[i];
        }
        r_data[idx] = data1[offset1] + data2[offset2];
    }

    free(result_stride1);
    free(result_stride2);
    free(r_stride);
}

void mul_ele_tensor(float *data1, float *data2, float *r_data, 
                    int *shape1, int *shape2, int *r_shape, 
                    int *stride1, int *stride2, 
                    int dim1, int dim2, int r_dim, int max_dim) {

    int *result_stride1 = (int *)malloc(max_dim * sizeof(int));
    int *result_stride2 = (int *)malloc(max_dim * sizeof(int));
    broadcast_stride(shape1, stride1, result_stride1, dim1, max_dim);
    broadcast_stride(shape2, stride2, result_stride2, dim2, max_dim);

    int total_elements = 1;
    for (int i = 0; i < max_dim; i++) {
        total_elements *= r_shape[i];
    }

    int *r_stride = (int *)malloc(max_dim * sizeof(int));
    cal_stride(r_shape, r_stride, max_dim);

    for (int idx = 0; idx < total_elements; idx++) {
        int offset1 = 0, offset2 = 0;
        int n_idx = idx;

        for (int i = 0; i < max_dim; i++) {
            int stride_idx = n_idx / r_stride[i];
            n_idx %= r_stride[i];

            offset1 += stride_idx * result_stride1[i];
            offset2 += stride_idx * result_stride2[i];
        }
        r_data[idx] = data1[offset1] * data2[offset2];
    }

    free(result_stride1);
    free(result_stride2);
    free(r_stride);
}

void pow_tensor(float *data, float *r_data, int size, float num){

    for (int k = 0; k < size; k++){
        r_data[k] = pow(data[k], num);
    }
}

void pow_two_tensor(float *data1, float *data2, float *r_data, 
                    int *shape1, int *shape2, int *r_shape, 
                    int *stride1, int *stride2, 
                    int dim1, int dim2, int r_dim, int max_dim) {

    int *result_stride1 = (int *)malloc(max_dim * sizeof(int));
    int *result_stride2 = (int *)malloc(max_dim * sizeof(int));
    broadcast_stride(shape1, stride1, result_stride1, dim1, max_dim);
    broadcast_stride(shape2, stride2, result_stride2, dim2, max_dim);

    int total_elements = 1;
    for (int i = 0; i < max_dim; i++) {
        total_elements *= r_shape[i];
    }

    int *r_stride = (int *)malloc(max_dim * sizeof(int));
    cal_stride(r_shape, r_stride, max_dim);

    for (int idx = 0; idx < total_elements; idx++) {
        int offset1 = 0, offset2 = 0;
        int n_idx = idx;

        for (int i = 0; i < max_dim; i++) {
            int stride_idx = n_idx / r_stride[i];
            n_idx %= r_stride[i];

            offset1 += stride_idx * result_stride1[i];
            offset2 += stride_idx * result_stride2[i];
        }
        r_data[idx] = pow(data1[offset1], data2[offset2]);
    }

    free(result_stride1);
    free(result_stride2);
    free(r_stride);
}

void sum_tensor(float* data1, int* shape1, int dim1, 
                float* r_data, int* r_shape, int axis, int keepdims) {
    int i, j, k;
    int result_size = 1;

    // Calculate the new shape
    for (i = 0, k = 0; i < dim1; i++) {
        if (i == axis) {
            if (keepdims) {
                r_shape[k++] = 1;
            }
        } else {
            r_shape[k++] = shape1[i];
            result_size *= shape1[i];
        }
    }

    // Initialize the result data array to zero
    for (i = 0; i < result_size; i++) {
        r_data[i] = 0.0;
    }

    // Calculate outer and inner sizes for summing along the specified axis
    int outer_size = 1, inner_size = 1;
    for (i = 0; i < axis; i++) outer_size *= shape1[i];
    for (i = axis + 1; i < dim1; i++) inner_size *= shape1[i];

    // Sum along the specified axis
    for (i = 0; i < outer_size; i++) {
        for (j = 0; j < shape1[axis]; j++) {
            for (k = 0; k < inner_size; k++) {
                int input_index = i * shape1[axis] * inner_size + j * inner_size + k;
                int result_index = i * inner_size + k;
                r_data[result_index] += data1[input_index];
            }
        }
    }
}


void display_tensor(FloatTensor *tensor){
    printf("Tensor [data = (");
    for (int i = 0; i < tensor->size; i++){
        printf("%f, ", tensor->data[i]);
    }
    printf("), Shape = (");
    for (int i = 0; i < tensor->dim; i++){
        printf("%d, ", tensor->shape[i]);
    }
    printf("), Dim = %d ]\n", tensor->dim);
}
