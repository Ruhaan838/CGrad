#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../storage/Float_tensor.h"
#include "../storage/methods.h"

#define len(arr) (sizeof(arr) / sizeof(arr[0]))

void cal_stride(FloatTensor* tensor){
    tensor->stride = calloc(tensor->dim, sizeof(int)); //allocate the size for stride
    tensor->stride[tensor->dim - 1] = 1; // the last dim is always 1
    for (int i = tensor->dim - 2; i >= 0; i--){
        tensor->stride[i] = tensor->stride[i + 1] * tensor->shape[i + 1];
    }
}
//caculate the stride for broadcast
void broadcast_stride(FloatTensor* tensor1, int* r_stride1, int max_dim){
    for (int i = 0; i<max_dim; i++){
        //basicly we are access the last dim of both tensor shape & stride and check if it's 1 or not and update the result stride so that help to get new stride.
        int dim_a = (i >= tensor1->dim) ? 1: tensor1->shape[tensor1->dim - 1 - i];
        // now we are change the result stride if the dim is 1 so we make the stride to 0 and if it's anything else we make it 1.
        r_stride1[max_dim - 1 - i] = (dim_a == 1) ? 0 : tensor1->stride[tensor1->dim - 1 - i];
    }
}

int broadcast_shape(FloatTensor* tensor1, FloatTensor* tensor2, int *ans) {
    // get the max of the both dims
    int max_dim = (tensor1->dim > tensor2->dim) ? tensor1->dim : tensor2->dim;
    int dima = tensor1->dim;
    int dimb = tensor2->dim;
    // is the ans is not null
    if (ans != NULL) {
        for (int i = 0; i < max_dim; i++) {
            //same as broadcast_stride
            int dim_a = (i >= dima) ? 1 : tensor1->shape[dima - 1 - i];
            int dim_b = (i >= dimb) ? 1 : tensor2->shape[dimb - 1 - i];
            //check the competable shape or not
            if (dim_a != 1 && dim_b != 1 && dim_a != dim_b) {
                return -1;  // Incompatible shapes
            }
            //update the ans shape of from last to first.
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

    cal_stride(newtensor);

    // Insert the data
    for (int i = 0; i < newtensor->size; i++){
        newtensor->data[i] = data[i];
    }

    return newtensor;
}

FloatTensor* add_tensor(FloatTensor* tensor1, FloatTensor* tensor2) {
    // take out the max dim and tell it's ndim_result
    int ndim_result = broadcast_shape(tensor1, tensor2, NULL);
    if (ndim_result == -1) {
        return NULL;
    }

    int* result_shape = (int*)malloc(ndim_result * sizeof(int));

    broadcast_shape(tensor1, tensor2, result_shape);//this time the result_shape is update

    int* result_stride1 = (int*)malloc(ndim_result * sizeof(int));
    int* result_stride2 = (int*)malloc(ndim_result * sizeof(int));
    broadcast_stride(tensor1, result_stride1, ndim_result);//caculate the strides
    broadcast_stride(tensor2, result_stride2, ndim_result);

    //same loop exits in add_tensor
    int total_elements = 1;
    for (int i = 0; i < ndim_result; i++) {
        total_elements *= result_shape[i];
    }

    float* result_data = (float*)malloc(total_elements * sizeof(float)); 
    FloatTensor* result = init_tensor(result_data, result_shape, ndim_result); 

    //now caculate the offset of the tensor at wich position the tensor data go
    // update the new tensor data
    for (int idx = 0; idx < total_elements; idx++) {//up to total ele
        int offset1 = 0, offset2 = 0;//assume like a[i][j] i:offset1, j:offset2
        int n_idx = idx;

        for (int i = 0; i < ndim_result; i++) {
            int stride_idx = n_idx / result->stride[i];
            n_idx %= result->stride[i];

            offset1 += stride_idx * result_stride1[i];
            offset2 += stride_idx * result_stride2[i];
        }
        // here the data add and also for other part the hole logic is same just change the sign like +, *, pow.
        result->data[idx] = tensor1->data[offset1] + tensor2->data[offset2];
    }

    free(result_stride1);//remove extra for memoery efficiency
    free(result_stride2);

    return result;
}

FloatTensor* mul_ele_tensor(FloatTensor* tensor1, FloatTensor* tensor2) {

    int ndim_result = broadcast_shape(tensor1, tensor2, NULL);
    if (ndim_result == -1) {

        return NULL;
    }

    int* result_shape = (int*)malloc(ndim_result * sizeof(int));
    broadcast_shape(tensor1, tensor2, result_shape);

    int* result_stride1 = (int*)malloc(ndim_result * sizeof(int));
    int* result_stride2 = (int*)malloc(ndim_result * sizeof(int));
    broadcast_stride(tensor1, result_stride1, ndim_result);
    broadcast_stride(tensor2, result_stride2, ndim_result);
    
    int total_elements = 1;
    for (int i = 0; i < ndim_result; i++) {
        total_elements *= result_shape[i];
    }

    float* result_data = (float*)malloc(total_elements * sizeof(float));

    FloatTensor* result = init_tensor(result_data, result_shape, ndim_result);

    for (int idx = 0; idx < total_elements; idx++) {
        int offset1 = 0, offset2 = 0;
        int n_idx = idx;

        for (int i = 0; i < ndim_result; i++) {
            int stride_idx = n_idx / result->stride[i];
            n_idx %= result->stride[i];

            offset1 += stride_idx * result_stride1[i];
            offset2 += stride_idx * result_stride2[i];
        }

        result->data[idx] = tensor1->data[offset1] * tensor2->data[offset2];
    }

    free(result_stride1);
    free(result_stride2);

    return result;
}

FloatTensor* pow_tensor(FloatTensor* tensor1, float num){

    float *data = (float*)malloc(tensor1->size * sizeof(float));
    for (int k = 0; k < tensor1->size; k++){
        data[k] = pow(tensor1->data[k], num);
    }
    FloatTensor* pow_ans_tensor = init_tensor(data, tensor1->shape, tensor1->dim);
    return pow_ans_tensor;
}

FloatTensor* pow_two_tensor(FloatTensor* tensor1, FloatTensor* tensor2) {

    int ndim_result = broadcast_shape(tensor1, tensor2, NULL);
    if (ndim_result == -1) {
        return NULL;
    }

    int* result_shape = (int*)malloc(ndim_result * sizeof(int));
    broadcast_shape(tensor1, tensor2, result_shape);

    int* result_stride1 = (int*)malloc(ndim_result * sizeof(int));
    int* result_stride2 = (int*)malloc(ndim_result * sizeof(int));
    broadcast_stride(tensor1, result_stride1, ndim_result);
    broadcast_stride(tensor2, result_stride2, ndim_result);

    int total_elements = 1;
    for (int i = 0; i < ndim_result; i++) {
        total_elements *= result_shape[i];
    }

    float* result_data = (float*)malloc(total_elements * sizeof(float));

    FloatTensor* result = init_tensor(result_data, result_shape, ndim_result);

    for (int idx = 0; idx < total_elements; idx++) {
        int offset1 = 0, offset2 = 0;
        int n_idx = idx;

        for (int i = 0; i < ndim_result; i++) {
            int stride_idx = n_idx / result->stride[i];
            n_idx %= result->stride[i];

            offset1 += stride_idx * result_stride1[i];
            offset2 += stride_idx * result_stride2[i];
        }

        result->data[idx] = pow(tensor1->data[offset1], tensor2->data[offset2]);
    }

    free(result_stride1);
    free(result_stride2);

    return result;
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