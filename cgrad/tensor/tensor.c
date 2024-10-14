#include <stdio.h>
#include <stdlib.h>
#include "storage.h"

#define len(arr) (sizeof(arr) / sizeof(arr[0]))

void cal_stride(CTensor* tensor){
    tensor->stride = calloc(tensor->dim, sizeof(int)); //allocate the size for stride
    tensor->stride[tensor->dim - 1] = 1; // the last dim is always 1
    for (int i = tensor->dim - 2; i >= 0; i--){
        tensor->stride[i] = tensor->stride[i + 1] * tensor->shape[i + 1];
    }
}

// tensor initialization
CTensor* init_tensor(float *data, int *shape, int ndim){
    CTensor* newtensor = (CTensor*)malloc(sizeof(CTensor)); // allocate memory for tensor struct
    if (newtensor == NULL){
        perror("Failed to allocate memory for CTensor");
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

CTensor* add_tensor(CTensor* tensor1, CTensor* tensor2){
    // Allocate exactly 'size' elements for the result
    float *data = (float*)malloc(tensor1->size * sizeof(float));

    // Perform the element-wise addition
    for (int k = 0; k < tensor1->size; k++){
        data[k] = tensor1->data[k] + tensor2->data[k];
    }

    // Initialize a new tensor with the result of the addition
    CTensor* added_tensor = init_tensor(data, tensor1->shape, tensor1->dim); 
    return added_tensor;
}

// element-wise multiplication
CTensor* mul_ele_tensor(CTensor* tensor1, CTensor* tensor2){

    float *data = (float*)mallco(tensor1->size * sizeof(float));
    // element wise multiplication
    for (int k = 0; k < tensor1->size; k++){
        data[k] = tensor1->data[k] * tensor2->data[k];
    }

    CTensor* mul_tensor = init_tensor(data, tensor1->shape, tensor1->dim);
    return mul_tensor;
}

void display_tensor(CTensor *tensor){
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

// int main(){
//     
// }