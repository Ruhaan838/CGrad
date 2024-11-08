cdef extern from "storage/Float_tensor.h":
    ctypedef struct FloatTensor:
        float *data
        int *shape
        int *stride
        int dim
        int size

cdef extern from "storage/methods.h":
    int broadcast_shape(int * shape1, int dim1, int * shape2, int dim2, int *ans)

    int matmul_broadcast_shape(int dim1, int dim2, int * shape1, int * shape2, int * shape3)

    FloatTensor * init_tensor(float *data, int *shape, int dim)

    void add_tensor(float *data1, float *data2, float *r_data,
                    int *shape1, int * shape2, int * r_shape,
                    int * stride1, int * stride2,
                    int dim1, int dim2, int r_dim, int max_dim)

    void mul_ele_tensor(float *data1, float *data2, float *r_data,
                        int *shape1, int *shape2, int *r_shape,
                        int *stride1, int *stride2,
                        int dim1, int dim2, int r_dim, int max_dim)

    FloatTensor * pow_two_tensor(FloatTensor * tensor1, FloatTensor * tensor2)

    void pow_tensor(float *data, float *r_data, int size, float num)

    void matmulNd(float * data1, int * shape1, int * stride1, int dim1,
                  float * data2, int * shape2, int * stride2, int dim2,
                  float * result_data, int * result_shape, int * result_size, int * result_dim)

    void transposeNd(float * input_data, int * input_shape, int input_dim,
                     float * transposed_data, int * new_shape, int * new_size)

    void sum_tensor(float * data1, int * shape1, int dim1,
                    float * r_data, int * r_shape, int axis, int keepdims)

    void display_tensor(FloatTensor *tensor)
