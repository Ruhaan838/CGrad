
cdef extern from "../storage/methods.h":
    void random_tensor(int *shape, int ndim, int min, int max, int seed, float* data)
    void random_tensor_n(int *shape, int ndim, int seed, float* data)
    void zeros_tensor(int* shape, int ndim, float* data)
    void ones_tensor(int *shape, int ndim, float* data)