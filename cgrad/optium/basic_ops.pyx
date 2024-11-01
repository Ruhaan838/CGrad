from cgrad.tensor.Tensorwrapper import Tensor
import numpy as np
from libc.stdlib cimport malloc, free
import time
from random import randint

cdef extern from "../storage/Float_tensor.h":
    ctypedef struct FloatTensor:
        float *data
        int *shape
        int *stride
        int dim
        int size

cdef extern from "../storage/methods.h":
    FloatTensor* random_tensor(int *shape, int ndim, int min, int max, int seed)
    FloatTensor* random_tensor_n(int *shape, int ndim, int seed)
    FloatTensor* zeros_tensor(int *shape, int ndim)
    FloatTensor* ones_tensor(int *shape, int ndim)

def randrange(shape, require_grad=False, min=0, max=10000):
    if not isinstance(shape, (list, tuple)) or not all(isinstance(dim, int) for dim in shape):
        raise TypeError(f"The shape should be a list or tuple of integers, not: {type(shape)}")

    cdef int ndim = len(shape)
    cdef int* c_shape = <int*>malloc(ndim * sizeof(int))

    for i in range(ndim):
        c_shape[i] = shape[i]

    cdef float* data = <float*>malloc(ndim * sizeof(float))
    cdef int seed = randint(0, 1000)

    random_tensor(c_shape, ndim, min, max, seed, data)
    
    new_random_data = np.array([data[i] for i in range(ndim)])
    new_shape = tuple(shape)
    new_random_data = new_random_data.reshape(new_shape)

    free(c_shape)
    free(data)

    return Tensor(new_random_data, require_grad=require_grad)


def rand(shape, require_grad=False):
    if not isinstance(shape, (list, tuple)) or not all(isinstance(dim, int) for dim in shape):
        raise TypeError(f"The shape should be a list or tuple of integers, not: {type(shape)}")

    cdef int ndim = len(shape)
    cdef int* c_shape = <int*>malloc(ndim * sizeof(int))

    for i in range(ndim):
        c_shape[i] = shape[i]

    cdef float* data = <float*>malloc(ndim * sizeof(float))
    cdef int seed = randint(0, 1000)

    random_tensor_n(c_shape, ndim, seed, data)

    new_random_data = np.array([round(data[i], 4) for i in range(ndim)])
    new_shape = tuple(shape)
    new_random_data = new_random_data.reshape(new_shape)

    free(c_shape)
    free(data)

    return Tensor(new_random_data, require_grad=require_grad)


def zeros(shape, require_grad=False):
    if not isinstance(shape, (list, tuple)) or not all(isinstance(dim, int) for dim in shape):
        raise TypeError(f"The shape should be a list or tuple of integers, not: {type(shape)}")

    cdef int ndim = len(shape)
    cdef int* c_shape = <int*>malloc(ndim * sizeof(int))

    for i in range(ndim):
        c_shape[i] = shape[i]

    cdef float* data = <float*>malloc(ndim * sizeof(float))

    zeros_tensor(c_shape, ndim, data)

    new_zeros_data = np.array([data[i] for i in range(ndim)])
    new_shape = tuple(shape)
    new_zeros_data = new_zeros_data.reshape(new_shape)

    free(c_shape)
    free(data)

    return Tensor(new_zeros_data, require_grad=require_grad)


def ones(shape, require_grad=False):
    if not isinstance(shape, (list, tuple)) or not all(isinstance(dim, int) for dim in shape):
        raise TypeError(f"The shape should be a list or tuple of integers, not: {type(shape)}")

    cdef int ndim = len(shape)
    cdef int* c_shape = <int*>malloc(ndim * sizeof(int))

    for i in range(ndim):
        c_shape[i] = shape[i]

    cdef float* data = <float*>malloc(ndim * sizeof(float))

    ones_tensor(c_shape, ndim, data)

    new_ones_data = np.array([data[i] for i in range(ndim)])
    new_shape = tuple(shape)
    new_ones_data = new_ones_data.reshape(new_shape)

    free(c_shape)
    free(data)

    return Tensor(new_ones_data, require_grad=require_grad)