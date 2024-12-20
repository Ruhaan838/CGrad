from cgrad.tensor import Tensor
import numpy as np
from libc.stdlib cimport malloc, free
from random import randint

def randrange(shape, requires_grad=False, min=0, max=10000):
    if not isinstance(shape, (list, tuple)) or not all(isinstance(dim, int) for dim in shape):
        raise TypeError(f"The shape should be a list or tuple of integers, not: {type(shape)}")

    cdef int ndim = len(shape)
    cdef int* c_shape = <int*>malloc(ndim * sizeof(int))
    cdef int size = 1;

    for i in range(ndim):
        c_shape[i] = shape[i]
        size *= shape[i]

    cdef float* data = <float*>malloc(size * sizeof(float))
    cdef int seed = randint(0, 1000)

    random_tensor(c_shape, ndim, min, max, seed, data)
    
    new_random_data = np.array([data[i] for i in range(size)])
    new_shape = tuple(shape)
    new_random_data = new_random_data.reshape(new_shape).tolist()

    free(c_shape)
    free(data)

    return Tensor(new_random_data, requires_grad=requires_grad)


def rand(shape, requires_grad=False):
    if not isinstance(shape, (list, tuple)) or not all(isinstance(dim, int) for dim in shape):
        raise TypeError(f"The shape should be a list or tuple of integers, not: {type(shape)}")

    cdef int ndim = len(shape)
    cdef int* c_shape = <int*>malloc(ndim * sizeof(int))
    cdef int size = 1;

    for i in range(ndim):
        c_shape[i] = shape[i]
        size *= shape[i]

    cdef float* data = <float*>malloc(size * sizeof(float))
    cdef int seed = randint(0, 1000)

    random_tensor_n(c_shape, ndim, seed, data)

    new_random_data = np.array([round(data[i], 4) for i in range(size)])
    new_shape = tuple(shape)
    new_random_data = new_random_data.reshape(new_shape).tolist()

    free(c_shape)
    free(data)

    return Tensor(new_random_data, requires_grad=requires_grad)


def zeros(shape, requires_grad=False):
    if not isinstance(shape, (list, tuple)) or not all(isinstance(dim, int) for dim in shape):
        raise TypeError(f"The shape should be a list or tuple of integers, not: {type(shape)}")

    cdef int ndim = len(shape)
    cdef int* c_shape = <int*>malloc(ndim * sizeof(int))
    cdef int size = 1;

    for i in range(ndim):
        c_shape[i] = shape[i]
        size *= shape[i]

    cdef float* data = <float*>malloc(size * sizeof(float))

    zeros_tensor(c_shape, ndim, data)

    new_zeros_data = np.array([data[i] for i in range(size)])
    new_shape = tuple(shape)
    new_zeros_data = new_zeros_data.reshape(new_shape).tolist()

    free(c_shape)
    free(data)

    return Tensor(new_zeros_data, requires_grad=requires_grad)


def ones(shape, requires_grad=False):
    if not isinstance(shape, (list, tuple)) or not all(isinstance(dim, int) for dim in shape):
        raise TypeError(f"The shape should be a list or tuple of integers, not: {type(shape)}")

    cdef int ndim = len(shape)
    cdef int* c_shape = <int*>malloc(ndim * sizeof(int))
    cdef int size = 1;

    for i in range(ndim):
        c_shape[i] = shape[i]
        size *= shape[i]

    cdef float* data = <float*>malloc(size * sizeof(float))

    ones_tensor(c_shape, ndim, data)

    new_ones_data = np.array([data[i] for i in range(size)])
    new_shape = tuple(shape)
    new_ones_data = new_ones_data.reshape(new_shape).tolist()

    free(c_shape)
    free(data)

    return Tensor(new_ones_data, requires_grad=requires_grad)