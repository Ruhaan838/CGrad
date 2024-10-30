from cgrad.tensor.Tensorwrapper import Tensor
import numpy as np
from libc.stdlib cimport malloc
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

def randrange(shape, require_grad = False, min=0, max=10000):

    if not isinstance(shape, (list, tuple)) or not all(isinstance(dim, int) for dim in shape):
        raise TypeError(f"The Shape is only support the type: list and tuple in int from not: {type(shape)}")

    cdef int* c_shape = <int*>malloc(sizeof(int) * len(shape)) 

    for i in range(len(shape)):
        c_shape[i] = <int>shape[i]
    seed = randint(0,1000)
    ndim = len(shape)
    new_random_tensor = random_tensor(c_shape, <int>ndim, <int>min, <int>max, <int> seed)

    if new_random_tensor == NULL:
        raise MemoryError("Unable to allocate memory for new_random tensor")
    
    new_random_data = np.array([new_random_tensor.data[i] for i in range(new_random_tensor.size)])
    new_shape = tuple(new_random_tensor.shape[i] for i in range(new_random_tensor.dim))
    new_random_data = new_random_data.reshape(new_shape)

    return Tensor(new_random_data, require_grad = require_grad)

def rand(shape, require_grad = False):

    if not isinstance(shape, (list, tuple)) or not all(isinstance(dim, int) for dim in shape):
        raise TypeError(f"The Shape is only support the type: list and tuple in int from not: {type(shape)}")

    cdef int ndim = len(shape)
    cdef int seed = randint(0,1000)
    cdef int* c_shape = <int*>malloc(ndim * sizeof(int)) 

    for i in range(len(shape)):
        c_shape[i] = <int>shape[i]

    new_random_tensor = random_tensor_n(c_shape, ndim, seed)

    if new_random_tensor == NULL:
        raise MemoryError("Unable to allocate memory for new_random tensor")
    
    new_random_data = np.array([round(new_random_tensor.data[i], 4) for i in range(new_random_tensor.size)])
    new_shape = tuple(new_random_tensor.shape[i] for i in range(new_random_tensor.dim))
    new_random_data = new_random_data.reshape(new_shape)

    return Tensor(new_random_data, require_grad = require_grad)