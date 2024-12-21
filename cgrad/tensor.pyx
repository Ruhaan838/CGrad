import numpy as np
from numpy cimport PyArray_Check
#import torch
#import tensorflow as tf
from typing import List, Tuple
from libc.stdlib cimport malloc, free
import pprint 
from cgrad.optium import zeros

from cgrad.autograd.grad_funcs import AutoGrad,GradMode
from cgrad.autograd.graph import BackwardGraph

cdef class Tensor:
    #tensor
    cdef FloatTensor* tensor
    cdef List _item
    cdef Tuple _shape
    cdef int _ndim

    #backward
    cdef set _prev
    cdef bint req_grad #requires_grad
    cdef Tensor _grad
    cdef object _backward_pass
    cdef str _name_backward

    def __init__(self, data, requires_grad = False):
        
        """ data """
        if isinstance(data, (int,float)):
            data = np.array([data], dtype=np.float16)
            self._shape = tuple(data.shape)

        elif isinstance(data, (List,Tuple)):
            data = np.array(data, dtype=np.float16)
            self._shape = tuple(data.shape)

        elif PyArray_Check(data):
            self._shape = tuple(data.shape)

        #elif isinstance(data, torch.Tensor):
        #    data = data.numpy()
        #    self._shape = tuple(data.shape)

        #elif isinstance(data, tf.Tensor):
        #    data = data.numpy()
        #    self._shape = tuple(data.shape)

        elif isinstance(data, Tensor):
            self._shape = tuple(data.shape)

        else:
            raise TypeError(f"Not Support the type ({type(data)})")
        
        """ requires_grad """
        if not isinstance(requires_grad, bool):
            raise TypeError(f"Not Support the type ({type(requires_grad)})")
        
        # real work #
        self.convert_and_init(data.reshape(-1).tolist(), self._shape)

        self._item = data.tolist() # assign the list
        self._ndim = len(self._shape) # assign the int

        self._prev = set()
        self.req_grad = requires_grad and GradMode.is_enabled()
        self._grad = None
        self._backward_pass = lambda grad: None
        self._name_backward = ""

    """Accessible Varibles"""
    @property
    def item(self):
        return self._item
    @property
    def shape(self):
        return self._shape
    @property
    def ndim(self):
        return self._ndim
    @property
    def requires_grad(self):
        return self.req_grad
    @property
    def grad(self):
        return self._grad
    @property
    def prev(self):
        return self._prev
    @property
    def numel(self):
        size = self.tensor.size
        return size

    """ Setters """
    @requires_grad.setter
    def requires_grad(self, value):
        if not GradMode.is_enabled():
            raise RuntimeError("Cannot set requires_grad in no_grad context")
        if not isinstance(value, bool):
            raise TypeError(f"Unsupported type ({type(value)}) for requires_grad")
        self.req_grad = value

    @grad.setter
    def grad(self, value):
        if not isinstance(value, Tensor):
            raise TypeError(f"Unsupport the type ({type(value)})")
        self._grad = value
    
    """ Methods """
    def transpose(self):
        return self._transpose_nd()
    
    def zero_(self):
        for i in range(self.tensor.size):
            self.tensor.data[i] = 0.0
        item1 = np.array([0.0] * self.tensor.size)
        self._item = item1.reshape(self.shape).tolist()

    def sum(self, axis=None, keepdims=False):
        arr_data = np.sum(self.item, axis=axis, keepdims=keepdims).tolist()
        ans = Tensor(arr_data, requires_grad=self.requires_grad)

        if ans.requires_grad:
            ans.__backward_init__("<SumBackward0>", AutoGrad.sum_grad(self))
            ans._prev.add(self)
        return ans
    
    def mean(self, axis=None, keepdims=False):
        arr_data = np.mean(self.item, axis=axis, keepdims=keepdims).tolist()
        ans = Tensor(arr_data, requires_grad=self.requires_grad)
        
        if ans.requires_grad:
            ans.__backward_init__("<MeanBackward0>", AutoGrad.mean_grad(self,ans))
            ans._prev.add(self)
        return ans
    
    #def median(self, axis=None, keepdims=False):
    #    arr_data = np.median(self.item, axis=axis, keepdims=keepdims).tolist()
    #    ans = Tensor(arr_data, requires_grad=self.requires_grad)

    #    if ans.requires_grad:
    #        ans.__backward_init__("<MedianBackward0>", AutoGrad.median_grad(self))
    #        ans._prev.add(self)
    #    return ans

    """ Backward """

    def _backward(self, grad):
        return self._backward_pass(grad)

    def backward(self, custom_grad=None):
        if not self.req_grad:
            raise RuntimeError("This tensor does not require a gradient.")
        
        if custom_grad is None:
            if len(self.item) != 1:
                raise ValueError("Provide the custom grad or use the scaler value to calculate the grad.")
            custom_grad = Tensor(np.ones(self.shape).tolist(), requires_grad=False)

        return BackwardGraph.execute(self, custom_grad)

    """ Arithmetic """
    def __add__(self, other):
        """
            Method for adding tensor.
        """
        if isinstance(other, Tensor):
            ans = self._add_tensor(other)
            if ans.requires_grad:
                ans.__backward_init__("<AddBackward0>", AutoGrad.add_grad_tensor(self, other, ans))
            return ans

        elif isinstance(other, (int, float)):
            ans = self._add_scalar(other)
            if ans.requires_grad:
                ans.__backward_init__("<AddBackward1>", AutoGrad.add_grad_tensor(self, other, ans))
            return ans

        else:
            raise TypeError(f"Unsupport the type ({type(other)})")
    
    def __radd__(self, other):

        if isinstance(other, (int, float)):
            ans = self._add_scalar(other)
            if ans.requires_grad:
                ans.__backward_init__("<AddBackward1>", AutoGrad.add_grad_tensor(self, other, ans))
            return ans
        else:
            raise TypeError(f"Unsupport the type ({type(other)})")
    
    def __sub__(self, other):
        """
        Method for Subtracting tensor.
        """
        if isinstance(other, Tensor):
            ans = self._add_tensor(other, sub=True)
            if ans.requires_grad:
                ans.__backward_init__("<SubBackward0>", AutoGrad.add_grad_tensor(self, other, ans, is_sub=True))
            return ans

        elif isinstance(other, (int, float)):
            ans = self._add_scalar(-other)
            if ans.requires_grad:
                ans.__backward_init__("<SubBackward1>", AutoGrad.add_grad_tensor(self, other, ans))
            return ans
        else:
            raise TypeError(f"Unsupported type ({type(other)})")

    def __rsub__(self, other):
        """
        Method for reverse subtraction.
        """
        if isinstance(other, (int, float)):
            ans = self - other
            if ans.requires_grad:
                ans.__backward_init__("<SubBackward1>", AutoGrad.add_grad_tensor(self, other, ans))
            return ans
        else:
            raise TypeError(f"Unsupported type ({type(other)})")


    def __mul__(self, other):
        """
            Method for Multiplying tensor.
        """
        if isinstance(other, Tensor):
            ans = self._mul_tensor(other)
            if ans.requires_grad:
                ans.__backward_init__("<MulBackward0>", AutoGrad.mul_grad_tensor(self, other, ans))
            return ans

        elif isinstance(other, (int, float)):
            ans = self._mul_scaler(other)
            if ans.requires_grad:
                ans.__backward_init__("<MulBackward1>", AutoGrad.mul_grad_tensor(self, other, ans))
            return ans  
        else:
            raise TypeError(f"Unsupport the type ({type(other)})")
    
    def __rmul__(self, other):

        if isinstance(other, (int, float)):
            ans = self._mul_scaler(other)
            if ans.requires_grad:
                ans.__backward_init__("<MulBackward1>", AutoGrad.mul_grad_tensor(self, other, ans))
            return ans  
        else:
            raise TypeError(f"Unsupport the type ({type(other)})")

    def __truediv__(self, other):
        """
            Method for Divide tensor.
        """
        if isinstance(other, Tensor):
            ans = self._mul_tensor(other, True)
            if ans.requires_grad:
                ans.__backward_init__("<DivBackward0>", AutoGrad.div_grad_tensor(self, other, ans))
            return ans

        elif isinstance(other, (int, float)):
            if other == 0:
                return Tensor((np.ones(self.shape) * np.inf).tolist(), requires_grad=self.requires_grad)
            ans = self._mul_scaler(other ** -1)
            if ans.requires_grad:
                ans.__backward_init__("<DivBackward1>", AutoGrad.div_grad_tensor(self, other, ans))
            return ans

        else:
            raise TypeError(f"Unsupport the type ({type(other)})")

    def __rtruediv__(self, other):

        if isinstance(other, (int, float)):
            new_self = self ** -1
            ans = new_self * other
            if ans.requires_grad:
                ans.__backward_init__("<DivBackward1>", AutoGrad.div_grad_tensor(self, other, ans))
            return ans

        else:
            raise TypeError(f"Unsupport the type ({type(other)})")
    
    def __matmul__(self, other):
        if isinstance(other, Tensor):
            ans = self._matmul(other)
            if ans.requires_grad:
                ans.__backward_init__("<MatMulBackward0>", AutoGrad.matmul_grad_tensor(self, other, ans))
            return ans

        else:
            raise TypeError(f"Unsupport the type ({type(other)})")
    
    def __pow__(self, other):
        """
            Method for Power: **.
        """
        if isinstance(other, Tensor):
            ans = self._pow_tensor(other)
            if ans.requires_grad:
                ans.__backward_init__("<PowBackward0>", None)
            return ans

        elif isinstance(other, (int, float)):
            ans = self._pow_scaler(other)
            if ans.requires_grad:
                ans.__backward_init__("<PowBackward1>", None)
            return ans

        else:
            raise TypeError(f"Unsupport the type ({type(other)})")
    
    def __neg__(self):
        return self * -1
    
    def __repr__(self):
        round_list = np.round(self._item, 4)
        formate_list = pprint.pformat(round_list.tolist(), width=80)

        if self.requires_grad:
            if self._name_backward == "":
                return f"Tensor(Data = {formate_list}, requires_grad = {self.requires_grad}, Shape = {self._shape})"
            else:
                return f"Tensor(Data = {formate_list}, GradFunction = {self._name_backward}, Shape = {self._shape})"
        else:
            return f"Tensor(Data = {formate_list}, Shape = {self._shape})"
    






















#""" Tensor helper methods (without us Tensor class nothing... haha..!) """#

    def __backward_init__(self, name_backward, backward_func=None):
        """
        Function that initializes the backpropagation function and its name.
        """
        self._name_backward = name_backward
        self._backward_pass = backward_func


    cdef void convert_and_init(self, data_list: list, arr_shape: tuple):
        """
        This function converts Python data_list and arr_shape into C types
        and initializes the FloatTensor using init_tensor.
        """
        cdef int i
        cdef int data_len = len(data_list)  # Initialize data_len properly

        # Allocate memory for data
        cdef float* c_data = <float*>malloc(data_len * sizeof(float))
        if c_data == NULL:
            raise MemoryError("Unable to allocate memory for tensor data")

        # Copy data from the Python list to the C array
        for i in range(data_len):
            c_data[i] = <float>data_list[i]

        # Allocate memory for shape
        cdef int shape_len = len(arr_shape)
        cdef int* c_shape = <int*>malloc(shape_len * sizeof(int))

        if c_shape == NULL:
            free(c_data)
            raise MemoryError("Unable to allocate memory for tensor shape")

        # Copy the shape data to the C array
        for i in range(shape_len):
            c_shape[i] = <int>arr_shape[i]

        # Initialize the tensor using the C function
        self.tensor = init_tensor(c_data, c_shape, shape_len)

        if self.tensor == NULL:
            free(c_data)
            free(c_shape)
            raise MemoryError("Failed to initialize tensor")

        free(c_data)
        free(c_shape)


    
    cdef _add_tensor(self, Tensor other, sub=False):
        """
        Helper function to add two tensors.
        """
        cdef int max_dim = broadcast_shape(self.tensor.shape, self.tensor.dim, other.tensor.shape, other.tensor.dim, NULL);
        cdef int *r_shape = <int*>malloc(max_dim * sizeof(int));
        cdef int allow = broadcast_shape(self.tensor.shape, self.tensor.dim, other.tensor.shape, other.tensor.dim, r_shape)

        if allow == -1:
            raise ValueError(f"Shapes of the tensors must be broadcasted but we found {self._shape} and {other._shape}")
        
        cdef int max_size = self.tensor.size if self.tensor.size > other.tensor.size else other.tensor.size
        cdef float *r_data = <float*>malloc(max_size * sizeof(float));

        if sub:
           for i in range(other.tensor.size):
               other.tensor.data[i] *= -1

        add_tensor(self.tensor.data, other.tensor.data, r_data, self.tensor.shape, other.tensor.shape, r_shape, self.tensor.stride, other.tensor.stride, self.tensor.dim, other.tensor.dim, max_dim, max_dim)

        if r_data is NULL:
            raise MemoryError("Failed to allocate memory for the result data.")

        new_added_data = np.array([r_data[i] for i in range(max_size)])
        new_shape = tuple(r_shape[i] for i in range(max_dim))
        new_added_data = new_added_data.reshape(new_shape).tolist()

        requires_grad = self.requires_grad or other.requires_grad
        ans_tensor = Tensor(new_added_data, requires_grad=requires_grad)

        if self.requires_grad:
            ans_tensor._prev.add(self)
        if other.requires_grad:
            ans_tensor._prev.add(other)

        free(r_data)
        free(r_shape)

        return ans_tensor

    cdef _add_scalar(self, double scalar):
        """
        Helper function to add a scalar to a tensor, broadcasting the scalar across the tensor.
        """

        cdef float* result_data = <float*>malloc(self.tensor.size * sizeof(float))
        if result_data == NULL:
            raise MemoryError("Failed to allocate memory for scalar addition.")

        for i in range(self.tensor.size):
            result_data[i] = self.tensor.data[i] + scalar

        size = 0
        for i in range(self.ndim):
            size *= self.shape[i]
        
        new_added_data = np.array([result_data[i] for i in range(self.tensor.size)])
        new_shape = tuple(self.tensor.shape[i] for i in range(self.tensor.dim))
        new_added_data = new_added_data.reshape(new_shape).tolist()

        requires_grad = self.requires_grad
        ans_tensor = Tensor(new_added_data, requires_grad=requires_grad)

        if requires_grad:
            self._prev.add(self)

        free(result_data)
        return ans_tensor

    cdef _mul_tensor(self, Tensor other, div = False):
        """
        Helper function for ele wise multiplication.
        """
        cdef int max_dim = broadcast_shape(self.tensor.shape, self.tensor.dim, other.tensor.shape, other.tensor.dim, NULL);
        cdef int *r_shape = <int*>malloc(max_dim * sizeof(int));
        cdef int allow = broadcast_shape(self.tensor.shape, self.tensor.dim, other.tensor.shape, other.tensor.dim, r_shape)

        if allow == -1:
            raise ValueError(f"Shapes of the tensors must be broadcasted but we found {self._shape} and {other._shape}")
        
        cdef int max_size = self.tensor.size if self.tensor.size > other.tensor.size else other.tensor.size
        cdef float *r_data = <float*>malloc(max_size * sizeof(float));

        if div:
            for i in range(other.tensor.size):
               other.tensor.data[i] = other.tensor.data[i] ** -1

        mul_ele_tensor(self.tensor.data, other.tensor.data, r_data, self.tensor.shape, other.tensor.shape, r_shape, self.tensor.stride, other.tensor.stride, self.tensor.dim, other.tensor.dim, max_dim, max_dim)

        if r_data is NULL:
            raise MemoryError("Failed to allocate memory for the result data.")

        new_added_data = np.array([r_data[i] for i in range(max_size)])
        new_shape = tuple(r_shape[i] for i in range(max_dim))
        new_added_data = new_added_data.reshape(new_shape).tolist()

        requires_grad = self.requires_grad or other.requires_grad

        if div:
            if other == zeros(other.shape):
                return Tensor((np.ones(new_shape)*np.inf).tolist(), requires_grad=requires_grad)

        ans_tensor = Tensor(new_added_data, requires_grad=requires_grad)

        if self.requires_grad:
            ans_tensor._prev.add(self)
        if other.requires_grad:
            ans_tensor._prev.add(other)


        free(r_data)
        free(r_shape)

        return ans_tensor
    
    cdef _mul_scaler(self, double scalar):
        """
            Helper function for multiply the and number with tensor.
        """

        cdef float* result_data = <float*>malloc(self.tensor.size * sizeof(float))
        if result_data == NULL:
            raise MemoryError("Failed to allocate memory for scalar multiplication.")

        for i in range(self.tensor.size):
            result_data[i] = self.tensor.data[i] * scalar

        new_mul_data = np.array([result_data[i] for i in range(self.tensor.size)])
        new_shape = tuple(self.tensor.shape[i] for i in range(self.tensor.dim))
        new_mul_data = new_mul_data.reshape(new_shape).tolist()

        requires_grad = self.requires_grad

        ans_tensor = Tensor(new_mul_data, requires_grad=requires_grad)

        if self.requires_grad:
            ans_tensor._prev.add(self)

        free(result_data)
        return ans_tensor
    
    cdef _pow_tensor(self, Tensor other):
        """
            Helper function for get the power with other tensor.
        """
        cdef int max_dim = broadcast_shape(self.tensor.shape, self.tensor.dim, other.tensor.shape, other.tensor.dim, NULL);
        cdef int *r_shape = <int*>malloc(max_dim * sizeof(int));
        cdef int allow = broadcast_shape(self.tensor.shape, self.tensor.dim, other.tensor.shape, other.tensor.dim, r_shape)

        if allow == -1:
            raise ValueError(f"Shapes of the tensors must be broadcasted but we found {self._shape} and {other._shape}")
        
        cdef int max_size = self.tensor.size if self.tensor.size > other.tensor.size else other.tensor.size
        cdef float *r_data = <float*>malloc(max_size * sizeof(float));

        mul_ele_tensor(self.tensor.data, other.tensor.data, r_data, self.tensor.shape, other.tensor.shape, r_shape, self.tensor.stride, other.tensor.stride, self.tensor.dim, other.tensor.dim, max_dim, max_dim)

        if r_data is NULL:
            raise MemoryError("Failed to allocate memory for the result data.")

        new_added_data = np.array([r_data[i] for i in range(max_size)])
        new_shape = tuple(r_shape[i] for i in range(max_dim))
        new_added_data = new_added_data.reshape(new_shape).tolist()

        requires_grad = self.requires_grad or other.requires_grad
        ans_tensor = Tensor(new_added_data, requires_grad)
        
        if self.requires_grad:
            self._prev.add(self)
        if other.requires_grad:
            self._prev.add(other)

        free(r_data)
        free(r_shape)

        return ans_tensor

    cdef _pow_scaler(self, float num):
        """
            Helper function for power with num
        """
        cdef float *data = <float*>malloc(self.tensor.size * sizeof(float));
        pow_tensor(self.tensor.data, data, self.tensor.size, num)

        if data == NULL:
            raise MemoryError("Failed to allocate the memory for the pow data.")
        
        new_pow_data = np.array([data[i] for i in range(self.tensor.size)])
        new_shape = tuple(self.tensor.shape[i] for i in range(self.tensor.dim))
        new_pow_data = new_pow_data.reshape(new_shape).tolist()
        
        requires_grad = self.requires_grad

        ans_tensor = Tensor(new_pow_data, requires_grad=requires_grad)

        if self.requires_grad:
            self._prev.add(self)

        return ans_tensor
    


    cdef _matmul(self, Tensor other):

        cdef int max_dim = matmul_broadcast_shape(self.tensor.dim, other.tensor.dim, self.tensor.shape, other.tensor.shape, NULL)
        
        if max_dim == -1:
            raise ValueError(f"Unable to do the Matrix Multiplication for Tensor1 with shape {self.shape} and Tensor2 with shape {other.shape}")

        cdef int* result_shape = <int*>malloc(max_dim * sizeof(int))
        if result_shape == NULL:
            raise MemoryError("Failed to allocate memory for result shape.")
        
        cdef int result_size = 1
        matmul_broadcast_shape(self.tensor.dim, other.tensor.dim, self.tensor.shape, other.tensor.shape, result_shape)
        for i in range(max_dim):
            result_size *= result_shape[i]
        
        cdef float* result_data = <float*>malloc(result_size * sizeof(float))
        if result_data == NULL:
            free(result_shape)
            raise MemoryError("Failed to allocate memory for matmul result data.")
        
        matmulNd(self.tensor.data, self.tensor.shape, self.tensor.stride, self.tensor.dim,
                other.tensor.data, other.tensor.shape, other.tensor.stride, other.tensor.dim,
                result_data, result_shape, &result_size, &max_dim)
        
        new_matmul_data = np.array([result_data[i] for i in range(result_size)])
        new_shape = tuple(result_shape[i] for i in range(max_dim))
        new_matmul_data = new_matmul_data.reshape(new_shape).tolist()
        
        requires_grad = self.requires_grad or other.requires_grad

        ans_tensor = Tensor(new_matmul_data, requires_grad=requires_grad)
 
        if self.requires_grad:
            ans_tensor._prev.add(self)
        if other.requires_grad:
            ans_tensor._prev.add(other)
        
        free(result_data)
        free(result_shape)
        
        return ans_tensor
    
    cdef _transpose_nd(self):

        if not isinstance(self, Tensor):
            raise TypeError(f"Unsupported type for transpose: {type(self)}")
        
        cdef int* transposed_shape = <int*>malloc(self.tensor.dim * sizeof(int))        
        cdef float* transposed_data = <float*>malloc(self.tensor.size * sizeof(float))

        if transposed_data == NULL:
            free(transposed_shape)
            raise MemoryError("Failed to allocate memory for transpose data.")
        
        transposeNd(self.tensor.data, self.tensor.shape, self.tensor.dim, transposed_data, transposed_shape, &self.tensor.size)
        
        new_ans_data = np.array([transposed_data[i] for i in range(self.tensor.size)])
        new_shape = tuple(transposed_shape[i] for i in range(self.tensor.dim))
        new_ans_data = new_ans_data.reshape(new_shape).tolist()
        
        requires_grad = self.requires_grad
        ans_tensor = Tensor(new_ans_data, requires_grad=requires_grad)
        
        
        free(transposed_data)
        free(transposed_shape)
        
        return ans_tensor