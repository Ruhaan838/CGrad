## for now this is depend on numpy but I try my hard to avoid this (🤞)
from libc.stdlib cimport malloc, free
import numpy as np
import pprint
from cgrad.autograd.grad_funcs import add_grad_tensor, mul_grad_tensor, div_grad_tensor, matmul_grad_tensor
from cgrad.autograd.graph import BackwardGraph

cdef extern from "../storage/Float_tensor.h":
    ctypedef struct FloatTensor:
        float *data
        int *shape
        int *stride
        int dim
        int size
        
cdef extern from "../storage/methods.h":

    int broadcast_shape(int* shape1, int dim1, int* shape2, int dim2, int *ans)

    int matmul_broadcast_shape(int dim1, int dim2, int* shape1, int* shape2, int* shape3)

    FloatTensor* init_tensor(float *data, int *shape, int dim)

    void add_tensor(float *data1, float *data2, float *r_data, 
                int *shape1, int* shape2, int* r_shape, 
                int* stride1, int* stride2, 
                int dim1, int dim2, int r_dim, int max_dim)

    void mul_ele_tensor(float *data1, float *data2, float *r_data, 
                    int *shape1, int *shape2, int *r_shape, 
                    int *stride1, int *stride2, 
                    int dim1, int dim2, int r_dim, int max_dim)

    FloatTensor* pow_two_tensor(FloatTensor* tensor1, FloatTensor* tensor2)

    void pow_tensor(float *data, float *r_data, int size, float num)

    void matmulNd(float* data1, int* shape1, int* stride1, int dim1,
              float* data2, int* shape2, int* stride2, int dim2,
              float* result_data, int* result_shape, int* result_size, int* result_dim)

    void transposeNd(float* input_data, int* input_shape, int input_dim,
                 float* transposed_data, int* new_shape, int* new_size)

    void sum_tensor(float* data1, int* shape1, int dim1, 
                float* r_data, int* r_shape, int axis, int keepdims)

    void display_tensor(FloatTensor *tensor)

cdef class Tensor:
    cdef FloatTensor* tensor
    cdef list _item
    cdef tuple _shape
    cdef int _ndim
    cdef set _prev
    cdef bint _re_grad #require_grad
    cdef Tensor _grad
    cdef object _backward_pass  #for store the function
    cdef str _name_backward

    def __init__(self, data: list| tuple| np.array| int| float, require_grad:bool = False):
        """
            Function that initalize the tensor using list, tuple, np.array, int or float
            Attributes
            ----------
            data : list | tuple | np.array | int | float
                Any Iterable 
        """
        try:
            if isinstance(data, (int, float)):#check the instance is int or float so it's convert it to list
                arr = np.array([data])  
                arr_shape = arr.shape  
            else:
                # TODO: chage this from numpy array to our own array for much faster then the numpy (try to hard 🤞)
                arr = np.array(data)  #convert to the np array for now later it will chage
                arr_shape = arr.shape

        except Exception as e:
            raise ValueError(f"Error in input data: {e}")

        if not isinstance(require_grad, bool):
            raise ValueError(f"require_grad is must in bool but you provide {type(require_grad)}")

        dim = len(arr_shape) #caculate the dim hope this is right 
        #flatten the data and its provide to the Tensor storage.
        data = arr.reshape(-1) 
        data_list = data.tolist()  

        self.convert_and_init(data_list, arr_shape) 

        #some acceable attributes
        self._prev = set() 
        self._item = arr.tolist()  
        self._shape = arr_shape
        self._ndim = dim
        self._re_grad = require_grad

        self._grad = None
        self._backward_pass = lambda: None
        self._name_backward = ""

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
    def grad(self):
        return self._grad

    @property    
    def require_grad(self):
        return self._re_grad
    
    @require_grad.setter
    def require_grad(self, value:bool):
        if isinstance(value, bool):
            self._re_grad = value
        else:
            raise ValueError("Unsported the type use bool values")

    @grad.setter
    def grad(self, value):
        if self.require_grad:
            if isinstance(value, Tensor):
                self._grad = value
            else:
                raise ValueError("Unsported the grad type")
        else:
            raise AttributeError("The require_grad is must True for set the grad.")

    def zero_(self):
        for i in range(self.tensor.size):
            self.tensor.data[i] = 0.0
        item1 = np.array([0.0] * self.tensor.size)
        self._item = item1.reshape(self.shape).tolist()  
        
    @property
    def prev(self):
        return self._prev

    def _backward(self):
        return self._backward_pass()
    
    def backward(self):
        if not self._re_grad:
            raise AttributeError("Please set require_grad=True to calculate the gradient.")
        return BackwardGraph.execute(self)

    def add(self,other):
        return self + other
    
    def sub(self, other):
        return self - other
    
    def mul(self, other):
        return self * other

    def pow(self, other):
        return self ** other
    
    def div(self, other):
        return self / other
    
    def matmul(self, other):
        return self @ other

    def transpose(self):
        return self._transpose_nd()

    def sum(self, axis=0, keepdims=False):
        return self.sum_t(axis, keepdims)

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

    def __getitem__(self, indx):
        if isinstance(indx, (int, slice, tuple)):
            return self._slice_data(np.array(self._item), indx)
        else:
            raise TypeError(f"Unsupported index type: {type(indx)}")

    def __back_init__(self, name_backward="", backward_func=None):
        """
        Function that init the backprop function and it's name
        """
        self._name_backward = name_backward
        self._backward_pass = backward_func 

# add method and also helper function
    def __add__(self, other):
        if isinstance(other, Tensor):
            ans = self._add_tensor(other)
            ans.require_grad = self.require_grad or other.require_grad 
            ans.__back_init__("<AddBackward>", add_grad_tensor(self, other, ans))
            return ans
        elif isinstance(other, (int, float)):
            ans = self._add_scalar(other)
            ans.require_grad = self.require_grad
            return ans
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __radd__(self, other):
        """
        Function for reverse addition also scalar.
        """
        if isinstance(other, (int, float)):
            ans = self._add_scalar(other)
            return ans
        else:
            raise TypeError(f"Unspported type for additation: {type(other)}")

#sub method and also helper funtion
    def __sub__(self, other):
        """Subtraction of a tensor or scalar from self."""
        if isinstance(other, Tensor):
            ans = self._add_tensor(other, sub=True)
            ans.require_grad = self.require_grad or other.require_grad
            ans.__back_init__("<SubBackward>", add_grad_tensor(self, other, ans, is_sub=True))
            return ans
        elif isinstance(other, (int, float)):
            ans = self._add_scalar(-other)
            return ans
        else:
            raise TypeError(f"Unsupported type for subtraction: {type(other)}")

    def __rsub__(self, other):
        """Handles scalar - tensor by reversing order."""
        if isinstance(other, (int, float)):
            ans = Tensor(other) - self
            return ans
        else:
            raise TypeError(f"Unsupported type for reverse subtraction: {type(other)}")

# mul and helper functions
    def __mul__(self, other):
        """
            Multiplication of a tensor or scalar from self.
        """
        if isinstance(other, Tensor):
            ans = self._mul_tensor(other)
            ans.require_grad = self.require_grad or other.require_grad  
            ans.__back_init__("<MulBackward>", mul_grad_tensor(self, other, ans))
            return ans
        elif isinstance(other, (int, float)):
            ans = self._mul_scaler(other)
            ans.require_grad = self.require_grad
            return ans
        else:
            raise TypeError(f"Unsupported type for multiplication: {type(other)}")
    
    def __rmul__(self, other):
        """
        Function for reverse mutiply also scalar.
        """
        if isinstance(other, (int, float)):
            ans = Tensor(other) * self
            return ans
        else:
            raise TypeError(f"Unspported type for multiplication: {type(other)}")

    def __pow__(self, other):
        """
            Function for do the power of any tenor
        """
        if isinstance(other, Tensor):
            ans = self._pow_tensor(other)
            return ans
        elif isinstance(other, (int, float)):
            ans = self._pow_scaler(other)
            return ans
        else:
            raise TypeError(f"Unspported type for power: {type(other)}")
    
    def __truediv__(self, other):
        """
            Function for devide the two tensor and scaler
        """
        if isinstance(other, Tensor):
            ans = self._mul_tensor(other, div=True)
            ans.require_grad = self.require_grad or other.require_grad
            ans.__back_init__("<DivBackword>", div_grad_tensor(self, other, ans))
            return ans

        elif isinstance(other, (int, float)):
            if other == 0:
                raise ArithmeticError("You can't devide the tensor with '0' ")
            ans = self._mul_scaler(other ** -1)
            return ans

        else:
            raise TypeError(f"Unspported type for devision: {type(other)}")
    
    def __rtruediv__(self, other):
        """
            Function for devide the two tensor and scaler
        """
        if isinstance(other, (int, float)):
            ans = Tensor(other) / self
            return ans
        else:
            raise TypeError(f"Unspported type for division: {type(other)}")

    def __matmul__(self, other):
        """
            Function for mutiply the N dim matrix
        """
        if isinstance(other, Tensor):
            ans = self._matmul(other)
            ans.require_grad = self.require_grad or other.require_grad
            ans.__back_init__("<MatMulBackword>", matmul_grad_tensor(self, other, ans))
            return ans
        else:
            raise TypeError(f"Unspport type for matrix multiplication {type(other)}")

    def _slice_data(self, data, indx):
        return Tensor(data[indx])

    cdef _add_tensor(self, Tensor other, sub=False):
        """
        Helper function to add two tensors. Requires both tensors to have the same shape.
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
        new_added_data = new_added_data.reshape(new_shape)
        ans_tensor = Tensor(new_added_data)
        ans_tensor._prev = set((self, other))

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

        new_add_tensor = init_tensor(result_data, self.tensor.shape, self.tensor.dim)

        if new_add_tensor == NULL:
            free(result_data)
            raise MemoryError("Failed to allocate memory for the result tensor.")

        new_added_data = np.array([new_add_tensor.data[i] for i in range(new_add_tensor.size)])
        new_shape = tuple(new_add_tensor.shape[i] for i in range(new_add_tensor.dim))
        new_added_data = new_added_data.reshape(new_shape)
        ans_tensor = Tensor(new_added_data)
        ans_tensor._prev = set((self, scalar))
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
        new_added_data = new_added_data.reshape(new_shape)

        ans_tensor = Tensor(new_added_data)
        ans_tensor._prev = set((self, other))

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

        new_mul_tensor = init_tensor(result_data, self.tensor.shape, self.tensor.dim)

        if new_mul_tensor == NULL:
            free(result_data)
            raise MemoryError("Failed to allocate memory for the result tensor.")

        new_mul_data = np.array([new_mul_tensor.data[i] for i in range(new_mul_tensor.size)])
        new_shape = tuple(new_mul_tensor.shape[i] for i in range(new_mul_tensor.dim))
        new_mul_data = new_mul_data.reshape(new_shape)

        ans_tensor = Tensor(new_mul_data)
        ans_tensor._prev = set((self, scalar))
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
        new_added_data = new_added_data.reshape(new_shape)

        ans_tensor = Tensor(new_added_data)
        ans_tensor._prev = set((self, other))

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
        new_pow_data = new_pow_data.reshape(new_shape)

        ans_tensor = Tensor(new_pow_data)
        ans_tensor._prev = set((self, num))
        return ans_tensor

    cdef _matmul(self, Tensor other):
        
        if not isinstance(self, Tensor) and not isinstance(other, Tensor):
            raise TypeError(f"Unsupported type for matmul: {type(self)} or {type(other)}")

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
        new_matmul_data = new_matmul_data.reshape(new_shape)
        
        ans_tensor = Tensor(new_matmul_data)
        ans_tensor._prev = set((self, other))
        
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
        new_ans_data = new_ans_data.reshape(new_shape)
        
        ans_tensor = Tensor(new_ans_data)
        
        free(transposed_data)
        free(transposed_shape)
        
        return ans_tensor

    
    cdef Tensor sum_t(self, int axis, bint keepdims):
        """
        Summing along the specified axis with optional keepdims.
        """
        if not isinstance(axis, int) or not isinstance(keepdims, bool):
            raise TypeError(f"Unsupported type for axis {type(axis)} and keepdims {type(keepdims)}")

        cdef int new_dim = self.tensor.dim if keepdims else self.tensor.dim - 1
        cdef float* data = <float*>malloc(self.tensor.size * sizeof(float))
        cdef int* shape = <int*>malloc(new_dim * sizeof(int))

        if data == NULL:
            raise MemoryError("Unable to allocate memory for result data")

        cdef int dim = 0 
        cdef int size = 0
        sum_tensor(self.tensor.data, self.tensor.shape, self.tensor.dim, data, shape, axis, keepdims)

        size = 1
        for i in range(new_dim):
            size *= shape[i]
        dim = new_dim

        result_data = np.array([data[i] for i in range(size)])
        result_shape = tuple(shape[i] for i in range(dim))
        result_data = result_data.reshape(result_shape)
        
        free(data)
        free(shape)

        return Tensor(result_data, require_grad=self.require_grad)


    def __repr__(self):
        round_list = np.round(self._item, 4)
        formate_list = pprint.pformat(round_list.tolist())

        
        if self._re_grad:
            if self._name_backward == "":
                
                return f"Tensor(Data = {formate_list}, require_grad = {self._re_grad}, Shape = {self._shape})"
            else:
                grad_str =  f"Tensor(Data = {formate_list}, GradFunction = {self._name_backward}, Shape = {self._shape})"
                return grad_str
        else:
            return f"Tensor(Data = {formate_list}, Shape = {self._shape})"
