## for now this is to depend on numpy but I try my hard to avoid this (🤞)
from libc.stdlib cimport malloc, free
import numpy as np

cdef extern from "storage/storage.h":
    ctypedef struct CTensor:
        float *data
        int *shape
        int *stride
        int dim
        int size
    
    int broadcast_shape(CTensor* tensor1, CTensor* tensor2, int *ans)
    CTensor* init_tensor(float *data, int *shape, int dim)
    CTensor* add_tensor(CTensor* tensor1, CTensor* tensor2)
    CTensor* mul_ele_tensor(CTensor* tensor1, CTensor* tenosr2)
    CTensor* pow_two_tensor(CTensor* tensor1, CTensor* tensor2)
    CTensor* pow_tensor(CTensor* tensor1, float num)
    
cdef class Tensor:
    cdef CTensor* tensor
    cdef list _item
    cdef tuple _shape
    cdef int _ndim
    cdef set _prev
    def __init__(self, data: list| tuple| np.array| int| float, _prev=()):
        """
            Function that initalize the tensor using list, tuple, np.array, int or float
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

        dim = len(arr_shape) #caculate the dim hope this is right 
        #flatten the data and its provide to the Tensor storage.
        data = arr.reshape(-1) 
        data_list = data.tolist()  

        self.__convert_and_init(data_list, arr_shape) 

        #some acceable attributes
        self._prev = set(_prev) 
        self._item = arr.tolist()  
        self._shape = arr_shape
        self._ndim = dim

    @property
    def item(self):
        return self._item
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def ndim(self):
        return self._ndim
    
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

    cdef void __convert_and_init(self, data_list: list, arr_shape: tuple):
        """
        This function converts Python data_list and arr_shape into C types
        and initializes the CTensor using init_tensor.
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

# add method and also helper function
    def __add__(self, other):
        """
        Function for adding tensors or adding a scalar to a tensor.
        """

        if isinstance(other, Tensor):
            return self._add_tensor(other)
        elif isinstance(other, (int, float)):
            return self._add_scalar(other)
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __radd__(self, other):
        """
        Function for reverse addition also scalar.
        """
        if isinstance(other, Tensor):
            return self._add_tensor(other)
        elif isinstance(other, (int, float)):
            return self._add_scalar(other)
        else:
            raise TypeError(f"Unspported type for additation: {type(other)}")

#sub method and also helper funtion
    def __sub__(self, other):
        """
            Function for subtracting tensor or scalar to a tensor.
        """
        if isinstance(other, Tensor):
            new_other = other * -1
            return self._add_tensor(new_other)
        elif isinstance(other, (int, float)):
            new_other = other * -1
            return self._add_scalar(new_other)
        else:
            raise TypeError(f"Unspported type for subtraction: {type(other)}")
    
    def __rsub__(self, other):
        if isinstance(other, Tensor):
            new_other = other * -1
            return self._add_tensor(new_other)
        elif isinstance(other, (int, float)):
            new_other = other * -1
            return self._add_scalar(new_other)
        else:
            raise TypeError(f"Unspported type for subtraction: {type(other)}")

# mul and helper functions
    def __mul__(self, other):
        """
        Function for multiply tensors or multiply a scalar to a tensor.
        """
        if isinstance(other, Tensor):
            return self._mul_tensor(other)
        elif isinstance(other, (int, float)):
            return self._mul_scaler(other)
        else:
            raise TypeError(f"Unspported type for multiplication: {type(other)}")
    
    def __rmul__(self, other):
        """
        Function for reverse mutiply also scalar.
        """
        if isinstance(other, Tensor):
            return self._mul_tensor(other)
        elif isinstance(other, (int, float)):
            return self._mul_scaler(other)
        else:
            raise TypeError(f"Unspported type for multiplication: {type(other)}")

    def __pow__(self, other):
        """
            Function for do the power of any tenor
        """
        if isinstance(other, Tensor):
            return self._pow_tensor(other)
        elif isinstance(other, (int, float)):
            return self._pow_scaler(other)
        else:
            raise TypeError(f"Unspported type for power: {type(other)}")
    
    def __truediv__(self, other):
        """
            Function for devide the two tensor and scaler
        """
        if isinstance(other, Tensor):
            return self._mul_tensor(other ** -1)

        elif isinstance(other, (int, float)):
            if other == 0:
                raise ArithmeticError("You can't devide the tensor with '0' ")
            return self._mul_scaler(other ** -1)

        else:
            raise TypeError(f"Unspported type for devision: {type(other)}")
    
    def __rtruediv__(self, other):
        """
            Function for devide the two tensor and scaler
        """
        if isinstance(other, Tensor):
            return self._mul_tensor(other ** -1)

        elif isinstance(other, (int, float)):
            if other == 0:
                raise ArithmeticError("You can't divide the tensor with '0' ")
            return self._mul_scaler(other ** -1)
            
        else:
            raise TypeError(f"Unspported type for division: {type(other)}")

    cdef _add_tensor(self, Tensor other):
        """
        Helper function to add two tensors. Requires both tensors to have the same shape.
        """
        cdef int* ans = <int*>malloc(sizeof(int));
        cdef int allow = broadcast_shape(self.tensor, other.tensor, ans)

        if allow == -1:
            raise ValueError(f"Shapes of the tensors must be but we found {self._shape} and {other._shape}")

        new_add_tensor = add_tensor(self.tensor, other.tensor)

        if new_add_tensor is NULL:
            raise MemoryError("Failed to allocate memory for the result tensor.")

        new_added_data = np.array([new_add_tensor.data[i] for i in range(new_add_tensor.size)])
        new_shape = tuple(new_add_tensor.shape[i] for i in range(new_add_tensor.dim))
        new_added_data = new_added_data.reshape(new_shape)
        return Tensor(new_added_data, (self, other))

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
        return Tensor(new_added_data, (self, scalar))

    cdef _mul_tensor(self, Tensor other):
        """
        Helper function for ele wise multiplication.
        """
        cdef int* ans = <int*>malloc(sizeof(int));
        cdef int allow = broadcast_shape(self.tensor, other.tensor, ans)

        if allow == -1:
            raise ValueError(f"Shapes of the tensors must be but we found {self._shape} and {other._shape}")
        
        new_mul_tensor = mul_ele_tensor(self.tensor, other.tensor)

        if new_mul_tensor == NULL:
            raise MemoryError("Failed to allocate memory for new_mul_tensor.")
        
        new_mul_data = np.array([new_mul_tensor.data[i] for i in range(new_mul_tensor.size)])
        new_shape = tuple(new_mul_tensor.shape[i] for i in range(new_mul_tensor.dim))
        new_mul_data = new_mul_data.reshape(new_shape)

        return Tensor(new_mul_data, (self, other))

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

        return Tensor(new_mul_data, (self, scalar))

    cdef _pow_tensor(self, Tensor other):
        """
            Helper function for get the power with other tensor.
        """
        cdef int* ans = <int*>malloc(sizeof(int));
        cdef int allow = broadcast_shape(self.tensor, other.tensor, ans)

        if allow == -1:
            raise ValueError(f"Shapes of the tensors must be but we found {self._shape} and {other._shape}")
        
        two_pow_tensor = pow_two_tensor(self.tensor, other.tensor)

        if two_pow_tensor == NULL:
            raise MemoryError("Failes to allocate the memory for new tensor for pow")
        
        two_pow_data = np.array([two_pow_tensor.data[i] for i in range(two_pow_tensor.size)])
        new_shape = tuple(two_pow_tensor.shape[i] for i in range(two_pow_tensor.dim))
        two_pow_data = two_pow_data.reshape(new_shape)

        return Tensor(two_pow_data, (self, other))

    cdef _pow_scaler(self, float num):
        """
            Helper function for power with num
        """
        new_pow_tensor = pow_tensor(self.tensor, num)

        if new_pow_tensor == NULL:
            raise MemoryError("Failed to allocate the memory for the pow tensor.")
        
        new_pow_data = np.array([new_pow_tensor.data[i] for i in range(new_pow_tensor.size)])
        new_shape = tuple(new_pow_tensor.shape[i] for i in range(new_pow_tensor.dim))
        new_pow_data = new_pow_data.reshape(new_shape)

        return Tensor(new_pow_data, (self, num))

    def __repr__(self):
        return f"Tensor(Data = {self._item}, Shape = {self._shape})"
