## for new this is to depend on numpy but I try my hard to avoid this (🤞)
from libc.stdlib cimport malloc, free
import numpy as np

cdef extern from "storage.h":
    ctypedef struct CTensor:
        float *data
        int *shape
        int *stride
        int dim
        int size
    
    CTensor* init_tensor(float *data, int *shape, int dim)
    CTensor* add_tensor(CTensor* tensro1, CTensor* tensor2)

cdef class Tensor:
    cdef CTensor* tensor
    cdef list _item
    cdef tuple _shape
    cdef int _ndim

    def __init__(self, data: list| tuple| np.array| int| float):
        """
            Function that initalize the tensor using list, tuple, np.array, int or float
        """
        try:
            if isinstance(data, (int, float)):#check the instance is int or float so it's convert it to list
                arr = np.array([data])  
                arr_shape = ()  
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

    def __add__(self, other):
        """
        Overloaded function for adding tensors or adding a scalar to a tensor.
        Dispatches to the appropriate helper function based on the type of 'other'.
        """

        if isinstance(other, Tensor):
            return self._add_tensor(other)
        elif isinstance(other, (int, float)):
            return self._add_scalar(other)
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}")

    cdef _add_tensor(self, Tensor other):
        """
        Helper function to add two tensors. Requires both tensors to have the same shape.
        """

        if self._shape != other._shape:
            raise ValueError("Shapes of the tensors must be the same for addition.")

       
        new_add_tensor = add_tensor(self.tensor, other.tensor)

        if new_add_tensor is NULL:
            raise MemoryError("Failed to allocate memory for the result tensor.")

        
        new_added_data = [new_add_tensor.data[i] for i in range(new_add_tensor.size)]
        new_shape = tuple(new_add_tensor.shape[i] for i in range(new_add_tensor.dim))
        
        return Tensor(new_added_data)

    cdef _add_scalar(self, double scalar):
        """
        Helper function to add a scalar to a tensor, broadcasting the scalar across the tensor.
        """

        cdef int i
        cdef float* result_data = <float*>malloc(self.tensor.size * sizeof(float))
        if result_data == NULL:
            raise MemoryError("Failed to allocate memory for scalar addition.")

        for i in range(self.tensor.size):
            result_data[i] = self.tensor.data[i] + scalar

        new_add_tensor = init_tensor(result_data, self.tensor.shape, self.tensor.dim)

        if new_add_tensor == NULL:
            free(result_data)
            raise MemoryError("Failed to allocate memory for the result tensor.")

        new_added_data = [new_add_tensor.data[i] for i in range(new_add_tensor.size)]
        new_shape = tuple(new_add_tensor.shape[i] for i in range(new_add_tensor.dim))
        
        return Tensor(new_added_data)

    def __repr__(self):
        return f"\nTensor(Data = {self._item}, Shape = {self._shape})\n"
