from typing import Tuple, Callable
from cgrad import Tensor
from contextlib import contextmanager

class AutoGrad:
    
    @staticmethod
    def init_grad(tensor:Tensor, input_shape:Tuple): ...
    """
        Initialisation the Grad of the tensor with zeros.
    """
    
    @staticmethod
    def accumulate_grad(tensor: Tensor, acc_tensor: Tensor): ...
    """
        Accumulate the Grad.
    """
    
    @staticmethod
    def accumulate_grad_matmul(tensor: Tensor, acc_tensor: Tensor): ...
    """Accumulate gradient for matmul operation."""
    
    @staticmethod
    def add_grad_tensor(tensor1: Tensor, tensor2: Tensor|int|float, ans_tensor:Tensor, is_sub: bool = False) -> Callable:...
    """Calculate gradients for addition/subtraction."""
    
    @staticmethod
    def mul_grad_tensor(tensor1: Tensor, tensor2: Tensor|int|float, ans_tensor:Tensor) -> Callable: ...
    """Calculate gradients for multiplication."""
    
    @staticmethod
    def div_grad_tensor(tensor1: Tensor, tensor2: Tensor|int|float, ans_tensor:Tensor) -> Callable: ...
    """Calculate gradients for division."""
    
    @staticmethod
    def matmul_grad_tensor(tensor1: Tensor, tensor2: Tensor, ans_tensor:Tensor) -> Callable: ...
    """Calculate gradients for matrix multiplication."""
    
    @staticmethod
    def sum_grad(tensor: Tensor) -> Callable:...
    """Calculate gradient for sum."""
    
    @staticmethod
    def mean_grad(tensor: Tensor) -> Callable:...
    """Calculate gradient for mean."""
    
    
    @contextmanager
    @staticmethod
    def no_grad() -> None: ... 

class GradMode:
    
    @classmethod
    def is_enabled(cls) -> bool:...
    @classmethod
    def set_enabled(cls, mode: bool) -> None:...