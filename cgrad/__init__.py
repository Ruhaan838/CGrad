from .tensor import Tensor
from .optium.basic_ops import rand, randrange, zeros, ones
from .autograd import AutoGrad, GradMode
from typing import Optional

__all__ = ["rand", "randrange", "zeros", "ones", "AutoGrad", "GradMode"]

def add(tensor1:Tensor, tensor2:Tensor) -> Tensor:
    return tensor1 + tensor2

def sub(tensor1:Tensor, tensor2:Tensor) -> Tensor:
    return tensor1 - tensor2

def mul(tensor1:Tensor, tensor2:Tensor) -> Tensor:
    return tensor1 * tensor2

def div(tensor1:Tensor, tensor2:Tensor) -> Tensor:
    return tensor1 / tensor2

def matmul(tensor1:Tensor, tensor2:Tensor) -> Tensor:
    return tensor1 @ tensor2

def transpose(tensor:Tensor) -> Tensor:
    return tensor.transpose()

def zeros_like(tensor:Tensor, requires_grad:Optional[bool]=False) -> Tensor:
    return zeros(tensor.shape, requires_grad=requires_grad)

def ones_like(tensor:Tensor, requires_grad:Optional[bool]=False) -> Tensor:
    return ones(tensor.shape, requires_grad=requires_grad)
