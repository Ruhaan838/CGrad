from .tensor import Tensor
from .optium.basic_ops import rand, randrange, zeros, ones
from .autograd import AutoGrad

__all__ = ["rand", "randrange", "zeros", "ones", "AutoGrad"]

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

def transpose(tensor1:Tensor) -> Tensor:
    return tensor1.transpose()

def zeros_like(tensor:Tensor) -> Tensor:
    return zeros(tensor.shape)

def ones_like(tensor:Tensor) -> Tensor:
    return ones(tensor.shape)
