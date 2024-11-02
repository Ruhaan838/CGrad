from cgrad.tensor.Tensorwrapper import Tensor
from cgrad.optium.basic_ops import zeros
import numpy as np

def init_grad(tensor: Tensor, output_shape):
    """Initializes the gradient for the tensor if it is None."""
    if tensor.grad is None:
        tensor.grad = zeros(output_shape, require_grad=False)

def accumulate_grad_matmul(tensor: Tensor, grad_increment):
    """Accumulates the gradient increment into the tensor's gradient."""
    grad_increment.require_grad = False
    if tensor.grad.shape != grad_increment.shape:
        grad_increment = Tensor(np.sum(grad_increment.item, axis=tuple(range(grad_increment.ndim - tensor.grad.ndim))).tolist())
    tensor.grad +=  grad_increment

def accumulate_grad(tensor:Tensor, grad_increment, is_sub=False):
    """Accumulates the gradient"""
    grad_increment.require_grad = False

    output_grad = (tensor.grad) + grad_increment
    if tensor.shape == output_grad.shape:
        tensor.grad = output_grad
        if is_sub:
            tensor.grad = tensor.grad * -1
    else:
        tensor.grad = output_grad.sum(axis=0, keepdims=True)
        if is_sub:
            tensor.grad = tensor.grad * -1

#function that caculate the grad for the + oprations
## c = a + b -> dc/da = 1; dc/db = 1
def add_grad_tensor(tensor1: Tensor, tensor2: Tensor, output: Tensor, is_sub=False):
    def _backward():
        if tensor1.require_grad:
            init_grad(tensor1, output.shape)
            accumulate_grad(tensor1, output.grad)

        if tensor2.require_grad:
            init_grad(tensor2, output.shape)
            accumulate_grad(tensor2, output.grad, is_sub)

    return _backward

#function that caculate the grad for the * oprations
# c = a * b -> dc/da = b; dc/db = a
def mul_grad_tensor(tensor1: Tensor, tensor2: Tensor, output: Tensor):
    def _backward():
        if tensor1.require_grad:
            init_grad(tensor1, output.shape)
            grad_increment = tensor2 * output.grad
            accumulate_grad(tensor1, grad_increment)

        if tensor2.require_grad:
            init_grad(tensor2, output.shape)
            grad_increment = tensor1 * output.grad
            accumulate_grad(tensor2, grad_increment)

    return _backward

#function that caculate the grad for the / oprations
# c = a / b -> dc/da = 1 / b; dc/db = -(a / b**2)
def div_grad_tensor(tensor1: Tensor, tensor2: Tensor, output: Tensor):
    def _backward():
        if tensor1.require_grad:
            init_grad(tensor1, output.shape)
            grad_increment = (1 / tensor2) * output.grad
            accumulate_grad(tensor1, grad_increment)

        if tensor2.require_grad:
            init_grad(tensor2, output.shape)
            dt_do = (-1 * tensor1 / tensor2 ** 2) * output.grad
            accumulate_grad(tensor2, dt_do)

    return _backward

#function that caculate the grad for the @ oprations
def matmul_grad_tensor(tensor1: Tensor, tensor2: Tensor, output: Tensor):
    def _backward():
        if tensor1.require_grad:
            init_grad(tensor1, tensor1.shape)
            grad_increment = output.grad @ tensor2.transpose()
            accumulate_grad_matmul(tensor1, grad_increment)

        if tensor2.require_grad:
            init_grad(tensor2, tensor2.shape)
            grad_increment = tensor1.transpose() @ output.grad
            accumulate_grad_matmul(tensor2, grad_increment)

    return _backward
