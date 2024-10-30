from cgrad.tensor.Tensorwrapper import Tensor
import numpy as np

def init_grad(tensor: Tensor, output_shape):
    """Initializes the gradient for the tensor if it is None."""
    if tensor.grad is None:
        tensor.grad = Tensor(np.zeros(output_shape).tolist())

def accumulate_grad(tensor: Tensor, grad_increment):
    """Accumulates the gradient increment into the tensor's gradient."""
    grad_increment.require_grad = False
    if tensor.grad.shape != grad_increment.shape:
        grad_increment = Tensor(np.sum(grad_increment.item, axis=tuple(range(grad_increment.ndim - tensor.grad.ndim))).tolist())
    tensor.grad = tensor.grad + grad_increment

#function that caculate the grad for the + oprations
## c = a + b -> dc/da = 1; dc/db = 1
def add_grad_tensor(tensor1: Tensor, tensor2: Tensor, output: Tensor):
    def _backward():
        if tensor1.require_grad:
            init_grad(tensor1, output.shape)
            accumulate_grad(tensor1, output.grad)

        if tensor2.require_grad:
            init_grad(tensor2, output.shape)
            accumulate_grad(tensor2, output.grad)

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
            accumulate_grad(tensor1, grad_increment)

        if tensor2.require_grad:
            init_grad(tensor2, tensor2.shape)
            grad_increment = tensor1.transpose() @ output.grad
            accumulate_grad(tensor2, grad_increment)

    return _backward

#helper function that do backword
def topo_sort_backward_pass_helper(v: Tensor, topo:list, visited:set):
    if v not in visited:
        visited.add(v)
        for child in v.prev:
            topo_sort_backward_pass_helper(child, topo, visited)
        topo.append(v)

#caculate the backword pass
def backward_node(out: Tensor):
    if out.grad == None:
        out.grad = Tensor(np.ones(out.shape).tolist())
    
    topo = []
    visited = set()
    
    topo_sort_backward_pass_helper(out, topo, visited)

    for node in reversed(topo):
        node._backward()
