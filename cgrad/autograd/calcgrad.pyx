from cgrad.tensor.Tensorwrapper import Tensor
import numpy as np

def init_grad(tensor: Tensor, output_shape):
    """Initializes the gradient for the tensor if it is None."""
    if tensor.grad is None:
        tensor.grad = Tensor(np.zeros(output_shape).tolist())

def accumulate_grad(tensor: Tensor, grad_increment):
    """Accumulates the gradient increment into the tensor's gradient."""
    output_grad = (np.array(tensor.grad.item) + grad_increment)
    if tensor.shape == output_grad.shape:
        tensor.grad = Tensor(output_grad.tolist())
    else:
        tensor.grad = Tensor(np.sum(output_grad, axis=0, keepdims=True).tolist())

#function that caculate the grad for the + oprations
## c = a + b -> dc/da = 1; dc/db = 1
def add_grad_tensor(tensor1: Tensor, tensor2: Tensor, output: Tensor):
    def _backward():
        if tensor1.require_grad:
            init_grad(tensor1, output.shape)
            accumulate_grad(tensor1, np.array(output.grad.item))

        if tensor2.require_grad:
            init_grad(tensor2, output.shape)
            accumulate_grad(tensor2, np.array(output.grad.item))

    return _backward

#function that caculate the grad for the * oprations
# c = a * b -> dc/da = b; dc/db = a
def mul_grad_tensor(tensor1: Tensor, tensor2: Tensor, output: Tensor):
    def _backward():
        if tensor1.require_grad:
            init_grad(tensor1, output.shape)
            grad_increment = np.array(tensor2.item) * np.array(output.grad.item)
            accumulate_grad(tensor1, grad_increment)

        if tensor2.require_grad:
            init_grad(tensor2, output.shape)
            grad_increment = np.array(tensor1.item) * np.array(output.grad.item)
            accumulate_grad(tensor2, grad_increment)

    return _backward

# c = a / b -> dc/da = 1 / b; dc/db = -(a / b**2)
def div_grad_tensor(tensor1: Tensor, tensor2: Tensor, output: Tensor):
    def _backward():
        if tensor1.require_grad:
            init_grad(tensor1, output.shape)
            grad_increment = (1 / np.array(tensor2.item)) * np.array(output.grad.item)
            accumulate_grad(tensor1, grad_increment)

        if tensor2.require_grad:
            init_grad(tensor2, output.shape)
            dt_do = (-np.array(tensor1.item) / np.array(tensor2.item) ** 2) * np.array(output.grad.item)
            accumulate_grad(tensor2, dt_do)

    return _backward

def matmul_grad_tensor(tensor1: Tensor, tensor2: Tensor, output: Tensor):
    pass

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
