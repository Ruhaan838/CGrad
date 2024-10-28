from cgrad.tensor.Tensorwrapper import Tensor
import numpy as np

#function that caculate the grad for the + oprations
def add_grad_tensor(tensor1: Tensor, tensor2: Tensor, output: Tensor):
    def _backward():
        if tensor1.require_grad:
            if tensor1.grad is None:
                tensor1.grad = np.zeros_like(output.grad).tolist()
            tensor1.grad = (np.array(tensor1.grad) + 1.0 * np.array(output.grad)).tolist()

        if tensor2.require_grad:
            if tensor2.grad is None:
                tensor2.grad = np.zeros_like(output.grad).tolist()
            tensor2.grad = (np.array(tensor2.grad) + 1.0 * np.array(output.grad)).tolist()
    return _backward

#function that caculate the grad for the * oprations
def mul_grad_tensor(tensor1: Tensor, tensor2: Tensor, output: Tensor):
    def _backward():
        if tensor1.require_grad:
            if tensor1.grad is None:
                tensor1.grad = np.zeros_like(output.grad).tolist()
            tensor1.grad = (np.array(tensor1.grad) + np.array(tensor2.item) * np.array(output.grad)).tolist()

        if tensor2.require_grad:
            if tensor2.grad is None:
                tensor2.grad = np.zeros_like(output.grad).tolist()
            tensor2.grad = (np.array(tensor2.grad) + np.array(tensor1.item) * np.array(output.grad)).tolist()
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
    if out.grad == np.zeros(out.shape).tolist():
        out.grad = np.ones(out.shape).tolist()
    
    topo = []
    visited = set()
    
    topo_sort_backward_pass_helper(out, topo, visited)

    for node in reversed(topo):
        node._backward()
