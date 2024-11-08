from cgrad.tensor import Tensor
from cgrad.optium.basic_ops import ones

cdef class BackwardGraph:

    cdef list topo
    cdef set visited

    @staticmethod
    def execute(output: Tensor, grad:Tensor):
        topo = []
        visited = set()
        
        def topo_sort_helper(node: Tensor):
            if node not in visited:
                visited.add(node)
                for child in node.prev:
                    topo_sort_helper(child)
                topo.append(node)
        
        if grad is None:
            grad = ones(output.shape, requires_grad=False)
        
        topo_sort_helper(output)
        
        for node in reversed(topo):
            node._backward(grad)