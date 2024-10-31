from cgrad.tensor.Tensorwrapper import Tensor
import numpy as np

cdef class BackwardGraph:

    cdef list topo
    cdef set visited

    @staticmethod
    def execute(output: Tensor):
        topo = []
        visited = set()
        
        def topo_sort_helper(node: Tensor):
            if node not in visited:
                visited.add(node)
                for child in node.prev:
                    topo_sort_helper(child)
                topo.append(node)
        
        if output.grad is None:
            output.grad = Tensor(np.ones(output.shape).tolist())
        
        topo_sort_helper(output)
        
        for node in reversed(topo):
            node._backward()