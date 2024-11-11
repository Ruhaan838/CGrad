from cgrad.tensor import Tensor
from cgrad.optium.basic_ops import zeros, ones
import numpy as np
from contextlib import contextmanager


cdef class AutoGrad:

    @staticmethod
    def init_grad(tensor:Tensor, input_shape:tuple, with_ones:bool=False):
        if tensor.grad is None and tensor.requires_grad:
            tensor.grad = zeros(shape=input_shape, requires_grad=False)
        if tensor.grad is None and tensor.requires_grad and with_ones:
            tensor.grad = ones(shape=input_shape, requires_grad=False)

    @staticmethod
    def accumulate_grad(tensor: Tensor, acc_tensor: Tensor):
        """Accumulate gradients into the tensor's gradient."""
        if tensor.grad is None:
            tensor.grad = zeros(tensor.shape)
        else:
            acc_tensor.requires_grad = False
            if tensor.grad.shape != acc_tensor.shape:
                acc_tensor = acc_tensor.sum(axis=0, keepdims=True)
            tensor.grad += acc_tensor

    @staticmethod
    def accumulate_grad_matmul(tensor: Tensor, acc_tensor: Tensor):
        """Accumulate gradient for matmul operation."""
        if tensor.grad is None:
            tensor.grad = zeros(acc_tensor.shape, requires_grad=False)
            
        acc_tensor.requires_grad = False
        if tensor.grad.shape != acc_tensor.shape:
            acc_tensor = acc_tensor.sum(axis=tuple(range(acc_tensor.ndim - tensor.grad.ndim)))
        tensor.grad += acc_tensor

    @staticmethod
    def add_grad_tensor(tensor1: Tensor, tensor2: Tensor|int|float, ans_tensor:Tensor, is_sub: bool = False):
        """Calculate gradients for addition/subtraction."""
        def _backward(grad):

            if tensor1.requires_grad:
                AutoGrad.init_grad(tensor1, ans_tensor.shape)
                AutoGrad.init_grad(ans_tensor, ans_tensor.shape, True)
                val_accumulate = ans_tensor.grad
                AutoGrad.accumulate_grad(tensor1, val_accumulate)

            if isinstance(tensor2, Tensor):
                if tensor2.requires_grad:
                    AutoGrad.init_grad(tensor2, ans_tensor.shape)
                    AutoGrad.init_grad(ans_tensor, ans_tensor.shape, True)
                    val_accumulate = ans_tensor.grad
                    if is_sub:
                        val_accumulate = -val_accumulate
                    AutoGrad.accumulate_grad(tensor2, val_accumulate)
            elif isinstance(tensor2, (int,float)):
                pass
        return _backward

    @staticmethod
    def mul_grad_tensor(tensor1: Tensor, tensor2: Tensor|int|float, ans_tensor:Tensor):
        """Calculate gradients for multiplication."""
        def _backward(grad):
            if isinstance(tensor2, Tensor):
                if tensor1.requires_grad:
                    AutoGrad.init_grad(tensor1, ans_tensor.shape)
                    AutoGrad.init_grad(ans_tensor, ans_tensor.shape, True)
                    val_accumulate = tensor2 * ans_tensor.grad
                    AutoGrad.accumulate_grad(tensor1, val_accumulate)

                if tensor2.requires_grad:
                    AutoGrad.init_grad(tensor2, ans_tensor.shape)
                    AutoGrad.init_grad(ans_tensor, ans_tensor.shape, True)
                    val_accumulate = tensor1 * ans_tensor.grad
                    AutoGrad.accumulate_grad(tensor2, val_accumulate)

            elif isinstance(tensor2, (int,float)):
                if tensor1.requires_grad:
                    AutoGrad.init_grad(tensor1, ans_tensor.shape)
                    AutoGrad.init_grad(ans_tensor, ans_tensor.shape, True)
                    val_accumulate = tensor2 * ans_tensor.grad
                    AutoGrad.accumulate_grad(tensor1, val_accumulate)

        return _backward
    
    @staticmethod
    def div_grad_tensor(tensor1: Tensor, tensor2: Tensor|int|float, ans_tensor:Tensor):
        """Calculate gradients for division."""
        def _backward(grad):
            if isinstance(tensor2, Tensor):
                if tensor1.requires_grad:
                    AutoGrad.init_grad(tensor1, ans_tensor.shape)
                    AutoGrad.init_grad(ans_tensor, ans_tensor.shape, True)
                    val_accumulate = (1 / tensor2) * ans_tensor.grad
                    AutoGrad.accumulate_grad(tensor1, val_accumulate)

                if tensor2.requires_grad:
                    AutoGrad.init_grad(tensor2, ans_tensor.shape)
                    AutoGrad.init_grad(ans_tensor, ans_tensor.shape, True)
                    val_accumulate = ((-tensor1) / (tensor2 ** 2)) * ans_tensor.grad
                    AutoGrad.accumulate_grad(tensor2, val_accumulate)

            elif isinstance(tensor2, (int, float)):
                if tensor1.requires_grad:
                    AutoGrad.init_grad(tensor1, ans_tensor.shape)
                    AutoGrad.init_grad(ans_tensor, ans_tensor.shape, True)
                    val_accumulate = (1/ tensor2) * ans_tensor.grad
                    AutoGrad.accumulate_grad(tensor1, val_accumulate)

        return _backward

    @staticmethod
    def matmul_grad_tensor(tensor1: Tensor, tensor2: Tensor, ans_tensor:Tensor):
        """Calculate gradients for matrix multiplication."""
        def _backward(grad):
            if tensor1.requires_grad:
                AutoGrad.init_grad(ans_tensor, ans_tensor.shape, True)
                val_accumulate = ans_tensor.grad @ tensor2.transpose()
                AutoGrad.accumulate_grad_matmul(tensor1, val_accumulate)
            if tensor2.requires_grad:
                AutoGrad.init_grad(ans_tensor, ans_tensor.shape, True)
                val_accumulate = tensor1.transpose() @ ans_tensor.grad
                AutoGrad.accumulate_grad_matmul(tensor2, val_accumulate)
        return _backward

    @staticmethod
    def sum_grad(tensor: Tensor):
        """Calculate gradient for sum."""
        def _backward(grad):
            if tensor.requires_grad:
                AutoGrad.init_grad(tensor, tensor.shape)
                tensor.grad += ones(tensor.shape) * grad
        return _backward

    @staticmethod
    def mean_grad(tensor: Tensor, ans_tensor:Tensor):
        """Calculate gradient for mean."""
        def _backward(grad):
            if tensor.requires_grad:
                AutoGrad.init_grad(tensor, tensor.shape)
                val_accumulate = (ones(ans_tensor.shape) * grad) / tensor.numel
                AutoGrad.accumulate_grad(tensor, val_accumulate)
        return _backward
    
    @contextmanager
    @staticmethod
    def no_grad():
        GradMode.set_enabled(False)
        try:
            yield
        finally:
            GradMode.set_enabled(True)

class GradMode:
    _enabled = True

    @classmethod
    def is_enabled(cls):
        return cls._enabled

    @classmethod
    def set_enabled(cls, mode: bool):
        cls._enabled = mode
