from cgrad.tensor import Tensor
from cgrad.optium.basic_ops import zeros, ones
import numpy as np
from contextlib import contextmanager

requires_grad_enabled = True

cdef class AutoGrad:

    @staticmethod
    def init_grad(tensor:Tensor, input_shape:tuple):
        if tensor.grad is None and tensor.requires_grad and requires_grad_enabled:
            tensor.grad = zeros(shape=input_shape, requires_grad=False)

    @staticmethod
    def accumulate_grad(tensor: Tensor, acc_tensor: Tensor):
        """Accumulate gradients into the tensor's gradient."""
        if tensor.grad is None and requires_grad_enabled:
            tensor.grad = zeros(tensor.shape)
        else:
            acc_tensor.requires_grad = False
            if tensor.grad.shape != acc_tensor.shape:
                acc_tensor = acc_tensor.sum(axis=0, keepdims=True)
            tensor.grad += acc_tensor

    @staticmethod
    def accumulate_grad_matmul(tensor: Tensor, acc_tensor: Tensor):
        """Accumulate gradient for matmul operation."""
        if tensor.grad.shape != acc_tensor.shape:
            acc_tensor = Tensor(np.sum(acc_tensor.item, axis=tuple(range(acc_tensor.ndim - tensor.grad.ndim))).tolist())
        tensor.grad += acc_tensor

    @staticmethod
    def add_grad_tensor(tensor1: Tensor, tensor2: Tensor|int|float, ans_tensor:Tensor, is_sub: bool = False):
        """Calculate gradients for addition/subtraction."""
        def _backward(grad):

            if tensor1.requires_grad and requires_grad_enabled:
                AutoGrad.init_grad(tensor1, ans_tensor.shape)
                val_accumulate = grad
                AutoGrad.accumulate_grad(tensor1, val_accumulate)

            if isinstance(tensor2, Tensor):
                if tensor2.requires_grad and requires_grad_enabled:
                    AutoGrad.init_grad(tensor2, ans_tensor.shape)
                    val_accumulate = grad
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
                if tensor1.requires_grad and requires_grad_enabled:
                    AutoGrad.init_grad(tensor1, ans_tensor.shape)
                    val_accumulate = tensor2 * ans_tensor.grad
                    AutoGrad.accumulate_grad(tensor1, val_accumulate)

                if tensor2.requires_grad and requires_grad_enabled:
                    AutoGrad.init_grad(tensor2, ans_tensor.shape)
                    val_accumulate = tensor1 * ans_tensor.grad
                    AutoGrad.accumulate_grad(tensor2, val_accumulate)

            elif isinstance(tensor2, (int,float)):
                if tensor1.requires_grad and requires_grad_enabled:
                    AutoGrad.init_grad(tensor1, ans_tensor.shape)
                    val_accumulate = tensor2 * ans_tensor.grad
                    AutoGrad.accumulate_grad(tensor1, val_accumulate)

        return _backward
    
    @staticmethod
    def div_grad_tensor(tensor1: Tensor, tensor2: Tensor|int|float, ans_tensor:Tensor):
        """Calculate gradients for division."""
        def _backward(grad):
            if isinstance(tensor2, Tensor):
                if tensor1.requires_grad and requires_grad_enabled:
                    AutoGrad.init_grad(tensor1, ans_tensor.shape)
                    val_accumulate = (1 / tensor2) * ans_tensor.grad
                    AutoGrad.accumulate_grad(tensor1, val_accumulate)

                if tensor2.requires_grad and requires_grad_enabled:
                    AutoGrad.init_grad(tensor2, ans_tensor.shape)
                    val_accumulate = ((-tensor1) / (tensor2 ** 2)) * ans_tensor.grad
                    AutoGrad.accumulate_grad(tensor2, val_accumulate)

            elif isinstance(tensor2, (int, float)):
                if tensor1.requires_grad and requires_grad_enabled:
                    AutoGrad.init_grad(tensor1, ans_tensor.shape)
                    val_accumulate = (1/ tensor2) * ans_tensor.grad
                    AutoGrad.accumulate_grad(tensor1, val_accumulate)

        return _backward

    @staticmethod
    def matmul_grad_tensor(tensor1: Tensor, tensor2: Tensor, ans_tensor:Tensor):
        """Calculate gradients for matrix multiplication."""
        def _backward(grad):
            if tensor1.requires_grad and requires_grad_enabled:
                AutoGrad.init_grad(tensor1, ans_tensor.shape)
                val_accumulate = ans_tensor.grad @ tensor2.transpose()
                AutoGrad.accumulate_grad_matmul(tensor1, val_accumulate)
            if tensor2.requires_grad and requires_grad_enabled:
                AutoGrad.init_grad(tensor2, ans_tensor.shape)
                val_accumulate = tensor1.transpose() @ ans_tensor.grad
                AutoGrad.accumulate_grad_matmul(tensor2, val_accumulate)
        return _backward

    @staticmethod
    def sum_grad(tensor: Tensor):
        """Calculate gradient for sum."""
        def _backward(grad):
            if tensor.requires_grad and requires_grad_enabled:
                AutoGrad.init_grad(tensor, tensor.shape)
                tensor.grad += ones(tensor.shape) * grad
        return _backward

    @staticmethod
    def mean_grad(tensor: Tensor):
        """Calculate gradient for mean."""
        def _backward(grad):
            if tensor.requires_grad and requires_grad_enabled:
                AutoGrad.init_grad(tensor, tensor.shape)
                tensor.grad += (ones(tensor.shape) * grad) / tensor.numel
        return _backward
    
    @contextmanager
    @staticmethod
    def no_grad():
        # Disable gradients temporarily
        GradMode.set_enabled(False)
        try:
            yield
        finally:
            # Re-enable gradients after block
            GradMode.set_enabled(True)

class GradMode:
    _enabled = True

    @classmethod
    def is_enabled(cls):
        return cls._enabled

    @classmethod
    def set_enabled(cls, mode: bool):
        cls._enabled = mode
