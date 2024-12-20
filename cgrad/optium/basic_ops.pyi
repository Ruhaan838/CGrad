from typing import List, Tuple, Optional
from cgrad.tensor import Tensor

def rand(shape: List[int] | Tuple[int], requires_grad:Optional[bool] = False) -> Tensor: ...
def randrange(shape: List[int] | Tuple[int], requires_grad:Optional[bool] = False, min:Optional[int] = 0, max:Optional[int] = 1000) -> Tensor: ...
def ones(shape: List[int] | Tuple[int], requires_grad:Optional[bool] = False) -> Tensor: ...
def zeros(shape: List[int] | Tuple[int], requires_grad:Optional[bool] = False) -> Tensor: ...

