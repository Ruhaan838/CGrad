# cgrad.Tensor

```python
cgrad.Tensor(data, requires_grad=False) -> Tensor
```

**Parameters**<br>
&emsp;&emsp;   **data** (array_like) - Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar.

**Keyword Arguments** <br>
- **requires_grad**(bool, Optional) - Allow this tensor to caculate the grad for backward pass.

<br>

### Examples:
### Arithmetic 
- Tensor is suppot the [Broadcast](https://stackoverflow.com/questions/51371070/how-does-pytorch-broadcasting-work) rule for Arithmetic.

- #### Addition and Subtraction
    -   Broadcast the shape of Tensor `x` from `(1, 3) to (2, 3)`
        ``` python
        x = cgrad.Tensor([[1,2,3]]) # (1, 3)
        y = cgrad.Tensor([[1,2,3],[2,3,4]]) # (2, 3)

        z = x + y # (2, 3)
        z = x - y # (2,3)
        ```

- #### Multiplication and Division
    - Broadcast the shape of Tensor `y` from `(1 ,2, 3) to (2, 2, 3)`
        ``` python
        x = cgrad.Tensor([[[1,2,3],[1,2,3]]]) # (1, 2, 3)
        y = cgrad.Tensor([[[1,2,3],[2,3,4]], [[1,2,3],[2,3,4]]]) # (2, 2, 3)

        z = x * y # (2, 2, 3)
        z = x / y # (2, 2, 3)
        ```

- #### Matrix Multiplication
    - Cgrad is also support the Nd Gemm.
        ``` python
        x = cgrad.Tensor([[1,2], [2,3]]) #(2, 2)
        y = cgrad.Tensor([[2,3], [3,4]]) #(2, 2)

        z = x @ y #(2, 2)
        ```

## Variables

- ### Tensor.item
    ```python
    Tensor.item -> List[float]
    ```
    - This will return the item in the Tensor in from of List.

- ### Tensor.shape
    ```python
    Tensor.shape -> Tuple[int]
    ```
    - This will return the shape of the tensor in from of Tuple.

- ### Tensor.ndim
    ```python
    Tensor.ndim -> int
    ```
    - This will return the dim of the tensor in from of int.

- ### Tensor.numel
    ```python
    Tensor.numel -> int
    ```
    - This will return the number of elements in the tensor in from of int.

## Setters and Variables

- ### Tensor.grad
    ```python
    Tensor.grad -> Tensor
    ```
    - This will return the current Gradient of the tensor in from of Tensor.
    - You can set the Tensor using:
    ```python
    Tensor.grad(value) -> Tensor
    ```
    - value: in Tensor from.

- ### Tensor.requires_grad
    ```python
    Tensor.requires_grad -> bool
    ```
    - This will give the status of the tensor grad caculation in from of bool.
    - You can set the Tensor using:
    ```python
    Tensor.requires_grad(value) -> None
    ```
    - value: in from of bool





## Methods

- ### Tensor.transpose

    ```
    Tensor.transpose() -> Tensor
    ```
    - Transpose the tensor for all batch.

- ### Tensor.backward
    ```
    Tensor.backward(custom_grad = None) -> None
    ```
    **Parameters**<br>
    &emsp;&emsp;   **custom_grad** (tensor_like) - Cgrad Tensor same as input shape use the ones_like or ones.

- ### Tensor.sum
    ```
    Tensor.sum(axis = None, keepdims = False) -> Tensor
    ```
    - Note: This time this .sum use the numpy's sum but after some time this will gone.

- ### Tensor.mean
    ```
    Tensor.mean(axis = None, keepdims = False) -> Tensor
    ```
    - Note: This time this .mean use the numpy's mean but after some time this will gone.

- ### Tensor.zeros_
    ```
    Tensor.zeros_() -> None
    ```
    - This will set the data of tensor to zero.