## cgrad.rand

```python
cgrad.rand(shape, requires_grad = False) -> Tensor
```

**Parameters**<br>
&emsp;&emsp;   **shape** (shape_like) -  Can be a list, tuple.

**Keyword Arguments** <br>
- **requires_grad**(bool, Optional) - Allow this tensor to caculate the grad for backward pass.

## cgrad.randomrange

```python
cgrad.randrange(shape, requires_grad = False, min = 0, max = 1000) -> Tensor
```
**Parameters**<br>
&emsp;&emsp;   **shape** (shape_like) -  Can be a list, tuple.

**Keyword Arguments** <br>
- **requires_grad**(bool, Optional) - Allow this tensor to caculate the grad for backward pass.
- **min** - Minimum number that starts.
- **max** - Maximum number that ends.

## cgrad.ones

```python
cgrad.ones(shape, requires_grad = False) -> Tensor
```
**Parameters**<br>
&emsp;&emsp;   **shape** (shape_like) -  Can be a list, tuple.

**Keyword Arguments** <br>
- **requires_grad**(bool, Optional) - Allow this tensor to caculate the grad for backward pass.

## cgrad.ones_like

```python
cgrad.ones_like(tensor, requires_grad = False) -> Tensor
```
**Parameters**<br>
&emsp;&emsp;   **tensor** (tensor_like) -  Cgrad Tensor.

**Keyword Arguments** <br>
- **requires_grad**(bool, Optional) - Allow this tensor to caculate the grad for backward pass.

## cgrad.zeros

```python
cgrad.zeros(shape, requires_grad = False) -> Tensor
```
**Parameters**<br>
&emsp;&emsp;   **shape** (shape_like) -  Can be a list, tuple.

**Keyword Arguments** <br>
- **requires_grad**(bool, Optional) - Allow this tensor to caculate the grad for backward pass.

## cgrad.zeros_like

```python
cgrad.zeros_like(tensor, requires_grad = False) -> Tensor
```
**Parameters**<br>
&emsp;&emsp;   **tensor** (tensor_like) -  Cgrad Tensor.

**Keyword Arguments** <br>
- **requires_grad**(bool, Optional) - Allow this tensor to caculate the grad for backward pass.

## cgrad.add

```
cgrad.add(tensor1, tensor2) -> Tensor
```
- add two tensor
**Parameters**<br>
&emsp;&emsp;   **tensor1** (tensor_like) -  Cgrad Tensor.
&emsp;&emsp;   **tensor2** (tensor_like) -  Cgrad Tensor.

## cgrad.sub

```
cgrad.sub(tensor1, tensor2) -> Tensor
```
- sub two tensor
**Parameters**<br>
&emsp;&emsp;   **tensor1** (tensor_like) -  Cgrad Tensor.
&emsp;&emsp;   **tensor2** (tensor_like) -  Cgrad Tensor.

## cgrad.mul

```
cgrad.mul(tensor1, tensor2) -> Tensor
```
- mul two tensor
**Parameters**<br>
&emsp;&emsp;   **tensor1** (tensor_like) -  Cgrad Tensor.
&emsp;&emsp;   **tensor2** (tensor_like) -  Cgrad Tensor.

## cgrad.div

```
cgrad.div(tensor1, tensor2) -> Tensor
```
- div two tensor
**Parameters**<br>
&emsp;&emsp;   **tensor1** (tensor_like) -  Cgrad Tensor.
&emsp;&emsp;   **tensor2** (tensor_like) -  Cgrad Tensor.

## cgrad.matmul

```
cgrad.matmul(tensor1, tensor2) -> Tensor
```
- matmul two tensor
**Parameters**<br>
&emsp;&emsp;   **tensor1** (tensor_like) -  Cgrad Tensor.
&emsp;&emsp;   **tensor2** (tensor_like) -  Cgrad Tensor.

## cgrad.transpose
```
cgrad.transpose(tensor) -> Tensor
```
- transpose the tensor in evry batch.
**Parameters**<br>
&emsp;&emsp;   **tensor** (tensor_like) -  Cgrad Tensor.



