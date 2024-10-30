# 🔥 CGrad

## ⏭️🥅 Next goal:
- [x] ~~Grad engine~~ -> new task: ~~matmul/div autograd.~~ -> scaler part still remaining.
- [x] ~~randn Generator~~ -> with seed
- [ ] Make the Tensor fast: Check the `tensor.c` and `Tensorwrapper.pyx` files again, and try to optimize them to make them faster.
- [ ] stop using numpy -> add the reshape, and other stuff.
- [ ] Build a Tensor for Int, Double, Long, etc. 
- [ ] Use the Fast matrix multiplication algorithm to reduce the time complexity.
- [ ] Make loss dir and make loss like "Tenh, ReLU, sigmoid, softmax" in a more optimistic way. -> Make the `loss` folder, but you also need to make the backward pass for it.
- [ ] Make Optimizer start with SGD in C not in pyx (aka cython) -> after SGD -> Adam ...
      
        
## ✨ Overview

Lightweight library for performing tensor operations. **CGrad** is a module designed to handle all gradient computations, and most matrix manipulation and numerical work generally required for tasks in machine learning and deep learning. 🤖📚

## 💡 Features

- 🌀 Support for n-dimensional tensor operations.
- 🤖 Automatic differentiation for gradient computation.
- 🛠️ Built-in functions for common tensor operations like addition, multiplication, dot product, etc.

## ⚙️ Installation

1. [`install MinGW`](https://gcc.gnu.org/install/binaries.html) for **Windows** user install latest MinGW.
2. [`install gcc`](https://formulae.brew.sh/formula/gcc) for **Mac** or **Linux** user install latest gcc.

3. clone the repository and install manually:

    ```bash
    git clone https://github.com/Ruhaan838/CGrad
    ```
    ``` 
    python setup.py build_ext --inplace
    pip install .
    ``` 

## 🚀 Getting Started

Here’s a simple guide to get you started with **CGrad**:

### 📥 Importing the module

```python
import cgrad
```

### 📦 Creating Tensors

You can create a tensor from a Python list or NumPy array:

```python
# Creating a tensor from a list
tensor = cgrad.Tensor([1.0, 2.0, 3.0])

# Creating a tensor with a specified shape
tensor = cgrad.Tensor([[1.0, 2.0], [3.0, 4.0]])
```

### 🔄 Basic Tensor Operations

CGrad supports basic operations like addition, multiplication, etc.:

```python
# Tensor addition 
a = cgrad.Tensor([1.0, 2.0, 3.0])
b = cgrad.Tensor([4.0, 5.0, 6.0])
result = a + b  # Element-wise addition

# Tensor multiplication 
c = cgrad.Tensor([[1.0, 2.0], [3.0, 4.0]])
d = cgrad.Tensor([[5.0, 6.0], [7.0, 8.0]])
result = c * d  # Element-wise multiplication
```

### 📐 Advance Tensor Operations

CGrad supports advanced operations like matrix multiplication etc.:
``` python
a = cgrad.randn((1,2,3))
b = cgrad.randn((5,3,2))
result = a @ b
```
Note: `cgrad.matmul` is still underdevelopment.

### 🔥 Gradient Computation

CGrad automatically tracks operations and computes gradients for backpropagation:

```python
# Defining tensors with gradient tracking 
x = cgrad.Tensor([2.0, 3.0], requires_grad=True)
y = cgrad.Tensor([1.0, 4.0], requires_grad=True)

# Performing operations 
z = x * y

# Backpropagation to compute gradients 
z.backward()

# Accessing gradients 
print(x.grad)  # Gradients of x
print(y.grad)  # Gradients of y
```

## 📖 Documentation

For more detailed information, please visit our [documentation website](docs/index.html).

## 🤝 Contributing

I ❤️ contributions! If you’d like to contribute to **CGrad**, please:
```
You can contribute to code improvement and documentation editing.
```
```
If any issue is found, report it on the GitHub issue
```
1. 🍴 Clone the repository.
2. 🌱 Create a new branch for your feature or bugfix.
3. ✉️ Submit a pull request.

## 📖 Reading

- Blog about how tensors work at the computer level.
[[link](http://blog.ezyang.com/2019/05/pytorch-internals/)]
- Cython Documentation. [[link](https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html)]

## 📝 License

📜 See [`LICENSE`](LICENSE) for more details.
