# 🔥 CGrad

## ✨ Overview

Lightweight and Efficient library for performing tensor operations. **CGrad** is a module designed to handle all gradient computations, and most matrix manipulation and numerical work generally required for tasks in machine learning and deep learning. 🤖📚

## 💡 Features

- 🌀 Support for n-dimensional tensor operations.
- 🤖 Automatic differentiation for gradient computation.
- 🛠️ Built-in functions for common tensor operations like addition, multiplication, dot product, etc.

## ⚙️ Installation

To install **CGrad**, you can use `pip`:

```bash
pip install cgrad
```

Or clone the repository and install manually:

```bash
git clone https://github.com/Ruhaan838/CGrad
python setup.py install
```

## 🚀 Getting Started

Here’s a simple guide to get you started with **CGrad**:

### 📥 Importing the module

```python
import cgrad as cg
```

### 📦 Creating Tensors

You can create a tensor from a Python list or NumPy array:

```python
# Creating a tensor from a list
tensor = cg.Tensor([1.0, 2.0, 3.0])

# Creating a tensor with a specified shape
tensor = cg.Tensor([[1.0, 2.0], [3.0, 4.0]])
```

### 🔄 Basic Tensor Operations

CGrad supports basic operations like addition, multiplication, etc.:

```python
# Tensor addition ➕
a = cg.Tensor([1.0, 2.0, 3.0])
b = cg.Tensor([4.0, 5.0, 6.0])
result = a + b  # Element-wise addition

# Tensor multiplication ✖️
c = cg.Tensor([[1.0, 2.0], [3.0, 4.0]])
d = cg.Tensor([[5.0, 6.0], [7.0, 8.0]])
result = c * d  # Element-wise multiplication
```

### 🔥 Gradient Computation

CGrad automatically tracks operations and computes gradients for backpropagation:

```python
# Defining tensors with gradient tracking 🌟
x = cg.Tensor([2.0, 3.0], requires_grad=True)
y = cg.Tensor([1.0, 4.0], requires_grad=True)

# Performing operations ⚙️
z = x * y + 2.0

# Backpropagation to compute gradients 🔄
z.backward()

# Accessing gradients 📈
print(x.grad)  # Gradients of x
print(y.grad)  # Gradients of y
```

## 🤝 Contributing

I ❤️ contributions! If you’d like to contribute to **CGrad**, please:

1. 🍴 Fork the repository.
2. 🌱 Create a new branch for your feature or bugfix.
3. ✉️ Submit a pull request.

## 📝 License

📜 See `LICENSE` for more details.
