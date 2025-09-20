## PyTorch Learning Notes
This document summarizes key concepts and notes for learning PyTorch, a popular open-source machine learning framework.
Machine Learning Methods

Supervised Learning: Models are trained on labeled data, where each input has a corresponding output. The goal is to learn a mapping from inputs to outputs.
Examples: Classification, Regression


Unsupervised Learning: Models work with unlabeled data to find patterns or structures, such as clustering or dimensionality reduction.
Examples: Clustering, Autoencoders


Reinforcement Learning: An agent learns by interacting with an environment, receiving rewards or penalties based on actions, aiming to maximize cumulative rewards.
Examples: Game playing, Robotics


---

## Key Tasks in Machine Learning

![classification and regression](https://github.com/ties2/ComputerVision-DataScience-Master/blob/main/lectures/machine%20learning/regression.png)

Classification: Predicting discrete labels or categories for input data.
Example: Identifying whether an email is spam or not.


Regression: Predicting continuous numerical values.
Example: Forecasting house prices based on features like size and location.

Note: Both of them use supervised learning method

**Classification** predicts discrete categories (e.g., spam vs. not spam), while **regression** predicts continuous values (e.g., house prices).

---
## Neural Networks

Neural Networks are machine learning models inspired by the human brain, consisting of interconnected layers of nodes (neurons). They process input data through weighted connections, applying activation functions to capture complex patterns. In PyTorch, neural networks are built using torch.nn.Module, enabling flexible architectures for tasks like classification and regression.

Key Components:

Input Layer: Receives raw data.
Hidden Layers: Extract features through transformations.
Output Layer: Produces predictions.
Activation Functions: (e.g., ReLU, Sigmoid) introduce non-linearity.
Loss Function: Measures prediction error.
Optimizer: Updates weights (e.g., SGD, Adam).

Example: A simple PyTorch neural network for classification:


``` python
import torch.nn as nn
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # Input: 10 features, Output: 5
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 2)   # Output: 2 classes
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
---

## Deep Learning vs. Machine Learning

- **Machine Learning (ML)**: Broad field of algorithms (e.g., linear regression, SVMs) for learning from data. Includes supervised, unsupervised, and reinforcement learning.
- **Deep Learning (DL)**: Subset of ML using multi-layered neural networks to model complex patterns in large datasets (e.g., images, text).
- **Differences**:
  - DL uses deep neural networks; ML includes simpler models.
  - DL needs large data and compute power; ML works with smaller datasets.
  - DL automates feature learning; ML often requires manual feature engineering.

**Example Deep Learning Model in PyTorch**:
```python
import torch.nn as nn
class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.layer = nn.Linear(10, 2)  # Simple neural network
    def forward(self, x):
        return self.layer(x)

```
---
## PyTorch Tensors

A **PyTorch tensor** is a multi-dimensional array for efficient numerical computations in machine learning, similar to NumPy arrays but with GPU support and gradient tracking.

- **Features**:
  - Multi-dimensional (scalars, vectors, matrices, etc.).
  - Supports GPU acceleration (e.g., CUDA).
  - Tracks gradients for backpropagation with `requires_grad=True`.

**Example**:
```python
import torch
# Create a 2D tensor
tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
# Enable gradient tracking
tensor.requires_grad_(True)
# Perform operation
result = tensor * 2
```
---
## Autograd in PyTorch

**Autograd** is PyTorch's automatic differentiation system that computes gradients for backpropagation, enabling neural network training.

- **Features**:
  - Tracks operations on tensors with `requires_grad=True`.
  - Builds a dynamic computational graph for automatic gradient calculation.
  - Used with optimizers to update model parameters.

**Example**:
```python
import torch
# Create tensor with gradient tracking
x = torch.tensor([2.0], requires_grad=True)
# Define a computation
y = x**2 + 3*x + 1
# Compute gradients
y.backward()
# Access gradient
print(x.grad)  # Output: 7.0 (dy/dx = 2x + 3, evaluated at x=2)
```
---
## PyTorch `optim` Module

The `torch.optim` module provides optimizers to update model parameters by minimizing the loss using gradients from `autograd`.

- **Key Optimizers**:
  - `SGD`: Stochastic Gradient Descent.
  - `Adam`: Adaptive optimizer, widely used in deep learning.
- **Usage**: Initialize with model parameters, compute loss, backpropagate, and update weights.

**Example**:
```python
import torch
import torch.nn as nn
import torch.optim as optim
model = nn.Linear(10, 2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# In training loop
loss = torch.tensor(0.5)  # Example loss
optimizer.zero_grad()     # Clear gradients
loss.backward()           # Compute gradients
optimizer.step()          # Update weights
```
---
## PyTorch DataLoader

The torch.utils.data.DataLoader handles efficient data loading, batching, shuffling, and parallel processing for training.

Features:

Batches data for efficient training.
Shuffles data to improve generalization.
Supports multi-threaded loading with num_workers.


Works with Dataset classes (e.g., TensorDataset).

Example:

```python
from torch.utils.data import DataLoader, TensorDataset
import torch
# Sample data
x = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# Iterate in training loop
for batch_x, batch_y in dataloader:
    # Process batch
    pass
```

---
## PyTorch Basics

Tensors: Core data structure in PyTorch, similar to NumPy arrays but optimized for GPU acceleration.
Example: torch.tensor([1, 2, 3])


Autograd: Automatic differentiation for computing gradients, essential for backpropagation.
Example: x.requires_grad_(True) to track computations.


Neural Networks: Built using torch.nn.Module, allowing flexible model design.
Example: Defining a simple feedforward network with nn.Linear.


Getting Started with PyTorch

Installation:pip install torch torchvision


Basic Workflow:
Load and preprocess data (e.g., using torchvision.datasets).
Define a model using torch.nn.
Specify a loss function (e.g., nn.CrossEntropyLoss for classification).
Choose an optimizer (e.g., torch.optim.SGD or torch.optim.Adam).
Train the model with a training loop, computing gradients and updating weights.


Some exercise for pytorch:

[link](https://github.com/ties2/ComputerVision-DataScience-Master/tree/main/lectures/Scientific%20Programming)


---
# pytorch learning

Status: In progress
URL: https://github.com/ties2/ComputerVision-DataScience-Master/blob/main/lectures/machine%20learning/pytorch.markdown
start date: September 19, 2025

- **Broadcasting**
    
    Broadcasting in PyTorch is a mechanism that allows a tensor with a smaller shape to be used in an operation with a tensor of a larger shape, without explicitly creating copies of the smaller tensor. It's a powerful tool that makes code more memory-efficient and concise.
    
    ### How Broadcasting Works
    
    Broadcasting works by "stretching" the smaller tensor along its dimensions to match the shape of the larger tensor. This stretching is virtual; no extra memory is allocated. For two tensors to be broadcastable, they must satisfy a set of rules:
    
    1. **Dimensions are compared from right to left.**
    2. **Two dimensions are compatible when:**
        - They are equal.
        - One of them is 1.
    
    If one tensor has fewer dimensions than the other, its shape is prepended with ones to match.
    
    For example, a tensor with shape `(3, 4)` can be broadcast with a tensor of shape `(4)`. The `(4)` tensor is treated as `(1, 4)`, and then it's stretched to `(3, 4)`.
    
    ### Examples
    
    - **Scalar and Tensor:** A scalar value (e.g., `5`) can be broadcast to any tensor. The scalar is treated as a tensor of shape `(1,)` and then expanded to match the other tensor's shape.
    
    ```mathematica
    import torch
    
    a = torch.tensor([[1, 2], [3, 4]])
    b = 5
    c = a + b
    # c will be [[6, 7], [8, 9]]
    ```
    
    - **Vector and Matrix:** A vector can be broadcast with a matrix if their dimensions are compatible.
    
    ```mathematica
    import torch
    
    a = torch.tensor([[1, 2, 3], [4, 5, 6]]) # shape (2, 3)
    b = torch.tensor([10, 20, 30])           # shape (3) -> treated as (1, 3)
    
    c = a + b
    # c will be [[11, 22, 33], [14, 25, 36]]
    ```
    

While both are used for numerical computation in Python, **PyTorch Tensors** and **NumPy arrays** have distinct differences, with the most significant being **GPU acceleration** and **automatic differentiation**.

| Feature | **PyTorch Tensor** | **NumPy Array** |
| --- | --- | --- |
| **Primary Use** | Deep learning, neural networks | Scientific computing, data analysis |
| **Hardware** | Optimized for GPU & CPU | CPU only (natively) |
| **Automatic Gradients** | Built-in via `autograd` | Not supported |
| **Mutability** | Mutable | Mutable |
| **API** | Similar to NumPy, but with more deep learning-specific functions | General-purpose mathematical and array manipulation functions |

### Key Reasons for Using Tensors

1. **GPU Acceleration:** PyTorch tensors can be easily moved to and from a GPU, allowing for massive parallel computation. This is crucial for the heavy matrix operations involved in training neural networks, leading to significant speedups.
2. **Automatic Differentiation:** Tensors are integrated with PyTorch's `autograd` system. This system automatically tracks all operations performed on a tensor and computes the gradients. This feature is the backbone of backpropagation, the algorithm used to train neural networks.
3. **Efficiency:** Tensors are more memory-efficient and computationally faster than Python lists or NumPy arrays for deep learning tasks. They provide a high-level API for complex mathematical operations, making the code cleaner and more readable.
4. **Specialized Operations:** PyTorch tensors support a wide range of functions optimized for deep learning, from basic arithmetic to advanced linear algebra and convolutions. This rich set of operations makes it a powerful tool for building and experimenting with neural network architectures.

we use. tensor for saving input , output and features of model

---

## make tensor in python

Here's a short explanation of each PyTorch tensor creation function:

- `t = torch.tensor(data)`: Creates a tensor from existing data (like a list or NumPy array). It copies the data.
- `t = torch.Tensor`: This is the base class for tensors. You typically don't use it directly for creation but rather for type checking or as a superclass. `torch.Tensor(data)` is a shorthand for `torch.FloatTensor(data)`.
- `t = torch.empty(size)`: Creates a tensor with the given `size` without initializing its elements. The values will be whatever is in memory.
- `t = torch.empty_like(data)`: Creates an uninitialized tensor with the same size as `data`.
- `t = torch.ones(size)`: Creates a tensor filled with ones.
- `t = torch.ones_like(data)`: Creates a tensor filled with ones with the same size as `data`.
- `t = torch.zeros(size)`: Creates a tensor filled with zeros.
- `t = torch.zeros_like(data)`: Creates a tensor filled with zeros with the same size as `data`.
- `t = torch.rand(size)`: Creates a tensor with random numbers from a uniform distribution (0 to 1).
- `t = torch.rand_like(data)`: Creates a tensor with random uniform numbers with the same size as `data`.
- `t = torch.randn(size)`: Creates a tensor with random numbers from a standard normal distribution (mean=0, variance=1).
- `t = torch.randn_like(data)`: Creates a tensor with random normal numbers with the same size as `data`.
- `t = torch.randint(low, high, size)`: Creates a tensor of integers chosen randomly between `low` (inclusive) and `high` (exclusive).
- `t = torch.randint_like(data, high)`: Creates a tensor of random integers with the same size as `data`. The integers are between 0 and `high` (exclusive).
- `t = torch.randperm(n)`: Creates a 1D tensor of a random permutation of integers from 0 to `n-1`.
- `t = torch.arange(start, stop, step)`: Creates a 1D tensor with a sequence of numbers from `start` to `stop`(exclusive), with a specified `step` size.
- `t = torch.linspace(start, stop, num)`: Creates a 1D tensor with a sequence of `num` evenly spaced numbers between `start` and `stop` (inclusive).
- `t = torch.from_numpy(array)`: Creates a tensor from a NumPy `array`. The created tensor and the NumPy array share the same memory, so changing one will change the other.

make 1D tensor manual:

t=torch.tensor()

- device (GPU or CPU ( default))
- dtype (consider type of data)
- requires_grad (default false , for track gradiant)

```
import torch
t=torch.tensor([2,4,6,8], device ='cpu',dtype=torch.float32)
print(t)
```

note: array by default is float32 in tensor and float64 in numpy array

**define device:**

```
t=torch.tensor([2,4,6,8], device ='cpu',dtype=torch.float32)
print(t,t.dtype)

n = np.array([2.,4.,6.,8.])
print(n,n.dtype)

if torch.cuda.is_available():
 mydevice =torch.device('cuda')
else:
 mydevice =torch.device('cpu')

print(mydevice)
```

make tensor with order number:

- manual
- arange()
    
    t=torch.arange(stop)
    
    t=torch.arange(start,stop)
    
    t=torch.arange(start,stop,step)
    
- linspace()
    
    torch.linspace(start,stop,num)
    

```mathematica
import torch

def create_tensors_with_arange_and_linspace():
    """
    Demonstrates the usage of torch.arange() and torch.linspace()
    with different arguments.
    """
    print("--- Using torch.arange() ---")
    
    # torch.arange(stop)
    # Creates a 1D tensor with a sequence of numbers from 0 to 'stop' (exclusive)
    print("1. torch.arange(5)")
    t1 = torch.arange(5)
    print(f"Tensor: {t1}")
    print(f"Shape: {t1.shape}\n")

    # torch.arange(start, stop)
    # Creates a 1D tensor with numbers from 'start' to 'stop' (exclusive)
    print("2. torch.arange(2, 8)")
    t2 = torch.arange(2, 8)
    print(f"Tensor: {t2}")
    print(f"Shape: {t2.shape}\n")

    # torch.arange(start, stop, step)
    # Creates a 1D tensor with numbers from 'start' to 'stop' (exclusive) with a given 'step'
    print("3. torch.arange(0, 10, 2)")
    t3 = torch.arange(0, 10, 2)
    print(f"Tensor: {t3}")
    print(f"Shape: {t3.shape}\n")
    
    print("--- Using torch.linspace() ---")

    # torch.linspace(start, stop, num)
    # Creates a 1D tensor with 'num' evenly spaced numbers between 'start' and 'stop' (inclusive)
    print("1. torch.linspace(0, 10, 5)")
    t4 = torch.linspace(0, 10, 5)
    print(f"Tensor: {t4}")
    print(f"Shape: {t4.shape}\n")
    
    print("2. torch.linspace(1, 10, 10)")
    t5 = torch.linspace(1, 10, 10)
    print(f"Tensor: {t5}")
    print(f"Shape: {t5.shape}\n")

if __name__ == "__main__":
    create_tensors_with_arange_and_linspace()
```

**features of tensor:**

- `t.shape`: A tuple representing the dimensions of the tensor (e.g., `(3, 4)` for a 2D tensor). It's the same as `t.size()`. s[0] raw, s[1] calumn
- `t.ndims`: The number of dimensions of the tensor, also available as `t.dim()`.
- `t.dtype`: The data type of the tensor's elements (e.g., `torch.float32`, `torch.int64`).
- `t.device`: The device where the tensor is stored, either `'cpu'` or `'cuda:0'`.
- `t.requires_grad`: A boolean indicating whether PyTorch is tracking operations on this tensor for automatic differentiation.
- `t.size()`: A function that returns a `torch.Size` object, which is a tuple representing the dimensions of the tensor. It's the same as `t.shape`.
- `t.numel()`: The total number of elements in the tensor.

---

**make 2D tensor na d 3D tensor**

3D = [2D,2D,…2D]

torch.tensor([1,2,3),(5,7,8),(6,9,2)]

```
if torch.cuda.is_available():
 mydevice =torch.device('cuda')
else:
 mydevice =torch.device('cpu')
print('1D tensor')
t1=torch.tensor([2,4,6,8], device =mydevice,dtype=torch.float32,requires_grad=True)
print(t1)
print(f"device:{t1.device}\n")
print(f"grand:{t1.requires_grad}\n")
print(f"type:{t1.dtype}\n")
print(f"shape:{t1.shape}\n")
print(f"size:{t1.size()}\n")
print(f"dim:{t1.ndim}\n")
print(f"total number:{t1.numel()}\n")
print("----------------------------------------")
print('different type of tensor with numpy array')
n = np.array([2.,4.,6.,8.])
print(n,n.dtype)
print("----------------------------------------") 
print('2D tensor')
t2=torch.tensor([(1,2,3),(5,7,8),(6,9,2)])
print(t2)
print(f"device:{t2.device}\n")
print(f"grand:{t2.requires_grad}\n")
print(f"type:{t2.dtype}\n")
print(f"shape:{t2.shape}\n")
print(f"size:{t2.size()}\n")
print(f"dim:{t2.ndim}\n")
print(f"total number:{t2.numel()}\n")
print("----------------------------------------") 
print('3D tensor')
t3=torch.tensor([
    [[1,2,3],[4,5,6]],
    [[0,0,0],[0,0,0]]
    
    ])
print(t3)
print(f"device:{t3.device}\n")
print(f"grand:{t3.requires_grad}\n")
print(f"type:{t3.dtype}\n")
print(f"shape:{t3.shape}\n")
print(f"size:{t3.size()}\n")
print(f"dim:{t3.ndim}\n")
print(f"total number:{t3.numel()}\n")
```

Note:You cannot directly create a PyTorch tensor with a string as its data type. PyTorch tensors are designed to handle numerical data—specifically, numbers like integers, floats, and booleans—which are necessary for mathematical operations in machine learning and deep learning.

To work with string data in PyTorch, you first need to convert it into a numerical representation. This process is common in Natural Language Processing (NLP) and typically involves steps like:

1. **Tokenization**: Breaking down the string into smaller units (words or subwords).
2. **Vocabulary Mapping**: Assigning a unique integer ID to each token.
3. **Embedding**: Converting the integer IDs into dense numerical vectors.

---

**make tensor with pytorch methods:** 

- `t = torch.empty(size)`: Creates a tensor of the specified `size` with uninitialized data. The values are random.
- `t = torch.empty_like(x)`: Creates a new, uninitialized tensor with the same size as an existing tensor `x`.
- `t = torch.ones(size)`: Creates a tensor of the specified `size` with all elements filled with the value `1`.
- `t = torch.ones_like(x)`: Creates a new tensor with the same size as `x` and fills all its elements with `1`.
- `t = torch.zeros(size)`: Creates a tensor of the specified `size` with all elements filled with the value `0`.
- `t = torch.zeros_like(x)`: Creates a new tensor with the same size as `x` and fills all its elements with `0`.

**random number generation**

- `t = torch.rand(size)`: Creates a tensor with random numbers from a **uniform distribution** (between 0 and 1).
- `t = torch.rand_like(x)`: Creates a new tensor with the **same size** as an existing tensor `x`, filled with random numbers from a uniform distribution.
- `t = torch.randn(size)`: Creates a tensor with random numbers from a **standard normal distribution** (mean=0, variance=1).
- `t = torch.randn_like(x)`: Creates a new tensor with the **same size** as `x`, filled with random numbers from a standard normal distribution.
- `t = torch.randint(low, high, size)`: Creates a tensor with **random integers** between `low` (inclusive) and `high`(exclusive).
- `t = torch.randint_like(x, high)`: Creates a new tensor with the **same size** as `x`, filled with random integers between 0 and `high` (exclusive).
- `t = torch.randperm(n)`: Creates a 1D tensor with a **random permutation** of integers from 0 to `n-1`.