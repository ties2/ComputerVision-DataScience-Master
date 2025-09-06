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




