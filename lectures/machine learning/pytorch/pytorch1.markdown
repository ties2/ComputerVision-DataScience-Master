## PyTorch Learning Notes
This document summarizes key concepts and notes for learning PyTorch, a popular open-source machine learning framework.
Machine Learning Methods

* Supervised Learning: Models are trained on labeled data, where each input has a corresponding output. The goal is to learn a mapping from inputs to outputs.
Examples: Classification, Regression

Classification: Predicts a category (e.g., "Spam" or "Not Spam," "Cat" or "Dog").

Regression: Predicts a numerical value (e.g., "Price" of a house, "Temperature" tomorrow)


* Unsupervised Learning: Models work with unlabeled data to find patterns or structures, such as clustering or dimensionality reduction.
Examples: Clustering, Autoencoders

Clustering: Groups similar data together (e.g., "Segmenting customers" based on behavior).

Dimensionality Reduction: Compresses data by finding the most important features (e.g., Autoencoders, PCA)

### clustering
Clustering is an unsupervised machine learning task that automatically groups similar data points together. The goal is to find natural structures in your data without using any pre-defined labels. Points in the same group (a "cluster") are more similar to each other than to points in other clusters.

Common Algorithm: K-Means.

Example: Grouping customers into different purchasing habit segments.

### Autoencoders
An Autoencoder is a type of unsupervised neural network used for compression and feature learning. It has two main parts:

Encoder: This part compresses the input data (like an image) into a much smaller, low-dimensional representation (called the "bottleneck" or "latent space").

Decoder: This part takes the compressed representation and tries to reconstruct the original input data as accurately as possible.

The network is trained to make the final output identical to the original input. The compressed "bottleneck" representation becomes a useful, dense summary of the data, which is great for dimensionality reduction or anomaly detection.


* Reinforcement Learning: An agent learns by interacting with an environment, receiving rewards or penalties based on actions, aiming to maximize cumulative rewards.
Examples: Game playing, Robotics

---
## When to Use Each Model Type

### Classification

When: You need to predict a category or class.

Question it Answers: "Which group does this belong to?"

Examples:

Spam or Not Spam?

Cat, Dog, or Bird?

Is this transaction fraudulent? (Yes/No)

### Regression

When: You need to predict a continuous numerical value.

Question it Answers: "How much?" or "How many?"

Examples:

What is the price of this house?

What will the temperature be tomorrow?

How many sales will we have next month?

### Clustering

When: You have unlabeled data and want to find natural groups based on similarity.

Question it Answers: "What are the hidden groups in my data?"

Examples:

Segmenting customers based on purchasing habits.

Grouping similar news articles together.

###vAutoencoder

When: You want to compress data (dimensionality reduction) or find anomalies.

Question it Answers: "What is a compressed summary of this data?" or "Is this data point 'normal'?"

Examples:

Removing noise from an image.

Detecting unusual activity on a network (anomaly detection).

A Special Note on Neural Networks

### Neural Network (NN)

This is an architecture or tool, not a problem type.

You use a Neural Network to perform other tasks.

When to Use It: When your problem is very complex (e.g., non-linear) and you have a lot of data.

Examples:

Use an NN for Classification (e.g., image recognition).

Use an NN for Regression (e.g., predicting complex stock movements).

An Autoencoder is a type of Neural Network.


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

Expalin code: 

import torch.nn as nn
This imports PyTorch's neural network library and gives it the common alias nn. This library contains all the building blocks for models, like layers (nn.Linear) and activation functions (nn.ReLU).

Class Definition

Python
class SimpleNN(nn.Module):

This starts defining your model as a Python class named SimpleNN.

It inherits from nn.Module, which is the base class for all neural network models in PyTorch. This gives your class a lot of built-in functionality, like tracking parameters.

The Constructor
Python
    def __init__(self):
        super(SimpleNN, self).__init__()
def __init__(self): is the constructor for the class. This code runs only once when you first create an object from this class (e.g., model = SimpleNN()).

Its job is to define and initialize all the layers the network will use.

super(...) is a required line that calls the constructor of the parent class (nn.Module) to set everything up correctly.

Defining the Layers

Python
        self.fc1 = nn.Linear(10, 5)
This creates the first layer, a fully connected (or "linear") layer, and assigns it to the attribute self.fc1.

nn.Linear(10, 5) means this layer expects an input with 10 features and will output 5 features.

Python
        self.relu = nn.ReLU()
This creates a ReLU activation function. This is a non-linear function (it simply changes all negative numbers to 0) that is applied after a layer. This non-linearity is what allows the network to learn complex patterns.

Python
        self.fc2 = nn.Linear(5, 2)
This creates the second and final linear layer, self.fc2.

It takes the 5 features from the previous layer as input and outputs 2 features. These 2 features are the final "scores" (or "logits") of the network.

The Forward Pass
Python
    def forward(self, x):
The forward method is the most important part. It defines how data flows through the layers you just defined.

This method is called automatically whenever you pass input data to your model (e.g., output = model(input_data)).

x represents the batch of input data.

The Data Flow Logic

Python
        x = self.relu(self.fc1(x))
This is the first step of the data flow.

self.fc1(x): The input x is passed through the first linear layer (fc1).

self.relu(...): The result is then passed through the ReLU activation function.

Python
        x = self.fc2(x)
The activated output from the previous step is now passed through the second linear layer (fc2).

Python
        return x
Finally, the network returns the result of the last layer. This x is now a tensor with 2 features, representing the model's final output.

Summary of the Model's Structure:

Input (10 features) → Linear Layer 1 (fc1) → (5 features) → ReLU Activation → (5 features) → Linear Layer 2 (fc2) → (2 features) → Output

---
### Layer in a Neural Network
A layer is the main building block of a neural network. Think of it as a "station" on an assembly line that processes data.


What it does: It takes information from the previous layer, performs a calculation, and passes the result to the next layer.

Learning: Layers contain parameters (or "weights") that are adjusted during training. This is how the network "learns."

Your Example: self.fc1 = nn.Linear(10, 5) creates a "Fully Connected" (or "Linear") layer. This layer's job is to take 10 input features and transform them into 5 output features by applying a matrix multiplication (and adding a bias)

### Activation Functions
An activation function is a simple function applied after a layer to introduce non-linearity.

Why it's needed: Without activation functions, a neural network, no matter how many layers, would just be a simple linear equation (like y = mx + b). It could only learn straight-line relationships.

What it does: It allows the network to learn complex, "curvy" patterns in the data. Think of it as a "decision-maker" or a "switch" at each station.

Your Example: self.relu = nn.ReLU() is the Rectified Linear Unit. It's a very simple switch:

If the input number is positive, it lets it pass through unchanged.

If the input number is negative, it changes it to zero.



1. For Hidden Layers (The "Workhorses")
These are used between layers to help the network learn complex patterns.

* ReLU (Rectified Linear Unit):

What it is: The default, most popular choice. It's fast and effective.

How it works: It's a simple switch. If the input is positive, it passes it on. If the input is negative, it outputs 0.

Formula: f(x) = max(0, x)

* Leaky ReLU:

What it is: A common variant of ReLU.

How it works: It's the same as ReLU, but instead of outputting 0 for negative numbers, it outputs a very small positive number (e.g., 0.01 * x). This helps prevent a problem called "dying ReLUs."

* Tanh (Hyperbolic Tangent):

What it is: A "classic" activation function.

How it works: It squashes all input values into a range between -1 and 1.

* Sigmoid (or Logistic):

What it is: Another "classic" function. It's now rarely used in hidden layers but is crucial for output layers.

How it works: It squashes all input values into a range between 0 and 1.

2. For Output Layers (Getting the Final Answer)
These are used on the very last layer to format the network's output into the answer you need.

* Softmax:

When to use: For multi-class classification (e.g., Cat vs. Dog vs. Bird).

What it does: It takes the final list of scores (logits) and converts them into probabilities that all add up to 1. For example, [1.7, -0.5, 3.0] might become [0.2, 0.0, 0.8].

* Sigmoid:

When to use: For binary classification (e.g., Spam vs. Not Spam).

What it does: It takes a single final score and converts it into a single probability between 0 and 1.

* None (Linear):

When to use: For regression (when you're predicting a number, like a house price or temperature).

What it does: You simply don't apply any activation function. The raw number from the last layer is your answer.



### Scores or Logits
These two terms, scores and logits, are often used interchangeably. They are the raw, final numerical outputs of the network before they are converted into probabilities.

What they represent: They are un-normalized values. A higher number means the model is more confident that the input belongs to that class.

Your Example: Your network ends with nn.Linear(5, 2). This means its final output is a tensor with 2 numbers, like [1.7, -0.5].

These two numbers are the logits.

They aren't probabilities (they don't add up to 1). They just show the model's "score" for each class.

Later, you would pass these logits to a Softmax function to turn them into probabilities (e.g., [0.90, 0.10]) or directly into a Cross-Entropy Loss function (which has Softmax built-in) to calculate the error.



### The forward Method (and its automatic call)
This is a key concept of how nn.Module works.

def forward(self, x):

This is the method where you define the path of your data. You are writing the "blueprint" for the assembly line, connecting the layers you defined in __init__.

"Called Automatically" (The __call__ method)

You never call model.forward(x) directly.

Instead, you just call the model object itself like a function: output = model(input_data).

When you do this, PyTorch's nn.Module base class automatically triggers its special __call__ method.

This __call__ method does some important background work (like registering hooks) and then it calls your forward method for you.

The Rule:

You define the logic inside def forward(self, x):.

You execute the logic by calling model(x)

### Common Neural Network Architectures

* Feedforward Neural Networks (FNN) / Multi-Layer Perceptrons (MLP)

The most basic type of neural network. Data moves in only one direction, from input to output, through hidden layers.

Use: Simple classification and regression tasks.

* Convolutional Neural Networks (CNN)

Designed to process data with a grid-like topology, such as images. They use "convolution" layers to automatically learn spatial hierarchies of features (e.g., edges, then shapes, then objects).

Use: Image recognition, video analysis, computer vision.

* Recurrent Neural Networks (RNN)

Designed for sequential data, like text or time series. They have loops that allow information to persist, giving them a form of "memory."

Use: Natural Language Processing (NLP), speech recognition, time series forecasting.

Variants:

Long Short-Term Memory (LSTM): A popular type of RNN that is better at learning long-term dependencies.

Gated Recurrent Unit (GRU): A simpler, more efficient variant of LSTM.

* Autoencoders (AE)

An unsupervised network that learns to compress data (encoding) and then reconstruct it (decoding).

Use: Dimensionality reduction, feature learning, and anomaly detection.

* Generative Adversarial Networks (GAN)

A system of two competing neural networks: a Generator (that creates fake data) and a Discriminator (that tries to tell fake from real). They train together until the Generator gets good at creating realistic data.

Use: Generating realistic images ("deepfakes"), art, and data augmentation.

* Transformer Networks

A more advanced architecture (based on an "attention" mechanism) that has largely replaced RNNs/LSTMs for NLP tasks. It is highly parallelizable and very effective at understanding context in sequential data.

Use: The basis for models like BERT, GPT, and modern machine translation.

* Graph Neural Networks (GNN)

Designed to work directly on data structured as a graph (nodes and edges).

Use: Social network analysis, recommendation systems, molecular chemistry.

---

### New & State-of-the-Art Models (Generative AI)

These models are "new" and define the cutting edge of AI. They are trained on massive datasets to generate new content.

1. Large Language Models (LLMs)

These models understand and generate human-like text. They are the power behind most modern chatbots and "agentic AI" systems.

GPT Series (OpenAI):

GPT-4o: The latest flagship model, known for its "omni-modal" capabilities—it can naturally understand and respond using text, audio, and vision all at once.

GPT-4: The highly capable and widely used predecessor.

Gemini Series (Google):

Gemini 2.5 Pro: Google's top-tier model, competing directly with GPT-4o in performance and multimodality.

Gemma 2: Google's family of powerful open-source models, built for developers and researchers.

Claude Series (Anthropic):

Claude 4.1 Opus: A top competitor to GPT-4, known for its large context window (for processing huge documents) and strong reasoning skills.

LLaMA Series (Meta):

LLaMA 3.1: The leading open-source model from Meta. Its release is a major driver of innovation, as it allows anyone to build on top-tier AI.

Mistral Series (Mistral AI):

A family of high-performing open-source models from a Paris-based startup, famous for their efficiency and power, even in smaller sizes.

2. Image Generation Models (Text-to-Image)

These models create detailed images from text descriptions.

DALL-E 3 (OpenAI): Tightly integrated with ChatGPT, known for its strong prompt-following and text-generation abilities within images.

Midjourney: Famous for creating highly artistic, stylized, and high-resolution images.

Stable Diffusion: The most popular open-source image model. It's highly customizable and has a massive community building tools for it.

Imagen (Google): Google's high-fidelity image generator, known for its photorealism and deep integration with Google's ecosystem.

3. Video & Audio Generation Models

This is a newer, rapidly emerging frontier.

Sora (OpenAI): A state-of-the-art text-to-video model that generates highly realistic and imaginative video clips up to a minute long.

Veo (Google): Google's primary competitor to Sora, designed to create high-definition, long-form video content.

Lyria (Google): A sophisticated AI model for generating music, capable of creating instrumental tracks and vocals in specific styles.

### Foundational "Classic" Models

These models are not "new," but they are arguably the most important in all of machine learning. They are the essential tools you learn first and are used in countless applications every day.

Linear Regression: Used for regression (predicting a number, like a house price).

Logistic Regression: Used for classification (predicting a category, like "spam" or "not spam").

Decision Trees / Random Forests: Versatile models used for both classification and regression, known for being easy to interpret.

Support Vector Machines (SVM): A powerful classification algorithm, highly effective at finding a clear boundary between groups.

K-Means Clustering: The most common unsupervised model, used to find hidden groups (clusters) in unlabeled data.

Principal Component Analysis (PCA): An unsupervised model used for dimensionality reduction (compressing data) by finding the most important features.



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

## Example:
```
import torch

t1= torch.linspace(1,5,5)

print(t1,t.dtype)
print("----------------------------------------") 
t= torch.empty(1,2)

print(t1,t.size())
print("----------------------------------------") 
t2= torch.ones(3,2,5)

print(t2,t.size())
print("----------------------------------------") 

t3= torch.tensor([[1,2,3],[4,5,6]])
a= torch.ones_like(t3)
print(t3,t3.size())
print(a,a.size())
print("----------------------------------------") 
n=5
t= torch.randperm(n)
t
```
output:
```
tensor([1., 2., 3., 4., 5.]) torch.int64
----------------------------------------
tensor([1., 2., 3., 4., 5.]) torch.Size([1, 2])
----------------------------------------
tensor([[[1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.]],

        [[1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.]],

        [[1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.]]]) torch.Size([1, 2])
----------------------------------------
tensor([[1, 2, 3],
        [4, 5, 6]]) torch.Size([2, 3])
tensor([[1, 1, 1],
        [1, 1, 1]]) torch.Size([2, 3])
----------------------------------------
tensor([0, 2, 1, 4, 3])
```
```
import torch

from matplotlib import pyplot as plt

t= torch.randn(2,100)

print(t.size())

plt.plot(t[0,:],t[1,:],'ob')

plt.show()
```
```
t= torch.randn(2,5)

print(t.size())

print(t)

ind= torch.randperm(t.size(1))

print(ind)

sel= t[:,ind[:2]]

print(sel)

```
output: 
torch.Size([2, 5])
tensor([[ 0.5783, -1.8835,  0.4670,  0.8298,  0.3463],
        [-0.8672,  0.1153,  1.0043,  0.7203,  1.7941]])
tensor([2, 1, 3, 0, 4])
tensor([[ 0.4670, -1.8835],
        [ 1.0043,  0.1153]])

## use flip 
```
# Original tensor
original_tensor = torch.tensor([4, -8, 5, 3, -2, 1])
print(f"Original Tensor: {original_tensor}\n")

# Flip the tensor along the only dimension (dimension 0)
flipped_tensor = torch.flip(original_tensor, dims=[0])
print(f"Flipped Tensor: {flipped_tensor}")
```
output:
Original Tensor: tensor([ 4, -8,  5,  3, -2,  1])

Flipped Tensor: tensor([ 1, -2,  3,  5, -8,  4])

---

# Comparison table mapping the specific PyTorch libraries you are using in your Fusion project to their equivalents in TensorFlow

1. Core Building Blocks (The "Lego Bricks")


|Situation |PyTorch |TensorFlow (Keras)|Description |
| ---- | ---- | ---- | -----|
The Main Library|import torch|import tensorflow as tf|The base library for math and tensors.
|Neural Network Layers|torch.nn|tf.keras.layers|"Contains the layers like Conv2d, ReLU, Linear."
2D Convolution|nn.Conv2d(...)|layers.Conv2D(...)|Used in your U-Net branch (Spatial).
1D Convolution|nn.Conv1d(...)|layers.Conv1D(...)|Used in your 1D-CNN branch (Spectral).
Fully Connected Layer|nn.Linear(...)|layers.Dense(...)|The final classification layers.
Activation Function|nn.ReLU() or F.relu()|layers.ReLU() or activation='relu'|Makes the network non-linear.

2. Data Handling (Loading Files)

|Situation|PyTorch |TensorFlow|Description|
| ---- | ---- | ---- | ----- |
|Data Container|torch.utils.data.Dataset|tf.data.Dataset|The class structure that holds your .npz files.
Batching & Shuffling|torch.utils.data.DataLoader|dataset.batch().shuffle()|"Takes single items and groups them into batches (e.g., 64 images)."
Image Augmentation|torchvision.transforms|tf.image or layers.Resizing|"Resizing, flipping, or normalizing images."
Tensors (Data format)|torch.Tensor|tf.Tensor|The multi-dimensional arrays (matrices) the GPU understands.|

3. Training & Optimization

|Situation|PyTorch |TensorFlow|
| ---- | ---- | ---- |
|Optimizer|torch.optim.Adam|tf.keras.optimizers.Adam|The algorithm that updates weights to reduce error.
|Loss Function|nn.CrossEntropyLoss|tf.keras.losses.CategoricalCrossentropy|"Calculates how ""wrong"" the prediction is."
|Learning Rate Decay|torch.optim.lr_scheduler|tf.keras.callbacks.LearningRateScheduler|Lowers learning rate as training improves (Plateau detection).
|Training Loop|Manual for loop |model.fit() (Built-in magic function)

4. The "Fusion" Specifics (Merging Networks)

|Situation|PyTorch|TensorFlow|Description
| ---- | ----- | ----- | -----|
|Concatenation|"torch.cat([x1, x2], dim=1)"|"tf.concat([x1, x2], axis=1)"|Merging the U-Net and 1D-CNN features together.
|Reshaping|x.view() or x.reshape()|"tf.reshape(x, ...)"|"Changing a tensor from (64, 64, 224) to (1, 224) etc."
|Moving to GPU|model.to('cuda')|Automatic|PyTorch requires manual .to('cuda'). TensorFlow finds the GPU automatically.

5. Deployment & Production

|Situation|PyTorch|TensorFlow|Description
| ---- | ---- | ---- | ---- |
|Saving Model|torch.save(model.state_dict())|model.save('path/to/model')|Saving your trained weights to disk.
|Serving API|TorchServe|TensorFlow Serving|The system that runs the model on a server.
|Mobile (Phone)|PyTorch Mobile / ExecuTorch|TensorFlow Lite|Running the model on Android/iOS.
|Web Browser|PyTorch Live (Limited)|TensorFlow.js|Running the model in Chrome/Edge.|

Summary: The "Keras" Confusion

You will often hear "Keras" when talking about TensorFlow.

Keras used to be a separate library.

Now, Keras IS the official high-level API for TensorFlow.

When you see tf.keras.layers, it is the TensorFlow equivalent of torch.nn.
