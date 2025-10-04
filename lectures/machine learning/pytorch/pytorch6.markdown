# PyTorch Learning Notes part 6

## Neuron: Human vs. Machine

A Neural Network (NN) is a computational system inspired by the human brain, designed to recognize patterns and solve complex problems like classification and prediction.


| Feature	| Biological Neuron (Human Brain)	|Artificial Neuron (ML/AI) |
| ----- | ----- | ----
| Complexity |	Extremely High. A living cell with thousands of internal chemical, electrical, and genetic processes.	| Very Low. A simple mathematical function (a node in a graph).
| Input/Output |	Receives chemical signals via dendrites; fires an electrical signal (action potential) through the axon.	|Receives numerical inputs; outputs a single calculated number.
| Mechanism	| Uses electrochemical reactions to accumulate inputs. When a threshold is met, it "fires" or activates.	|Calculates a weighted sum of its inputs, adds a bias, and passes the result through an activation function.
| Learning	|Synaptic strength changes through complex biological processes like long-term potentiation.|	Synaptic weights (w) are adjusted purely by mathematical optimization (Backpropagation/Gradient Descent).
| Speed	| Slow (milliseconds)	|Extremely Fast (nanoseconds, processed in parallel by GPUs)

### The Concept
At its core, a neural network is a system of simple, interconnected units called neurons (or nodes) arranged in layers.

1. Layers: The network is organized into three main types of layers:

* Input Layer: Receives the raw data (features).

* Hidden Layers: Perform the heavy computational lifting by extracting complex patterns and features from the data.

* Output Layer: Provides the final result (e.g., a predicted class or a numerical value).

2. Connections (Weights): Each connection between neurons has an associated number called a weight. This weight determines the influence one neuron has on the next. These weights are the parameters the network learns during training.

3. Activation: Each neuron takes the weighted sum of its inputs, adds a bias, and then passes the result through an activation function (a mathematical function) to decide whether and how strongly to "fire" or activate.

### How it Learns (Training)

The network learns through a process called Backpropagation.

1. Forward Pass: Data is fed from the input layer, through the hidden layers, to the output layer to make a prediction.

2. Loss Calculation: The difference between the network's prediction and the true answer is measured as the loss (error).

3. Weight Adjustment: Backpropagation uses the loss to calculate how much each individual weight contributed to the error. The network then slightly adjusts these weights to reduce the error in the next pass, slowly improving its ability to make accurate predictions

 ## activation function
 
 activation function is the non-linear "switch" that gives a neural network its power. Without it, the network would just be performing linear regression.

Here are the details on several common activation functions:

| Function | Type | Formula - Range | Use Case & Feature |
| ----- | ----- | ----- | -----|
| Threshold / Step	| Discontinued	| Output is 0 or 1.	| Obsolete. Used in early Perceptrons. Cannot be used for modern training because its derivative is zero everywhere (it cannot learn).|
| Sigmoid	| S-shaped, Smooth	| Range: (0,1)	| Historically popular for binary classification (since the output looks like a probability). Suffer from the vanishing gradient problem when inputs are very large or very small.|
| Tanh (Hyperbolic Tangent)	| S-shaped, Smooth	| Range: (−1,1)	| Similar to Sigmoid but centered at zero. Often performs better than Sigmoid because centering the data around zero helps the optimization process.
| ReLU (Rectified Linear Unit)	| Piecewise Linear	| Range: [0,∞)	| Most Popular Default. Simple and highly efficient. It avoids the vanishing gradient problem for positive inputs, leading to faster convergence in deep networks.|
| Leaky ReLU	| Variation of ReLU	| Range: (−∞,∞)	| Addresses the "dying ReLU" problem by allowing a small, non-zero gradient (e.g., 0.01x) for negative inputs.
| Softmax	| Probabilistic	| Output sums to 1.0	| Used exclusively in the output layer of a network solving multi-class classification problems. It converts the raw scores into a probability distribution.|

### Short Explanations

* Threshold (Step Function): Acts as a simple on/off switch. If the input exceeds a certain value, the output is 1; otherwise, it's 0. It's too rigid for modern backpropagation, so it's rarely used.

* Sigmoid and Tanh: These are smooth, non-linear functions. They introduce the necessary complexity but struggle when the input is extreme (very positive or very negative). In these regions, the gradient (slope) becomes near zero, causing the learning process to stall (the vanishing gradient issue).

* ReLU and Leaky ReLU: These are the standard choices today. Their simple linear structure for positive values allows the gradient to remain strong, speeding up training. Leaky ReLU fixes the problem where neurons can become permanently "dead" (always outputting zero) by giving them a tiny slope for negative inputs.

* Softmax: Unlike the others (which are applied to hidden layers), Softmax is only used on the final output layer. It ensures that the model's predictions for all possible classes add up to 1, effectively giving you a clean probability distribution.

### Python implementations for those key activation functions

```
import numpy as np
import math

# --- 1. Threshold (Step) Function ---
def step_function(x, threshold=0.0):
    """
    Implements the Threshold or Step function.
    Output is 1 if input (x) exceeds the threshold, 0 otherwise.
    This function is non-differentiable at the threshold, making it unsuitable for backpropagation.
    """
    return np.where(x >= threshold, 1, 0)

# --- 2. Sigmoid Function ---
def sigmoid(x):
    """
    Implements the Sigmoid activation function.
    Range: (0, 1). Useful for binary classification output (probabilities).
    """
    # Using np.exp for element-wise exponentiation in the array
    return 1 / (1 + np.exp(-x))

# --- 3. Tanh (Hyperbolic Tangent) Function ---
def tanh_function(x):
    """
    Implements the Tanh activation function.
    Range: (-1, 1). Zero-centered, often performs better than Sigmoid in hidden layers.
    """
    return np.tanh(x)

# --- 4. ReLU (Rectified Linear Unit) Function ---
def relu(x):
    """
    Implements the ReLU activation function.
    Output is max(0, x). Most commonly used default activation in deep learning.
    """
    return np.maximum(0, x)

# --- 5. Leaky ReLU Function ---
def leaky_relu(x, alpha=0.01):
    """
    Implements the Leaky ReLU function.
    Returns x for x > 0, and (alpha * x) for x <= 0, preventing 'dying neurons'.
    """
    return np.where(x > 0, x, alpha * x)

# --- 6. Softmax Function ---
def softmax(x):
    """
    Implements the Softmax function.
    Used in the output layer for multi-class classification. Converts scores (logits) 
    into a probability distribution that sums to 1.
    """
    # Subtracting the maximum input for numerical stability (common practice)
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0) # Sum across the entire array for 1D input

# --- Demonstration ---

# Define a sample input array representing the weighted sum of inputs (or 'logits')
input_data = np.array([-2.0, 0.0, 3.0])

print(f"--- Input Data: {input_data} ---")

print(f"1. Step Function (Threshold=0): {step_function(input_data)}")
print(f"2. Sigmoid (Range 0 to 1):      {sigmoid(input_data)}")
print(f"3. Tanh (Range -1 to 1):        {tanh_function(input_data)}")
print(f"4. ReLU (max(0, x)):            {relu(input_data)}")
print(f"5. Leaky ReLU (alpha=0.01):     {leaky_relu(input_data)}")
print(f"6. Softmax (Probabilities sum to 1): {softmax(input_data)}")
print(f"   (Sum of Softmax outputs: {np.sum(softmax(input_data)):.4f})")
```
output:
```
--- Input Data: [-2.  0.  3.] ---
1. Step Function (Threshold=0): [0 1 1]
2. Sigmoid (Range 0 to 1):      [0.11920292 0.5        0.95257413]
3. Tanh (Range -1 to 1):        [-0.96402758  0.          0.99505475]
4. ReLU (max(0, x)):            [0. 0. 3.]
5. Leaky ReLU (alpha=0.01):     [-0.02  0.    3.  ]
6. Softmax (Probabilities sum to 1): [0.00637746 0.04712342 0.94649912]
   (Sum of Softmax outputs: 1.0000)
```
Analysis of Activation Function Outputs
The input array represents three weighted sums received by three different neurons: a strong negative input (−2.0), a neutral input (0.0), and a strong positive input (3.0).

|Input	|Step/Threshold	|Sigmoid	|Tanh	|ReLU	|Leaky ReLU	|Softmax |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
−2.0 (Strong Negative)	|0	|0.12	|−0.96	|0.0	|−0.02	|0.006 |
|0.0 (Neutral)	|1	|0.50	|0.0	|0.0	|0.0	|0.047|
|3.0 (Strong Positive)	|1	|0.95	|0.995	|3.0	|3.0	|0.946


1. The On/Off Switch: Threshold & ReLU

* Step Function: This is the most basic switch. The negative input is turned Off (0), while the 0.0 and 3.0 inputs are both turned completely On (1). It loses all sense of magnitude.

* ReLU (Rectified Linear Unit): This is the modern digital switch. Any negative input is killed (set to 0.0). The 0.0 input is 0.0. Critically, the strong positive input of 3.0 is passed through unchanged (3.0), preserving its magnitude and maintaining a strong gradient for learning.

2. The Squashers: Sigmoid & Tanh

These functions are called "squashers" because they compress the input into a limited range:

* Sigmoid: It squashes all inputs into the (0,1) range.

    * The strong negative input (−2.0) is squashed close to 0 (0.12).

    * The strong positive input (3.0) is squashed close to 1 (0.95).

    * The 0.0 input always lands exactly at 0.50.

* Tanh (Hyperbolic Tangent): It squashes all inputs into the (−1,1) range.

    * It treats the 0.0 input perfectly, mapping it to 0.0.

    * The strong negative input is mapped close to −1 (−0.96), and the positive input close to 1 (0.995). Tanh is generally preferred over Sigmoid because its output is centered around zero.

3. The Specialist: Softmax

* Softmax: This function converts the raw inputs (logits) into a probability distribution that sums to 1.0.

    * The strong positive value (3.0) is assigned the highest probability (0.946).

    * The negative and neutral values are assigned very low probabilities, essentially indicating the network is highly confident that the true class is the one corresponding to the 3.0 input.

4. The Fixer: Leaky ReLU

* Leaky ReLU: This is used to fix ReLU's problem with negative inputs (the "dying neuron" problem).

    * Instead of setting −2.0 to 0.0 (like ReLU), it assigns a tiny, non-zero value (−0.02) to ensure that the neuron still has a small gradient and can potentially learn again later.

Example:

```
x= torch.arange(-5,5,0.1)

y1= torch.sign(x)

plt.plot(x,y1,'r.',linewidth=2)

plt.grid('minor')

plt.xlabel('v',fontsize=15)

plt.ylabel('y',fontsize=15)

plt.show()
```