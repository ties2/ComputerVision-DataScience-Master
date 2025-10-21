# The mathematical functions 

used in machine learning particularly in neural networks, are primarily called Activation Functions. Here's a list of the most common ones, along with a brief explanation and their typical use:

📐 Activation Functions in Machine Learning
These functions introduce non-linearity into the network, allowing it to learn complex patterns.

|Function	|Formula	|Output Range	|Key Use & Characteristics
| ---- | ---- | ---- | ----|
| Sigmoid (or Logistic) | $$ \sigma(x) = \frac{1}{1 + e^{-x}} $$ | (0,1) | Used for binary classification (last layer) to output a probability. Prone to the vanishing gradient problem.|
| Hyperbolic Tangent (Tanh)| $$ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$ | (−1,1) | A zero-centered alternative to Sigmoid. Generally performs better than Sigmoid in hidden layers but still suffers from vanishing gradient. |
|Rectified Linear Unit (ReLU) | f(x)=max(0,x) | [0,∞) | The most popular choice for hidden layers today. Fast to compute and helps mitigate the vanishing gradient problem. |
| Leaky ReLU | $$ f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \le 0 \end{cases} $$ |(−∞,∞) | A variant of ReLU that solves the "dying ReLU" problem by allowing a small, non-zero gradient (α is a small constant like 0.01) for negative inputs. 
softmax | 
$$
 P(y=j|\mathbf{x}) = \frac{e^{\mathbf{x}_j}}{\sum_{k=1}^K e^{\mathbf{x}_k}} 
 $$  | (0,1) | Used exclusively in the output layer of a neural network for multi-class classification (where there are three or more possible outcomes). |