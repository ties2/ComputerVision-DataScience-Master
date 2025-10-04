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