
# Hyper Parameter Tuning 

## Introduction

In machine learning, it's important to distinguish between two types of "parameters":

  * **Parameters:** These are the internal variables of the model itself, which are learned during the training process. For example, the weights and biases in a neural network or the coefficients in a linear regression.

  * **Hyper-parameters:** These are the high-level configuration settings for the learning process and the model's architecture. They are set before training begins and are not learned from the data.

The main goal of **Hyper-parameter Optimization (HPO)** is to find the specific set of hyper-parameters that allows a model to achieve its maximum performance on a given dataset. This optimization process is separate from the model training itself.

-----

## Example with Polynomial Regression

A simple way to understand this difference is with polynomial regression.

Consider the task of fitting a polynomial curve to a set of data points.

  * **Parameters:** The coefficients of the polynomial (e.g., $a$, $b$, $c$ in the equation $y=a+bx^{1}+cx^{2}$) are the model parameters. They are found by the training (fitting) process.
  * **Hyper-parameter:** The **degree** of the polynomial (e.g., degree 1, 2, or 3 ) is the hyper-parameter. We have to choose this value *before* fitting the model.

choosing the wrong degree has consequences:

  * A low degree (e.g., 1) might **underfit** the data, failing to capture the underlying trend.
  * A very high degree (e.g., 20) might **overfit** the data, capturing noise and fitting the training points perfectly but failing to generalize to new, unseen data.


## Hyper-parameters in Neural Networks

Neural networks have many hyper-parameters that can be tuned. They generally fall into two categories:

1. **Model Complexity / Architecture**:

      * Number of layers
      * Numbers of neurons per layer
      * Neural network architecture (e.g., convolutional, recurrent) 

2.  **Training / Regularization**:

      * **Learning-rate**: Controls how much to change the model in response to the estimated error.
      * **Batch size**: The number of training examples used in one iteration.
      * **Momentum**: Helps accelerate gradient descent in the relevant direction.
      * **Weight decay**: A regularization technique to prevent overfitting.
      * **Early stopping**: Stops training when performance on a validation set stops improving.
      * **Weight clipping** 
      And many others.

-----

Manually tuning hyper-parameters often involves "reading the tea leaves" of training and validation loss graphs.

  * **Graph 1 :**

<p align="center">
  <img src="https://github.com/ties2/ComputerVision-DataScience-Master/blob/main/lectures/machine%20learning/Graph1.png" alt="Graph1" width="300" />
</p>


* What's happening? The validation loss is consistently much lower than the training loss.
* Observation: The validation set is likely unrepresentative or simply too easy to solve.

  * **Graph 2 (Page 11):**

<p align="center">
  <img src="https://github.com/ties2/ComputerVision-DataScience-Master/blob/main/lectures/machine%20learning/Graph2.png" alt="Graph2" width="300" />
</p>


* What's happening? Both the training and validation loss curves are extremely noisy and jumping around wildly.
* bservation: The training process is oscillating.
* What to change? This is a classic sign that the **learning rate is too high**. Lowering it would likely stabilize the training.

  * **Graph 3 (Page 12):**

<p align="center">
  <img src="https://github.com/ties2/ComputerVision-DataScience-Master/blob/main/lectures/machine%20learning/Graph3.png" alt="Graph3" width="300" />
</p>

* What's happening? The training loss (blue line) keeps decreasing, but the validation loss (orange line) has flattened out and started to rise.
* Observation: The model is **overfitting**. It's learning the training data too well but is losing its ability to generalize.
* What to change?This is a signal to stop training (i.e., use **early stopping** ). You could also introduce stronger regularization (like weight decay ) or increase model complexity if both losses are high (underfitting).

-----

## Methods for Hyper-parameter Optimization

There are several strategies for finding the best hyper-parameters:

  * **Manual Tuning**: Using experience and the "manual tuning quiz" approach to guide the search.
  * **Grid Search** : Define a grid of possible values for each hyper-parameter and train a model for *every single combination*. This is exhaustive but can be very computationally expensive.
  * **Random Search**: Define a range or distribution for each hyper-parameter and randomly sample combinations from that space. It is often more efficient than Grid Search.
  * **Bayesian Optimization**: An "optimization-based"  method that uses the results from previous runs to intelligently choose the next set of hyper-parameters to try. It aims to find the optimum in fewer steps.
  * **Other Optimization-Based Methods**:
      * Evolutionary algorithms (like Genetic Algorithms) 
      * Gradient-based methods 
      * Reinforcement learning 

### Finding Specific Hyper-parameters

  * **Finding Epochs (Early Stopping)**: Instead of tuning the number of epochs as a hyper-parameter, it's common practice to train for many epochs and use **Early Stopping**. This method monitors a validation metric (like `val_loss` or `val_acc` ) and stops training when that metric has not improved for a specified number of "patience" epochs.
  * **Finding Learning Rate (Automated Method)** : A popular technique is to plot the loss while exponentially increasing the learning rate (on a log scale).
      * The loss will typically plateau (rate is too low) , then descend (good rates) , and finally explode (rate is too high).
      * The optimal learning rate is usually found at the point of steepest descent, just before the loss starts to increase[cite: 344].

-----

## Best Practices and Key Takeaways [cite: 354]

  * Hyper-parameter tuning is **crucial** for getting good performance from neural networks.
  * **Start simple** and try tuning one parameter at a time.
  * Always use **cross-validation** to get a reliable estimate of your model's performance.
  * **Track your experiments\!** Use tools like TensorBoard to log your hyper-parameters and resulting metrics[cite: 357].
  * There are many methods, from simple grid search to advanced automated techniques.
  *  **systematic approach** is essential for success.
-----

## Exercise: SVC Grid Search [cite: 364]

This exercise involves finding the optimal hyper-parameters for a **Support Vector Classifier (SVC)**  using a **Grid Search**.

  * **Goal:** Find the best values for the hyper-parameters `'C'` and `'gamma'`.
  * **Model:** SVC with a `'Radial Basis Function'` kernel.
  * **Scoring:** The best combination should be determined by `accuracy`.
  * **Task:** Complete the Python function `find_hyper_parameters` on page 22.
  * **Reference:** For more understanding, see the scikit-learn example: `https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html`.

The images on page 21 visualize how the decision boundary of the SVC changes with different values of $C$ and $gamma$.
-----

## Genetic Algorithms [cite: 15, 421]

Genetic Algorithms (GAs) are a type of optimization-based method that can be used for hyper-parameter tuning.

  * **Concept:** A search heuristic inspired by Charles Darwin's theory of natural selection.
  * **Process:** It starts with an initial population of candidate solutions. The "best" (most fit) individuals are selected for reproduction (crossover) and mutation to create a new generation.
  * **Limitation:** GAs **do not guarantee** finding the global optimal solution. They can sometimes get stuck in a "local minimum".

### GA Terminology [cite: 462]

  * **Population:** The set of all candidate solutions (e.g., `Candidate Solution 1...m`).
  * **Chromosome (or Individual):** A single candidate solution (e.g., `Candidate Solution 2`).
  * **Gene:** A single component of a chromosome (e.g., `x2_1`).

### GA Steps

The process follows a loop:

1.  **Start** & **Generate Initial Population**: Create a set of random solutions.
2.  **Calculate Fitness**  Evaluate each solution (chromosome) to get a "fitness value" indicating how good it is.
3. **Selection**: Choose which individuals will reproduce.
      * **Elitism:** Automatically keep the top 'k' best individuals for the next generation.
      * **Roulette Wheel:** Give each individual a chance to be selected proportional to its fitness.
4.  **Crossover** : Mix the genes of two "parent" individuals to create "offspring" (a new solution).
      * **Ordered Crossover:** Swaps alternating segments between parents.
      * **Uniform Crossover:** Flips a coin for each gene to decide which parent it will come from .
5.  **Mutation** : Apply small, random changes to an individual's genes to introduce new genetic material[cite: 551].
      * **Swap:** Randomly swap two genes.
      * **Scramble:** Randomly shuffle the genes within a subset of the chromosome.
6.  **Termination**: Check if the stopping criteria are met.
      * **No:** Go back to the **Selection** step with the new generation.
      * **Yes:** **End** and return the best solution found.

### GA Termination (Convergence)

The algorithm stops when one of these conditions is met:

  * A minimum solution criterion is reached (e.g., 95% accuracy).
  * A maximum number of generations has passed.
  * A time/computation budget is exhausted, or a user manually stops it.

### Using GAs for HPO

Genetic algorithms can be applied to hyper-parameter optimization:

  * **Chromosome:** A set of hyper-parameters (e.g., `n_layers = 5`, `n_neurons = 256`, `learning_rate = 0.1`).
  * **Fitness:** The resulting score (e.g., accuracy) of the model trained with those hyper-parameters (e.g., `92%`).

### Example: Travelling Salesman Problem (TSP) 

GAs are famously used to solve problems like the TSP, which asks: "What's the shortest route to visit all locations and return?". As you add more stops, the problem becomes exponentially harder to solve perfectly.

  * **Chromosome:** A list of cities in a specific order (a route).
  * **Fitness:** The total distance of the route (lower is better).
  * **Exercise (Page 40):** Implement a **mutation** function that randomly swaps two cities in a route with a given `mutationrate`.
  * **Exercise (Page 43):** Compare a GA using **Elitism** (which keeps the best solutions) against one without. The results show that the Elitism-based solution found a much shorter path (distance 788.38) than the one without (distance 1035.72).