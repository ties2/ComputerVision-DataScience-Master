
# Chapter 1: Introduction and Motivation 


The Core Goal of Machine Learning

Machine learning is defined as the design of algorithms that automatically extract valuable information from data. The ultimate goal is not just to fit the data you have, but to find a model that generalizes well to unseen data.


The Three Components of ML

To understand machine learning mathematically, you must distinguish between three core components:

* Data (Vectors):

Data is the core of ML. While data exists in many forms (images, text, audio), mathematically, it must be converted into a numerical format.

In this book, data is represented as vectors.

Practical Note: Whether you view a vector as an array of numbers (CS view) or an arrow with direction (Physics view), linear algebra provides the rules to manipulate them.

* The Model:

A model is a mathematical description of the process that generates the data.

Models are simplified versions of reality designed to capture relevant patterns.

Practical Note: A good model allows you to predict real-world outcomes without running real-world experiments.

* Learning (Optimization):

Learning is the process of adjusting the internal parameters of the model to improve performance.

This is mathematically analogous to climbing a hill to reach a peak (finding the maximum of a performance measure).

### Crucial Terminology: "Algorithm"

The chapter highlights a common source of confusion in ML terminology. The word "Algorithm" is often used to describe two very different things:

The Predictor: The system that takes an input and produces an output (inference).

The Training System: The method that adjusts the parameters of the predictor (learning).

The Four Pillars of Machine Learning

The book structures the mathematical foundations (Part I) to support four major machine learning problems (Part II).

* Regression: Predicting continuous values (e.g., salary based on age).

* Dimensionality Reduction: Compressing high-dimensional data into smaller, manageable representations (e.g., compressing images).

* Density Estimation: Finding the probability distribution that describes a dataset (e.g., clustering data).

* Classification: Predicting discrete labels (e.g., spam vs. non-spam)


---

# Chapter 2: Linear Algebra 


---

# Chapter 3: Analytic Geometry 


---

# Chapter 4: Matrix Decompositions 


---

# Chapter 5: Vector Calculus 

---

# Chapter 6: Probability and Distributions 

---

# Chapter 7: Continuous Optimization 


---

# Chapter 8: When Models Meet Data

This chapter formally connects the mathematical foundations from the first part of the book to the core problems in machine learning. It frames the learning process by establishing the relationships between data, models, and learning (parameter estimation).

1. Data, Models, and Learning

* Data: Data is assumed to be represented numerically as vectors (features). In supervised learning, this data consists of example-label pairs, e.g., $(x_n, y_n)$


* Models: The book presents two primary views of a model:

    * As a Function: A predictor f(x,θ) is a function (e.g., a linear function $f(x) = \theta^T x + \theta_0$)
that maps an input feature vector x to a prediction, guided by parameters θ.

    * As a Probability Distribution: A model can be a probabilistic distribution (e.g., p(y∣x,θ)) that quantifies uncertainty about the data or parameters.

* Learning: This is the process of finding the best parameters θ for a chosen model based on the training data. The chapter focuses on three main frameworks for learning.

---

2. Empirical Risk Minimization (ERM)

ERM is a non-probabilistic framework for learning. The goal is to find parameters θ for a predictor f(x,θ) that minimize the empirical risk.

* Loss Function $\ell(y, \hat{y})$ A function that measures the penalty for making a prediction y^ when the true label is y (e.g., squared loss 
$(y - \hat{y})^2$)

* Empirical Risk $R_emp$ : The average loss across the training set. This is what is minimized during training.

* Expected Risk R true: The "true" risk, or the average loss over all possible unseen data, $R_{\text{true}}(f) = \mathbb{E}_{x,y}[\ell(y, f(x))]$ This is what we want to minimize to ensure the model generalizes well.
* Regularization: To prevent overfitting (where the model fits the training data perfectly but fails on new data), a penalty term (regularizer) is added to the objective. This creates a trade-off between fitting the data and model simplicity (e.g., $\min_\theta R_{\text{emp}} + \lambda ||\theta||^2)$

* Cross-Validation: A method used to estimate the expected risk (generalization performance) by repeatedly splitting the data into training and validation sets.

3. Parameter Estimation (Probabilistic View)

This section reframes the learning problem using probability.

* Maximum Likelihood Estimation (MLE): This approach finds the parameters θ that maximize the likelihood p(Y∣X,θ)—the probability of observing the training data given the parameters. For i.i.d. data, this involves maximizing a product of probabilities, $p(Y|X, \theta) = \prod_{n=1}^N p(y_n | x_n, \theta)$, which is equivalent to minimizing the negative log-likelihood (a sum).

    * Connection to ERM: For a Gaussian likelihood $p(y|x, \theta) = \mathcal{N}(y | f(x, \theta), \sigma^2)$ , MLE is identical to ERM with a squared loss function.

* Maximum A Posteriori (MAP) Estimation: This method uses Bayes' theorem to find the most probable parameters after observing the data. It maximizes the posterior p(θ∣X,Y), which is proportional to the likelihood times a prior: p(θ∣X,Y)∝p(Y∣X,θ)p(θ).

    * Connection to Regularization: The prior p(θ) serves the same purpose as a regularizer in ERM. For example, a Gaussian prior on θ is equivalent to the $λ∣∣θ∣∣^2$ (L2) regularizer.

4. Probabilistic Modeling and Bayesian Inference

This framework extends the probabilistic view by treating parameters θ as random variables themselves, rather than just finding a single "best" (point) estimate.

* Bayesian Inference: The goal is not to find a single θ, but to compute the full posterior distribution p(θ∣X,Y) using Bayes' theorem.

* Prediction: Predictions for a new data point $x_new$ are made by marginalizing (integrating out) the parameters: $p(x_{\text{new}} | X, Y) = \int p(x_{\text{new}} | \theta) p(\theta | X, Y) d\theta$ . This provides a prediction that is averaged over all plausible parameter values, naturally incorporating uncertainty.

* Latent Variables: The chapter introduces models that use additional unobserved latent variables z (beyond parameters θ) to represent hidden structure or simplify the model, such as $p(x|\theta) = \int p(x|z, \theta) p(z) dz$ .
----

5. Directed Graphical Models (Bayesian Networks)

This section introduces a visual language for describing the structure of probabilistic models.

* Structure: Nodes represent random variables, and arrows represent conditional dependencies.

* Factorization: The graph defines how the joint distribution factorizes into a product of simpler, local conditional probabilities (each variable given its parents), e.g., $p(x_1, \dots, x_K) = \prod_{k=1}^K p(x_k | \text{Pa}(x_k))$

* Conditional Independence: The graph provides a simple way to read conditional independence relationships (e.g., A⊥⊥B∣C) between variables using rules known as d-separation.

---

6. Model Selection

This final section discusses how to choose between different models (e.g., polynomial degrees) or hyperparameters (e.g., the regularization strength λ).

* Nested Cross-Validation: A non-probabilistic method that uses an outer loop to estimate generalization error and an inner loop to select the best hyperparameters on a validation set.

* Bayesian Model Selection: A probabilistic method that compares models by computing their marginal likelihood (or evidence), $p(D|M) = \int p(D|\theta, M) p(\theta|M) d\theta$
This integral inherently penalizes overly complex models, acting as an "automatic Occam's razor". Models are compared using the Bayes factor, which is the ratio of their marginal likelihoods, $p(D|M_1) / p(D|M_2)$


Visualizing Overfitting vs. Underfitting (Concept from 8.3.3)

Below is a Python visualization illustrating the core concept of model fitting discussed in the chapter.

<p align="center">
  <img src="https://github.com/ties2/ComputerVision-DataScience-Master/blob/main/docs/summary/mathematics%20for%20ML/Code_Generated_Image-2.png" alt="Practical Methodology" width="800" />
</p>


---

# Chapter 9: Linear Regression

Key Concepts

* Problem Formulation: The goal of regression is to find a function f that maps inputs $x \in \mathbb{R}^D$ to real-valued outputs $y \in \mathbb{R}$.

* Probabilistic Model: The model assumes the observed target y is the function's output f(x) plus i.i.d. Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$

* Likelihood: This leads to a Gaussian likelihood function $p(y|x, \theta) = \mathcal{N}(y | f(x, \theta), \sigma^2)$ where θ are the model parameters.

* Linear Regression: This term means the model is linear in its parameters θ, not necessarily in its inputs x. The basic model is $f(x) = x^T\theta$ but this can be extended using basis functions (features) ϕ(x), resulting in $f(x) = \phi(x)^T\theta$ .

### Parameter Estimation: Point Estimates

* Maximum Likelihood Estimation (MLE): This approach finds the parameters $\theta_{ML}$ that maximize the likelihood of the training data.

* MLE as Least Squares: For a Gaussian likelihood, maximizing the log-likelihood is equivalent to minimizing the sum of squared errors.

    * Objective: $L(\theta) \propto ||y - \Phi\theta||^2$ , where Φ is the design matrix of features.

    * Solution: The MLE has a closed-form solution $\theta_{ML} = (\Phi^T\Phi)^{-1}\Phi^T y$

* Overfitting: MLE can overfit the data, especially with flexible models (e.g., high-degree polynomials). This results in low training error but high test error.

* Maximum A Posteriori (MAP) Estimation: This method introduces a prior distribution p(θ) to control overfitting by penalizing complex parameters.

* MAP as Regularized Least Squares: A Gaussian prior $p(\theta) = \mathcal{N}(0, b^2I)$ is equivalent to L2 regularization (also called regularized least squares).

    * Objective: $\min_\theta ||y - \Phi\theta||^2 + \lambda ||\theta||^2$
    * Solution: The MAP estimate is $\theta_{MAP} = (\Phi^T\Phi + \lambda I)^{-1}\Phi^T y$ This adds a term to the matrix, making it invertible and more numerically stable

### Bayesian Linear Regression

* Core Idea: Instead of finding a single "best" θ, this approach computes the full posterior distribution over the parameters, p(θ∣X,Y)

* Conjugate Prior: Using a Gaussian prior p(θ) (which is conjugate to the Gaussian likelihood) results in a posterior p(θ∣X,Y) that is also a Gaussian.

* Posterior Predictive Distribution: To make a prediction for a new input $x_*$, the parameters are marginalized (integrated out): $p(y_* | X, Y, x_*) = \int p(y_* | x_*, \theta) p(\theta | X, Y) d\theta$

* Uncertainty: The resulting prediction is a Gaussian whose variance accounts for both the observation noise $σ ^2$ and the parameter uncertainty (from the posterior covariance $S_N$ )This naturally captures model confidence.

* Marginal Likelihood: The model evidence p(Y∣X) can be computed in closed form. This is used for model selection (e.g., choosing the best polynomial degree) as it inherently penalizes "complex" models (an "automatic Occam's razor" from Chapter 8).

### Geometric Interpretation

* MLE as Orthogonal Projection: The maximum likelihood solution  
$\hat{y} = \Phi\theta_{ML}$ is geometrically equivalent to the orthogonal projection of the observed target vector y onto the subspace spanned by the columns of the feature matrix Φ.

* This means the "least squares" solution finds the vector y^ 
in the feature subspace that is closest (in Euclidean distance) to the true observations y

---

# Chapter 12: Classification with SVM

This chapter introduces the Support Vector Machine (SVM), a model used for **binary classification**. The goal is to take input data (represented as feature vectors) and predict a label from one of two discrete classes, such as $\{+1, -1\}$.
The SVM is a powerful classifier based on finding an optimal separating boundary, or hyperplane, between the two classes.

### 1. The Separating Hyperplane

The core idea of the SVM is to find a **linear separator**. For data in a $D$-dimensional space, this separator is a $(D-1)$-dimensional **hyperplane**.

* **Definition:** A hyperplane is defined by the function $f(x) = \langle w, x \rangle + b = 0$.
    * $w$ is a vector normal (orthogonal) to the hyperplane, defining its orientation.
    * $b$ is a scalar intercept (or bias) term that shifts the hyperplane off the origin.
* **Classification Rule:** A new data point $x_{\text{test}}$ is classified based on which side of the hyperplane it falls on. The prediction is $\text{sign}(f(x_{\text{test}}))$.
* **Correctness:** For a training point $x_n$ with label $y_n$, it is classified correctly if $y_n(\langle w, x_n \rangle + b) \ge 0$.

---

### 2. Finding the "Best" Hyperplane: The Margin

For data that is linearly separable, there are infinitely many hyperplanes that can perfectly separate the two classes. The SVM finds the single "best" one by selecting the hyperplane that maximizes the **margin**.
The margin is the distance from the separating hyperplane to the closest data point from either class. This "maximum margin" hyperplane is the one that is furthest from all training examples, which intuitively leads to better generalization.
This goal can be formulated as a convex optimization problem. The "traditional" and most common formulation (known as the **hard margin SVM**) is:

> **Minimize:** $\frac{1}{2}\|w\|^2$
> **Subject to:** $y_n(\langle w, x_n \rangle + b) \ge 1$ for all $n=1, \dots, N$

In this formulation, minimizing $\frac{1}{2}\|w\|^2$ is equivalent to maximizing the margin, which is $1/\|w\|$. The constraint ensures all points are correctly classified and are at least a "distance" of 1 from the hyperplane.

---

### 3. Handling Overlap: The Soft Margin SVM

The hard margin SVM fails if the data is not perfectly linearly separable. To handle real-world data that may have overlapping classes, the **soft margin SVM** is used.
This approach allows some examples to be misclassified or to fall inside the margin. It does this by introducing **slack variables** $\xi_n \ge 0$ for each data point.

* $\xi_n = 0$ if the point is correctly classified and outside the margin.
* $0 < \xi_n < 1$ if the point is correctly classified but *inside* the margin.
* $\xi_n \ge 1$ if the point is *misclassified*.

This changes the optimization problem to a trade-off: we want to maximize the margin (minimize $\|w\|^2$) while also minimizing the total amount of slack (minimize $\sum \xi_n$).

> **Minimize:** $\frac{1}{2}\|w\|^2 + C \sum_{n=1}^N \xi_n$


> **Subject to:** $y_n(\langle w, x_n \rangle + b) \ge 1 - \xi_n$ and $\xi_n \ge 0$

The term $\frac{1}{2}\|w\|^2$ is the **regularizer**, and $C \sum \xi_n$ is the **loss term**. The **regularization parameter** $C$ controls the trade-off:

* **Large $C$**: Penalizes slack heavily, leading to a narrower margin and fewer margin violations (similar to the hard margin SVM).
* **Small $C$**: Penalizes slack less, allowing for more margin violations in exchange for a wider, "simpler" margin.

This formulation is equivalent to an empirical risk minimization problem using the **hinge loss** function, $\ell(t) = \max\{0, 1 - t\}$.

---

### 4. The Dual SVM and Support Vectors

The problem above is the **primal SVM**. Its number of parameters grows with the number of **features** $D$. By using Lagrange multipliers (a constrained optimization technique), we can derive the **dual SVM**, where the number of parameters grows with the number of **data points** $N$.
This dual form is crucial because it reveals that the optimal $w$ is a linear combination of the **training examples**:
$$w = \sum_{n=1}^N \alpha_n y_n x_n$$

The $\alpha_n$ are the Lagrange multipliers found by the dual problem. Most $\alpha_n$ will be zero. The few data points $x_n$ for which $\alpha_n > 0$ are called **support vectors**. These are the only points that define the margin and the hyperplane; all other data points are irrelevant.

---

### 5. Nonlinear Classification: The Kernel Trick

The most powerful part of the SVM comes from its dual formulation. The dual problem only ever uses the data in the form of inner products, $\langle x_i, x_j \rangle$.
The **kernel trick** replaces this inner product with a **kernel function**, $k(x_i, x_j)$.
$$k(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$$

* $\phi(x)$ is an implicit, nonlinear feature map that projects the data into a much higher-dimensional (even infinite-dimensional) space.
* The SVM then finds a *linear* separator in this high-dimensional space.
* When mapped back to the original data space, this linear hyperplane becomes a complex, *nonlinear* decision boundary.

This allows SVMs to capture highly nonlinear patterns without ever explicitly computing the high-dimensional coordinates, making them incredibly flexible and efficient.

### Summary of Solution

Ultimately, solving an SVM (both primal and dual forms) is a **convex quadratic programming problem**, which is a well-understood class of optimization problem that can be solved efficiently.