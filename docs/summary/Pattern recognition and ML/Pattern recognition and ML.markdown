# Chapter 1: Introduction - Summary

This chapter introduces the fundamental concepts of **pattern recognition** and **machine learning**, laying the groundwork for the rest of the book by emphasizing the importance of a **probabilistic approach**.

---

## 1. The Core Problem and Learning Types

**Pattern Recognition**  
The field is concerned with the automatic discovery of regularities in data using computer algorithms, which are then used to classify data into different categories or predict continuous values.

- **Supervised Learning**:  
  The model is trained using a training set consisting of input vectors **x** and their corresponding known target vectors **t**.  
  - **Classification**: Assign the input vector to one of a finite number of discrete categories (e.g., classifying a handwritten digit as 0–9).  
  - **Regression**: The desired output consists of one or more continuous variables (e.g., predicting the yield in a chemical process).  

- **Unsupervised Learning**:  
  The training data consists only of input vectors **x**, without corresponding target values.

---

## 2. Probability Theory: The Language of Uncertainty

- Provides the consistent mathematical framework for quantifying and manipulating uncertainty, which arises from:
  - Noise in measurements  
  - Finite size of data sets  

- When combined with **Decision Theory**, it enables making **optimal predictions** even when information is incomplete or ambiguous.

---

## 3. Key Concepts Illustrated: Polynomial Curve Fitting (Section 1.1)

The example of **fitting a polynomial curve** to data illustrates key concepts:

- **Overfitting**:  
  A model with too much flexibility (e.g., high-order polynomial) can fit the noise in the training data, leading to poor generalization on new data.

- **Regularization**:  
  Technique used to control model complexity and prevent overfitting.

---

## 4. The Curse of Dimensionality

Highlights problems in working with data in **high-dimensional spaces** (many input variables).

- **Geometric Breakdown**:  
  Our intuition from 3D fails in high dimensions.  
  - In high dimensions, most of the volume of a sphere is concentrated in a thin shell near the surface.  

- **Implication**:  
  The amount of data needed to cover the space grows **exponentially** with the number of dimensions, making modeling and density estimation much harder.

---

## 5. Decision Theory

When combined with probabilistic models, **Decision Theory** enables **optimal decision-making** in pattern recognition tasks.

- **Minimizing Expected Loss**:  
  Optimal decision = one that minimizes the expected loss (risk), defined using a user-specified loss function (or cost matrix).

- **The Reject Option**:  
  In classification, errors are likely when posterior probabilities of multiple classes are similar (uncertain).  
  - A threshold **θ** can be used to defer uncertain decisions to a human expert or another system, reducing errors in classified cases.

---

## 6. Information Theory

Introduced as a final perspective for viewing machine learning. Provides tools to **quantify uncertainty** in random variables.

- **Entropy**: Measures the uncertainty of a single variable.  
- **Mutual Information**: Reduction in uncertainty of one variable given another.

# Chapter 2: Probability Distribution

Here are the summaries for Chapter 3 and Chapter 4.

## 📖 Summary of Chapter 3: Linear Models for Regression

This chapter focuses on regression models that predict one or more continuous target variables $t$ based on an input vector $x$[cite: 2654]. The models are "linear models" because they are linear functions of their adjustable parameters (weights $w$), although they can be nonlinear with respect to the input variables $x$.

This nonlinearity is achieved by using a set of fixed, nonlinear functions of the input, known as **basis functions** $\phi(x)$[. The model is then a linear combination of these basis functions.

### Key Concepts:

* **Maximum Likelihood and Least Squares:** The chapter first explores finding the parameters $w$ by minimizing a **sum-of-squares error function**. This approach is shown to be equivalent to the method of **maximum likelihood** under the assumption that the target variable $t$ has a Gaussian distribution given $x$.
* **Over-fitting and Regularization:** A key problem with maximum likelihood is **over-fitting**, where the model fits the training data noise rather than the underlying trend. This can be controlled by adding a penalty term to the error function, known as **regularization**. A common choice is a quadratic regularizer (sum-of-squares of the weights), also known as **ridge regression** or **weight decay**.
* **The Bias-Variance Decomposition:** This is a frequentist concept for understanding model complexity[cite: 2665]. The expected error of a model is decomposed into the sum of $(\text{bias})^2 + \text{variance} + \text{noise}$[cite: 2666].
    * **Bias** measures how much the average model prediction (over all possible data sets) differs from the true underlying function. Simple models (like low-order polynomials) have high bias.
    * **Variance** measures how much the model's prediction changes in response to different training data sets. Complex models (like high-order polynomials) have high variance.
    * There is a **trade-off** between bias and variance; flexible models have low bias and high variance, while rigid models have high bias and low variance.
* **Bayesian Linear Regression:** The chapter presents a Bayesian approach to regression, which avoids the over-fitting problem of maximum likelihood.
    * A **prior distribution** (typically Gaussian) is introduced over the parameters $w$.
    * The posterior distribution $p(w|t)$ is then computed. Predictions are made by **marginalizing** (integrating) over this posterior distribution, rather than using a single point estimate of $w$.
    * The predictive distribution is a Gaussian, whose variance captures the uncertainty in the predictions.
    * The mean of the predictive distribution can be expressed in terms of an **"equivalent kernel"** or "smoother matrix," which shows that the prediction for a new input is a linear combination of the training set target values.
* **Bayesian Model Comparison:** The Bayesian framework provides a way to select the "best" model or set its complexity parameters (like the regularization coefficient) directly from the training data, without needing a separate validation set.
    * This is done by comparing models using the **model evidence** (also called marginal likelihood), which is the probability of the observed data given the model $p(D|\mathcal{M}_i)$.
    * The evidence framework naturally penalizes overly complex models and favors the model with the best balance of data fit and complexity.
    * A practical framework called the **"evidence approximation"** (or empirical Bayes) is presented, where the marginal likelihood is maximized to find optimal values for hyperparameters like $\alpha$ (the prior precision) and $\beta$ (the noise precision). This gives rise to the concept of the **"effective number of parameters"** ($\gamma$), which is a measure of how many parameters in the model are actually determined by the data.
* **Limitations:** The chapter concludes by noting the main limitation of these models: the **"curse of dimensionality."** Because the basis functions are fixed and not adaptive, the number of basis functions required often grows exponentially with the dimensionality $D$ of the input space.



# Chapter 4: Linear Models for Classification

This chapter applies the linear model framework to classification problems, where the goal is to assign an input vector $x$ to one of $K$ discrete classes. The models are linear in the parameters, and the decision boundaries they create are linear functions of the input $x$ (hyperplanes).

The model is a **generalized linear model**, where a linear combination of features $a = w^T\phi(x)$ is transformed by a nonlinear **activation function** $y = f(a)$ to produce the model output (e.g., a posterior probability).

The chapter explores three distinct approaches to classification.

### 1. Discriminant Functions
This approach finds a function that maps $x$ directly to a class label.
* **Methods:** The chapter discusses several discriminant functions:
    * **Least Squares for Classification:** This involves fitting a linear regression model to target vectors (e.g., 1-of-K encoding). However, this method has "severe problems," as it is not robust to outliers and can produce very poor decision boundaries.
    * **Fisher's Linear Discriminant:** This is a dimensionality reduction technique that finds a projection $w$ that maximizes the separation between classes (maximizing between-class variance while minimizing within-class variance).
    * **The Perceptron Algorithm:** An iterative algorithm that learns a separating hyperplane by minimizing the "perceptron criterion," an error function defined over misclassified training points.

### 2. Probabilistic Generative Models
This approach models the class-conditional densities $p(x|C_k)$ and the class priors $p(C_k)$, and then uses Bayes' theorem to find the posterior $p(C_k|x)$.
* **Key Result:** The chapter shows that if the class-conditional densities $p(x|C_k)$ are assumed to be **multivariate Gaussian distributions with a shared covariance matrix**, the resulting posterior probability $p(C_k|x)$ is a **logistic sigmoid** (for $K=2$ classes) or a **softmax function** (for $K \ge 2$ classes) of a linear function of $x$.
* If the covariances are not shared, the discriminant is quadratic.
* The **"naive Bayes"** model, which assumes features are conditionally independent given the class, also results in a linear classifier.

### 3. Probabilistic Discriminative Models
This approach models the posterior $p(C_k|x)$ directly, avoiding the need to model the class-conditional densities.
* **Logistic Regression:** This is the primary discriminative model for two-class classification. It models $p(C_1|x)$ directly using the **logistic sigmoid** function $\sigma(w^T\phi(x))$.
* **Training:** Because there is no closed-form solution, the model is trained by maximum likelihood, which is equivalent to minimizing a **cross-entropy error function**. This optimization is performed using an iterative algorithm called **iterative reweighted least squares (IRLS)**, which is based on the Newton-Raphson update.
* **Multiclass Logistic Regression:** For $K > 2$ classes, the logistic sigmoid is replaced by the **softmax function**, and the corresponding cross-entropy error is minimized, again using IRLS.
* **Probit Regression:** An alternative model that uses the **probit function** (the cumulative distribution function of a Gaussian) as its activation function instead of the logistic sigmoid.

### Bayesian Approach
* **The Laplace Approximation:** Exact Bayesian inference for logistic regression is intractable. The chapter introduces the **Laplace approximation**, a general framework for approximating a probability distribution with a Gaussian centered at its mode (the maximum a posteriori, or MAP, solution).
* **Bayesian Logistic Regression:** The Laplace approximation is applied to create a Bayesian logistic regression model. This provides a Gaussian approximation to the posterior distribution $p(w|t)$, which can then be used to make probabilistic predictions.

---

# Chapter 6: Kernel Methods

Here is a more complete summary of Chapter 6: Kernel Methods, including the key mathematical formulations in LaTeX.

[cite_start]This chapter introduces a new class of "memory-based" models that use the training data points directly to make predictions[cite: 1136]. [cite_start]The central idea is to reformulate linear models (from Chapters 3 and 4) into an equivalent **dual representation** where predictions are based on combinations of **kernel functions** evaluated at the training data points[cite: 1137].

---

## 6.1 Dual Representations

The chapter starts by showing how a linear model, like regularized least squares, can be re-cast. The original (primal) problem is to find the $M$-dimensional parameter vector $\mathbf{w}$ by minimizing:
[cite_start]$$J(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^{N} \{\mathbf{w}^T\phi(\mathbf{x}_n) - t_n\}^2 + \frac{\lambda}{2}\mathbf{w}^T\mathbf{w}$$ [cite: 1138]

The solution for $\mathbf{w}$ can be shown to be a linear combination of the feature vectors $\phi(\mathbf{x}_n)$ from the training data:
[cite_start]$$\mathbf{w} = \sum_{n=1}^{N} a_n \phi(\mathbf{x}_n) = \mathbf{\Phi}^T \mathbf{a}$$ [cite: 1138]

[cite_start]By substituting this back into $J(\mathbf{w})$, we get a *dual representation* of the problem, which involves maximizing an objective $J(\mathbf{a})$ with respect to the $N$-dimensional vector $\mathbf{a}$[cite: 1138]. The final solution for $\mathbf{a}$ is:
[cite_start]$$\mathbf{a} = (\mathbf{K} + \lambda \mathbf{I}_N)^{-1} \mathbf{t}$$ [cite: 1139]

Here, $\mathbf{K}$ is the $N \times N$ **Gram matrix**, which is the core of the kernel method. Its elements are computed from the kernel function $k(\mathbf{x}, \mathbf{x}')$:
[cite_start]$$K_{nm} = k(\mathbf{x}_n, \mathbf{x}_m) = \phi(\mathbf{x}_n)^T \phi(\mathbf{x}_m)$$ [cite: 1139]

The prediction for a new input $\mathbf{x}$ is then given by:
[cite_start]$$y(\mathbf{x}) = \mathbf{w}^T\phi(\mathbf{x}) = \mathbf{a}^T\mathbf{\Phi}\phi(\mathbf{x}) = \mathbf{k}(\mathbf{x})^T (\mathbf{K} + \lambda \mathbf{I}_N)^{-1} \mathbf{t}$$ [cite: 1139]
[cite_start]where $\mathbf{k}(\mathbf{x})$ is a vector with elements $k(\mathbf{x}_n, \mathbf{x})$[cite: 1139].

This is the **kernel trick**: the algorithm is now formulated entirely in terms of the kernel function $k(\mathbf{x}, \mathbf{x}')$. [cite_start]We never need to explicitly know the feature mapping $\phi(\mathbf{x})$, which could be infinite-dimensional[cite: 1139].

---

## 6.2 Constructing Kernels

For a function to be a *valid* kernel, it must correspond to a dot product in some feature space. [cite_start]The necessary and sufficient condition for this is that the **Gram matrix** $\mathbf{K}$ must be **positive semidefinite** for any set of data points $\{\mathbf{x}_n\}$[cite: 1140].

Valid kernels can be created by combining simpler ones:
* [cite_start]$k(\mathbf{x}, \mathbf{x}') = k_1(\mathbf{x}, \mathbf{x}') + k_2(\mathbf{x}, \mathbf{x}')$ [cite: 1141]
* [cite_start]$k(\mathbf{x}, \mathbf{x}') = k_1(\mathbf{x}, \mathbf{x}') k_2(\mathbf{x}, \mathbf{x}')$ [cite: 1141]
* [cite_start]$k(\mathbf{x}, \mathbf{x}') = q(k_1(\mathbf{x}, \mathbf{x}'))$ (where $q$ is a polynomial with non-negative coefficients) [cite: 1141]
* [cite_start]$k(\mathbf{x}, \mathbf{x}') = \exp(k_1(\mathbf{x}, \mathbf{x}'))$ [cite: 1141]

A common example is the 'Gaussian' kernel (which corresponds to an infinite-dimensional feature space):
[cite_start]$$k(\mathbf{x}, \mathbf{x}') = \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\sigma^2}\right)$$ [cite: 1142]

The chapter also introduces **generative kernels**, which can be built from generative models. [cite_start]A key example is the **Fisher kernel**, which uses the gradient of the log likelihood of a generative model to define the feature space[cite: 1143].

---

## 6.3 Radial Basis Function (RBF) Networks

[cite_start]This section connects kernels to **Radial Basis Function (RBF) networks**, which are linear models that use basis functions $\phi_j(\mathbf{x}) = h(\|\mathbf{x} - \mathbf{\mu}_j\|)$ that depend only on the radial distance from a set of "centers" $\mathbf{\mu}_j$[cite: 1145].

A common RBF model is the **Nadaraya-Watson model** (or kernel regression). [cite_start]This is a non-parametric method that centers a kernel function (like a Gaussian) on *every* training data point[cite: 1147]. [cite_start]The prediction is a weighted average of the training targets $t_n$, where the weight for each point is its kernel similarity to the new input $\mathbf{x}$[cite: 1147].

---

## 6.4 Gaussian Processes

Gaussian Processes (GPs) are the probabilistic, non-parametric heart of the chapter. [cite_start]A GP is a powerful Bayesian approach that avoids defining a parametric function $\mathbf{w}$ and instead defines a prior probability distribution *directly over the space of functions*[cite: 1148].

* [cite_start]**Definition:** A Gaussian Process is a collection of random variables, any finite number of which have a joint Gaussian distribution[cite: 1149].
* [cite_start]**Specification:** A GP is fully specified by a mean function $m(\mathbf{x})$ (often assumed to be zero) and a covariance function $k(\mathbf{x}_n, \mathbf{x}_m)$, which is the kernel[cite: 1149].
    [cite_start]$$E[y(\mathbf{x}_n)y(\mathbf{x}_m)] = k(\mathbf{x}_n, \mathbf{x}_m)$$ [cite: 1149]

###  GPs for Regression

[cite_start]For regression, we assume the observed targets $t_n$ are the function values $y(\mathbf{x}_n)$ plus some Gaussian noise $\epsilon_n \sim N(0, \beta^{-1})$[cite: 1150].

* [cite_start]**Joint Distribution:** The joint distribution of the training targets $\mathbf{t}_N$ and the target $t_{N+1}$ for a new test point $\mathbf{x}_{N+1}$ is a Gaussian $p(\mathbf{t}_{N+1}) = N(\mathbf{t}_{N+1} | \mathbf{0}, \mathbf{C}_{N+1})$[cite: 1152].
* **Covariance Matrix:** The covariance matrix $\mathbf{C}$ is built from the kernel $k$ and the noise $\beta^{-1}$:
    [cite_start]$$C(\mathbf{x}_n, \mathbf{x}_m) = k(\mathbf{x}_n, \mathbf{x}_m) + \beta^{-1}\delta_{nm}$$ [cite: 1152]
* [cite_start]**Predictive Distribution:** Because this is a joint Gaussian, the predictive distribution $p(t_{N+1} | \mathbf{t}_N)$ is also Gaussian[cite: 1152]. Its mean and variance are:
    * [cite_start]**Mean:** $m(\mathbf{x}_{N+1}) = \mathbf{k}^T \mathbf{C}_N^{-1} \mathbf{t}$ [cite: 1153]
    * [cite_start]**Variance:** $\sigma^2(\mathbf{x}_{N+1}) = c - \mathbf{k}^T \mathbf{C}_N^{-1} \mathbf{k}$ [cite: 1153]
    (where $\mathbf{k}$ is the vector of kernel similarities between the test point and training points, and $c = k(\mathbf{x}_{N+1}, \mathbf{x}_{N+1}) [cite_start]+ \beta^{-1}$)[cite: 1152, 1153].
    This explicitly gives a mean prediction (a weighted sum of training targets) and a measure of uncertainty for that prediction.

### Learning Hyperparameters

The kernel function $k$ depends on **hyperparameters** $\mathbf{\theta}$ (e.g., the width $\sigma^2$ of a Gaussian kernel, or the noise $\beta$). [cite_start]These are learned not by cross-validation, but by maximizing the **marginal likelihood** $p(\mathbf{t} | \mathbf{\theta})$[cite: 1156]. The log marginal likelihood is given by:
$$\ln p(\mathbf{t}|\mathbf{\theta}) = -\frac{1}{2}\ln |\mathbf{C}_N| - [cite_start]\frac{1}{2}\mathbf{t}^T\mathbf{C}_N^{-1}\mathbf{t} - \frac{N}{2}\ln(2\pi)$$ [cite: 1157]
[cite_start]This can be maximized using gradient-based optimization[cite: 1157].

### Automatic Relevance Determination (ARD)

[cite_start]ARD is a powerful feature of GPs where a separate length-scale parameter $\eta_i$ is used for each input dimension[cite: 1158]. For example:
[cite_start]$$k(\mathbf{x}, \mathbf{x}') = \theta_0 \exp\left(-\frac{1}{2}\sum_i \eta_i(x_i - x'_i)^2\right)$$ [cite: 1158]
[cite_start]When the marginal likelihood is maximized, if an input $\mathbf{x}_i$ is irrelevant to predicting the target, its corresponding $\eta_i$ will be driven to a very small value, effectively pruning that input from the model[cite: 1158].

###  GPs for Classification

* [cite_start]For classification, the GP output $a(\mathbf{x})$ is passed through a **logistic sigmoid function** $y = \sigma(a(\mathbf{x}))$ to produce a probability $p(t=1|\mathbf{x}) = y$[cite: 1159].
* [cite_start]This makes the model **analytically intractable** because the resulting distribution is not Gaussian[cite: 1159].
* [cite_start]The chapter uses the **Laplace approximation** (from Chapter 4) to find a Gaussian approximation to the posterior distribution, allowing for approximate Bayesian inference[cite: 1161, 1162].

[cite_start]Finally, the chapter notes a deep connection: a Bayesian neural network (from Chapter 5) with a prior over its parameters becomes a Gaussian process in the limit of an infinite number of hidden units[cite: 1164].


