
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

Here is a comprehensive summary of **Chapter 2: Linear Algebra**, which establishes the fundamental language used to describe and manipulate data in machine learning.

### **Overview**
Linear algebra provides the formal language to handle high-dimensional data. In machine learning, data is represented as **vectors**, and operations on that data are represented as **matrices** and **linear mappings**. This chapter builds the algebraic structures necessary to solve systems of linear equations and understand the geometry of vector spaces.

---

### **1. Systems of Linear Equations & Matrices**
The central practical problem in linear algebra is solving systems of linear equations.
* **Representation:** A system of linear equations is compactly represented as $Ax = b$, where $A$ is a matrix of coefficients, $x$ is the vector of unknowns, and $b$ is the result vector.
* **Geometric Interpretation:** Each linear equation represents a line (in 2D) or a hyperplane (in higher dimensions). The solution to the system is the intersection of these geometric objects.
* **Matrices:** A matrix is a rectangular array of numbers. The chapter defines key operations:
    * **Matrix Multiplication:** This is **non-commutative**, meaning $AB \neq BA$.
    * **Inverse ($A^{-1}$):** A square matrix $A$ is invertible (regular) if $AB = I_n = BA$. If no inverse exists, the matrix is singular.
    * **Transpose ($A^\top$):** Created by swapping rows and columns. A matrix is **symmetric** if $A = A^\top$.



---

### **2. Solving Systems via Gaussian Elimination**
Gaussian elimination is the constructive algorithm used to solve systems of linear equations and find matrix inverses.
* **Elementary Transformations:** To solve a system, we manipulate the **augmented matrix** $[A|b]$ using three operations that do not change the solution set: swapping rows, multiplying a row by a non-zero scalar, and adding multiples of rows to one another.
* **Row-Echelon Form (REF):** The goal is to transform the matrix into a "staircase" structure where leading coefficients (pivots) move to the right as you go down the rows.
* **The Solution Structure:**
    * **Particular Solution:** A specific vector that solves the inhomogeneous system $Ax=b$.
    * **General Solution:** To capture *all* solutions, we add the particular solution to the solution of the homogeneous system $Ax=0$ (the kernel).
    * **The Minus-1 Trick:** A practical shorthand for reading the kernel (null space) directly from a matrix in reduced row-echelon form by extending the matrix with rows of $-1$ on the diagonal.

---

### **3. Vector Spaces and Independence**
This section formalizes the environment in which vectors "live."
* **Groups:** A set with an operation (like addition) that satisfies closure, associativity, has a neutral element, and inverse elements. An **Abelian group** is one where the operation is also commutative.
* **Vector Space:** A set $V$ is a vector space if it allows for vector addition (inner operation) and scalar multiplication (outer operation), satisfying specific distributivity and associativity rules.
* **Subspaces:** A subset $U \subseteq V$ is a subspace if it is non-empty (contains the zero vector) and is closed under addition and scalar multiplication. Intuitively, if you stay within a subspace and perform linear operations, you never leave that subspace.
* **Linear Independence:** A set of vectors is linearly independent if none of them can be written as a linear combination of the others. If $\sum \lambda_i x_i = 0$ implies that all $\lambda_i = 0$, the vectors are independent.



---

### **4. Basis and Rank**
These concepts quantify the "size" and structure of vector spaces.
* **Generating Set and Span:** A set of vectors is a generating set if their linear combinations can create (span) every vector in the space.
* **Basis:** A **minimal** generating set and a **maximal** linearly independent set. Every vector in the space has a unique coordinate representation with respect to a specific basis.
* **Dimension:** The number of vectors in a basis of a vector space.
* **Rank:** The number of linearly independent columns (or rows) in a matrix. A matrix has full rank if its rank equals the smaller of its row/column count. Key properties include:
    * $rk(A) = rk(A^\top)$.
    * A square matrix is invertible if and only if it has full rank.

---

### **5. Linear Mappings**
Mappings preserve the structure of the vector space.
* **Definition:** A mapping $\Phi: V \rightarrow W$ is linear if $\Phi(x+y) = \Phi(x) + \Phi(y)$ and $\Phi(\lambda x) = \lambda \Phi(x)$.
* **Matrix Representation:** Any linear mapping between finite-dimensional vector spaces can be represented by a matrix. The coefficients of this matrix depend on the chosen **ordered bases** of the input and output spaces.
* **Basis Change:** If we change the basis of the vector space (e.g., to a new coordinate system), the transformation matrix changes via the relationship $\tilde{A} = T^{-1}AS$, where $S$ and $T$ are the transformation matrices of the basis change.
* **Image and Kernel:**
    * **Kernel (Null Space):** The subspace of vectors mapped to 0 ($Ax=0$).
    * **Image (Range):** The subspace of vectors that can be "reached" by the mapping.
    * **Rank-Nullity Theorem:** For a matrix $A \in \mathbb{R}^{m \times n}$, the dimension of the kernel plus the dimension of the image equals $n$.



---

### **6. Affine Spaces**
* **Affine Subspace:** A linear subspace that is offset from the origin (shifted by a vector $x_0$). It typically represents lines or planes that do not pass through the origin.
* **Affine Mapping:** A composition of a linear mapping and a translation: $\phi(x) = Ax + a$.

---

# Chapter 3: Analytic Geometry 

### Chapter 3: Analytic Geometry

Chapter 3 bridges the gap between the algebraic manipulations of vectors (Linear Algebra) and geometric intuition. It establishes the mathematical frameworks required to quantify similarity, distance, and orientation between vectors, which are fundamental for machine learning algorithms like Support Vector Machines, Linear Regression, and Principal Component Analysis (PCA).

Here is a summary of the core concepts covered in the chapter:

**1. Norms (Measuring Length)**
The chapter begins by defining the **norm**, a function that assigns a "length" to a vector. While the Euclidean norm ($l_2$) is the most common (representing physical distance), the text also introduces the Manhattan norm ($l_1$). A valid norm must satisfy specific properties: it must be non-negative, homogeneous (scaling the vector scales the norm), and satisfy the triangle inequality.

**2. Inner Products (Measuring Angles and Similarity)**
The inner product is introduced as a more general concept than the standard dot product. It allows for the rigorous definition of geometric properties in vector spaces. Key takeaways include:
* **Angles:** The inner product allows the calculation of the angle between two vectors using the Cauchy-Schwarz inequality.
* **Similarity:** Intuitively, the inner product measures how aligned (similar) two vectors are.
* **Induced Norms:** An inner product naturally induces a norm, linking the concepts of length and angle.

**3. Orthogonality**
Using the inner product, the chapter defines **orthogonality**. Two vectors are orthogonal if their inner product is zero. This generalizes the concept of perpendicularity to high-dimensional spaces.
* **Orthonormal Basis (ONB):** A basis where all vectors are orthogonal to one another and have unit length. The **Gram-Schmidt process** is introduced as a method to iteratively construct an ONB from a set of linearly independent vectors.
* **Orthogonal Complement:** This describes the set of all vectors that are orthogonal to a specific subspace, used essentially to define hyperplanes (which separate data in classification tasks).

**4. Distances and Metrics**
Building on norms, the text defines **metrics**, which measure the distance between two vectors. While norms measure the length of a single vector, metrics measure the distance between two points ($x$ and $y$). The most common metric in machine learning is the Euclidean distance.

**5. Orthogonal Projections**
This is a critical section for later machine learning applications. An orthogonal projection finds the vector within a specific subspace that is "closest" to the original vector.
* **Projection onto Lines and Subspaces:** The text derives the formulas for projecting a vector onto a line (1D subspace) and general $M$-dimensional subspaces.
* **Projection Matrix:** A matrix $P_\pi$ can be constructed to perform this projection via linear transformation.
* **Application:** Minimizing the distance between a vector and a subspace is the mathematical foundation of **Linear Regression** (minimizing squares) and **PCA** (finding the subspace that retains the most variance).

**6. Rotations**
The chapter concludes by discussing rotations. Rotations are linear transformations that preserve both the lengths of vectors and the angles between them. They are defined by orthogonal matrices with a determinant of 1. The text details how to construct rotation matrices for 2D and 3D spaces (e.g., rotating around a specific axis).


---

# Chapter 4: Matrix Decompositions 

### Chapter 3: Analytic Geometry

This chapter establishes the geometric intuition required to understand vectors and matrices, moving beyond algebraic manipulation to defining similarity, distance, and orientation. These concepts are the bedrock of algorithms like Support Vector Machines (SVMs) and Principal Component Analysis (PCA).

**1. Norms (Measuring Length)**
The norm is a function that assigns a length to a vector.
* **Euclidean Norm ($l_2$):** The standard physical distance.
* **Manhattan Norm ($l_1$):** The sum of absolute differences.
* **Properties:** A valid norm must be non-negative, homogeneous (scaling the vector scales the norm), and satisfy the triangle inequality.

**2. Inner Products (Measuring Angles and Similarity)**
The inner product generalizes the dot product to define geometric relationships in vector spaces.
* **Angles:** Using the Cauchy-Schwarz inequality, the inner product allows the calculation of angles between vectors.
* **Geometric Interpretation:** It measures alignment; if the inner product is large, vectors are similar (pointing in the same direction).
* **Induced Norms:** Every inner product induces a norm, linking length and angle.

**3. Orthogonality**
Two vectors are orthogonal if their inner product is zero. This concept is crucial for separating signals and dimensions.
* **Orthonormal Basis (ONB):** A set of basis vectors that are all orthogonal to each other and have a unit length of 1.
* **Gram-Schmidt Process:** A method to iteratively construct an orthonormal basis from a set of linearly independent vectors.
* **Orthogonal Complement:** The set of vectors orthogonal to a specific subspace, used to define hyperplanes in high-dimensional space.

**4. Distances and Metrics**
While norms measure the length of a single vector, metrics measure the distance between two distinct vectors ($x$ and $y$). The Euclidean distance is the most common metric used in machine learning.

**5. Orthogonal Projections**
This section details how to project a vector onto a lower-dimensional subspace (e.g., projecting a 3D point onto a 2D plane) such that the "error" or distance is minimized.
* **Optimization:** The orthogonal projection finds the vector in the subspace closest to the original vector.
* **Applications:** This minimizes squared error, serving as the mathematical basis for **Linear Regression** and **PCA**.

**6. Rotations**
Rotations are linear transformations that preserve lengths and angles (distances between points remain constant). They are represented by orthogonal matrices with a determinant of 1.

---

### Chapter 4: Matrix Decompositions

This chapter focuses on breaking matrices down into interpretable constituent parts. These decompositions are primary tools for data compression, dimensionality reduction, and solving linear systems efficiently.

**1. Determinant and Trace**
These functions map square matrices to real numbers to characterize their properties.
* **Determinant:** Represents the signed volume of the region spanned by the matrix's column vectors. It indicates invertibility; if the determinant is non-zero, the matrix can be inverted.
* **Trace:** The sum of the diagonal elements. It is an invariant property, meaning $\text{tr}(ABC) = \text{tr}(BCA)$.

**2. Eigenvalues and Eigenvectors**
These characterize a linear mapping by identifying axes that do not rotate during transformation.
* **Concept:** An eigenvector $x$ is a vector that, when transformed by $A$, is only scaled by a scalar $\lambda$ (the eigenvalue): $Ax = \lambda x$.
* **Characteristic Polynomial:** Eigenvalues are found by finding the roots of $p_A(\lambda) = \det(A - \lambda I)$.
* **Interpretation:** The eigenvector indicates the direction of stretching or shrinking, while the eigenvalue indicates the magnitude of that stretch.

**3. Cholesky Decomposition**
A specific decomposition for symmetric, positive definite matrices (like covariance matrices).
* **The Factorization:** $A = LL^\top$, where $L$ is a lower-triangular matrix.
* **Utility:** It is roughly twice as efficient as other methods for solving linear systems involving symmetric matrices and is heavily used in optimization and simulation.

**4. Eigendecomposition and Diagonalization**
This decomposes a square matrix into a canonical form using its eigenvalues.
* **Diagonalization:** A matrix $A$ can be written as $A = PDP^{-1}$, where $D$ is a diagonal matrix of eigenvalues and $P$ is a matrix of eigenvectors.
* **Spectral Theorem:** Symmetric matrices can always be diagonalized, and their eigenvectors are orthogonal (form an orthonormal basis).

**5. Singular Value Decomposition (SVD)**
Often called the "fundamental theorem of linear algebra," SVD applies to **all** matrices, not just square ones.
* **The Factorization:** $A = U\Sigma V^\top$.
    * $U$: Orthogonal matrix (Left-singular vectors).
    * $\Sigma$: Diagonal matrix containing singular values (non-negative).
    * $V$: Orthogonal matrix (Right-singular vectors).
* **Geometric Meaning:** Every linear mapping consists of a rotation ($V^\top$), followed by a scaling ($\Sigma$), followed by another rotation ($U$).

**6. Matrix Approximation**
SVD allows for the "lossy compression" of matrices.
* **Low-Rank Approximation:** By keeping only the largest singular values in $\Sigma$ and setting the small ones to zero, one creates a lower-rank matrix that is the "best" approximation of the original matrix (minimizing the Frobenius norm difference). This is the math behind image compression and denoising.
* **Eckart-Young Theorem:** Provides the theoretical guarantee that the truncated SVD is the optimal low-rank approximation.

**7. Matrix Phylogeny**
The chapter closes by organizing matrices into a hierarchy based on properties (Symmetric, Orthogonal, PSD) and linking them to the specific decompositions available for each type.

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