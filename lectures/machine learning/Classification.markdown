# Classification in Computer Vision

**Goal:** To assign an input vector  (e.g., an image or feature set) to one of  discrete classes.

## 1. The Linear Approach

The simplest way to classify data is to draw a line (or hyperplane) through it. These models are called **Generalized Linear Models**. They take a linear combination of inputs and pass them through a nonlinear **activation function**  to produce an output .

### A. Three Ways to Solve the Problem

There are three distinct strategies to find this linear boundary:

1. **Discriminant Functions (Direct Mapping):**
* **Concept:** Find a function that maps an input  directly to a class label (0 or 1).
* **Least Squares:** Fits a regression line to class labels. *Note: This is often poor for classification as it is sensitive to outliers.*
* **Fisher’s Linear Discriminant:** Reduces the data dimensionality. It tries to maximize the separation *between* classes while minimizing the spread *within* classes.
* **Perceptron:** An iterative algorithm that learns the boundary by minimizing error, updating only when it makes a mistake.


2. **Probabilistic Generative Models (Modeling the Data):**
* **Concept:** Model the "shape" of the data for each class () and the class probability (), then use Bayes' theorem to find the posterior probability.
* **Key Insight:** If you assume the data for each class follows a **Gaussian distribution** with a shared covariance matrix, the decision boundary is guaranteed to be linear.
* **Naive Bayes:** A simplified version assuming all features are independent.


3. **Probabilistic Discriminative Models (Modeling the Probability):**
* **Concept:** Model the posterior probability  directly.
* **Logistic Regression:** The standard for binary classification, using the **logistic sigmoid** function.
* **Training:** Solved via **Iterative Reweighted Least Squares (IRLS)** (since there is no closed-form solution).
* **Multiclass:** Uses the **Softmax function** instead of the sigmoid.



> **Bayesian Twist:** Exact Bayesian inference for Logistic Regression is hard. The **Laplace Approximation** is used to approximate the complex posterior distribution with a simpler Gaussian centered at the peak.

---

## 2. Kernel Methods (The "Memory" Approach)

Instead of learning fixed parameters , these methods use the training data points themselves to make predictions.

### A. The Kernel Trick

* **Dual Representation:** Linear models can be rewritten so that data points only appear as dot products.
* **The Trick:** You can replace the simple dot product with a **Kernel Function** . This allows you to operate in a high-dimensional (even infinite) feature space without ever calculating the coordinates.
* **Gram Matrix ():** An  matrix representing the similarity between every pair of training points.

### B. Gaussian Processes (GPs)

A GP is a probabilistic method that defines a distribution over *functions* rather than parameters.

* **Regression:** Predicts a target by averaging training targets, weighted by their kernel similarity to the new input. It outputs a **mean** (prediction) and **variance** (uncertainty).
* **Automatic Relevance Determination (ARD):** GPs can automatically learn which input features are important by adjusting length-scale parameters.

---

## 3. Support Vector Machines (SVM)

The SVM is a powerful binary classifier that focuses on the "gap" between classes.

### A. Maximizing the Margin

* **Concept:** The best separating hyperplane is the one that has the largest distance (**margin**) to the nearest data points.
* **Hard Margin:** Assumes classes are perfectly separable. It minimizes  to maximize the margin.

### B. Handling Real-World Data (Soft Margin)

* **Concept:** Real data overlaps. We introduce **slack variables** () to allow some points to be misclassified.
* **The Trade-off ():** You tune a parameter .
* **Large :** Strict. Penalizes mistakes heavily (leads to narrow margins).
* **Small :** Loose. Allows more mistakes (leads to wider margins).



### C. The "Support Vectors"

* The SVM solution depends *only* on a subset of data points called **Support Vectors**.
* These are the points that lie exactly on the margin (or are misclassified). All other correctly classified points away from the margin are irrelevant to the decision boundary.

### D. Nonlinear Classification

* SVMs use the **Kernel Trick** (mentioned above) to project data into higher dimensions.
* A **linear** boundary in that high-dimensional space corresponds to a complex, **nonlinear** boundary in the original image space.

---
