
# Chapter 1: Introduction and Motivation 

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

Data: Data is assumed to be represented numerically as vectors (features). In supervised learning, this data consists of example-label pairs, e.g., (x 
n
​	
 ,y 
n
​	
 ).

Models: The book presents two primary views of a model:

As a Function: A predictor f(x,θ) is a function (e.g., a linear function f(x)=θ 
T
 x+θ 
0
​	
 ) that maps an input feature vector x to a prediction, guided by parameters θ.

As a Probability Distribution: A model can be a probabilistic distribution (e.g., p(y∣x,θ)) that quantifies uncertainty about the data or parameters.

Learning: This is the process of finding the best parameters θ for a chosen model based on the training data. The chapter focuses on three main frameworks for learning.

2. Empirical Risk Minimization (ERM)

ERM is a non-probabilistic framework for learning. The goal is to find parameters θ for a predictor f(x,θ) that minimize the empirical risk.


Loss Function ℓ(y, 
y
^
​	
 ): A function that measures the penalty for making a prediction  
y
^
​	
  when the true label is y (e.g., squared loss (y− 
y
^
​	
 ) 
2
 ).

Empirical Risk R 
emp
​	
 : The average loss across the training set. This is what is minimized during training.

Expected Risk R 
true
​	
 : The "true" risk, or the average loss over all possible unseen data, R 
true
​	
 (f)=E 
x,y
​	
 [ℓ(y,f(x))]. This is what we want to minimize to ensure the model generalizes well.

Regularization: To prevent overfitting (where the model fits the training data perfectly but fails on new data), a penalty term (regularizer) is added to the objective. This creates a trade-off between fitting the data and model simplicity (e.g., min 
θ
​	
 R 
emp
​	
 +λ∣∣θ∣∣ 
2
 ).


Cross-Validation: A method used to estimate the expected risk (generalization performance) by repeatedly splitting the data into training and validation sets.

3. Parameter Estimation (Probabilistic View)

This section reframes the learning problem using probability.


Maximum Likelihood Estimation (MLE): This approach finds the parameters θ that maximize the likelihood p(Y∣X,θ)—the probability of observing the training data given the parameters. For i.i.d. data, this involves maximizing a product of probabilities, p(Y∣X,θ)=∏ 
n=1
N
​	
 p(y 
n
​	
 ∣x 
n
​	
 ,θ), which is equivalent to minimizing the negative log-likelihood (a sum).



Connection to ERM: For a Gaussian likelihood p(y∣x,θ)=N(y∣f(x,θ),σ 
2
 ), MLE is identical to ERM with a squared loss function.

Maximum A Posteriori (MAP) Estimation: This method uses Bayes' theorem to find the most probable parameters after observing the data. It maximizes the posterior p(θ∣X,Y), which is proportional to the likelihood times a prior: p(θ∣X,Y)∝p(Y∣X,θ)p(θ).

Connection to Regularization: The prior p(θ) serves the same purpose as a regularizer in ERM. For example, a Gaussian prior on θ is equivalent to the λ∣∣θ∣∣ 
2
  (L2) regularizer.

4. Probabilistic Modeling and Bayesian Inference

This framework extends the probabilistic view by treating parameters θ as random variables themselves, rather than just finding a single "best" (point) estimate.

Bayesian Inference: The goal is not to find a single θ, but to compute the full posterior distribution p(θ∣X,Y) using Bayes' theorem.

Prediction: Predictions for a new data point x 
new
​	
  are made by marginalizing (integrating out) the parameters: p(x 
new
​	
 ∣X,Y)=∫p(x 
new
​	
 ∣θ)p(θ∣X,Y)dθ. This provides a prediction that is averaged over all plausible parameter values, naturally incorporating uncertainty.

Latent Variables: The chapter introduces models that use additional unobserved latent variables z (beyond parameters θ) to represent hidden structure or simplify the model, such as p(x∣θ)=∫p(x∣z,θ)p(z)dz.

5. Directed Graphical Models (Bayesian Networks)

This section introduces a visual language for describing the structure of probabilistic models.

Structure: Nodes represent random variables, and arrows represent conditional dependencies.

Factorization: The graph defines how the joint distribution factorizes into a product of simpler, local conditional probabilities (each variable given its parents), e.g., p(x 
1
​	
 ,…,x 
K
​	
 )=∏ 
k=1
K
​	
 p(x 
k
​	
 ∣Pa(x 
k
​	
 )).

Conditional Independence: The graph provides a simple way to read conditional independence relationships (e.g., A⊥⊥B∣C) between variables using rules known as d-separation.

6. Model Selection

This final section discusses how to choose between different models (e.g., polynomial degrees) or hyperparameters (e.g., the regularization strength λ).

Nested Cross-Validation: A non-probabilistic method that uses an outer loop to estimate generalization error and an inner loop to select the best hyperparameters on a validation set.

Bayesian Model Selection: A probabilistic method that compares models by computing their marginal likelihood (or evidence), p(D∣M)=∫p(D∣θ,M)p(θ∣M)dθ. This integral inherently penalizes overly complex models, acting as an "automatic Occam's razor". Models are compared using the Bayes factor, which is the ratio of their marginal likelihoods, p(D∣M 
1
​	
 )/p(D∣M 
2
​	
 ).