# Machine learning Models

Machine learning models can be categorized in several ways, but the most common and fundamental method is based on how the model learns from the data.

Here are the three primary categories, followed by secondary ways to classify them:

1. Learning Type (The Main Categories)
This classification focuses on the nature of the data and the goal of the learning process.

### A. Supervised Learning 

In supervised learning, the model is trained on labeled data, meaning the input features (X) are provided along with their correct output answers or labels (y). The model learns the mapping function from X to y

| Subcategory	|Goal	|Example Models 
| ---- | ---- | ---- |
|Classification	|To predict a discrete label (category).|Logistic Regression, KNN, Decision Trees, SVM, Random Forest.
|Regression	|To predict a continuous value (quantity).	|Linear Regression, Lasso, Ridge, Support Vector Regression.

### B. Unsupervised Learning 

In unsupervised learning, the model is trained on unlabeled data. The goal is for the model to find hidden patterns, structures, or relationships within the data on its own.

|Subcategory	|Goal	|Example Models
| ---- | ---- | ---- |
|Clustering	|To group similar data points together into clusters.	|K-Means, DBSCAN, Hierarchical Clustering.
|Dimensionality Reduction	|To reduce the number of features while retaining most of the important information.	|Principal Component Analysis (PCA), t-SNE.
|Association	|To discover rules that describe large portions of the data (e.g., market basket analysis).	|Apriori, Eclat.

### C. Reinforcement Learning 

This type of learning involves an agent placed in an environment. The agent learns the optimal sequence of actions by receiving rewards for good actions and penalties for bad ones. The goal is to maximize the cumulative reward.



Key Use: Robotics, game playing (e.g., AlphaGo), and autonomous systems.

Example Models: Q-Learning, SARSA, Deep Q Networks (DQN).

---

2. Model Structure

Models can also be categorized based on their underlying mathematical structure:

|Category	|Description	|Example Models
| ---- | ---- | ---- |
|Linear Models	|Models that assume a linear relationship between input and output.	|Linear Regression, Logistic Regression, basic SVM.
|Non-Linear Models	|Models that can capture complex, curved relationships in the data.	|Neural Networks, Decision Trees, KNN, non-linear SVMs (using kernels).
|Parametric Models	|Models that have a fixed number of parameters (coefficients) determined from the data. Once trained, the data can be discarded.	|Linear Regression, Logistic Regression.
|Non-Parametric Models	|Models whose number of parameters grows with the size of the training data. The model must keep all or part of the training data to make predictions.	|K-Nearest Neighbors (KNN), Decision Trees.
|Ensemble Models	|Models that combine predictions from multiple individual models (often Decision Trees) to improve overall accuracy and stability.	|Random Forest, Gradient Boosting (XGBoost, LightGBM).


note: neural network without activation functions becomes a linear model


The hyperparameters are the "knobs and dials" you turn to tune a model's performance.

Here are the most critical and frequently used hyperparameters for the foundational machine learning models we discussed, categorized by model type.

1. Linear and Basic Models

|Model	|Hyperparameter	|Description	|Default (in sklearn)
| ---- | ---- | ---- | ---- |
|Logistic Regression	|penalty	|Type of regularization. Common choices are 'l1' (Lasso) for feature selection or 'l2' (Ridge) for shrinking coefficients.	|'l2'
|Logistic Regression|C	|Inverse of regularization strength. Smaller values mean stronger regularization. Used to control overfitting/underfitting tradeoff.	|1.0
|Logistic Regression|solver	|Algorithm used for optimization. Must be compatible with the chosen penalty (e.g., 'liblinear' works well for L1 and small datasets).	|'lbfgs'
|K-Nearest Neighbors (KNN)	|n_neighbors	|The number of neighbors (k) to consider when making a prediction. (This is the most critical parameter.)	|5
|K-Nearest Neighbors (KNN)|weights	|How to weight the neighbors. 'uniform' (all neighbors count equally) or 'distance' (closer neighbors count more).	|'uniform'
|K-Nearest Neighbors (KNN)|metric	|The distance metric to use. Common choices are 'minkowski' (Euclidean distance when p=2) or 'manhattan'.	|'minkowski'|
|Support Vector Machines (SVM)	|C	|Regularization parameter. Controls the penalty for misclassified points. Large C means low tolerance for error (potential overfitting).	|1.0
|Support Vector Machines (SVM)|kernel	|Defines the function used to map the input data into a higher-dimensional space. Choices include 'linear', 'poly', or 'rbf' (Radial Basis Function).	|'rbf'
|Support Vector Machines (SVM)|gamma	|Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels. Controls how much influence a single training sample has (low γ means smooth boundaries).	|'scale'

---

2. Tree-Based and Ensemble Models
These models are typically used when aiming for high predictive accuracy on structured data.

|Model	|Hyperparameter	|Description	|Default (in sklearn)
| ---- | ---- | ---- | ---- |
|Decision Tree	|max_depth	|Maximum depth (number of levels) the tree can grow. Controls complexity and prevents overfitting.	|None (full depth)
|Decision Tree |min_samples_leaf	|Minimum number of samples required to be at a leaf node. Used for smoothing the model.|	1
|Decision Tree |criterion	|The function to measure the quality of a split. 'gini' (Gini impurity) or 'entropy' (Information Gain).	|'gini'
|Random Forest	|n_estimators	|The number of trees in the forest. Generally, more trees give better performance (up to a point) but are computationally slower.	|100
|Random Forest |max_features	|The number of features to consider when looking for the best split (e.g., 'sqrt' or 'log2'). Controls the diversity of the trees.	|'sqrt'
|Random Forest| max_depth	|(Inherited from the base Decision Tree) See above.	|None
|Gradient Boosting (e.g., XGBoost)	|learning_rate	|Shrinkage factor applied to the contribution of each tree. A lower learning rate requires more estimators but usually results in a more robust model.	|0.1
|Gradient Boosting (e.g., XGBoost)	|n_estimators	|The number of boosting stages (number of trees to build).	|100
|Gradient Boosting (e.g., XGBoost)	|subsample	|The fraction of samples to be used for fitting the individual base learners. Used to prevent overfitting (similar to Stochastic Gradient Descent).	|1.0

---

3. Deep Learning (ANN)
While Deep Learning models involve complex architectures (CNN, RNN, etc.), the key hyperparameters for a basic fully connected Artificial Neural Network (ANN), often implemented via sklearn.neural_network.MLPClassifier, are:


|Hyperparameter	|Description	|Typical Range/Choices
| ---- | ---- | ---- 
|hidden_layer_sizes	|The structure of the network. A tuple defining the number of neurons in each hidden layer (e.g., (100, 50, 25) for 3 layers).	|Varies widely by problem size.
|activation	|The activation function for the hidden layers. Choices include 'relu' (most common), 'tanh', or 'logistic' (Sigmoid).	|'relu'
|solver	|The optimization algorithm (e.g., 'adam', 'lbfgs', 'sgd'). Adam is often the standard choice.	|'adam'
|alpha	|The L2 regularization term (similar to penalty='l2' in LogReg). Increases stability and prevents large weights.	|0.0001