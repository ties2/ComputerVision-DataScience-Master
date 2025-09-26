# PyTorch Learning Notes part 4

## work with sklearn library

Scikit-learn (often abbreviated as sklearn) is a powerful and widely-used open-source machine learning library for Python. It provides a consistent and simple interface for a wide variety of machine learning tasks. Built on top of popular Python libraries like NumPy and SciPy, it's an essential tool for data scientists and machine learning practitioners.

### Key Features of Scikit-learn

Scikit-learn is known for its comprehensive and user-friendly features that cover the entire machine learning workflow.

* Diverse Algorithms: It includes a vast collection of machine learning algorithms for both supervised and unsupervised learning, such as classification, regression, and clustering. You can find implementations for everything from linear regression to support vector machines and k-means.

* Consistent API: The library's API is incredibly consistent. The standard fit(), transform(), and predict() methods make it easy to swap different models and preprocessing steps in your pipeline.

* Data Preprocessing Tools: Scikit-learn offers a robust set of tools for preparing your data. This includes functions for scaling numerical features (StandardScaler, MinMaxScaler), encoding categorical variables (OneHotEncoder), and handling missing values (SimpleImputer).

* Model Evaluation: The library provides a rich set of metrics and tools to evaluate model performance, including functions for cross-validation, hyperparameter tuning (GridSearchCV), and generating various performance reports like a confusion matrix.

* Pipelines: You can chain multiple steps, such as preprocessing and modeling, into a single Pipeline object. This helps streamline your workflow and prevents data leakage, a common issue in machine learning.

### sample of use sklearn library

```
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load the iris dataset
# This is a classic dataset for classification tasks.
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
# This is a crucial step to evaluate how well the model generalizes to new, unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Logistic Regression model instance
model = LogisticRegression(max_iter=200)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# You can also use the model to predict a new single sample
new_sample = np.array([[5.0, 3.5, 1.3, 0.3]])
prediction = model.predict(new_sample)
print(f"Prediction for new sample: {iris.target_names[prediction][0]}")
```

