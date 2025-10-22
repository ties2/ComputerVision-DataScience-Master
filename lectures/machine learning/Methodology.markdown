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