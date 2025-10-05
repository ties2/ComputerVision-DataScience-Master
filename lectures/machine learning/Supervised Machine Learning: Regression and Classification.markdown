# Supervised Machine Learning

Supervised Machine Learning is a type of AI where the algorithm learns to predict an outcome by studying example data that is already labeled (i.e., it has the correct answers).

The two main tasks under Supervised ML are defined by the type of output they predict:

1. Classification

Goal: To predict a discrete category or label.

Output: A finite set of classes (e.g., 0 or 1, Red, Green, or Blue).

Example: Is this transaction fraudulent or not fraudulent?

2. Regression

Goal: To predict a continuous, real-valued number.

Output: Any value within a range (e.g., 10.5, 1000.75, −5.0).

Example: Predicting the stock price next week or the rainfall in inches.

|Task	|Output Type	|Example Question
| ---- | ---- | -----|
|Classification	|Discrete Category (Label)	|Is this email Spam?
|Regression	|Continuous Value (Number)	|What will the temperature be?

That's a great follow-up! Unsupervised Learning is the counterpart to Supervised Learning, focusing on finding hidden structures in data that hasn't been labeled.

Unsupervised Machine Learning
Unsupervised Learning is a type of machine learning where the algorithm is given a dataset with no labels and no predefined answers.

Goal: To explore the data and automatically discover patterns, groupings, or relationships hidden within the features themselves.

The Learning: The model is not told what the correct output should be; it just tries to make sense of the structure on its own.

Use Case: Market segmentation, anomaly detection, data compression.

Clustering
Clustering is the most common and classic task within Unsupervised Learning.

Goal: To group similar data points together.

Mechanism: The algorithm measures the similarity (or distance) between every data point and groups those that are closest together into a cluster.

Output: Sets of data points (clusters) where members of the same cluster are more similar to each other than to members of other clusters. The algorithm might label them Cluster A, Cluster B, etc., but it doesn't know what those clusters represent (e.g., it doesn't know Cluster A means "high-income customers").

Common Clustering Algorithms:

K-Means: Divides data into K predefined, non-overlapping clusters.

DBSCAN: Finds clusters of varying shapes and sizes based on density, rather than assuming spherical shapes.

Hierarchical Clustering: Builds a hierarchy of clusters, useful for visualizing data organization.