# Statistical Models: Bayes Theorem & Naive Bayes

1. Introduction
Probability theory is the backbone of machine learning. It provides a framework for making decisions under uncertainty. In classification tasks, we rarely know with 100% certainty what a specific input is; instead, we calculate the probability that an input belongs to a certain class.

This guide focuses on Bayes' Theorem and its application in the Naive Bayes Classifier.

2. Bayes' Theorem
Bayes' Theorem is a mathematical formula for determining conditional probability. It describes the probability of an event, based on prior knowledge of conditions that might be related to the event.

The Formula

Shutterstock

$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$
​	
 
Where:

P(A∣B) (Posterior): The probability of hypothesis A being true given that evidence B has occurred. This is what we want to calculate.

P(B∣A) (Likelihood): The probability of seeing evidence B if hypothesis A is true.

P(A) (Prior): The initial probability of hypothesis A being true, before seeing the evidence.

P(B) (Evidence): The total probability of seeing the evidence B under all circumstances.

In the Context of Classification

If we are classifying an image (let's say an image from CIFAR-10) into a class (e.g., "Airplane"), the formula becomes:

$P(\text{Class}|\text{Data}) = \frac{P(\text{Data}|\text{Class}) \cdot P(\text{Class})}{P(\text{Data})}$
​	
 
3. The Naive Bayes Classifier
Calculating the true probability is computationally expensive because images have high dimensions (many pixels). To simplify this, we use the Naive Bayes approach.

Why is it "Naive"?

It makes a very strong (and often technically incorrect) assumption: Independence. It assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.

Example: In an image of a face, a "nose" feature usually implies eyes are nearby. A Naive classifier ignores this relationship and treats the nose and eyes as completely independent events.

The Naive Bayes Equation

For an input vector X=(x 

$P(y|X) \propto P(y) \cdot \prod_{i=1}^{n} P(x_i|y)$

We calculate the probability of the class y multiplied by the probability of every single individual pixel belonging to that class, and then select the class with the highest score.

Types of Naive Bayes

Gaussian Naive Bayes: Used when features are continuous and assumed to follow a normal (Gaussian) distribution. (Best for image pixel intensities).

Multinomial Naive Bayes: Used for discrete counts (e.g., text classification/word counts).

Bernoulli Naive Bayes: Used for binary/boolean features.

4. Exercise: CIFAR-10 Classification
The goal is to fit a Naive Bayes classifier on the CIFAR-10 dataset and evaluate its accuracy.

Step-by-Step Implementation Guide

A. Preprocessing

Naive Bayes in Scikit-Learn (specifically GaussianNB) generally expects 2D arrays as input (Samples, Features). CIFAR-10 images are 3D (Height, Width, Channels).

Flatten the images: Convert the 32×32×3 image into a flat vector of size 3072.

Normalization: Scale pixel values (0-255) to a range of 0-1 or standardize them.

B. Implementation (Mental Draft)

You will likely use the GaussianNB class from Scikit-Learn.

```
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 1. Initialize the model
gnb = GaussianNB()

# 2. Train the model (X_train must be flattened)
gnb.fit(X_train_flat, y_train)

# 3. Predict
y_pred = gnb.predict(X_test_flat)

# 4. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```

C. Comparison Questions

As you analyze your results, answer these questions:

Accuracy: Naive Bayes usually performs worse than KNN or CNNs on raw pixels. Why? (Hint: Does the "independence" assumption hold true for neighboring pixels in an image?)

Speed: Compare the training and prediction time of Naive Bayes vs. KNN. Naive Bayes should be significantly faster at prediction time. Why?

1. Answer regarding Accuracy

Question: Why does Naive Bayes usually perform worse than KNN or CNNs on raw pixels? Answer: The main reason is the "Independence Assumption."

The Reality of Images: In an image, neighboring pixels are highly correlated. If one pixel is blue (e.g., the sky), the adjacent pixel is likely blue as well. Furthermore, objects are defined by "shapes" and "patterns," which result from the specific spatial arrangement of pixels.

The Naive Bayes Problem: This model assumes that every pixel is completely independent of the others. To this model, it makes no difference whether an "eye" pixel is located next to a "nose" pixel or in the far corner of the image.

Result: Naive Bayes ignores Spatial Structure and pixel relationships, treating the image like a "bag of jumbled pixels." In contrast, CNNs are designed specifically to understand these neighborhood relationships, and KNN (though simple) compares the overall global structure of one image to another.

2. Answer regarding Speed

Question: Why is Naive Bayes much faster than KNN during the Prediction phase? Answer: The difference lies in "Modeling" (Eager Learning) vs. "Lazy Learning."

KNN (Lazy Learner):

Training: It essentially has no training phase; it simply stores the data.

Prediction: When a new image arrives, it must calculate the distance between that image and every single one of the 50,000 images in the training data (in CIFAR-10) to find its neighbors. This process is computationally heavy and slow.

Time Complexity: Dependent on the number of training samples (O(N)).

Naive Bayes (Eager Learner):

Training: During training, it calculates the mean and variance of pixels for each class and stores them as a compact formula. (The original data is no longer needed).

Prediction: When a new image arrives, it simply plugs the data into the pre-built formula and calculates the result. No comparison with past data is required.

Time Complexity: Independent of the number of training samples and is very fast (O(1)).

General Conclusion: Naive Bayes is fast but less accurate (because it ignores complex relationships), whereas KNN can be more accurate but is extremely slow at runtime.

---

Naive Bayes is not "bad"; it is just specialized. While it struggles with complex images (like CIFAR-10) where pixel relationships matter, it is arguably the best choice for several other specific domains.

Here are the projects where Naive Bayes is successful and appropriate:

1. Text Classification (The #1 Use Case)

Naive Bayes is famous for being excellent at processing text.

Spam Filtering: This is the classic example. If an email contains words like "Free," "Winner," and "Cash," it is likely spam. The model treats these words as independent "keywords" (Bag of Words), which works very well for identifying topics.

Sentiment Analysis: Determining if a movie review is positive or negative based on the count of words like "excellent," "boring," "loved," etc.

Why it works: In text, the presence of specific words is often more important than the exact grammar or order (spatial structure), so the "Independence Assumption" is less damaging here.

2. Real-Time Prediction

Applications: Real-time ad targeting, content moderation on websites, or high-speed recommendation engines.

Why it works: As you noted, the prediction speed is blazing fast. If you need a result in milliseconds, Naive Bayes is often better than a deep neural network.

3. Medical Diagnosis

Applications: Predicting a disease (e.g., Diabetes, Cancer) based on a list of independent symptoms or test results (Age, Glucose Level, Blood Pressure).

Why it works: Medical features are often not as spatially correlated as image pixels. Furthermore, Naive Bayes provides a probability score (e.g., "75% chance of flu"), which is more useful to doctors than a simple "Yes/No."

4. Recommendation Systems

Applications: "Collaborative Filtering" (e.g., recommending a product because other users who bought X also bought Y).

Why it works: It handles large, sparse datasets (where users have only rated a few items out of thousands) very efficiently.

Summary Comparison

Domain	Is Naive Bayes Good?	Why?
Images (Fusion/Vision)	 No	Images rely on spatial structure (pixels next to each other matter). The "Independence Assumption" breaks this.
Text (NLP/Spam)	 Yes	Text topics rely on specific keywords. Treating words as independent features works surprisingly well.
Real-Time Systems Yes	It requires very little CPU power to predict.