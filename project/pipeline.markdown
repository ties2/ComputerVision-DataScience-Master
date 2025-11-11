Here is a list of common issues that can occur at each stage of a machine learning project.

##  Preprocessing

* **Data Leakage:** This is a critical error where information from your validation or test set (or the target itself) accidentally "leaks" into your training data. For example, scaling or imputing your *entire* dataset *before* splitting it into train/test sets. The model learns information it wouldn't have in the real world, leading to a deceptively high-performance score.
* **Missing or Corrupt Values:** Data is often incomplete (NaNs, nulls) or contains errors (e.g., "age = -99"). How you handle this (deleting, imputing with mean/median, or using a model) can significantly bias your results.
* **Class Imbalance:** In classification tasks (like fraud or disease detection), one class may be extremely rare (e.g., 99% "not fraud," 1% "fraud"). A model can achieve 99% accuracy by just predicting "not fraud" every time, making it useless. This requires techniques like oversampling (SMOTE) or using better metrics (F1-score, Precision-Recall).
* **Unscaled Features:** If one feature is `age` (18-65) and another is `salary` (50,000-500,000), the `salary` feature will numerically dominate the model's learning process. Most models require features to be scaled (e.g., standardized or normalized).
* **Improper Encoding:** Models only understand numbers. Failing to correctly convert categorical data (like "Red," "Green," "Blue") into a numerical format (like one-hot encoding) will cause errors.

---

##  Training

* **Overfitting:** The most common problem. The model learns the training data *too well*, including its noise and random fluctuations. It performs perfectly on data it has seen but fails to generalize to new, unseen data.
* **Underfitting:** The model is too simple to capture the underlying patterns in the data. It performs poorly on *both* the training data and new data.
* **Vanishing Gradients:** (As you mentioned) In very deep networks, the gradients (error signals) become exponentially smaller as they are passed backward. The early layers of the network learn extremely slowly or stop learning altogether.
* **Exploding Gradients:** The opposite problem. Gradients become exponentially large, causing the model's weights to update erratically. This leads to unstable training where the loss (error) can suddenly become `NaN` (Not a Number).
* **Slow or No Convergence:** The model's loss doesn't decrease, or it bounces around wildly. This is almost always caused by a **bad learning rate** (either too high, causing it to overshoot, or too low, causing it to get stuck).

---

##  Validating

* **Validation Set Mismatch:** The validation data is not representative of the training data (or the real world). For example, your training set has images from all seasons, but your validation set only has images from winter. Your validation score will be misleading.
* **Overfitting to the Validation Set:** This happens during hyperparameter tuning. If you relentlessly tune your model to get the best possible score on your *one* validation set, you are effectively "overfitting" to that specific set. The model may not perform as well on the final test set.
* **Using the Wrong Metric:** Using **accuracy** on an imbalanced dataset is a classic example. You must choose a metric that actually reflects your project's goal (e.g., **Precision** for minimizing false positives or **Recall** for minimizing false negatives).

---

##  Testing

* **Poor Generalization:** This is the ultimate "failure" state. Your model performed well in training and validation, but when shown the final, held-out test set, its performance is poor. This confirms overfitting, a validation/test set mismatch, or data leakage.
* **Distribution Shift (or Covariate Shift):** The data in the "real world" (i.e., your test set or production data) has different properties (e.g., different mean, variance) than your training data. For example, you trained a model on data from one country and are testing it on data from another.
* **Concept Drift:** This is a problem over *time*. The underlying *relationship* between the features and the target variable changes. For example, a spam filter trained in 2020 won't know how to handle new spam tactics from 2025. The model's performance will degrade in production.