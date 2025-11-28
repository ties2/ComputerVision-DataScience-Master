# Data Exploration Overview

Data exploration is the initial phase of data analysis, aimed at understanding a dataset's structure, identifying patterns, detecting anomalies, and preparing for deeper modeling. This document outlines three key components: summarizing statistics, visualizing data, and handling missing values or outliers.

Data exploration—through summarizing statistics visualization, and cleaning—transforms raw data into a model-ready state, improving analysis accuracy and revealing actionable insights.

### The 5 V’s of Big Data

Volume, Velocity, Variety, Veracity, Value

* Volume

The size of the dataset.
* Velocity

The speed at which data is created, collected, processed and analyzed.
* Variety

The different types and formats of the data.
    * Structured: databases
    * Semi-structured: XML, JSON, CSV
    * Unstructured: Images, videos, audio, text
* Veracity

The quality and accuracy of data (inconsistencies, uncertainty, bias in data).
* Value
The usefulness of the data.



## Understanding the characteristics of (image) data in the dataset:
* Size
    * Number of samples or size of images.
    * Resolution
* Validity
    * How balanced is the dataset regarding the classes available in the dataset?
    * What is the quality of the dataset?
        * In the case of images: variability of e.g. brightness, sharpness, contrast, noise
        * Annotation quality (accuracy, consistency)
* Relationship between samples
    * What is the relation between for example classes across samples in the dataset.
* Data format
    * What is the format of the files (.jpg, .png, .json, .xml)
    * How are the images and annotations organized?

## Summarizing Statistics

Summarizing statistics condense a dataset's key characteristics, providing insights into its distribution, central tendency, and variability.

* Mean: The average, sensitive to outliers. Example: For [100, 120, 110, 500], mean = 207.5 (skewed by 500).
* Median: The middle value when sorted, robust to outliers. Example: For [100, 110, 120, 500], median = 115.
* Other Metrics:
    * Mode: Most frequent value.
    * Standard Deviation/Variance: Measures data spread.
    * Quartiles/Percentiles: Show data distribution (e.g., Q1, Q3).


* Use Case: In Python, use df.describe() in Pandas to compute these metrics, revealing data imbalances or skewness.

## Visualizing Data
Visualizations transform data into graphical forms, making patterns, trends, and relationships intuitive.

* Histograms: Show frequency distribution of a variable, revealing shape (e.g., normal or skewed). Example: Plotting ages might show a peak at 30-40 years.
Scatter Plots: Display relationships between two variables, indicating correlations or clusters. Example: Height vs. weight shows positive trends.

* Other Visuals:
    * Box Plots: Summarize quartiles, median, and outliers.
    * Line Charts: Track changes over time (e.g., stock prices).
    * Heatmaps: Show correlations between variables.


* Use Case: Use Matplotlib/Seaborn in Python (plt.hist(), sns.scatterplot()) to confirm statistical insights and spot data quality issues.

## Handling Missing Values and Outliers
Cleaning data by addressing missing values and outliers ensures reliable inputs for modeling.

* Missing Values:
    * Detection: Use df.isnull().sum() in Pandas to count missing entries.
    * Strategies:
        * Deletion: Drop rows/columns if missing data is minimal (df.dropna()).
        * Imputation: Fill with mean/median (df.fillna(df.mean())), mode, or advanced methods like KNN.
        * Flagging: Mark missingness in a new column for models to learn from.


   * Why?: Models like regression require complete data; imputation preserves dataset size.


* Outliers:
    * Detection: Identify via stats (>3 SD from mean) or box plots (beyond Q1-1.5IQR or Q3+1.5IQR).
    * Strategies:
        * Removal: Remove if erroneous, but preserve valid extremes.
        * Capping: Replace with boundary values (e.g., 99th percentile).
        * Transformation: Apply log/square-root to reduce impact.


   * Why?: Outliers can skew means, inflate variance, or cause overfitting.


* Use Case: After cleaning, re-check with updated stats/visuals to ensure data quality.


I'd be happy to provide a concise summary of the academic essay on data preparation challenges.

Here is a summary of the key problems and solutions discussed in the essay:

## Data Preparation Challenges

The majority of machine learning model failures stem from neglecting critical issues during the initial steps of Data Exploration (EDA) and Preprocessing. Mastering these steps is essential for building trustworthy models.

I. Challenges in Data Exploration (EDA)

1. Misinterpreting Skew and Distribution: Many models assume data is normally distributed.

* Problem: Heavily skewed features (like income or extreme values) can mislead linear models and distance-based algorithms.

* Solution: Employ power transformations (e.g., Box-Cox or Yeo-Johnson) to stabilize variance and make distributions more Gaussian.

2. Handling Outliers: Outliers drastically inflate mean and standard deviation.

* Problem: They distort scaling operations and can confuse algorithms.

* Solution: Use robust detection methods like Isolation Forest. For correction, capping (winsorization) is preferred over deletion to limit the extreme values' influence while retaining the data point.

II. Challenges in Data Preprocessing

1. Data Leakage (The Insidious Threat): This is the most damaging error, where information from the test set unintentionally influences the training set.

* Problem: It leads to wildly optimistic performance scores during testing that vanish in production. This often happens by fitting scalers or imputers to the entire dataset before splitting.

* Solution: Strictly perform all fit operations only on the training data. The learned parameters must then be applied to the test/validation sets using the transform method. This process should be enforced using Pipelines.

2. Class Imbalance: Significant disparity in the count of samples between the majority and minority classes.

* Problem: Models become biased toward the majority class, achieving high overall accuracy while failing to predict the crucial minority class (poor recall).

* Solution: Use advanced oversampling techniques like SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic examples of the minority class, thus balancing the dataset.

3. High Cardinality: Categorical features having a vast number of unique values (e.g., user IDs).

* Problem: Simple one-hot encoding creates too many columns, causing the curse of dimensionality and increasing the risk of overfitting.

* Solution: Use target-guided encoding (e.g., Mean Encoding), replacing categories with the average target value for that group, while using K-fold cross-validation to prevent leakage within the encoding process.
---

Why Resampling is ONLY for the Training Set
The purpose of the validation and test sets is to simulate real-world, unseen data and provide an unbiased evaluation of your model's performance.

1. Preventing Data Leakage: If you use techniques like RandomOverSampler on the test set, you are introducing artificial, duplicated, or synthetically generated samples into the data that will be used for final evaluation. This inflates the scores (like accuracy and recall) and gives you a false sense of security about your model's true capability. This is a severe form of Data Leakage.

2. Maintaining Real-World Distribution: Your original dataset's imbalance (for validation and test sets) reflects the true distribution of Gamma rays (Class 1) and Hadrons (Class 0) in the real-world observations. The model should be tested on this natural imbalance to see how it performs in a realistic scenario.

---

# Data Exploration: Reading Plots

When exploring a new dataset, we visualize variables to understand their relationships (Scatter Plots) and their distributions (Histograms).

1. Scatter Plots: Relationship Between Two Variables

Scatter plots compare two continuous variables (e.g., Height vs. Weight, Price vs. Square Footage). When analyzing them, look for three specific characteristics:

A. Direction (The Trend)

Positive Correlation: As variable X increases, variable Y increases. The dots go "uphill" from left to right.

Negative Correlation: As variable X increases, variable Y decreases. The dots go "downhill."

No Correlation: There is no visible pattern. Knowing X gives you no clue about Y.

B. Strength (The Spread)

Strong: The data points are tightly clustered around the trend line or curve.

Weak: The data points are loosely scattered, like a diffuse cloud, making the trend harder to see.

C. Shape (The Form)

Linear: The points roughly form a straight line.

Non-Linear: The points form a curve (e.g., U-shape, exponential curve).

2. Histograms: Distribution of One Variable

While scatter plots show relationships, histograms show the frequency of data points for a single variable.

Normal Distribution (Bell Curve): Symmetrical with a peak in the middle. Most data is average; extremes are rare.

Skewed Left/Right: The "tail" stretches out to one side.

Right Skewed: Tail is on the right (e.g., Income data—most earn average, a few earn billions).

Left Skewed: Tail is on the left (e.g., Age at death—most are older, fewer are very young).

Bimodal: Two distinct peaks. This suggests there might be two different groups mixed together in your data (e.g., plotting heights of both men and women on one graph).

3. Analysis of Typical Plot Patterns

Below is an analysis of the four most common scatter plot types you will encounter (similar to your upload), which I have recreated in the visual below.

Plot A: Strong Positive Linear

Observation: The dots form a tight, upward-sloping line.

Analysis: There is a strong direct relationship. If you know x, you can predict y with high accuracy.

Real-world example: Years of experience vs. Salary.

Plot B: Weak Negative Linear

Observation: The dots generally drift downwards, but they are spread out like a swarm.

Analysis: As x increases, y tends to decrease, but other factors (noise) are clearly affecting the data. Predictions will be less accurate here.

Real-world example: Hours of video games played vs. Exam grades (a trend exists, but it's not the only factor).

Plot C: No Correlation (Null)

Observation: A random "cloud" or "blob" of dots.

Analysis: The variables are independent. Changing x has no effect on y.

Real-world example: Shoe size vs. IQ score.

Plot D: Strong Non-Linear (Curvilinear)

Observation: The dots form an inverted "U" shape (parabola).

Analysis: The relationship changes direction. y increases as x increases, but only up to a certain point, after which y drops. A standard linear regression would fail here.

Real-world example: Stress levels vs. Performance (performance improves with some stress, but crashes if stress gets too high).

<p align="center">
  <img src="https://github.com/ties2/ComputerVision-DataScience-Master/blob/main/lectures/machine%20learning/Code_Generated_Image.png" alt="Practical Methodology" width="500" />
</p>