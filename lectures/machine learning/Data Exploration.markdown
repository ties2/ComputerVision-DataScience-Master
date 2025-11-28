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

---

# How to Analyze a Histogram

When you analyze a histogram, you are essentially describing the "personality" of the data. You generally look for four main characteristics, often remembered by the acronym SOCS:


<p align="center">
  <img src="https://github.com/ties2/ComputerVision-DataScience-Master/blob/main/docs/summary/mathematics%20for%20ML/Code_Generated_Image.png" alt="Practical Methodology" width="800" />
</p>


1. Shape (The Pattern)

Modality (Peaks): How many "humps" does the graph have?

Unimodal: One clear peak (e.g., normal distribution).

Bimodal: Two peaks. This often suggests two different groups were mixed together (e.g., heights of adults, mixing men and women).

Uniform: No peaks; the bars are roughly the same height (e.g., rolling a fair die).

Skewness (The Tail): Ideally, data is symmetric (mirror image). If it's not, it is "skewed."

Skewed Right: The "tail" extends to the right (higher numbers). This is common with data that has a minimum but no maximum, like income or housing prices.

Skewed Left: The "tail" extends to the left (lower numbers). This is common with data that has a maximum limit, like test scores (most students pass, a few fail).

2. Outliers (Unusual Values)

Are there any bars that stand distinctly apart from the rest of the group? These may indicate data errors or interesting anomalies.

3. Center (The Typical Value)

Median: The middle point. In skewed graphs, the median is usually the best measure of the center.

Mean: The average. In skewed graphs, the mean is pulled toward the tail (outliers).

4. Spread (The Variation)

How wide is the graph? A wide graph means high variability (inconsistent data); a narrow graph means low variability (consistent data).

Visual Analysis of Histogram Types

I have generated examples of these distributions below to help you visualize the analysis.

()

1. Symmetric (Unimodal)

Analysis: The left and right sides are mirror images. The Mean and Median are exactly in the center.

Real-world Example: Heights of people, standardized test scores, errors in measurements.

2. Skewed Left (Negative Skew)

Analysis: The "mass" of the data is on the right, but the tail drags out to the left. The Mean is typically less than the Median here because the low outliers pull the average down.

Real-world Example: Age at death (most people are old, fewer are young), scores on an easy test.

3. Skewed Right (Positive Skew)

Analysis: The "mass" is on the left, but the tail drags out to the right. The Mean is typically greater than the Median here because the high outliers pull the average up.

Real-world Example: House prices, salaries, number of children in a family.

4. Uniform

Analysis: There are no peaks; every outcome is roughly equally likely.

Real-world Example: Rolling a die (1-6 have equal chance), lottery numbers.

5. Bimodal

Analysis: Two distinct peaks. This strongly suggests that there are two distinct populations hidden in your dataset.

Real-world Example: The lunch rush at a restaurant (peaks at 12 PM and 7 PM), running speeds (joggers vs. sprinters).

6. Multimodal

Analysis: Three or more distinct peaks. This indicates complex data with multiple sub-groups.

Real-world Example: Test scores in a class containing students from three different grade levels

---

# Understanding Box Plots


<p align="center">
  <img src="https://github.com/ties2/ComputerVision-DataScience-Master/blob/main/lectures/machine%20learning/Code_Generated_Image%20copy.png" alt="Practical Methodology" width="500" />
</p>



A Box Plot (or Box-and-Whisker Plot) is a standardized way of displaying the distribution of data based on a five-number summary. It is particularly useful for comparing distributions between different groups and identifying outliers.

()

The Five-Number Summary

Minimum (Whisker end): The lowest data point excluding any outliers. It is calculated as Q1−1.5×IQR.

First Quartile (Q1): The median of the lower half of the dataset (25th percentile). 25% of the data lies below this line.

Median (Q2): The middle value of the dataset (50th percentile). It splits the data into two equal halves.

Third Quartile (Q3): The median of the upper half of the dataset (75th percentile). 75% of the data lies below this line.

Maximum (Whisker end): The highest data point excluding any outliers. It is calculated as Q3+1.5×IQR.

Key Elements

The Box (Interquartile Range - IQR): The central box represents the middle 50% of the data, from Q1 to Q3. The length of the box is the Interquartile Range (IQR=Q3−Q1). A wider box indicates more variability in the middle of the data.

The Whiskers: These lines extend from the box to the Minimum and Maximum values. They show the range of the rest of the data.

Outliers: Individual points plotted beyond the whiskers. These are data points that are statistically significantly different from the rest of the data (usually defined as being more than 1.5×IQR away from the quartiles).

When to Use a Box Plot

Comparisons: Ideally suited for comparing distributions across groups (e.g., test scores of Class A vs. Class B).

Outlier Detection: Quickly highlights extreme values.

Summary: Provides a quick visual summary of the data's central tendency and spread without the detail of a histogram.

<p align="center">
  <img src="https://github.com/ties2/ComputerVision-DataScience-Master/blob/main/lectures/machine%20learning/Code_Generated_Image-2.png" alt="Practical Methodology" width="800" />
</p>

Here is a detailed visualization that connects the Box Plot to the actual Distribution of the data. This provides a "better" picture by showing why the box plot looks the way it does.

()

How to Read This Detailed View

This image reveals what a box plot simplifies.

The Box (The Middle 50%):

Look at the Orange Box at the bottom. The left edge aligns with Q1 (25%) and the right edge aligns with Q3 (75%).

Comparing it to the blue graph above, you can see that the "Box" captures the bulk of the data (the highest part of the histogram curve).

The Median (The Center Line):

The Blue Dashed Line cuts through the exact middle of the data.

Notice how the Median is not in the center of the "peaks" (modes). Because this data is "bimodal" (has two humps), the median sits in the valley between them. A simple box plot hides this "two-hump" shape, which is why seeing it with a histogram is powerful.

The Whiskers (The Spread):

The horizontal lines extending from the box cover the rest of the data. They stretch out to cover the "tails" of the blue distribution curve.

Summary Table: When to use which?

Feature	Histogram / Density Plot	Box Plot
Detail Level	High (Shows every bump and valley)	Low (Summarizes into 5 numbers)
Best For	Seeing the "shape" (e.g., bimodal, skew)	Comparing many groups side-by-side
Weakness	Can look cluttered with many groups	Hides distinct peaks (modes)

---

## make a dataset suitable for analysis and machine learning. These steps fall under the umbrella of **Data Preprocessing** and **Data Cleaning**.

### 1. Structuring the Data
Machine learning models generally require data in a tabular or vector format.
* **What to do:** Convert unstructured data (like raw text, logs, or JSON files) into rows (samples) and columns (features).
* **Why:** Algorithms need mathematical inputs.
* **Example:** Converting a folder of emails into a CSV file with columns for `Sender`, `Subject`, `Body_Length`, and `Is_Spam`.

### 2. Handling Missing Values
Real-world data is rarely complete. Missing values can crash models or skew predictions.
* **Imputation (Filling in):** Replace missing values with the **Mean** (average), **Median** (middle value), or **Mode** (most frequent value) of that column.
    * *Best for:* Numerical data where the missingness is random.
* **Deletion:** Remove rows (samples) or columns (features) that have too many missing values.
    * *Best for:* Very large datasets where losing a few rows doesn't hurt, or columns that are mostly empty.
* **Prediction:** Use a separate ML model (like K-Nearest Neighbors) to guess the missing value based on other features.

### 3. Handling Outliers
Outliers are data points significantly different from the majority. They can distort statistical measures like the mean.
* **Mathematical Definition:** As discussed previously, points falling below $Q1 - 1.5 \times IQR$ or above $Q3 + 1.5 \times IQR$ (from a Box Plot context).
* **Removal:** Delete the rows containing outliers if they are likely due to measurement error or bad data entry.
* **Capping (Winsorizing):** Set a maximum/minimum limit. Any value above the limit is set to that limit (e.g., setting all incomes above \$1M to exactly \$1M).
* **Transformations:** Apply a log transformation (e.g., $log(x)$) to squeeze high outliers closer to the rest of the data. This is common for "skewed right" data like salaries.

### 4. Normalization & Standardization
Different features often have different units (e.g., Age in years vs. Salary in dollars). Large numbers can dominate the model.
* **Normalization (Min-Max Scaling):** Scales all values to fit between 0 and 1.
    $$X_{new} = \frac{X - X_{min}}{X_{max} - X_{min}}$$
    * *Best for:* Algorithms like Neural Networks or when the data doesn't follow a normal distribution.
* **Standardization (Z-score Scaling):** Centers data around 0 with a standard deviation of 1.
    $$X_{new} = \frac{X - \mu}{\sigma}$$
    * *Best for:* Algorithms that assume a Gaussian distribution (like Logistic Regression or SVMs) and makes them robust to outliers.

### 5. Visual Inspection (Specifically for Images)
Automated checks might miss qualitative issues that human eyes catch instantly.
* **Random Sampling:** Don't check just the first 10 images; pick random indices to check the variety.
* **Check for Artifacts:** Look for compression blocks (jpeg artifacts), watermarks, or corruption (grey boxes).
* **Check Variance:** Ensure the dataset isn't just the same image repeated with slightly different brightness.
* **Sanity Check:** If the folder is labeled "Cats," ensure the images actually contain cats.

### 6. Fixing Errors (Label Cleaning)
"Garbage in, Garbage out." If your target labels (what you are trying to predict) are wrong, the model will learn wrong patterns.
* **Consistency Checks:** Ensure distinct categories don't overlap (e.g., having both "USA" and "United States" as separate categories).
* **Manual Review:** For critical datasets, humans review a subset of labels to estimate the error rate.
* **Heuristics:** If a house is listed as having 200 bedrooms and 1 bathroom, it is likely an error that needs correction or removal.

---

What is an Outlier?

<p align="center">
  <img src="https://github.com/ties2/ComputerVision-DataScience-Master/blob/main/lectures/machine%20learning/outlier.png" alt="Practical Methodology" width="600" />
</p>

An outlier is a data point that differs significantly from other observations. It lies far away from the main group of data points. Think of it as a "rebel" in your dataset that doesn't follow the general trend.

()

Why do Outliers Matter?

As shown in the graph above, even a single outlier (the red 'x') can drastically change the results of your analysis.

The Green Line: Represents the relationship (trend) of the normal data.

The Red Line: Shows how the model gets "pulled" down by the outlier. The model tries to minimize error for all points, so it sacrifices accuracy on the majority to accommodate the one anomaly.

This sensitivity is why detecting and handling outliers is a critical step in data cleaning.

1. Causes: Where do they come from?

Data Entry Errors: Human error (e.g., typing "1000" instead of "10.00").

Measurement Errors: Sensor malfunction or experimental error.

Natural Variation: Sometimes extreme values are real and valid (e.g., Jeff Bezos' income in a dataset of average salaries). These are often the most interesting points for "Anomaly Detection."

2. Detection: How to find them?

Method A: The Z-Score (Standard Deviation)

This method assumes your data follows a normal (bell curve) distribution. It measures how many standard deviations (σ) a data point is from the mean (μ).

Rule of Thumb: If a data point has a Z-score greater than 3 or less than -3, it is widely considered an outlier.

$Z = \frac{x - \mu}{\sigma}$
​	

Method B: The IQR Method (Box Plot)

This method is robust (less sensitive) to extreme values and is what Box Plots use. It relies on the Interquartile Range (IQR=Q3−Q1).

Low Outlier: <Q1−1.5×IQR

High Outlier: >Q3+1.5×IQR






