# Statistical Concepts for Machine Learning Engineers

A comprehensive guide covering every statistical concept you need to master for machine learning.

---

## Table of Contents

1. [Descriptive Statistics](#1-descriptive-statistics)
2. [Probability Fundamentals](#2-probability-fundamentals)
3. [Probability Distributions](#3-probability-distributions)
4. [Statistical Inference](#4-statistical-inference)
5. [Hypothesis Testing](#5-hypothesis-testing)
6. [Regression Analysis](#6-regression-analysis)
7. [Bayesian Statistics](#7-bayesian-statistics)
8. [Dimensionality Reduction](#8-dimensionality-reduction)
9. [Sampling Methods](#9-sampling-methods)
10. [Information Theory](#10-information-theory)
11. [Statistical Learning Theory](#11-statistical-learning-theory)
12. [Experimental Design](#12-experimental-design)

---

## 1. Descriptive Statistics

Descriptive statistics summarize and describe the main features of a dataset.

### 1.1 Measures of Central Tendency

These tell us where the "center" of our data lies.

#### Mean (Average)
The sum of all values divided by the number of values.

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

**Example:**
```
Dataset: [85, 90, 78, 92, 88]
Mean = (85 + 90 + 78 + 92 + 88) / 5 = 86.6
```

**ML Application:** Used in loss functions, normalization, and as a baseline prediction.

**Warning:** Sensitive to outliers. If one student scored 20, the mean drops to 70.6.

#### Median
The middle value when data is sorted. For even-sized datasets, it's the average of two middle values.

**Example:**
```
Dataset: [85, 90, 78, 92, 88]
Sorted:  [78, 85, 88, 90, 92]
Median = 88 (middle value)

Dataset: [78, 85, 88, 90, 92, 95]
Median = (88 + 90) / 2 = 89
```

**ML Application:** More robust to outliers. Used in robust statistics and when data is skewed.

#### Mode
The most frequently occurring value.

**Example:**
```
Dataset: [red, blue, red, green, red, blue]
Mode = red (appears 3 times)
```

**ML Application:** Useful for categorical data, imputing missing categorical values.

### 1.2 Measures of Dispersion

These tell us how spread out our data is.

#### Variance
The average squared deviation from the mean.

**Population Variance:**
$$\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2$$

**Sample Variance (Bessel's correction):**
$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

**Example:**
```
Dataset: [2, 4, 4, 4, 5, 5, 7, 9]
Mean = 5

Deviations: [-3, -1, -1, -1, 0, 0, 2, 4]
Squared:    [9, 1, 1, 1, 0, 0, 4, 16]
Sum = 32

Population Variance = 32/8 = 4
Sample Variance = 32/7 ≈ 4.57
```

**Why n-1?** When estimating population variance from a sample, using n underestimates it. Dividing by n-1 corrects this bias.

#### Standard Deviation
The square root of variance. Same units as original data.

$$\sigma = \sqrt{\sigma^2}$$

**Example:**
```
If variance = 4, then standard deviation = 2
```

**ML Application:** Feature scaling, understanding model uncertainty, regularization.

#### Range
The difference between maximum and minimum values.

```
Dataset: [2, 4, 4, 4, 5, 5, 7, 9]
Range = 9 - 2 = 7
```

#### Interquartile Range (IQR)
The range of the middle 50% of data.

```
IQR = Q3 - Q1

Dataset: [2, 4, 4, 4, 5, 5, 7, 9]
Q1 (25th percentile) = 4
Q3 (75th percentile) = 6
IQR = 6 - 4 = 2
```

**ML Application:** Outlier detection. Values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR are often considered outliers.

### 1.3 Measures of Shape

#### Skewness
Measures asymmetry of the distribution.

$$\text{Skewness} = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^3$$

- **Positive skew (right-skewed):** Tail extends to the right (e.g., income distribution)
- **Negative skew (left-skewed):** Tail extends to the left (e.g., age at retirement)
- **Zero skew:** Symmetric distribution

```
Right-skewed example: [1, 2, 2, 3, 3, 3, 4, 4, 10, 50]
Most values cluster left, long tail to right
```

**ML Application:** Helps choose transformations (log transform for right-skewed data).

#### Kurtosis
Measures the "tailedness" of a distribution.

$$\text{Kurtosis} = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^4 - 3$$

- **Positive (leptokurtic):** Heavy tails, more outliers than normal
- **Negative (platykurtic):** Light tails, fewer outliers than normal
- **Zero (mesokurtic):** Similar to normal distribution

**ML Application:** Understanding outlier prevalence, risk assessment.

### 1.4 Covariance and Correlation

#### Covariance
Measures how two variables change together.

$$\text{Cov}(X, Y) = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$$

**Example:**
```
Height (X): [160, 170, 180, 175, 165]
Weight (Y): [55, 70, 80, 75, 60]

Mean X = 170, Mean Y = 68

Deviations X: [-10, 0, 10, 5, -5]
Deviations Y: [-13, 2, 12, 7, -8]

Products: [130, 0, 120, 35, 40]
Sum = 325

Covariance = 325/4 = 81.25 (positive: as height ↑, weight ↑)
```

**Problem:** Scale-dependent. Hard to interpret magnitude.

#### Pearson Correlation Coefficient
Standardized covariance between -1 and 1.

$$r = \frac{\text{Cov}(X, Y)}{s_X \cdot s_Y}$$

- **r = 1:** Perfect positive linear relationship
- **r = -1:** Perfect negative linear relationship
- **r = 0:** No linear relationship (but could have non-linear relationship!)

**Example:**
```
Cov(X, Y) = 81.25
Std(X) ≈ 7.91
Std(Y) ≈ 10.37

r = 81.25 / (7.91 × 10.37) ≈ 0.99 (strong positive correlation)
```

**ML Application:** Feature selection, multicollinearity detection, understanding relationships.

#### Spearman Rank Correlation
Correlation based on ranks, not values. Captures monotonic relationships.

**Example:**
```
X: [1, 2, 3, 4, 5]
Y: [1, 4, 9, 16, 25]  (Y = X²)

Pearson r ≈ 0.98 (not quite 1 because relationship is non-linear)
Spearman ρ = 1.0 (perfect monotonic relationship)
```

**ML Application:** When data has outliers or non-linear monotonic relationships.

---

## 2. Probability Fundamentals

### 2.1 Basic Probability Concepts

#### Sample Space (Ω)
The set of all possible outcomes.

```
Coin flip: Ω = {Heads, Tails}
Die roll: Ω = {1, 2, 3, 4, 5, 6}
```

#### Event
A subset of the sample space.

```
Event A = "rolling an even number" = {2, 4, 6}
```

#### Probability
A number between 0 and 1 representing likelihood.

$$P(A) = \frac{\text{favorable outcomes}}{\text{total outcomes}}$$

**Example:**
```
P(even number on die) = 3/6 = 0.5
```

### 2.2 Probability Rules

#### Complement Rule
$$P(A') = 1 - P(A)$$

```
P(not rolling a 6) = 1 - P(rolling a 6) = 1 - 1/6 = 5/6
```

#### Addition Rule
For any two events:
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

For mutually exclusive events (can't happen together):
$$P(A \cup B) = P(A) + P(B)$$

**Example:**
```
P(rolling 1 or even) = P(1) + P(even) - P(1 and even)
                     = 1/6 + 3/6 - 0 = 4/6 = 2/3
```

#### Multiplication Rule
$$P(A \cap B) = P(A) \cdot P(B|A)$$

For independent events:
$$P(A \cap B) = P(A) \cdot P(B)$$

**Example:**
```
P(two heads in a row) = P(H) × P(H) = 0.5 × 0.5 = 0.25
```

### 2.3 Conditional Probability

The probability of A given that B has occurred.

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

**Example: Medical Testing**
```
Disease prevalence: P(D) = 0.01 (1% have disease)
Test sensitivity: P(+|D) = 0.95 (95% true positive rate)
Test specificity: P(-|D') = 0.90 (90% true negative rate)

What's P(D|+)? (Probability of disease given positive test)

P(+) = P(+|D)P(D) + P(+|D')P(D')
     = 0.95 × 0.01 + 0.10 × 0.99
     = 0.0095 + 0.099 = 0.1085

P(D|+) = P(+|D)P(D) / P(+)
       = 0.0095 / 0.1085 ≈ 0.088 (only 8.8%!)
```

**ML Application:** Understanding precision/recall, Naive Bayes classifier.

### 2.4 Bayes' Theorem

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

Components:
- **P(A):** Prior probability (belief before seeing data)
- **P(B|A):** Likelihood (probability of data given hypothesis)
- **P(A|B):** Posterior probability (updated belief after seeing data)
- **P(B):** Evidence (normalizing constant)

**Example: Email Spam Classification**
```
Prior: P(Spam) = 0.3

Word "free" appears:
P("free"|Spam) = 0.8
P("free"|Not Spam) = 0.1

P("free") = P("free"|Spam)P(Spam) + P("free"|Not Spam)P(Not Spam)
          = 0.8 × 0.3 + 0.1 × 0.7 = 0.31

P(Spam|"free") = (0.8 × 0.3) / 0.31 ≈ 0.77
```

**ML Application:** Naive Bayes, Bayesian neural networks, probabilistic graphical models.

### 2.5 Independence

Two events are independent if:
$$P(A \cap B) = P(A) \cdot P(B)$$

Equivalently:
$$P(A|B) = P(A)$$

**Example:**
```
Rolling a die twice: outcomes are independent
Drawing cards without replacement: outcomes are dependent
```

### 2.6 Law of Total Probability

If B₁, B₂, ..., Bₙ partition the sample space:

$$P(A) = \sum_{i=1}^{n} P(A|B_i) \cdot P(B_i)$$

**Example:**
```
Factory has 3 machines producing widgets:
- Machine 1: 50% of production, 2% defect rate
- Machine 2: 30% of production, 3% defect rate
- Machine 3: 20% of production, 5% defect rate

P(Defect) = 0.02×0.50 + 0.03×0.30 + 0.05×0.20
          = 0.01 + 0.009 + 0.01 = 0.029 (2.9%)
```

---

## 3. Probability Distributions

### 3.1 Discrete Distributions

#### Bernoulli Distribution
Single trial with two outcomes (success/failure).

$$P(X = k) = p^k(1-p)^{1-k}, \quad k \in \{0, 1\}$$

- **Mean:** μ = p
- **Variance:** σ² = p(1-p)

**Example:**
```
Coin flip with P(Heads) = 0.6
X = 1 if heads, X = 0 if tails
P(X = 1) = 0.6
P(X = 0) = 0.4
```

**ML Application:** Binary classification, logistic regression output.

#### Binomial Distribution
Number of successes in n independent Bernoulli trials.

$$P(X = k) = \binom{n}{k}p^k(1-p)^{n-k}$$

- **Mean:** μ = np
- **Variance:** σ² = np(1-p)

**Example:**
```
10 coin flips, P(Heads) = 0.5
P(exactly 7 heads) = C(10,7) × 0.5^7 × 0.5^3
                   = 120 × 0.0078 × 0.125 ≈ 0.117
```

**ML Application:** Classification metrics, A/B testing.

#### Poisson Distribution
Number of events in a fixed interval (rare events).

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

- **Mean:** μ = λ
- **Variance:** σ² = λ

**Example:**
```
Average 3 emails per hour (λ = 3)
P(5 emails in next hour) = (3^5 × e^-3) / 5!
                         = (243 × 0.0498) / 120 ≈ 0.101
```

**ML Application:** Count data modeling, rare event detection.

#### Multinomial Distribution
Generalization of binomial for k > 2 outcomes.

$$P(X_1=x_1, ..., X_k=x_k) = \frac{n!}{x_1!...x_k!}p_1^{x_1}...p_k^{x_k}$$

**Example:**
```
Rolling a die 12 times:
P(each number appears exactly twice)
= 12! / (2!)^6 × (1/6)^12
```

**ML Application:** Multi-class classification, topic modeling.

### 3.2 Continuous Distributions

#### Uniform Distribution
All values in range equally likely.

$$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$

- **Mean:** μ = (a + b) / 2
- **Variance:** σ² = (b - a)² / 12

**Example:**
```
Random number between 0 and 10:
P(3 < X < 7) = (7 - 3) / (10 - 0) = 0.4
```

**ML Application:** Random initialization, random sampling.

#### Normal (Gaussian) Distribution
The bell curve. Most important distribution in statistics.

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

- **Mean:** μ
- **Variance:** σ²

**68-95-99.7 Rule:**
```
68% of data within 1 standard deviation of mean
95% of data within 2 standard deviations
99.7% of data within 3 standard deviations
```

**Standard Normal (Z-distribution):** μ = 0, σ = 1

**Example:**
```
Heights: μ = 170 cm, σ = 10 cm
P(Height > 180) = P(Z > (180-170)/10) = P(Z > 1) ≈ 0.159
```

**ML Application:** Assumption in many algorithms, error modeling, initialization.

#### Log-Normal Distribution
If ln(X) is normally distributed, X is log-normal.

$$f(x) = \frac{1}{x\sigma\sqrt{2\pi}}e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}$$

**Example:**
```
Stock prices, income distribution, file sizes
Always positive, right-skewed
```

**ML Application:** Modeling positive-only data, multiplicative processes.

#### Exponential Distribution
Time between events in a Poisson process.

$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

- **Mean:** μ = 1/λ
- **Variance:** σ² = 1/λ²

**Memoryless property:** P(X > s + t | X > s) = P(X > t)

**Example:**
```
Average 5 minutes between bus arrivals (λ = 1/5)
P(wait > 10 min) = e^(-10/5) = e^-2 ≈ 0.135
```

**ML Application:** Survival analysis, time-to-event modeling.

#### Beta Distribution
Probability distribution over probabilities (values between 0 and 1).

$$f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}$$

- **Mean:** μ = α / (α + β)
- **Variance:** σ² = αβ / [(α + β)²(α + β + 1)]

**Example:**
```
α = 2, β = 5: Mode near 0.2, right-skewed
α = 5, β = 2: Mode near 0.8, left-skewed
α = 5, β = 5: Symmetric, mode at 0.5
α = 1, β = 1: Uniform distribution
```

**ML Application:** Prior for Bernoulli parameter, Thompson sampling, Bayesian A/B testing.

#### Gamma Distribution
Generalization of exponential; time until k events occur.

$$f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$$

**ML Application:** Prior for precision (inverse variance), modeling wait times.

### 3.3 Multivariate Distributions

#### Multivariate Normal Distribution
Generalization of normal to multiple dimensions.

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{k/2}|\Sigma|^{1/2}}e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})}$$

- **μ:** Mean vector
- **Σ:** Covariance matrix

**Example (2D):**
```
μ = [0, 0]
Σ = [[1, 0.8],
     [0.8, 1]]

Variables are positively correlated (ρ = 0.8)
```

**ML Application:** Gaussian Mixture Models, Gaussian processes, multivariate analysis.

---

## 4. Statistical Inference

### 4.1 Point Estimation

Estimating a single value for a population parameter.

#### Properties of Good Estimators

**Unbiasedness:** E[θ̂] = θ
```
Sample mean is unbiased: E[X̄] = μ
```

**Consistency:** θ̂ → θ as n → ∞
```
As sample size increases, estimate approaches true value
```

**Efficiency:** Minimum variance among unbiased estimators
```
Sample mean is more efficient than sample median for normal data
```

#### Maximum Likelihood Estimation (MLE)

Find parameter that maximizes probability of observed data.

$$\hat{\theta}_{MLE} = \arg\max_\theta L(\theta|x) = \arg\max_\theta \prod_{i=1}^{n} f(x_i|\theta)$$

Usually maximize log-likelihood:
$$\hat{\theta}_{MLE} = \arg\max_\theta \sum_{i=1}^{n} \log f(x_i|\theta)$$

**Example: Bernoulli MLE**
```
Data: [1, 1, 0, 1, 0] (3 successes, 2 failures)
L(p) = p³(1-p)²
log L(p) = 3 log(p) + 2 log(1-p)

d/dp [log L] = 3/p - 2/(1-p) = 0
3(1-p) = 2p
3 = 5p
p̂ = 0.6 (sample proportion)
```

**ML Application:** Training neural networks (cross-entropy loss), logistic regression.

#### Method of Moments

Set sample moments equal to population moments.

**Example: Normal distribution**
```
First moment: X̄ = μ → μ̂ = X̄
Second moment: (1/n)Σ(Xi - X̄)² = σ² → σ̂² = sample variance
```

### 4.2 Interval Estimation (Confidence Intervals)

A range of plausible values for a parameter.

#### Confidence Interval for Mean (σ known)

$$\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

**Example:**
```
n = 100, X̄ = 50, σ = 10, 95% CI
z_0.025 = 1.96

CI = 50 ± 1.96 × (10/√100)
   = 50 ± 1.96
   = (48.04, 51.96)
```

**Interpretation:** If we repeated this procedure many times, 95% of the intervals would contain the true mean.

#### Confidence Interval for Mean (σ unknown)

Use t-distribution:
$$\bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$

**Example:**
```
n = 25, X̄ = 50, s = 10, 95% CI
t_0.025,24 ≈ 2.064

CI = 50 ± 2.064 × (10/√25)
   = 50 ± 4.13
   = (45.87, 54.13)
```

#### Confidence Interval for Proportion

$$\hat{p} \pm z_{\alpha/2} \cdot \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

**Example:**
```
n = 400, 240 successes, p̂ = 0.6, 95% CI

CI = 0.6 ± 1.96 × √(0.6 × 0.4 / 400)
   = 0.6 ± 0.048
   = (0.552, 0.648)
```

**ML Application:** Uncertainty quantification, model comparison.

### 4.3 The Central Limit Theorem

For large n, the sampling distribution of the mean approaches normal, regardless of population distribution.

$$\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right) \text{ as } n \to \infty$$

**Example:**
```
Rolling a die (uniform discrete):
- Single roll: definitely not normal
- Mean of 30 rolls: approximately normal with μ = 3.5, σ = √(35/12)/√30

This is why we can use z-tests even for non-normal data!
```

**ML Application:** Justifies many statistical tests, explains why batch means are normally distributed.

### 4.4 Bootstrap Methods

Estimate sampling distribution by resampling with replacement.

**Algorithm:**
```python
def bootstrap_mean_ci(data, n_bootstrap=10000, alpha=0.05):
    means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    
    lower = np.percentile(means, 100 * alpha/2)
    upper = np.percentile(means, 100 * (1 - alpha/2))
    return lower, upper
```

**ML Application:** Model uncertainty, when theoretical distributions are unknown.

---

## 5. Hypothesis Testing

### 5.1 Framework

1. **Null Hypothesis (H₀):** Default assumption (usually "no effect")
2. **Alternative Hypothesis (H₁):** What we're trying to show
3. **Test Statistic:** Summarizes evidence against H₀
4. **P-value:** Probability of seeing result as extreme as observed, if H₀ true
5. **Decision:** Reject H₀ if p-value < α (significance level)

### 5.2 Types of Errors

|               | H₀ True         | H₀ False        |
|---------------|-----------------|-----------------|
| **Reject H₀** | Type I Error (α)| Correct! (Power)|
| **Keep H₀**   | Correct!        | Type II Error (β)|

- **Type I Error (α):** False positive (convicting innocent)
- **Type II Error (β):** False negative (freeing guilty)
- **Power = 1 - β:** Probability of detecting true effect

**Example:**
```
Medical test:
- Type I: Healthy person diagnosed with disease (unnecessary treatment)
- Type II: Sick person cleared as healthy (disease progresses)
```

### 5.3 Common Tests

#### One-Sample t-Test

Tests if mean equals hypothesized value.

$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$$

**Example:**
```
H₀: μ = 100 (IQ test mean)
Data: n = 30, X̄ = 105, s = 15

t = (105 - 100) / (15/√30) = 5 / 2.74 = 1.83
df = 29
p-value ≈ 0.078 (two-tailed)

At α = 0.05: Don't reject H₀
```

#### Two-Sample t-Test

Tests if two means are equal.

**Equal variances:**
$$t = \frac{\bar{x}_1 - \bar{x}_2}{s_p\sqrt{1/n_1 + 1/n_2}}$$

**Unequal variances (Welch's t-test):**
$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$$

**Example:**
```
Group A: n₁ = 50, X̄₁ = 75, s₁ = 10
Group B: n₂ = 50, X̄₂ = 72, s₂ = 12

Using Welch's:
t = (75 - 72) / √(100/50 + 144/50)
  = 3 / √4.88 = 1.36
```

**ML Application:** A/B testing, comparing model performance.

#### Chi-Square Test

Tests association between categorical variables.

$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

**Example:**
```
           | Click | No Click | Total
Treatment  |   45  |   155    |  200
Control    |   30  |   170    |  200
Total      |   75  |   325    |  400

Expected (Treatment, Click) = (200 × 75) / 400 = 37.5

χ² = (45-37.5)²/37.5 + (155-162.5)²/162.5 + ...
   = 1.5 + 0.35 + 1.5 + 0.35 = 3.7

df = (2-1)(2-1) = 1
p-value ≈ 0.054
```

#### ANOVA (Analysis of Variance)

Tests if means of 3+ groups are equal.

$$F = \frac{\text{Between-group variance}}{\text{Within-group variance}}$$

**Example:**
```
Three teaching methods:
Method A: [85, 90, 78]
Method B: [92, 88, 95]
Method C: [75, 80, 77]

F-statistic tests if at least one method differs
```

**ML Application:** Feature importance, comparing multiple models.

#### Mann-Whitney U Test

Non-parametric alternative to t-test (doesn't assume normality).

**Example:**
```
Used when:
- Small sample sizes
- Ordinal data
- Non-normal distributions
```

### 5.4 Multiple Testing Correction

When running many tests, false positives accumulate.

**Bonferroni Correction:**
$$\alpha_{adj} = \frac{\alpha}{m}$$

**Example:**
```
Testing 20 hypotheses at α = 0.05
Without correction: expect 1 false positive
With Bonferroni: α_adj = 0.05/20 = 0.0025
```

**False Discovery Rate (FDR):**
Controls expected proportion of false positives among rejections.

**ML Application:** Feature selection with many features, genomics.

### 5.5 Effect Size

Statistical significance ≠ practical significance!

#### Cohen's d
Standardized difference between means:
$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$$

- **Small:** d = 0.2
- **Medium:** d = 0.5
- **Large:** d = 0.8

**Example:**
```
Two training methods:
X̄₁ = 80, X̄₂ = 75, s_pooled = 10
d = (80 - 75) / 10 = 0.5 (medium effect)
```

**ML Application:** Understanding if improvements are meaningful.

---

## 6. Regression Analysis

### 6.1 Simple Linear Regression

Models relationship between one predictor and one response.

$$Y = \beta_0 + \beta_1 X + \epsilon$$

**Least Squares Estimates:**
$$\hat{\beta}_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} = \frac{\text{Cov}(X,Y)}{\text{Var}(X)}$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}$$

**Example:**
```
Study hours (X): [1, 2, 3, 4, 5]
Test scores (Y): [52, 58, 65, 70, 74]

X̄ = 3, Ȳ = 63.8
Cov(X,Y) = 27, Var(X) = 2.5

β₁ = 27/2.5 = 10.8 (each hour → 10.8 point increase)
β₀ = 63.8 - 10.8×3 = 31.4

Ŷ = 31.4 + 10.8X
```

### 6.2 Multiple Linear Regression

Multiple predictors:
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p + \epsilon$$

Matrix form:
$$\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

**Least Squares Solution:**
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$$

### 6.3 Assumptions (LINE)

1. **Linearity:** Relationship between X and Y is linear
2. **Independence:** Errors are independent
3. **Normality:** Errors are normally distributed
4. **Equal variance (Homoscedasticity):** Constant error variance

**Checking assumptions:**
```
- Residual vs. Fitted plot: Check linearity and homoscedasticity
- Q-Q plot: Check normality
- Residual vs. Order: Check independence
```

### 6.4 Model Evaluation Metrics

#### R² (Coefficient of Determination)
Proportion of variance explained.

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- R² = 1: Perfect prediction
- R² = 0: No better than mean

**Adjusted R²:** Penalizes adding variables
$$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

#### Mean Squared Error (MSE)
$$MSE = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$$

#### Root Mean Squared Error (RMSE)
$$RMSE = \sqrt{MSE}$$

Same units as Y, interpretable as "average error."

#### Mean Absolute Error (MAE)
$$MAE = \frac{1}{n}\sum|y_i - \hat{y}_i|$$

More robust to outliers than MSE.

### 6.5 Regularization

Adds penalty to prevent overfitting.

#### Ridge Regression (L2)
$$\hat{\boldsymbol{\beta}} = \arg\min_\beta \left[\sum(y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \lambda\sum\beta_j^2\right]$$

- Shrinks coefficients toward zero
- Keeps all features

#### Lasso Regression (L1)
$$\hat{\boldsymbol{\beta}} = \arg\min_\beta \left[\sum(y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \lambda\sum|\beta_j|\right]$$

- Can shrink coefficients to exactly zero
- Performs feature selection

#### Elastic Net
Combination of L1 and L2:
$$\lambda_1\sum|\beta_j| + \lambda_2\sum\beta_j^2$$

**Example:**
```
λ = 0: Standard OLS
λ → ∞: All coefficients → 0

Cross-validation finds optimal λ
```

**ML Application:** Preventing overfitting, feature selection.

### 6.6 Polynomial Regression

Models non-linear relationships:
$$Y = \beta_0 + \beta_1 X + \beta_2 X^2 + ... + \beta_d X^d + \epsilon$$

**Caution:** High-degree polynomials overfit easily!

### 6.7 Logistic Regression

For binary classification:
$$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}} = \sigma(\beta_0 + \beta_1 X)$$

**Log-odds (logit):**
$$\log\left(\frac{P}{1-P}\right) = \beta_0 + \beta_1 X$$

**Interpretation:**
```
β₁ = 0.5 means:
- One unit increase in X multiplies odds by e^0.5 ≈ 1.65
- 65% increase in odds
```

**ML Application:** Classification baseline, interpretable models.

---

## 7. Bayesian Statistics

### 7.1 Bayesian vs. Frequentist

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| Probability | Long-run frequency | Degree of belief |
| Parameters | Fixed, unknown | Random variables |
| Data | Random | Fixed (observed) |
| Inference | Point estimates, CIs | Posterior distribution |

### 7.2 Bayesian Inference

$$\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}$$

$$P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}$$

**Example: Coin Flip**
```
Prior: θ ~ Beta(2, 2) (slight belief in fairness)
Data: 7 heads in 10 flips
Likelihood: Binomial(10, θ)

Posterior: θ | Data ~ Beta(2+7, 2+3) = Beta(9, 5)
Posterior mean: 9/14 ≈ 0.64
```

### 7.3 Conjugate Priors

Prior and posterior from same family.

| Likelihood | Prior | Posterior |
|------------|-------|-----------|
| Bernoulli/Binomial | Beta | Beta |
| Poisson | Gamma | Gamma |
| Normal (known σ) | Normal | Normal |
| Normal (known μ) | Inverse-Gamma | Inverse-Gamma |

**Example:**
```
Likelihood: Normal with known σ²
Prior: μ ~ N(μ₀, σ₀²)

Posterior: μ | Data ~ N(μ_n, σ_n²)

where:
μ_n = (σ₀⁻² μ₀ + n σ⁻² X̄) / (σ₀⁻² + n σ⁻²)
σ_n² = 1 / (σ₀⁻² + n σ⁻²)
```

### 7.4 Prior Selection

#### Informative Priors
Incorporate prior knowledge.
```
Expert says success rate is around 60%, fairly confident
Prior: Beta(12, 8) → mean 0.6, std ≈ 0.1
```

#### Weakly Informative Priors
Regularize without strong assumptions.
```
For regression coefficients: N(0, 10)
Allows large values but penalizes extreme ones
```

#### Non-Informative (Flat) Priors
Let data speak.
```
Uniform prior: P(θ) ∝ 1
Jeffrey's prior: P(θ) ∝ √I(θ) (Fisher information)
```

### 7.5 Credible Intervals

Bayesian analog to confidence intervals.

**95% Credible Interval:** Region containing 95% of posterior probability.

```
Posterior: θ ~ Beta(9, 5)
95% CI: [0.38, 0.85]

Interpretation: 95% probability θ is in this interval
(More intuitive than frequentist CI!)
```

### 7.6 MCMC Methods

For complex posteriors without closed form.

#### Metropolis-Hastings
```python
def metropolis_hastings(log_posterior, proposal, x0, n_samples):
    samples = [x0]
    x = x0
    for _ in range(n_samples):
        x_new = proposal(x)
        log_ratio = log_posterior(x_new) - log_posterior(x)
        if np.log(np.random.random()) < log_ratio:
            x = x_new
        samples.append(x)
    return samples
```

#### Gibbs Sampling
Sample each variable conditionally on others.
```
For multivariate posterior P(θ₁, θ₂):
1. Sample θ₁ ~ P(θ₁ | θ₂, data)
2. Sample θ₂ ~ P(θ₂ | θ₁, data)
3. Repeat
```

**ML Application:** Bayesian neural networks, probabilistic programming, uncertainty quantification.

---

## 8. Dimensionality Reduction

### 8.1 Principal Component Analysis (PCA)

Finds orthogonal directions of maximum variance.

**Algorithm:**
1. Center data: X - μ
2. Compute covariance matrix: Σ = XᵀX / (n-1)
3. Eigendecomposition: Σ = VΛVᵀ
4. Project onto top k eigenvectors

**Example:**
```
2D data with correlation:
     ●
   ●   ●
 ●   ●   ●
   ●   ●
     ●

PC1: Direction of maximum spread (diagonal)
PC2: Perpendicular to PC1

After PCA:
- PC1 captures most variance
- PC2 captures remaining variance
- Components are uncorrelated
```

**Variance Explained:**
$$\text{Explained ratio}_k = \frac{\lambda_k}{\sum_i \lambda_i}$$

**ML Application:** Dimensionality reduction, visualization, noise reduction.

### 8.2 Singular Value Decomposition (SVD)

Any matrix can be decomposed as:
$$\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$

- **U:** Left singular vectors (n × n)
- **Σ:** Singular values on diagonal (n × p)
- **V:** Right singular vectors (p × p)

**Relationship to PCA:**
- Right singular vectors V = principal components
- Singular values σᵢ = √(λᵢ × (n-1))

**ML Application:** Matrix factorization, recommender systems, latent semantic analysis.

### 8.3 t-SNE

Non-linear dimensionality reduction for visualization.

**Key Idea:**
1. Convert high-dim distances to probabilities (Gaussian)
2. Convert low-dim distances to probabilities (t-distribution)
3. Minimize KL divergence between them

**Example:**
```
MNIST digits in 784D → 2D
- Clusters form for each digit
- Preserves local structure
- Does NOT preserve global distances!
```

**Hyperparameters:**
- **Perplexity:** Balance between local/global (~30)

**ML Application:** Visualizing embeddings, exploring clusters.

### 8.4 UMAP

Uniform Manifold Approximation and Projection.

Similar goals to t-SNE but:
- Faster
- Better global structure preservation
- More robust hyperparameters

**ML Application:** Alternative to t-SNE for visualization.

---

## 9. Sampling Methods

### 9.1 Random Sampling

#### Simple Random Sampling
Each item equally likely to be selected.

```python
sample = np.random.choice(population, size=n, replace=False)
```

#### Stratified Sampling
Proportional sampling from subgroups.

```
Population: 70% Class A, 30% Class B
Sample n=100: 70 from A, 30 from B
```

**Use when:** Subgroups have different characteristics.

#### Cluster Sampling
Randomly select clusters, sample all within.

```
Select 10 random schools
Survey all students in those schools
```

**Use when:** Population naturally clusters.

### 9.2 Resampling Methods

#### Cross-Validation
Estimate generalization error.

**K-Fold CV:**
```
1. Split data into K folds
2. For each fold:
   - Train on K-1 folds
   - Validate on remaining fold
3. Average performance across folds
```

**Leave-One-Out CV:** K = n (computationally expensive)

**Stratified K-Fold:** Maintains class proportions in each fold.

**Example:**
```
5-Fold CV with n=100:
Fold 1: Train on 80, test on 20
Fold 2: Train on 80 (different), test on 20 (different)
...
Average accuracy: mean of 5 fold accuracies
```

#### Bootstrap
Sample with replacement, same size as original.

```python
def bootstrap_statistic(data, statistic, n_bootstrap=1000):
    stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        stats.append(statistic(sample))
    return stats
```

**Use for:** Confidence intervals, standard errors when theory is hard.

### 9.3 Importance Sampling

Sample from easy distribution, reweight.

$$E_p[f(x)] = E_q\left[f(x)\frac{p(x)}{q(x)}\right]$$

**Example:**
```
Estimating P(X > 5) where X ~ N(0,1)
- Direct sampling: ~0.0000003 (rare event!)
- Importance sampling: sample from N(5,1), reweight
```

**ML Application:** Reinforcement learning, rare event estimation.

---

## 10. Information Theory

### 10.1 Entropy

Measure of uncertainty/randomness.

$$H(X) = -\sum_{i} P(x_i) \log_2 P(x_i)$$

**Example:**
```
Fair coin: H = -0.5 log₂(0.5) - 0.5 log₂(0.5) = 1 bit
Biased coin (90%/10%): H = -0.9 log₂(0.9) - 0.1 log₂(0.1) ≈ 0.47 bits

More certainty → lower entropy
```

**Properties:**
- H(X) ≥ 0
- Maximum when uniform distribution
- H(X) = 0 when deterministic

**ML Application:** Decision trees (information gain), feature selection.

### 10.2 Cross-Entropy

Measures cost of using distribution q when true distribution is p.

$$H(p, q) = -\sum_{i} p(x_i) \log q(x_i)$$

**Example:**
```
True labels: p = [1, 0] (class 0)
Predicted: q = [0.9, 0.1]

Cross-entropy = -1 × log(0.9) - 0 × log(0.1) = 0.105

If prediction were [0.6, 0.4]:
Cross-entropy = -1 × log(0.6) = 0.511 (worse!)
```

**ML Application:** Loss function for classification.

### 10.3 KL Divergence

Measures how different q is from p.

$$D_{KL}(p \| q) = \sum_{i} p(x_i) \log \frac{p(x_i)}{q(x_i)} = H(p, q) - H(p)$$

**Properties:**
- D_KL ≥ 0
- D_KL = 0 iff p = q
- NOT symmetric: D_KL(p‖q) ≠ D_KL(q‖p)

**Example:**
```
p = [0.5, 0.5]
q = [0.9, 0.1]

D_KL(p‖q) = 0.5 log(0.5/0.9) + 0.5 log(0.5/0.1)
          = 0.5(-0.85) + 0.5(2.32)
          = 0.74 bits
```

**ML Application:** VAEs, policy gradients, distribution matching.

### 10.4 Mutual Information

Information shared between two variables.

$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

$$I(X; Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$$

**Properties:**
- I(X; Y) ≥ 0
- I(X; Y) = 0 iff X and Y are independent
- I(X; Y) = H(X) iff X is deterministic function of Y

**ML Application:** Feature selection, representation learning, InfoGAN.

### 10.5 Information Gain

Reduction in entropy from knowing a feature.

$$IG(T, a) = H(T) - \sum_{v \in values(a)} \frac{|T_v|}{|T|} H(T_v)$$

**Example (Decision Tree):**
```
Dataset: 14 examples, 9 yes, 5 no
H(S) = -9/14 log(9/14) - 5/14 log(5/14) = 0.940

Feature "Outlook" splits into:
- Sunny: 5 examples (2 yes, 3 no)
- Overcast: 4 examples (4 yes, 0 no)
- Rain: 5 examples (3 yes, 2 no)

H(Sunny) = 0.971
H(Overcast) = 0 (pure!)
H(Rain) = 0.971

IG = 0.940 - (5/14 × 0.971 + 4/14 × 0 + 5/14 × 0.971)
   = 0.940 - 0.694 = 0.246 bits
```

**ML Application:** Decision tree splitting criterion.

---

## 11. Statistical Learning Theory

### 11.1 Bias-Variance Tradeoff

Total error = Bias² + Variance + Irreducible Error

$$E[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$

**Bias:** Error from wrong assumptions (underfitting)
**Variance:** Error from sensitivity to training data (overfitting)

**Example:**
```
Fitting polynomial to noisy sine wave:
- Degree 1 (line): High bias, low variance
- Degree 3: Balanced
- Degree 15: Low bias, high variance (wiggly!)
```

```
Error
  │
  │    Total
  │   Error    ╱
  │  ╲       ╱
  │   ╲    ╱
  │    ╲ ╱
  │  Variance
  │      ╲
  │       ╲
  │  Bias  ╲────
  └───────────────
       Model Complexity
```

**ML Application:** Model selection, regularization tuning.

### 11.2 Overfitting and Underfitting

**Underfitting (High Bias):**
- Model too simple
- High training error
- High test error
- Solutions: More features, complex model

**Overfitting (High Variance):**
- Model too complex
- Low training error
- High test error
- Solutions: Regularization, more data, simpler model

**Detection:**
```
Training Error:  5%
Validation Error: 25%
→ Overfitting (20% gap)

Training Error:  25%
Validation Error: 27%
→ Underfitting (both high)
```

### 11.3 Regularization Theory

Adds complexity penalty to loss function:

$$\text{Loss}_{regularized} = \text{Loss}_{data} + \lambda \cdot \text{Complexity}$$

**Effect:**
- Constrains hypothesis space
- Reduces variance at cost of some bias
- Prevents overfitting

**Examples:**
- L1/L2 regularization (Ridge/Lasso)
- Dropout
- Early stopping
- Data augmentation

### 11.4 VC Dimension

Measures model capacity/complexity.

**Definition:** Maximum number of points model can shatter (classify in all 2ⁿ ways).

**Examples:**
```
Linear classifier in 2D:
- Can shatter any 3 non-collinear points
- Cannot shatter 4 points (XOR problem)
- VC dimension = 3

Linear classifier in d dimensions:
- VC dimension = d + 1
```

**Generalization Bound:**
$$\text{Test Error} \leq \text{Train Error} + O\left(\sqrt{\frac{VC}{n}}\right)$$

**ML Application:** Understanding model capacity.

### 11.5 PAC Learning

Probably Approximately Correct learning framework.

**Definition:** Algorithm PAC-learns if, with probability ≥ 1-δ, it outputs hypothesis with error ≤ ε.

**Sample Complexity:**
$$n \geq \frac{1}{\epsilon}\left(\ln|H| + \ln\frac{1}{\delta}\right)$$

**Example:**
```
To learn with ε = 0.05 error, δ = 0.05 failure probability
From hypothesis class of size 1000:

n ≥ (1/0.05)(ln(1000) + ln(20))
  ≥ 20 × (6.9 + 3.0)
  ≥ 198 samples
```

---

## 12. Experimental Design

### 12.1 A/B Testing

Compare two variants to determine which performs better.

**Steps:**
1. Define metric (conversion rate, CTR, etc.)
2. Determine sample size (power analysis)
3. Randomly assign users
4. Run experiment
5. Analyze results (hypothesis test)

**Sample Size Calculation:**
$$n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2p(1-p)}{(\text{MDE})^2}$$

**Example:**
```
Baseline conversion: 10%
Minimum Detectable Effect: 1% absolute (10% → 11%)
α = 0.05, Power = 0.80

n ≈ 14,752 per group
```

### 12.2 Power Analysis

**Power:** Probability of detecting a true effect.

**Factors affecting power:**
- Sample size (↑ n → ↑ power)
- Effect size (↑ effect → ↑ power)
- Significance level (↑ α → ↑ power, but more false positives)
- Variance (↑ variance → ↓ power)

**Example:**
```
Want 80% power to detect d = 0.5 at α = 0.05

Required n per group ≈ 64
(from power tables or calculation)
```

### 12.3 Randomization and Control

**Why randomize?**
- Eliminates selection bias
- Balances confounders (known and unknown)
- Enables causal inference

**Control group:** Baseline for comparison
- Placebo control
- Active control (current best)
- Historical control (weaker)

**Stratified randomization:**
Block on important variables to ensure balance.

### 12.4 Confounding and Causation

**Confounder:** Variable that affects both treatment and outcome.

```
Example: Ice cream sales and drownings are correlated.
Confounder: Hot weather increases both!

Ice Cream ← Hot Weather → Drownings
          ↘          ↙
            Correlation
              (spurious)
```

**Simpson's Paradox:**
Trend in aggregated data reverses in subgroups.

```
Hospital A: 90% survival (treats severe cases)
Hospital B: 95% survival (treats mild cases)

Overall A looks worse, but within severity levels:
- Severe cases: A beats B
- Mild cases: A beats B

A is actually better!
```

### 12.5 Causal Inference

**Potential Outcomes Framework:**
- Y(1): Outcome if treated
- Y(0): Outcome if not treated
- Causal effect: Y(1) - Y(0)

**Problem:** We only observe one potential outcome!

**Solutions:**
1. Randomization (A/B testing)
2. Instrumental variables
3. Difference-in-differences
4. Regression discontinuity
5. Propensity score matching

**Example: Propensity Score**
```
Estimate P(Treatment | Covariates)
Match treated/control with similar propensity
Compare outcomes within matched pairs
```

**ML Application:** Uplift modeling, causal ML, treatment effect estimation.

---

## Quick Reference: When to Use What

| Situation | Method |
|-----------|--------|
| Comparing two means | t-test |
| Comparing 3+ means | ANOVA |
| Categorical association | Chi-square test |
| Predicting continuous outcome | Linear regression |
| Predicting binary outcome | Logistic regression |
| Non-normal data comparison | Mann-Whitney U |
| Many features, few samples | Regularization (Lasso/Ridge) |
| Uncertainty in predictions | Bayesian methods |
| Visualizing high-dim data | t-SNE / UMAP |
| Model selection | Cross-validation |
| Feature importance | Mutual information, permutation importance |
| Rare event probability | Importance sampling |
| A/B test analysis | Two-proportion z-test |

---

## Common Pitfalls to Avoid

1. **P-hacking:** Running many tests until one is significant
2. **Confusing correlation with causation**
3. **Ignoring multiple testing correction**
4. **Using parametric tests on non-normal data**
5. **Overfitting to training data**
6. **Ignoring class imbalance**
7. **Data leakage in cross-validation**
8. **Simpson's paradox in aggregated data**
9. **Interpreting statistical significance as practical significance**
10. **Assuming independence when data is correlated**

---

## Recommended Reading

1. *The Elements of Statistical Learning* - Hastie, Tibshirani, Friedman
2. *Pattern Recognition and Machine Learning* - Bishop
3. *All of Statistics* - Wasserman
4. *Bayesian Data Analysis* - Gelman et al.
5. *An Introduction to Statistical Learning* - James et al.

---

*This guide covers the essential statistical concepts for machine learning. Master these, and you'll have a solid foundation for understanding and developing ML algorithms.*