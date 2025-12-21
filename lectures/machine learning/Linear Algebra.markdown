# Linear Algebra: The Backbone of Artificial Intelligence

Linear Algebra, in simplest terms, is the mathematics of **"Lists"** and **"Tables"** of numbers.
While regular mathematics deals with single numbers (like $x=5$), Linear Algebra deals with sets of numbers to describe a multi-dimensional world.

Here, I explain the key concepts simply:

---

## 1. The Building Blocks

* **Scalar:** A single number.
    * *Example:* Car speed (e.g., 50 km/h).
* **Vector:** A list of numbers that usually represents direction and magnitude.
    * *Example:* Movement in a computer game. You don't just go forward; you go right and up at the same time. Your vector becomes: `[x, y, z]`
    * *Mathematically:* $v=[2, 5]$
* **Matrix:** A table of numbers (like an Excel spreadsheet). It is a collection of vectors arranged side-by-side.
    * *Example:* A black-and-white image. Each cell in the table represents the brightness of one pixel.

---

## 2. Why is it called "Linear"?

Because in this mathematics, we do not **curve** space. Everything is like straight lines or flat planes.

* **Allowed:** Stretching, rotating, scaling.
* **Not Allowed:** Curving or using exponents (like $x^2$ has no place here).

> **An Image is a Matrix:** When you load a photo, the computer sees it as a large table of numbers (pixels).

**Blurring is a Matrix Operation:** To blur an image, the computer averages the values of matrix numbers with their neighbors (a type of matrix multiplication called **Convolution**).

**TensorBoard:** When you plot graphs, you are visualizing these vectors of numbers.

---

## 3. The Magic of Matrices: From Blurring to AI

Linear Algebra is the language of data. If you want to teach a computer to "understand" an image (Artificial Intelligence), you must convert that image into a matrix and perform calculations on it using the rules of Linear Algebra.

Let's see how mathematics causes an image to **Blur**. In the world of image processing and AI, this operation is called **Convolution**.

### A) Image as a Matrix (Input)

```
0   0   0
0  100  0
0   0   0
```

### B) Filter or Kernel ( The Tool for Change)

```
0.1  0.1  0.1
0.1  0.1  0.1
0.1  0.1  0.1
```

### C) Mathematical Operation (Convolution)

The kernel slides over the image, multiplies overlapping values, and sums them, spreading intensity and creating blur.

---

## 4. Artificial Intelligence: Automated Linear Algebra

Deep learning automates the search for the best matrices (weights) through training.

---

## 5. The Language of Tensors: Understanding Shapes

**PyTorch Standard:** `[N, C, H, W]`

Flatten example:

```python
x = x.view(x.size(0), -1)
```

---

## 6. The Mathematical Backbone

Key concepts include vector norms, inner products, projections, matrix operations, and decompositions.

---

## 7. Sample Project: Eigenfaces

Eigenfaces demonstrate PCA and eigenvectors as facial building blocks.

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load Data (Every image is a vector)
print("Loading faces...")
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w = lfw_people.images.shape
X = lfw_people.data

# 2. Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Matrix Decomposition (PCA & Eigenfaces)
# Reducing dimensions from 1850 pixels to 150 features
n_components = 150
pca = PCA(n_components=n_components, whiten=True).fit(X_scaled)
eigenfaces = pca.components_.reshape((n_components, h, w))

# 4. Display Eigenfaces (The Building Blocks)
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()

print("These are the Eigenfaces (The mathematical building blocks of faces):")
plot_gallery(eigenfaces, ["Eigenface %d" % i for i in range(12)], h, w)

```