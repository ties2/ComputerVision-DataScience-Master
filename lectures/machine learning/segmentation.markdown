# Segmentation for Computer Vision

**Goal:** To understand how to separate objects of interest from the background using color analysis, thresholding techniques, and contour detection.

---

## 1. What is Segmentation?

Segmentation is the process of partitioning an image into multiple segments (sets of pixels) to simplify its representation.

* **Classification:** "Is there a cat in this image?"
* **Object Detection:** "Where are the cats in this image?" (Bounding boxes) .


* **Instance Segmentation:** "Which specific pixels belong to each individual cat?" (Exact boundaries) .



---

## 2. The Role of Color (Chapter 6 & Histograms)

Before segmenting, we often analyze the distribution of pixel intensities or colors using **Histograms**.

* A **histogram** plots the frequency of pixel intensity values (0–255).


* **Peaks** represent dominant regions (e.g., background vs. object).
* 
**Valleys** between peaks are ideal places to set a **threshold** to separate these regions.



> **Self-Study Tip:** If an image has two distinct peaks (bimodal), it is easy to separate the foreground from the background. If the histogram is messy, simple separation might fail.

---

## 3. Thresholding Techniques (Chapter 10 Fundamentals)

Thresholding is the simplest method of segmentation. It converts a grayscale image into a binary image (black and white).

### A. Simple (Global) Thresholding

We pick a specific value (e.g., 127).

* 
**Rule:** If a pixel value is greater than the threshold, set it to 1 (white); otherwise, set it to 0 (black).


* **Limitation:** It applies the *same* threshold to every pixel. If one part of the image is in shadow, it might fail.



### B. Adaptive Thresholding

Used when lighting is uneven (e.g., the "Sudoku" example).

* 
**How it works:** Instead of one global value, the algorithm calculates a threshold for small regions (blocks) of the image.


* 
**Calculation:** It computes the average intensity inside a localized window (block) and subtracts a constant  to determine the threshold for that specific area.



### C. Otsu’s Binarization

This is an automated global thresholding method.

* **Benefit:** You do not need to manually guess the threshold value. Otsu's algorithm automatically calculates the best global threshold by analyzing the histogram to minimize variance between the two classes (black and white).


* 
**Best for:** Noisy images where the histogram has two clear peaks (bimodal).



---

## 4. Finding Contours

Once an image is thresholded (binary), we can extract the shapes of objects.

* 
**Definition:** Contour detection finds boundaries by detecting changes in color or intensity.


* 
**Pre-requisite:** For accurate detection in OpenCV, the **object to be detected should be white** and the background should be black.


* 
**Output:** Contours can be used to draw bounding boxes, count objects (like coins), or calculate areas.



---

## 5. Practical Workflow (The Pipeline)

For self-study, implement this standard pipeline using the concepts above:

1. **Input:** Load the image (e.g., coins or sudoku grid).
2. **Preprocessing:** Convert to Grayscale (simplifies data) or a specific Color Space (HSV) if color is key.
3. **Thresholding:**
* Try **Otsu** first if the lighting is even.
* Use **Adaptive** if there are shadows or gradients.


4. **Post-Processing:** Use **Find Contours** to locate the white regions on the black background.
5. **Result:** Draw bounding boxes or count the distinct objects found.

---

## 6. Summary Checklist

| Technique | When to use | Key Concept |
| --- | --- | --- |
| **Global Threshold** | Even lighting, high contrast | `Pixel > T = 1` |
| **Adaptive Threshold** | Uneven lighting (shadows) | Computes `T` locally per block |
| **Otsu's Method** | Unknown threshold value | Automatically finds best `T` from histogram |
| **Contours** | To count/measure objects | Finds boundaries of white objects on black |

---


