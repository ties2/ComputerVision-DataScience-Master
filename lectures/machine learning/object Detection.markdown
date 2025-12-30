# Object Detection

**Goal:** The primary goal of object detection is to identify and locate objects within an image, usually by predicting **bounding boxes** around them.

---

## 1. Classical Computer Vision Approach

Before deep learning, object detection relied on manually created methods. This approach is effective when the object of interest is **clearly separable** from the background (e.g., by color or contrast).

### Case Study: Brazilian Coins

A classic example is detecting Brazilian coins placed on a white background.

* **The Basic Pipeline (Contours):**
* A standard approach uses **OpenCV contour finding**.


* 
**Process:** Threshold the image (binary)  Find white regions on black background  Draw bounding boxes around these contours.


* **Limitation:** This relies heavily on perfect thresholding. If the lighting is uneven or the coins touch, simple contours may fail (merging multiple coins into one).




* **The Advanced Pipeline (Hough Transform):**
* To improve upon simple contours, more sophisticated geometric techniques are used.
* **Hough Circle Transform:** A feature extraction technique used to detect circles in imperfect images. It is more robust than simple contouring for circular objects like coins.



---

## 2. Deep Learning Approach

Modern object detection largely uses Deep Learning models to **directly predict** bounding boxes without manual feature engineering.

### Key Models

Deep learning pipelines are robust and can handle complex backgrounds where objects are not clearly separable. Common architectures include:

* **YOLO (You Only Look Once):** Known for real-time speed.
* **RetinaNet:** Uses a focal loss function to handle class imbalance.
* **EfficientDet:** Optimizes efficiency and accuracy.

### Advanced Techniques

* **Tiling:** Processing high-resolution images by cutting them into smaller "tiles" to detect small objects that might be lost when resizing the full image.

---

## Summary Comparison

| Feature | Classical CV Detection | Deep Learning Detection |
| --- | --- | --- | 
| **Method** | Manual Feature Extraction (Contours, Hough) | Learned Features (CNNs) |
| **Requirement** | Object clearly separable from background | Large labeled dataset|
| **Example** | Brazilian Coins on white backgroun |Detecting cars ,pedestrians in traffic| 
| **Tools** | OpenCV (Thresholding, Morphological Ops)| YOLO, RetinaNet, EfficientDet |