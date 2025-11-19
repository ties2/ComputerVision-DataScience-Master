# Chapter 3 intensity transformations and spatial filtering

# Chapter 4:

# Chapter 5:

# Chapter 6:

# Chapter 7:

# Chapter 8:

# Chapter 9:




# Chapter 10:

Image segmentation is the process of partitioning a digital image into multiple sub-regions or sets of pixels. The goal is to change the representation of an image into something that is more meaningful and easier to analyze. Segmentation algorithms generally rely on one of two basic properties of intensity values: discontinuity (boundaries/edges) and similarity (regions)

1. Point, Line, and Edge Detection

This approach segments images by finding abrupt local changes in intensity.

Summary: Detection is often performed using spatial masks (kernels) that compute the first or second derivative of the image.

Points: Detected using the Laplacian, which isolates single pixels with significantly different intensities from their neighbors.

Lines: Detected using specific masks sensitive to horizontal, vertical, or diagonal orientations.

Edges: The most common discontinuity. An edge is a boundary between two regions with distinct intensity properties.

Practical Application (Edge Detectors):

Sobel & Prewitt Operators: Simple, fast methods that compute the gradient magnitude to find edges. Good for general purpose but sensitive to noise.

Marr-Hildreth: Uses the Laplacian of a Gaussian (LoG) to find zero-crossings. It detects edges at different scales but can create closed loops ("spaghetti effect").

Canny Edge Detector: Considered the superior standard. It uses a multi-stage algorithm (smoothing, gradient calculation, non-maxima suppression, and hysteresis thresholding) to detect weak edges while minimizing noise and false positives. It produces thin, continuous edges ,.

Hough Transform: A global processing technique used to link edge points into shapes (like lines or circles) even if there are gaps in the boundary. It is highly effective for finding straight lines in noisy images.

2. Thresholding

Thresholding separates objects from the background by selecting an intensity value (threshold) T. Any pixel with intensity >T belongs to the object (foreground), and others belong to the background.

Summary:

Global Thresholding: Uses a single T for the entire image. Works best when the histogram is bimodal (has two distinct peaks).

Variable (Local/Adaptive) Thresholding: The threshold T changes over the image based on local neighborhood properties (e.g., local mean or standard deviation). Essential for images with uneven illumination.

Practical Application (Otsu’s Method):

Otsu’s Method: An optimum global thresholding technique that automatically calculates the best threshold by maximizing the variance between the object and background classes. It is purely statistical and requires no prior knowledge of the image features.

Tip: If an image is noisy, smooth it first (e.g., Gaussian blur) before applying Otsu’s method to get a much cleaner segmentation.

3. Region-Based Segmentation

Instead of finding boundaries, these methods try to find the regions themselves directly.

Region Growing: Starts with "seed" points and grows regions by appending neighboring pixels that have similar properties (e.g., intensity, texture, color).

Practical: Critical for segmenting objects that have similar properties but irregular shapes. Selection of seed points and the "similarity predicate" (rules for joining) are the key design choices.

Region Splitting and Merging: A top-down approach that subdivides an image into quadrants (quadtrees) if a region is not uniform, and then merges adjacent regions that are similar.

4. Segmentation Using Clustering and Superpixels

k-Means Clustering: An iterative algorithm that partitions data into k groups. In image segmentation, it groups pixels based on intensity or color. It is simple but requires you to specify the number of regions (k) in advance.

Superpixels (SLIC): This method groups pixels into perceptually meaningful atomic regions (superpixels) to replace the rigid pixel grid. The Simple Linear Iterative Clustering (SLIC) algorithm is fast and efficient, generating superpixels based on color similarity and proximity.

Practical: Use superpixels as a pre-processing step to reduce the computational load for complex segmentation algorithms.

5. Segmentation Using Graph Cuts

Summary: Represents the image as a graph where pixels are nodes and edges represent the similarity between them. Segmentation becomes a problem of finding a "cut" through the graph that partitions it into disjoint sets (foreground and background) with minimum cost.

Practical Application: Very powerful for segmenting objects when you have some idea of what the object and background look like (e.g., via user strokes). It finds a globally optimal solution but can be computationally expensive.

6. Segmentation Using Morphological Watersheds

Summary: Visualizes the image as a topographic surface where light pixels are "high" and dark pixels are "low". It simulates flooding from regional minima; "dams" built where water from different basins meets become the segmentation lines.

Practical Application: Excellent for separating touching objects (e.g., overlapping cells in microscopy).

Problem: It often leads to "over-segmentation" (too many small regions) due to noise.

Solution: Use markers (pre-defined internal and external seed areas) to control the flooding and restrict the number of resulting regions.

7. Motion in Segmentation

Summary: Uses motion as a strong cue to separate objects from the background.

Practical Application:

Difference Images: Subtracting a reference image from the current frame detects moving objects.

Accumulative Difference Image (ADI): Helps track the path and speed of moving objects over time by keeping a history of changes

# Chpeter 11:

Chapter 11: Feature Extraction (often titled Representation and Description in classic contexts).

This chapter bridges the gap between segmentation (isolating objects) and classification (recognizing objects). Once a region is segmented, it must be represented in a compact form (Representation) and then measured/quantified (Description) to be useful for computer processing.

1. Background & Core Concepts

Feature Detection vs. Description:

Detection finds a feature (e.g., finding a corner).

Description assigns a quantitative attribute to it (e.g., measuring the angle or orientation of that corner).

Invariance: A good descriptor should be insensitive to variations in:

Scale: The size of the object shouldn't change its description.

Translation: The position in the image shouldn't matter.

Rotation: The orientation shouldn't change the unique identity of the descriptor.

2. Boundary Preprocessing (Representation)

Before measuring a boundary, it is often necessary to convert the list of pixels into a more useful format.

Chain Codes (Freeman Chain Codes):

Concept: Represent a boundary as a sequence of directional steps (0–7 for 8-connectivity) rather than a list of coordinates.

Practical Tip: Raw chain codes are sensitive to rotation and starting points. Use the first difference (difference between adjacent codes) to make it rotation invariant, and treat it as a circular sequence (select the integer of minimum magnitude) to normalize the starting point.

Polygonal Approximations (MPP):

Concept: Represents a digital boundary as a polygon with the fewest vertices possible while maintaining the basic shape. The Minimum Perimeter Polygon (MPP) algorithm visualizes the boundary as a rubber band shrinking around the object constrained by an inner and outer wall.

Practical Tip: Great for data reduction (compression) and smoothing out noise along a boundary before analysis.

Signatures:

Concept: Reduces a 2D boundary to a 1D function, such as plotting the distance from the centroid to the boundary as a function of angle.

Practical Tip: Useful for distinguishing shapes like circles (constant signature) vs. squares (signature with 4 peaks). It reduces 2D matching problems to easier 1D signal matching.

Skeletons (Medial Axis Transform - MAT):

Concept: Reduces a region to a graph or "stick figure" representing the structural shape. Ideally, it is the set of points equidistant from the region's boundaries.

Practical Tip: Essential for analyzing biological shapes or handwriting where the thickness of the object is irrelevant, but the structure is critical.

3. Boundary Feature Descriptors

Once a boundary is represented, we calculate numbers to describe it.

Shape Numbers: The "first difference" of a chain code of smallest magnitude. It defines the "shape" independent of orientation.

Fourier Descriptors:

Concept: Treat the x,y coordinates of a boundary as complex numbers (x+jy) and apply the Discrete Fourier Transform (DFT).

Practical Tip: The low-frequency coefficients capture the general shape, while high-frequency coefficients capture fine detail (and noise). To smooth a boundary or match shapes robustly, keep only the first few Fourier descriptors (e.g., the first 10–20) and discard the rest. This makes the descriptor insensitive to noise.

Statistical Moments: Using the mean, variance, and skewness of a 1D signature (e.g., distance vs. angle) to describe the boundary shape.

4. Region Feature Descriptors

Describing the entire interior of an object, not just the edge.

Simple Descriptors:

Compactness: Perimeter 
2
 /Area. (Low for circles, high for complex shapes).

Circularity: 4π(Area)/Perimeter 
2
 . (1 for a perfect circle).

Eccentricity: Ratio of the major axis to the minor axis (measure of elongation).

Topological Descriptors (Euler Number):

Concept: Describes the connectivity of a region. Formula: E=C−H (Connected Components - Holes).

Practical Tip: Highly robust to rubber-sheet deformations. A letter "B" always has 2 holes, regardless of how you stretch or rotate the font.

Texture:

Statistical Approaches: Uses the intensity histogram of a region.

Smooth textures have narrow histograms (low variance).

Coarse textures have broad histograms (high variance).

Co-occurrence Matrix: A powerful statistical tool that tracks how often pairs of pixels with specific values occur in specific spatial relationships (e.g., "how often does a gray pixel appear next to a white pixel?"). It yields descriptors like Contrast (measure of local variations) and Energy (measure of uniformity).

Spectral Approaches: Uses the Fourier spectrum to detect global periodicity (e.g., grid patterns or repeated textures).

Moment Invariants (Hu Moments):

Concept: A set of 7 statistical moments calculated from the image intensity function.

Practical Tip: These 7 numbers are invariant to translation, rotation, and scale. If you need to recognize an object (like a letter or a silhouette) regardless of its size or angle in the image, compare its Hu Moments against a database.

5. Principal Components (PCA)

Concept: A mathematical transform (Hotelling Transform) that aligns data along the directions of highest variance (eigenvectors of the covariance matrix).

Practical Tip: Used to align objects before recognition. If you have an object that is rotated arbitrarily, PCA finds the "major axis" (direction of greatest spread). You can then rotate the object so this axis aligns with the vertical, standardizing the object's orientation before extracting features.

6. Whole-Image Features (Advanced)

Harris-Stephens Corner Detector:

Finds points in an image where intensity changes rapidly in all directions (corners), as opposed to edges (change in one direction) or flat regions (no change).

Practical Tip: These "interest points" are excellent anchors for image matching (e.g., stitching panoramas) because they are distinct and stable.

SIFT (Scale-Invariant Feature Transform):

Extracts keypoints that are invariant to scaling and rotation and partially invariant to illumination. It creates a unique "fingerprint" for key parts of an image, allowing for robust object recognition even in cluttered or occluded scenes