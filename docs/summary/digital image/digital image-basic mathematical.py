import numpy as np
import matplotlib.pyplot as plt

# --- 0. Setup: Create a simple synthetic image (a 2D discrete function) ---
# Image dimensions: 64x64 pixels, 8-bit grayscale (values 0-255)
SIZE = 64
save_path='./docs/summary/digital image'

# Create a base image with a diagonal gradient for visual interest
x, y = np.meshgrid(np.linspace(0, 1, SIZE), np.linspace(0, 1, SIZE))
base_image = (np.sin(x * np.pi * 5) * np.cos(y * np.pi * 5) + 1) / 2
base_image = (base_image * 255).astype(np.uint8)

# Create a noisy version for restoration/statistical tests
noisy_image = base_image + np.random.randint(-20, 20, (SIZE, SIZE), dtype=np.int16)
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

print("Original Image Matrix Shape:", base_image.shape)

# Helper function to display results
def display_result(images, titles, save_path):
    """Displays images side-by-side and optionally saves them."""
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    if len(images) == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        # Use 'gray' colormap for grayscale image data
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to: {save_path}")
        plt.close(fig) # Close the figure after saving to prevent memory leaks
    else:
        plt.show()


# =====================================================================
# 1. Linear Algebra (Vectors and Matrices)
# =====================================================================

# Operation: Simple Image Scaling (Brightness adjustment)
# This is a scalar multiplication applied element-wise across the matrix.
scale_factor = 1.5
bright_image = np.clip(base_image * scale_factor, 0, 255).astype(np.uint8)

# Operation: Matrix Subtraction (Difference Image)
# Used for motion detection or change detection.
# Simulate a slight shift (subtraction is pixel-wise)
shifted_image = np.roll(base_image, shift=2, axis=1) # Shift 2 pixels right
difference_image = np.abs(base_image.astype(np.int16) - shifted_image.astype(np.int16)).astype(np.uint8)

# Display Linear Algebra Results (SAVED to file)
display_result(
    [base_image, bright_image, difference_image],
    ['Original Image', f'Scaled (x{scale_factor})', 'Difference (Change Detected)'],
    save_path='./docs/summary/digital image/1_linear_algebra_results.png'
)


# =====================================================================
# 2. Set and Logical Operations
# =====================================================================

# Set Operation: Union (Equivalent to element-wise maximum)
# Create a simple rectangular mask (A)
mask_A = np.zeros_like(base_image, dtype=np.uint8)
mask_A[10:40, 10:40] = 255 # Region 1

# Create a second rectangular mask (B)
mask_B = np.zeros_like(base_image, dtype=np.uint8)
mask_B[30:50, 30:50] = 255 # Region 2 (overlaps Region 1)

# Union A U B is performed by element-wise maximum (max(A, B))
union_result = np.maximum(mask_A, mask_B)

# Logical Operation: NOT (Complement)
# In binary images, NOT(A) is 1-A. In 8-bit, 255-A.
complement_A = 255 - mask_A

# Display Set/Logical Results (SAVED to file)
display_result(
    [mask_A, mask_B, union_result, complement_A],
    ['Mask A', 'Mask B', 'Union (max(A, B))', 'Complement (NOT A)'],
    save_path='./docs/summary/digital image/2_set_logical_ops.png'
)


# =====================================================================
# 3. Filtering and Convolution (The core operation for spatial filtering)
# 5. Calculus and Differential Operators (Uses convolution with a kernel)
# =====================================================================

# In DIP, convolution is the key mathematical operation for both filtering (5)
# and applying differential operators (3). We define a simple 2D convolution
# function for demonstration.
def convolve_2d(image, kernel):
    """
    Simplified 2D convolution implementation for demonstration.
    Assumes odd-sized square kernel and handles boundaries by padding.
    """
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2
    
    # Pad the image to handle borders
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros_like(image, dtype=np.float32)

    # Perform the convolution (sum of products)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the neighborhood (the region under the kernel)
            neighborhood = padded_img[i:i+kH, j:j+kW]
            # Sum of products is the convolution result for this pixel
            output[i, j] = np.sum(neighborhood * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)


# 5. Filtering (Smoothing/Blurring)
# Kernel: 3x3 Averaging (Low-Pass Filter)
averaging_kernel = np.ones((3, 3), dtype=np.float32) / 9
blurred_image = convolve_2d(noisy_image, averaging_kernel)


# 3. Calculus/Differential Operators (Edge Detection)
# Kernel: Sobel X Operator (First-order derivative approximation in x direction)
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# Applying the kernel using convolution approximates the gradient (edge detection)
edges_x = convolve_2d(base_image, sobel_x)


# Display Convolution/Filtering/Derivative Results (SAVED to file)
display_result(
    [noisy_image, blurred_image, edges_x],
    ['Noisy Input', 'Filtered (Averaging/Blur)', 'Edges (Sobel X/Gradient)'],
    save_path='./docs/summary/digital image/3_convolution_calculus_results.png'
)


# =====================================================================
# 4. Transforms (Signal Processing)
# =====================================================================

# Operation: 2D Discrete Fourier Transform (FT)
# Converts image from spatial domain (pixels) to frequency domain.
f_transform = np.fft.fft2(base_image.astype(np.float32))

# Shift the zero-frequency component to the center for visualization (DC component)
f_shift = np.fft.fftshift(f_transform)

# Magnitude Spectrum: Log-scaled for better visualization (low values mask high ones)
magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-9) # adding epsilon to avoid log(0)

# The frequency domain image shows structure: high-frequency details (edges)
# are far from the center; low-frequency content (smooth areas) is near the center.

# Display Transform Results (SAVED to file)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(base_image, cmap='gray', vmin=0, vmax=255)
plt.title('Spatial Domain Image')
plt.axis('off')

plt.subplot(1, 2, 2)
# Display magnitude spectrum (not a typical image, so vmin/vmax range may vary)
plt.imshow(magnitude_spectrum, cmap='magma')
plt.title('Frequency Domain (Log Magnitude Spectrum)')
plt.axis('off')
plt.tight_layout()
plt.savefig('./docs/summary/digital image/4_fourier_transform_results.png')
print("Saved visualization to: 4_fourier_transform_results.png")
plt.close()


# =====================================================================
# 6. Probability and Statistics
# =====================================================================

# Tool: Histogram Processing (Statistical Distribution)
# The histogram shows the count of pixels for each intensity level (0-255).
# Used for contrast enhancement (Histogram Equalization) and noise analysis.

plt.figure(figsize=(10, 5))

# Calculate and plot the histogram of the original image
hist_original, bins = np.histogram(base_image.flatten(), bins=256, range=[0, 256])
plt.plot(hist_original, color='blue', label='Original Image')

# Calculate and plot the histogram of the difference image (edges)
# The difference image (edges_x) has many pixels near zero (dark areas)
hist_edges, bins = np.histogram(edges_x.flatten(), bins=256, range=[0, 256])
plt.plot(hist_edges, color='red', label='Edges Image (Concentrated around 0)')

plt.title('Image Histograms (Pixel Intensity Distribution)')
plt.xlabel('Pixel Intensity Value (0-255)')
plt.ylabel('Number of Pixels (Frequency)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('./docs/summary/digital image/5_histogram_statistics.png')
print("Saved visualization to: 5_histogram_statistics.png")
plt.close()

# Example of Statistic usage: Mean Intensity
mean_intensity = np.mean(base_image)
print(f"\nStatistical Property: Mean Intensity of Original Image = {mean_intensity:.2f}")

# Example of Statistic usage: Standard Deviation (Measure of contrast/spread)
std_dev = np.std(base_image)
print(f"Statistical Property: Standard Deviation of Original Image = {std_dev:.2f}")
# A higher standard deviation usually means higher contrast.
