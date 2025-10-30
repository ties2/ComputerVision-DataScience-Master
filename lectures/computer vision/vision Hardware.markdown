# Vision Hardware

The "Vision Hardware" lecture emphasizes using physics and "analogue processing" getting the image right before it hits the sensor to create the best possible data for digital processing.

## Illumination

Proper lighting is the most critical factor. The choice of illumination can solve a problem before any code is written. 

### Techniques include:

* Back lighting: Silhouettes an object, perfect for measuring shape and size.

* Dark field: Uses low-angle light to highlight surface imperfections like scratches or engravings. The camera only sees light reflected off these features.

* Bright field (Full/Partial): Standard lighting where the camera captures the light reflected directly off the object's surface.

* Diffuse Lighting (Dome, Axial, Flat): Uses scattered, non-directional light to eliminate glare and shadows, ideal for shiny or reflective surfaces.

## Camera Parameters (The Exposure Triangle)

Three main camera parameters that must be balanced:

* Aperture (Iris Opening):

This controls the Depth of Field (DoF)—how much of the image is in focus from front to back.

It's measured in f-stops (e.g., F1.4 to F32).

    * A large aperture (small f-number, e.g., F1.4)  lets in more light but has a shallow DoF (blurry background).

    * A small aperture (large f-number, e.g., F32)  lets in less light but has a deep DoF (everything in focus).

* Shutter Speed (Exposure Time):

This controls motion blur.

It's measured in seconds (e.g., 1/1000s to 1/2s).

    * A fast shutter speed (e.g., 1/1000)  freezes motion but requires more light.

    * A slow shutter speed (e.g., 1/2)  creates motion blur but works well in low light.

* ISO / Gain:

This controls the sensor's sensitivity to light.

It's measured in ISO values (e.g., ISO 50 to ISO 25600).

    * Low ISO (e.g., ISO 100)  is less sensitive, requires more light, and produces a clean, noise-free image.

    * High ISO (e.g., ISO 25600)  is very sensitive, works in dark conditions, but introduces digital noise (grain).

* Optics and Lenses

* Pinhole Camera Model: This is the fundamental model for computer vision. It describes how a 3D point in the world P(X,Y,Z) is projected onto a 2D point p(x,y) on the camera's image plane through a "center of projection".

* Lens Formula: When adding optics (a lens), the relationship between the object, lens, and sensor is described by the Gaussian lens formula

## Filters

Filters are another form of "analogue processing" placed in front of the lens or sensor.

* Color Filters: Used to enhance contrast. For example, a red filter will lighten red objects and darken (block) other colors.

* Polarization Filter: Used to manage glare and reflections from surfaces like glass or water. It works by filtering out unpolarized light, often using two crossed polarizers.

* Bayer Filter: This filter is built into most color camera sensors. It's a mosaic of red, green, and blue filters placed over the sensor array so that each pixel captures the intensity for only one color. This "color coded" pattern is then "decoded" through a process called demosaicing to create a full RGB image

* Image sensors typically capture only one color (Red, Green, or Blue) at each pixel location. A Bayer filter is a common pattern placed over the sensor pixels to achieve this. The pattern specified here is GRBG:

```
G R G R ...
B G B G ...
G R G R ...
B G B G ...
...
```

This means your input image (a 2D NumPy array) has pixel values corresponding to this pattern. To get a full-color RGB image, we need to estimate the missing two color values at each pixel location. This process is called demosaicing or debayering.

bilinear interpolation and specifies that the output RGB image should be half the height and width of the input RAW image. This simplifies the process greatly, as we don't need complex interpolation for missing pixels within the original grid. Instead, for each 2x2 block in the RAW image, we determine the R, G, and B values for a single pixel in the output RGB image.

Steps (for GRBG pattern and H/2, W/2 output):

Consider a 2x2 block in the input image starting at (row, col) where row and col are even numbers:

image[row, col]      = Green (G1)

image[row, col+1]    = Red   (R)

image[row+1, col]    = Blue  (B)

image[row+1, col+1]  = Green (G2)

This 2x2 block corresponds to one pixel at (row/2, col/2) in the output RGB image.

Red Channel: The Red value comes directly from the top-right pixel: R = image[row, col+1].

Green Channel: The Green value is the average of the two green pixels in the block: G = (image[row, col] + image[row+1, col+1]) / 2.

Blue Channel: The Blue value comes directly from the bottom-left pixel: B = image[row+1, col].

We repeat this for all non-overlapping 2x2 blocks in the input image.

---

Correct artifacts in a RAW monochrome image caused by variations in pixel sensitivity (photo-response non-uniformity or PRNU) and sensor dark current/offset noise.

Concept: Ideally, if you point a camera with a perfect sensor at a perfectly uniform white surface, every pixel should record the same brightness value. Similarly, with the lens cap on (no light), every pixel should record zero. In reality, this doesn't happen due to manufacturing imperfections and thermal noise.

Dark Frame (dark): An image taken with the lens cap on (or in complete darkness) with the same exposure settings as the raw image. It captures the baseline signal (offset and dark current) that each pixel outputs even without light.

Flat Frame (flat): An image taken of a uniformly illuminated, featureless surface (like a lightbox or a clear sky) with the same exposure settings. It captures how each pixel responds differently to the same amount of light (PRNU). Pixels that are less sensitive will appear darker, and vice-versa.

Raw Image (raw): The actual image you want to correct.

Formula: The standard formula aims to normalize the pixel responses:

Corrected = Gain * (Raw - Dark) / (Flat - Dark)

Where:

(Raw - Dark) subtracts the baseline noise/offset.

(Flat - Dark) represents the actual sensitivity variation of each pixel (how much it responded above the dark level).

Dividing (Raw - Dark) by (Flat - Dark) normalizes the pixel's response relative to its sensitivity.

Gain is a scaling factor, often chosen as the average sensitivity Mean(Flat - Dark), to bring the overall brightness of the corrected image back to a reasonable level.

So the formula becomes:

Corrected = Mean(Flat - Dark) * (Raw - Dark) / (Flat - Dark)