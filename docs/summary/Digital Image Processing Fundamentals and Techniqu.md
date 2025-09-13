# Digital Image Processing Fundamentals and Techniques

subject: Image Processing
any other review: No
chap 1: Yes
chap 2: No
chap 3: No
start date: September 12, 2025

chap 1 : (17- ? page)

- Understand the concept of a digital image.
- Have a broad overview of the historical under-
pinnings of the field of digital image process-
ing.
- Understand the definition and scope of digi-
tal image processing.
- Know the fundamentals of the electromag-
netic spectrum and its relationship to image
generation.
- Be aware of the different fields in which digi-
tal image processing methods are applied.
- Be familiar with the basic processes involved
in image processing.
- Be familiar with the components that make
up a general-purpose digital image process-
ing system.
- Be familiar with the scope of the literature
where image processing work is reported

---

An image may be defined as a two-dimensional function, f x y ( , ), where x and y are spatial (plane) coordinates, and the amplitude of f at any pair of coordinates ( , is called the intensity or gray level of the image at that point. When x, y, and the intensity values of f are all finite, discrete quantities, we call the image a digital image.

digital image is composed of a finite number of ele-
ments:

- picture elements
- image elements
- pixels
- pels

The area of image analysis (also called image
understanding) is in between image processing and computer vision

- Low-level processes involve primitive operations such as image
preprocessing to reduce noise, contrast enhancement, and image sharpening. A low-level process is characterized by the fact that both its inputs and outputs are images.
- Mid-level processing of images involves tasks such as segmentation (partitioning an image into regions or objects), description of those objects to reduce them to a form suitable for computer processing, and classification (recognition) of individual
objects. A mid-level process is characterized by the fact that its inputs generally are images, but its outputs are attributes extracted from those images (e.g., edges,contours, and the identity of individual objects)
- higher-level processing involves “making sense” of an ensemble of recognized objects, as in image analysis, and, at the far end of the continuum, performing the cognitive functions normally
associated with human vision

---

## 1.2 THE ORIGINS OF DIGITAL IMAGE PROCESSING

## 1.3 EXAMPLES OF FIELDS THAT USE DIGITAL IMAGE PROCESSING

**GAMMA-RAY IMAGING**

Gamma-ray imaging is used in **nuclear medicine** (like bone scans and PET scans to find diseases) and **astronomical observations** (to study objects like supernova remnants). It detects radiation from injected isotopes in patients or from natural cosmic sources.

**X-RAY IMAGING**

X-rays are primarily used for **medical diagnostics** (like chest X-rays) and **industrial imaging**. They are generated in a vacuum tube where electrons strike an anode, producing penetrating radiation that creates images based on absorption through objects

Angiography uses X-rays and injected **contrast dye** to create images of blood vessels (angiograms) to detect blockages or irregularities. **CAT scans** use X-rays to generate 3D cross-sectional "slices" of the body. X-rays are also used in **industrial inspection** (e.g., checking circuit boards) and **astronomy**

Ultraviolet imaging is used in applications like **fluorescence microscopy** and **astronomy**. In microscopy, UV light excites electrons in a specimen, causing it to emit visible light (fluorescence), which is then captured to create high-contrast images for biological study.

Imaging in the **visible light** band is the most common, used everywhere from photography to microscopy. The **infrared band** is often used alongside it for applications like remote sensing, night vision, and industrial inspection, as it can detect heat and see through certain materials.

**IMAGING IN THE MICROWAVE BAND**

maging in the **microwave band** is primarily used for **radar**, which can create detailed images of the Earth's surface in any weather, day or night, by transmitting microwave pulses and processing the reflected signals

**IMAGING IN THE RADIO BAND**

The main uses of imaging with **radio waves** are in **medicine (MRI)** and **astronomy**. MRI uses magnetic fields and radio pulses to create detailed internal images of the body. In astronomy, radio waves reveal different features of celestial objects, like pulsars, that are not visible in other parts of the spectrum

**OTHER IMAGING MODALITIES**

- acoustic imaging
- electron microscopy
- synthetic (computer-gen-erated) imaging

## **1.4 FUNDAMENTAL STEPS IN DIGITAL IMAGE PROCESSING**

**Image acquisition**

Image acquisition is **the process of capturing visual information from the real world and converting it into a digital format that a computer can process**, such as taking a photo with a smartphone or a medical scan. This first critical step in image processing and analysis involves a physical sensing device, like a camera or a scanner, which captures energy (often light) from a scene and transforms it into an electrical signal, and then a digitizer that converts this signal into a digital image

**Image filtering and enhancement**

Image filtering and enhancement are **processes that use filters or operators to adjust pixel values**, with filtering primarily focusing on noise reduction and the manipulation of spatial frequencies for tasks like blurring or edge detection, while enhancement aims to improve an image's overall visual quality or suitability for human or machine analysis by tasks such as contrast adjustment, sharpening, and denoising

**Image restoration**

Image restoration involves techniques to improve the quality of degraded images. This can include removing noise, repairing damage, and enhancing details. Methods range from simple filters to complex algorithms using machine learning. The goal is to recover a clear and visually appealing image, often by addressing issues like blur, scratches, or faded colors. Successful restoration relies on understanding the source of image degradation and applying the appropriate techniques to correct it

**Color image multiresolution processing**

analyzes images at multiple scales using techniques like image pyramids and wavelets to improve features, reduce complexity, and enhance robustness. It involves representing and processing images at coarse to fine resolutions, capturing both short-range pixel-level details and long-range relationships for tasks such as feature detection, texture segmentation, and object recognition. Common multiresolution tools include wavelets and pyramid structures, which allow for more efficient algorithms by breaking down the image into different levels of detail.

**Wavelets and other image transforms**

Wavelets and other image transforms, such as [Discrete Cosine Transform (DCT)](https://www.google.com/search?sca_esv=0594ea3e66c3f3c4&cs=0&sxsrf=AE3TifNwDQ3znr5G3A9mBXfQbeEruiTZRg%3A1757747278973&q=Discrete+Cosine+Transform+%28DCT%29&sa=X&ved=2ahUKEwi05YTxltWPAxWtm_0HHZJxAjYQxccNegQIAhAB&mstk=AUtExfCdCQ-CYFz0SdUPWsKul2-HL05O0fJohT6Ho-EjHwVhzjA0oMfIvrlInHgj278eRWG11tAV-9Hr4UynNwbKpq5YboQkZmRhX0AGReC8jmRNBVnBJAK40jBZnWIArahZ084JBwCYa_Y7u_Q1z_iRZvTKAB7IumDREcmBwMF2npPmvqML2dp5LWkL6r8UD-ee9vbRk2Dc77aKJwz9Zl_JUlt_XNOOmxldGKQ0YSYMVNgkef-_O-cA28Ei2KoNVsOoFW5S2EzyArEnyiyHp-D1hrBv&csui=3), are **methods that decompose an image into different components, enabling applications like data compression and noise reduction**. Wavelet transforms decompose images using localized, wavelike oscillations called wavelets, which provide both time and frequency information simultaneously and achieve a multi-resolution analysis.

**Compression and watermarking**

Compression is the process of reducing file size by removing redundant data, while digital watermarking embeds information into multimedia content to protect copyright and ensure authenticity. The two are often studied together because compression techniques can weaken or remove embedded watermarks, so robust watermarks must be designed to survive compression and other signal processing attacks, such as those used on the internet. 

**Morphological processing**

Morphological processing is a set of operations in computer vision and image analysis that modifies an image by probing it with a structuring element, a small shape or template, to extract shape-based features like boundaries, skeletons, and regions.

**Segmentation**

Segmentation in computer vision is the process of dividing a digital image into multiple sets of pixels, known as segments or regions, to simplify its complexity and make it easier to analyze and understand. It involves assigning a class label to every pixel in an image based on shared characteristics like color, texture, or intensity, which helps in tasks such as object recognition, medical imaging analysis, and enabling autonomous vehicles to "see" and navigate the environment

**Feature extraction** 

the process of transforming raw pixel data from an image into a more digestible set of numerical features, such as edges, corners, textures, and object parts. This transformation reduces data complexity, enabling machines to effectively process visual information for tasks like object recognition, classification, and image segmentation by creating a "fingerprint" of the image's key characteristics. Methods range from manual or algorithmic approaches like SIFT to automatic feature learning in deep neural networks (CNNs).

**Image pattern classification**

Image pattern classification in computer vision involves categorizing images based on visual patterns. This is a core task encompassing diverse techniques, from traditional methods like feature extraction and handcrafted classifiers to deep learning approaches leveraging convolutional neural networks (CNNs)

---

## 1.5 COMPONENTS OF AN IMAGE PROCESSING SYSTEM

Two subsystems are required to acquire digital images. The first is a physical sensor that responds to the energy radiated by the object we wish to image. The second, called a digitizer, is a device for converting the output of the physical sensing device into digital form. For instance, in a digital video camera, the sensors (CCD chips) produce an electrical output proportional to light intensity. The digitizer converts these outputs to digital data