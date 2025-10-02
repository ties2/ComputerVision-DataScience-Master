# Digital Image Processing Fundamentals and Techniques

# chap 1 : (17- 46 page)

- Understand the concept of a digital image.
- Have a broad overview of the historical underpinnings of the field of digital image processing.
- Understand the definition and scope of digital image processing.
- Know the fundamentals of the electromagnetic spectrum and its relationship to image
generation.
- Be aware of the different fields in which digital image processing methods are applied.
- Be familiar with the basic processes involved
in image processing.
- Be familiar with the components that make
up a general-purpose digital image processing system.
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

**Components of a general-purpose image processing system**

- computer
- mass storage
- image displays
- hardcopy
- Specialized image processing hardware
- Image processing software
- Image sensors

---
# chap 2: digital image fundementals
(47 -  page)

* Have an understanding of some important
functions and limitations of human vision.
* Be familiar with the electromagnetic energy
spectrum, including basic properties of light.
* Know how digital images are generated and
represented.
* Understand the basics of image sampling and
quantization.
* Be familiar with spatial and intensity resolu-
tion and their effects on image appearance.
* Have an understanding of basic geometric
relationships between image pixels.
* Be familiar with the principal mathematical
tools used in digital image processing.
* Be able to apply a variety of introductory dig-
ital image processing techniques.

## 2.1 ELEMENTS OF VISUAL PERCEPTION

Human Visual System (Section 2.1): Summarizes visual perception, including image formation in the eye, and its capabilities for brightness adaptation and discrimination. It notes that while digital processing is mathematical, human intuition and visual judgment are crucial, and electronic sensors often exceed the eye's resolution.


## 2.2 LIGHT AND THE ELECTROMAGNETIC SPECTRUM

Discusses light, the electromagnetic spectrum, and their imaging characteristics.

A photon is the fundamental particle, or quantum, of electromagnetic (EM) radiation, including visible light. It's a discrete packet of energy that has:

Zero rest mass and no electric charge.

It travels at the speed of light (c) in a vacuum.

It exhibits wave-particle duality, meaning it has properties of both a particle and a wave.

Photon and Wavelength Relationship
The wavelength (λ) is a characteristic of the photon's wave nature, representing the spatial distance over which the wave's shape repeats.

The photon's energy (E) is inversely proportional to its wavelength. This relationship is described by the Planck-Einstein relation:

E= 
hc /
λ

​	
 
Where:

E is the energy of the photon (in Joules or electron-volts).

h is Planck's constant (6.626×10 
−34
  J⋅s).

c is the speed of light (2.998×10 
8
  m/s).

λ is the wavelength (in meters).

This inverse relationship means:

Short wavelength (e.g., gamma rays, X-rays, blue light) corresponds to high frequency and high energy photons.

Long wavelength (e.g., radio waves, microwaves, red light) corresponds to low frequency and low energy photons.

The relationship between wavelength (λ), frequency (f), and the speed of light (c) is also fundamental: c=fλ, which shows that wavelength and frequency are also inversely related.

https://www.youtube.com/watch?v=tbaTbESqycc

## 2.3 IMAGE SENSING AND ACQUISITION

Explains imaging sensors and their use in generating digital images.

1. Image Acquisition Using a Single Sensing Element
This method uses a single sensor (like a photodiode, which outputs a voltage proportional to light intensity, often with a filter for color selectivity).

Mechanism: A 2-D image is created by using precise mechanical relative displacement in both x and y directions between the sensor and the object/film.

Example: A high-precision drum scanner where a film negative rotates in one dimension (by the drum) and the sensor moves in the perpendicular dimension (by a lead screw). Light passes through the film, is modulated by its density, and is then captured and digitized by the sensor.

Pros & Cons: It produces high-resolution images but is slow and generally not portable.

Nomenclature: Systems that acquire images by light passing through the medium are called transmission microdensitometers; those using reflected light are reflection microdensitometers.

2. Image Acquisition Using Sensor Strips
This method uses an in-line sensor strip (a 1-D array of sensors) to capture one line of the image at a time.

Mechanism (Planar Imaging): Motion perpendicular to the strip provides the second dimension, completing the 2-D image.

Examples:

Flatbed Scanners: The most common use.

Airborne Imaging: The strip is mounted perpendicular to the direction of flight to image a geographical area.

Mechanism (Cross-Sectional Imaging): Sensors in a ring configuration are used with a rotating X-ray source to capture data used for Computerized Axial Tomography (CAT), which produces cross-sectional ("slice") images. Reconstruction algorithms are needed to convert the raw sensed data into a meaningful image.