---
layout: page
title: "Project 2"
description: "Fun with Filters and Frequencies"
img: assets/img/3_project/part2/2.4/irregular_blending.jpg
importance: 3
category: CS180
related_publications: false
---

## Project Overview

This project explores fundamental concepts in image processing through filters and frequency domain analysis. Part 1 focuses on implementing edge detection algorithms from scratch, comparing different convolution methods, and understanding how filters reveal image structure through derivatives and frequency analysis.

---

## Part 1: Filters and Edges

---

### Part 1.1: Convolution Implementation and Comparison

In this section, we implement convolution operations using pure NumPy and compare them with `scipy.signal.convolve2d`. We explore different implementation approaches and analyze their performance characteristics.

#### Implementation Approaches

**1. Four-Loop Convolution (Naive Implementation)**
```python
def convolve_4loops(image, kernel):

    img = np.array(img, dtype=float)
    kernel = np.array(kernel, dtype=float)
    
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2

    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), 'constant')
    
    result = np.zeros_like(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(kernel.shape[0]):
                for l in range(kernel.shape[1]):
                    result[i, j] += padded_img[i+k, j+l] * kernel[k, l]
    
    return result
```

**2. Two-Loop Convolution (Optimized)**  
```python
def conv_two_loops(img, kernel):

    img = np.array(img, dtype=float)
    kernel = np.array(kernel, dtype=float)
    
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), 'constant')
    
    result = np.zeros_like(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            result[i, j] = np.sum(region * kernel)
    
    return result
```

**3. SciPy Reference Implementation**
```python
from scipy.signal import convolve2d
result = convolve2d(image, kernel, mode='same')
```

#### Results and Analysis

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part1/1.1/1_original_image.png" title="Original Image" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Original Image</strong><br>
            Test image for convolution comparison
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part1/1.1/2_four_loops_convolution.png" title="Four-Loop Result" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Four-Loop Convolution</strong><br>
            Naive implementation result
        </div>
    </div>
</div>

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part1/1.1/3_two_loops_convolution.png" title="Two-Loop Result" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Two-Loop Convolution</strong><br>
            Optimized NumPy implementation
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part1/1.1/4_scipy_convolution.png" title="SciPy Result" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>SciPy Convolution</strong><br>
            Reference implementation result
        </div>
    </div>
</div>

#### Performance and Boundary Handling Analysis

**Runtime Comparison:**
- **Four-Loop Implementation**: 21.09 seconds
- **Two-Loop Implementation**: 8.71 seconds  
- **SciPy Implementation**: 0.14 seconds

**Key Insights:**
- SciPy is ~152× faster than four-loop implementation
- Vectorization provides ~2.4× speedup
- Custom implementations: zero-padding; SciPy: multiple boundary conditions

---

### Part 1.2: Partial Derivatives and Edge Detection

Edge detection through gradient computation reveals image structure by highlighting regions of rapid intensity change. We implement finite difference operators to compute partial derivatives and construct gradient-based edge maps.

#### Mathematical Foundation

**Finite Difference Operators:**

X-direction (vertical edges): $D_x = \begin{bmatrix} 1 & -1 \end{bmatrix}$

Y-direction (horizontal edges): $D_y = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$

**Gradient Magnitude:**
$$|\nabla I| = \sqrt{(\frac{\partial I}{\partial x})^2 + (\frac{\partial I}{\partial y})^2}$$

#### Results

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part1/1.2/1_original_cameraman.png" title="Original Cameraman" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Original Image</strong><br>
            Cameraman test image
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part1/1.2/2_partial_derivative_x.png" title="Partial Derivative X" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>∂I/∂x</strong><br>
            Vertical edge detection
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part1/1.2/3_partial_derivative_y.png" title="Partial Derivative Y" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>∂I/∂y</strong><br>
            Horizontal edge detection
        </div>
    </div>
</div>

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part1/1.2/4_gradient_magnitude.png" title="Gradient Magnitude" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Gradient Magnitude</strong><br>
            Combined edge strength
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part1/1.2/5_binary_edges.png" title="Binary Edge Map" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Binary Edge Map</strong><br>
            Thresholded at 0.15
        </div>
    </div>
</div>

#### Edge Detection Analysis

**Threshold Selection:**
- **Threshold = 0.15**: Balances edge completeness with noise suppression
- Lower values capture noise; higher values miss weak edges

**Trade-offs:**
- Finite difference method is noise-sensitive but preserves structural edges
- Simple thresholding may fragment thin edges

---

### Part 1.3: Gaussian and DoG Filters

Gaussian smoothing reduces noise before edge detection, while Difference of Gaussians (DoG) filters provide a more robust approach to edge detection by combining smoothing and differentiation in a single operation.

#### Filter Construction

**Gaussian Kernel Generation:**
```python
import cv2
# Create separable Gaussian kernels
kernel_1d = cv2.getGaussianKernel(ksize=5, sigma=1.0)
gaussian_2d = kernel_1d @ kernel_1d.T
```

**DoG Filter Construction:**
$$\text{DoG}_x = G_\sigma * D_x$$
$$\text{DoG}_y = G_\sigma * D_y$$

Where $G_\sigma$ is the Gaussian kernel and $D_x, D_y$ are finite difference operators.

#### Filter Visualizations

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part1/1.3/1_3_gaussian_kernel.png" title="Gaussian Kernel" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Gaussian Kernel</strong><br>
            σ = 1.0, size = 5×5
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part1/1.3/2_3_dog_x_filter.png" title="DoG X Filter" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>DoG X Filter</strong><br>
            Gaussian derivative in X
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part1/1.3/3_3_dog_y_filter.png" title="DoG Y Filter" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>DoG Y Filter</strong><br>
            Gaussian derivative in Y
        </div>
    </div>
</div>

#### Application Results

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part1/1.3/5_3_gaussian_blurred.png" title="Gaussian Smoothed" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Gaussian Smoothed</strong><br>
            Noise reduction preprocessing
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part1/1.3/6_3_dog_gradient.png" title="DoG Gradient" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>DoG Gradient Magnitude</strong><br>
            Single-step edge detection
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part1/1.3/4_3_edge_comparison.png" title="Method Comparison" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Method Comparison</strong><br>
            Finite difference vs DoG
        </div>
    </div>
</div>

#### Comparison: Finite Difference vs. DoG Method

**Finite Difference:** Simple but noise-sensitive, requires separate smoothing
**DoG Method:** Built-in noise suppression, single operation, parameter-dependent

**Key Results:**
- DoG produces cleaner edge maps with better noise robustness
- DoG is computationally more efficient (single vs. two-step process)
- Both methods provide similar edge localization accuracy

---

## Part 2: Fun with Frequencies

---

### Part 2.1: Image "Sharpening" with Unsharp Mask Filter

The unsharp mask filter is a classic image sharpening technique that enhances edges and fine details by amplifying high-frequency components. Despite its name, the process involves creating a "mask" from a blurred (unsharp) version of the image.

#### Mathematical Foundation

The unsharp mask operation can be expressed as:

$$\text{Sharpened} = \text{Original} + \alpha \times (\text{Original} - \text{Blurred})$$

Where:
- $\alpha$ is the sharpening strength parameter
- $(\text{Original} - \text{Blurred})$ represents the high-frequency details

This can be rewritten as:
$$\text{Sharpened} = (1 + \alpha) \times \text{Original} - \alpha \times \text{Blurred}$$

#### How Unsharp Mask Works

1. **Blur the original image** using a Gaussian filter to remove high frequencies
2. **Extract high frequencies** by subtracting the blurred version from the original
3. **Add amplified high frequencies** back to the original image

The relationship to frequency domain:
- **Low frequencies**: Preserved from the original image
- **High frequencies**: Amplified by factor $\alpha$
- **Result**: Enhanced edge contrast and detail sharpness

#### Results

**Taj Mahal Sharpening:**

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.1/taj.jpg" title="Original Taj Mahal" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Original Image</strong><br>
            Taj Mahal photograph
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.1/taj_blurred.jpg" title="Blurred Taj" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Blurred Version</strong><br>
            Gaussian blur (σ = 2.0)
        </div>
    </div>
</div>

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.1/taj_high_freq.jpg" title="High Frequency" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>High Frequency Details</strong><br>
            Original - Blurred
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.1/taj_sharpened.jpg" title="Sharpened Taj" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Sharpened Result</strong><br>
            α = 1.0, enhanced architectural details
        </div>
    </div>
</div>

**Building Architecture Sharpening:**

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.1/building.png" title="Original Building" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Original Building</strong><br>
            Modern architecture
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.1/building_blur_1.jpg" title="Pre-blurred Building" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Pre-blurred Version</strong><br>
            Gaussian blur (σ = 1.0)
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.1/blurred_image.jpg" title="Blurred Building" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Further Blurred</strong><br>
            For unsharp mask (σ = 2.0)
        </div>
    </div>
</div>

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.1/high_freq_image.jpg" title="High Frequency Building" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>High Frequency Details</strong><br>
            Pre-blurred - Further Blurred
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.1/sharpened_image.jpg" title="Sharpened Building" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Sharpened Result</strong><br>
            α = 1.0, applied to pre-blurred image
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        <div class="caption" style="text-align: center; margin-top: 50%;">
            <strong>Process Flow:</strong><br>
            Original → Pre-blur (σ=1) → Unsharp mask → Final result
        </div>
    </div>
</div>

#### Analysis: Varying Sharpening Amount

**α Parameter Effects:**
- **α = 0.5**: Subtle enhancement
- **α = 1.0**: Moderate sharpening (optimal balance)
- **α = 1.5**: Strong sharpening
- **α > 2.0**: Over-sharpening artifacts

**Key Points:** Unsharp mask enhances edges and fine textures; excessive values cause halos and noise amplification.

---

### Part 2.2: Hybrid Images

Hybrid images are static images that change in interpretation based on viewing distance. They combine the high-frequency components of one image with the low-frequency components of another, creating a fascinating perceptual illusion.

#### Technical Approach

**Process Overview:**
1. **Image Alignment**: Register two images to the same coordinate system
2. **Frequency Separation**: Apply complementary filters (low-pass and high-pass)
3. **Combination**: Add filtered components to create hybrid result
4. **Analysis**: Examine frequency domain properties

**Mathematical Foundation:**
$$\text{Hybrid} = \text{LowPass}(\text{Image}_1) + \text{HighPass}(\text{Image}_2)$$

#### Featured Example: Derek + Nutmeg

**Complete Process Visualization:**

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.2/DerekPicture.jpg" title="Derek Original" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Derek (Original)</strong><br>
            Source for low frequencies
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.2/nutmeg.jpg" title="Nutmeg Original" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Nutmeg (Original)</strong><br>
            Source for high frequencies
        </div>
    </div>
</div>

**Frequency Domain Analysis:**

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.2/fft_low_freq_filtered.jpg" title="Derek FFT" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Derek FFT</strong><br>
            Frequency spectrum
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.2/fft_high_freq_filtered.jpg" title="Nutmeg FFT" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Nutmeg FFT</strong><br>
            Frequency spectrum
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.2/fft_hybrid_result.jpg" title="Hybrid FFT" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Hybrid FFT</strong><br>
            Combined spectrum
        </div>
    </div>
</div>

**Filtered Components:**

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.2/low_freq_filtered.jpg" title="Derek Low-pass" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Derek Low-pass Filtered</strong><br>
            Cutoff frequency: 15 cycles/image
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.2/high_freq_filtered.jpg" title="Nutmeg High-pass" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Nutmeg High-pass Filtered</strong><br>
            Cutoff frequency: 15 cycles/image
        </div>
    </div>
</div>

**Final Hybrid Result:**

<div class="row justify-content-center">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.2/hybrid1.jpg" title="Derek-Nutmeg Hybrid" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Derek + Nutmeg Hybrid</strong><br>
            View from distance to see Derek, up close to see Nutmeg
        </div>
    </div>
</div>

#### Additional Hybrid Examples

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.2/hybrid2.jpg" title="Custom Hybrid 1" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Custom Hybrid 1</strong><br>
            Up close: Mona Lisa, From distance: Einstein
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.2/hybrid3.jpg" title="Custom Hybrid 2" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Custom Hybrid 2</strong><br>
            Up close: David sculpture, From distance: Statue of Liberty
        </div>
    </div>
</div>

#### Cutoff Frequency Selection

**Parameter Guidelines:**
- **< 10 cycles/image**: Insufficient detail
- **10-20 cycles/image**: Optimal range
- **> 25 cycles/image**: Competing details

**Success Factors:** Proper alignment, complementary content, contrast matching

---

### Part 2.3: Gaussian and Laplacian Stacks

Multi-resolution image analysis using Gaussian and Laplacian pyramids enables sophisticated blending operations that preserve natural transitions across different frequency bands.

#### Mathematical Framework

**Gaussian Stack:**
$$G_0 = \text{Original Image}$$
$$G_k = G_{k-1} * \text{Gaussian}(\sigma_k)$$

**Laplacian Stack:**
$$L_k = G_k - G_{k+1}$$
$$L_{\text{final}} = G_{\text{final}}$$

#### Orange and Apple Blending Process


**Multi-resolution Blending Process:**

The multi-resolution blending technique operates by decomposing images into frequency bands using Laplacian stacks, blending each band separately with the mask at corresponding resolution levels, and then reconstructing the final result. This approach ensures smooth transitions across all spatial frequencies.

**Process Steps:**
1. **Decomposition**: Both source images are decomposed into Laplacian stacks (frequency bands)
2. **Mask Processing**: The blending mask is also decomposed into a Gaussian stack
3. **Band-wise Blending**: Each frequency band is blended using the corresponding mask level
4. **Reconstruction**: All blended bands are summed to create the final seamless result

<div class="row justify-content-center">
    <div class="col-md-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.3/visualization.jpg" title="Blending Visualization" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Multi-resolution Blending Visualization</strong><br>
            Shows the complete process: original images, Laplacian decomposition, mask-based blending at each level, and final reconstruction. Each row represents a different frequency band, demonstrating how details at different scales are seamlessly combined.
        </div>
    </div>
</div>

**Final Reconstructed Result:**

<div class="row justify-content-center">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.3/reconstructed_image.jpg" title="Oraple Result" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>"Oraple" - Blended Result</strong><br>
            Seamless multi-frequency blending
        </div>
    </div>
</div>

#### Multi-resolution Blending Results

The multi-resolution blending technique demonstrates superior performance, showing:

1. **Frequency Band Preservation**: Each level maintains specific spatial frequency information
2. **Natural Transitions**: Smooth blending without visible seams
3. **Detail Preservation**: Fine textures maintained across the blend boundary
4. **Scalability**: Technique works across different image sizes and content types

---

### Part 2.4: Multiresolution Blending with Irregular Masks

Advanced blending using irregular masks enables more sophisticated image compositions that follow natural object boundaries rather than simple geometric shapes.

#### Custom Mask Creation Method

This project implements an **Interactive Drawing-Based Mask Creation** system that provides real-time, user-controlled mask generation:

**Key Features:**
- **Interactive Brush Tool**: Circular brush with adjustable size for precise painting
- **Real-time Visualization**: Semi-transparent overlay shows mask boundaries during creation
- **Dual Editing Modes**: Paint (left-click) and erase (right-click) functionality
- **Automatic Smoothing**: Gaussian blur post-processing ensures smooth transitions

The system seamlessly integrates with multi-resolution pyramid blending, automatically generating mask pyramids for each frequency band while preserving user-defined boundaries.

#### Original Source Images

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.4/frog.png" title="Frog Original" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Frog and Cucumber(Original)</strong><br>
            Source image for irregular blending
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.4/dt.jpg" title="Man Original" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>David Tao (Original)</strong><br>
            Chinese singer - source for base portrait
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.4/woman.jpg" title="Woman Original" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Yifei Liu (Original)</strong><br>
            Chinese actress - source for facial features
        </div>
    </div>
    
</div>

#### Results

**Creative Irregular Blending:**

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.4/irregular_blending.jpg" title="Complex Irregular Blend" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Complex Irregular Blend</strong><br>
            Multiple irregular regions
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.4/eye.jpg" title="Eye Blending" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Eye Feature Blending</strong><br>
            Woman's eyes transplanted to man's face using irregular mask
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3_project/part2/2.4/mouth.jpg" title="Mouth Blending" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Mouth Feature Blending</strong><br>
            Woman's mouth transplanted to man's face using irregular mask
        </div>
    </div>
    
</div>

---
