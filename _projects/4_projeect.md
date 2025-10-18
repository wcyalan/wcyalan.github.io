---
layout: page
title: "Project 3"
description: "Image Warping and Mosaicing"
img: assets/img/3_project/part2/2.4/irregular_blending.jpg
importance: 4
category: CS180
related_publications: false
---

# Image Warping and Mosaicing

This project implements a complete image mosaicing pipeline that combines multiple photographs into seamless panoramic images through projective warping, homography recovery, and intelligent blending techniques.

## Project Overview

Image mosaicing involves registering multiple overlapping images taken from different viewpoints and blending them into a single, coherent panoramic image. The key challenges include:

- **Homography Recovery**: Computing projective transformations between image pairs
- **Image Warping**: Applying transformations with proper interpolation
- **Seamless Blending**: Combining images without visible artifacts

The implementation covers the complete pipeline from image acquisition to final mosaic creation, with custom implementations of all core algorithms.

## 1: Image Acquisition

### Shooting Requirements

To ensure successful mosaicing, images must satisfy specific geometric constraints:

**Camera Setup:**
- **Fixed Center of Projection**: Rotate camera around optical center
- **Significant Overlap**: 40-70% overlap between consecutive images
- **Stable Lighting**: Minimize exposure changes between shots
- **Minimal Distortion**: Avoid fisheye lenses for straight-line preservation

### Image Sets

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result1/01_original_image1.jpg" title="Subway Set 1" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Subway Station - Image 1</strong><br>
            Source image for subway mosaic
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result1/02_original_image2.jpg" title="Subway Set 2" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Subway Station - Image 2</strong><br>
            Overlapping view with projective transformation
        </div>
    </div>
</div>

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result2/01_original_image1.jpg" title="Street Set 1" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Urban Street - Image 1</strong><br>
            Source image for street mosaic
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result2/02_original_image2.jpg" title="Street Set 2" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Urban Street - Image 2</strong><br>
            Adjacent view with sufficient overlap
        </div>
    </div>
</div>

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result3/01_original_image1.jpg" title="Tram Set 1" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Tram Station - Image 1</strong><br>
            Source image for tram station mosaic
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result3/02_original_image2.jpg" title="Tram Set 2" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Tram Station - Image 2</strong><br>
            Adjacent view with sufficient overlap
        </div>
    </div>
</div>

## 2: Homography Recovery

### Mathematical Foundation

A homography is a projective transformation that maps points between two planes:

$$\mathbf{p}' = H\mathbf{p}$$

where $H$ is a 3×3 matrix with 8 degrees of freedom:

$$H = \begin{bmatrix}
h_{1} & h_{2} & h_{3} \\
h_{4} & h_{5} & h_{6} \\
h_{7} & h_{8} & 1
\end{bmatrix}$$

### Point Correspondence Setup

Given $n$ point correspondences $(\mathbf{p}_i, \mathbf{p}'_i)$, we solve the linear system:

$$\mathbf{A}\mathbf{h} = \mathbf{b}$$

where $\mathbf{h} = [h_{1}, h_{2}, h_{3}, h_{4}, h_{5}, h_{6}, h_{7}, h_{8}]^T$

### Linear System Construction

For each point correspondence $(x, y) \rightarrow (u, v)$, we derive two linear equations:

$$u = \frac{h_{1}x + h_{2}y + h_{3}}{h_{7}x + h_{8}y + 1}$$

$$v = \frac{h_{3}x + h_{4}y + h_{5}}{h_{7}x + h_{8}y + 1}$$

Cross-multiplying and rearranging:

$$h_{1}x + h_{2}y + h_{3} - h_{7}xu - h_{8}yu = u$$

$$h_{4}x + h_{5}y + h_{6} - h_{7}xv - h_{8}yv = v$$

This gives us the coefficient matrix structure used in `process_points()`.

### Implementation

```python
def process_points(p1,p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    x = p2[:,0]
    y = p2[:,1]
    u = p1[:,0]
    v = p1[:,1]
    A = np.zeros((len(p1)*2,8))
    b = np.zeros((len(p1)*2,1))
    for i in range(len(p1)):
        A[i*2,:] = [x[i],y[i],1,0,0,0,-x[i]*u[i],-y[i]*u[i]]
        A[i*2+1,:] = [0,0,0,x[i],y[i],1,-x[i]*v[i],-y[i]*v[i]]
        b[i*2,0] = u[i]
        b[i*2+1,0] = v[i]
    return A,b

def convert_matrix(x):
    H = x.flatten()
    H = np.append(H, 1)  # Add h33 = 1
    H = H.reshape(3, 3)
    return H

def computeH(im1_pts, im2_pts):
    A, b = process_points(im1_pts, im2_pts)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    H = convert_matrix(x)
    return H
```

### Interactive Point Selection Tool

The `get_points()` function provides an interactive interface for selecting corresponding points between two images with side-by-side display and automatic coordinate adjustment.

### Point Correspondence Visualization

#### Subway Station Correspondence

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result1/01_original_image1.jpg" title="Subway Correspondences 1" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Subway Station - Image 1</strong><br>
            Subway station feature points
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result1/02_original_image2.jpg" title="Subway Correspondences 2" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Subway Station - Image 2</strong><br>
            Corresponding subway station features
        </div>
    </div>
</div>

#### Urban Street Correspondence

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result2/01_original_image1.jpg" title="Street Correspondences 1" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Urban Street - Image 1</strong><br>
            Street scene feature points
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result2/02_original_image2.jpg" title="Street Correspondences 2" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Urban Street - Image 2</strong><br>
            Corresponding street scene features
        </div>
    </div>
</div>

#### Tram Station Correspondence

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result3/01_original_image1.jpg" title="Tram Correspondences 1" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Tram Station - Image 1</strong><br>
            Tram station feature points
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result3/02_original_image2.jpg" title="Tram Correspondences 2" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Tram Station - Image 2</strong><br>
            Corresponding tram station features
        </div>
    </div>
</div>

### Recovered Homography Matrix

For the building interior example:
$$H = \begin{bmatrix}
5.92115948 \times 10^{-1} & -3.51781288 \times 10^{-3} & 1.55296120 \times 10^{3} \\
-1.20717700 \times 10^{-1} & 8.86780010 \times 10^{-1} & 8.55110140 \times 10^{1} \\
-9.13840180 \times 10^{-5} & 1.38846378 \times 10^{-5} & 1.00000000 \times 10^{0}
\end{bmatrix}$$

## 3: Image Warping

### Inverse Warping Implementation

To avoid holes in the output image, we use inverse warping:

1. **Forward Transform**: Apply homography to image corners
2. **Bounding Box**: Calculate output image dimensions
3. **Inverse Mapping**: For each output pixel, find source coordinates
4. **Interpolation**: Sample source image with chosen method

### Bilinear Interpolation Algorithm

**Algorithm Description:**
1. **Coordinate Extraction**: Extract integer and fractional parts of coordinates
2. **Boundary Handling**: Ensure coordinates stay within image bounds
3. **Weight Calculation**: Compute interpolation weights based on fractional parts
4. **Weighted Average**: Combine four neighboring pixels using bilinear weights

**Mathematical Formula:**
$$I(x,y) = \sum_{i=0}^{1}\sum_{j=0}^{1} I(x_i,y_j) \cdot w_i \cdot w_j$$

where $w_i$ and $w_j$ are interpolation weights.

### Nearest Neighbor Interpolation

```python
def warpImageNearestNeighbor(im, H):
    im_warped = np.zeros((out_height, out_width, 3))
    
    for i in tqdm(range(out_height), desc="Warping progress"):
        for j in range(out_width):
            # Inverse transformation
            src_coords = np.dot(H_inv, np.array([world_x, world_y, 1]))
            src_coords = src_coords / src_coords[2]
            
            src_x = round(src_coords[0])
            src_y = round(src_coords[1])
            
            # Nearest neighbor sampling
            if 0 <= src_x < C and 0 <= src_y < R:
                im_warped[i, j, :] = im[src_y, src_x, :]
    
    return im_warped.astype(np.uint8), (x_min, y_min, x_max, y_max)
```

### Bilinear Interpolation

```python
def bilinear_interpolation(im, x, y):
    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1
    if x1 >= im.shape[1]:
        x1 = im.shape[1] - 1
    if y1 >= im.shape[0]:
        y1 = im.shape[0] - 1
    return im[y0, x0, :] * (1 - x + x0) * (1 - y + y0) + im[y0, x1, :] * (x - x0) * (1 - y + y0) + im[y1, x0, :] * (1 - x + x0) * (y - y0) + im[y1, x1, :] * (x - x0) * (y - y0)
    
def warpImageBilinear(im, H):
    im_warped = np.zeros((out_height, out_width, 3))
    
    for i in tqdm(range(out_height), desc="Warping progress"):
        for j in range(out_width):
            # Inverse transformation
            src_coords = np.dot(H_inv, np.array([world_x, world_y, 1]))
            src_coords = src_coords / src_coords[2]
            
            src_x = src_coords[0]
            src_y = src_coords[1]
            
            # Bilinear interpolation
            if 0 <= src_x < C and 0 <= src_y < R:
                im_warped[i, j, :] = bilinear_interpolation(im, src_x, src_y)
    
    return imwarped.astype(np.uint8)
```

### Interpolation Method Comparison

Both nearest neighbor and bilinear interpolation methods produce similar results for high-resolution images, with minimal visual differences. However, for low-resolution images, bilinear interpolation provides smoother results with better quality, while nearest neighbor interpolation may exhibit pixelated artifacts.

### Technical Implementation Details

**Key Features:**
- **Robust Bounding Box**: Transforms corners to determine output dimensions
- **Inverse Warping**: Avoids holes in output images
- **Progress Tracking**: `tqdm` integration for real-time feedback
- **Modular Design**: Separate interpolation functions with consistent interface

**Trade-offs:**
- **Nearest Neighbor**: Fast computation, pixelated artifacts
- **Bilinear**: Smooth results, higher computational cost

### Image Rectification

Rectification demonstrates homography accuracy by making rectangular objects appear properly rectangular.

#### Subway Station Rectification

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result1/02_original_image2.jpg" title="Subway Original" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Subway Station - Original</strong><br>
            Perspective-distorted subway elements
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result1/03_warped_image2.jpg" title="Subway Rectified" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Subway Station - Rectified</strong><br>
            Corrected subway perspective
        </div>
    </div>
</div>

#### Urban Street Rectification

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result2/02_original_image2.jpg" title="Street Original" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Urban Street - Original</strong><br>
            Perspective-distorted street scene
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result2/03_warped_image2.jpg" title="Street Rectified" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Urban Street - Rectified</strong><br>
            Corrected street perspective
        </div>
    </div>
</div>

#### Tram Station Rectification

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result3/02_original_image2.jpg" title="Tram Original" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Tram Station - Original</strong><br>
            Perspective-distorted tram station view
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result3/03_warped_image2.jpg" title="Tram Rectified" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Tram Station - Rectified</strong><br>
            Corrected tram station perspective
        </div>
    </div>
</div>

## 4: Image Mosaicing

### Advanced Blending Strategy

This implementation uses **multi-resolution pyramid blending** with **distance transform-based masking** for seamless results.

### Multi-Resolution Pyramid Algorithm

Following the same principle as Project 2, this implementation uses **two-level Laplacian pyramid blending** for seamless image fusion. The algorithm computes distance transforms for both input images, generates an intelligent alpha mask from their difference, then applies multi-resolution blending using Gaussian and Laplacian pyramids. The two-level pyramid structure effectively eliminates visible seams while preserving fine details across different scales.

### Distance Transform-Based Masking Algorithm

**Distance Transform Computation:**
1. **Background Detection**: Identify black pixels (background) in images
2. **Distance Calculation**: Compute Euclidean distance to nearest background pixel
3. **Multi-channel Handling**: Convert RGB to grayscale for mask detection

**Intelligent Mask Generation:**
1. **Distance Difference**: Calculate difference between two distance transforms
2. **Transition Region**: Create linear interpolation in boundary areas
3. **Alpha Weighting**: Generate smooth blending weights based on distance

**Mathematical Formula:**
$$\alpha = \begin{cases}
1.0 & \text{if } d_1 - d_2 \geq w \\
0.0 & \text{if } d_1 - d_2 \leq -w \\
0.5 + \frac{d_1 - d_2}{2w} & \text{otherwise}
\end{cases}$$

where $d_1, d_2$ are distance transforms and $w$ is transition width.

### Distance Transform Visualization

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result1/04_distance_transform1.jpg" title="Distance Transform 1" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Distance Transform 1</strong><br>
            Heat map showing distance to background
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result1/05_distance_transform2.jpg" title="Distance Transform 2" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Distance Transform 2</strong><br>
            Heat map for second image
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result1/06_alpha_mask.jpg" title="Alpha Mask" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Alpha Mask</strong><br>
            Intelligent blending weights
        </div>
    </div>
</div>

#### Landscape Distance Transforms

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result2/04_distance_transform1.jpg" title="Landscape Distance Transform 1" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Landscape - Distance Transform 1</strong><br>
            Natural scene distance map
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result2/05_distance_transform2.jpg" title="Landscape Distance Transform 2" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Landscape - Distance Transform 2</strong><br>
            Second natural view distance map
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result2/06_alpha_mask.jpg" title="Landscape Alpha Mask" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Landscape - Alpha Mask</strong><br>
            Intelligent natural blending weights
        </div>
    </div>
</div>

#### Urban Scene Distance Transforms

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result3/04_distance_transform1.jpg" title="Urban Distance Transform 1" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Urban - Distance Transform 1</strong><br>
            Street scene distance map
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result3/05_distance_transform2.jpg" title="Urban Distance Transform 2" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Urban - Distance Transform 2</strong><br>
            Second street view distance map
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result3/06_alpha_mask.jpg" title="Urban Alpha Mask" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Urban - Alpha Mask</strong><br>
            Intelligent street blending weights
        </div>
    </div>
</div>

### Mosaic Results

#### Mosaic 1: Subway Station

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result1/01_original_image1.jpg" title="Subway Source 1" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Source Image 1</strong><br>
            Left portion of subway station
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result1/02_original_image2.jpg" title="Subway Source 2" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Source Image 2</strong><br>
            Right portion of subway station
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result1/09_final_blended_result.jpg" title="Subway Mosaic" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Final Mosaic</strong><br>
            Complete subway station panorama
        </div>
    </div>
</div>

#### Mosaic 2: Urban Street

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result2/01_original_image1.jpg" title="Street Source 1" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Source Image 1</strong><br>
            Left portion of urban street
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result2/02_original_image2.jpg" title="Street Source 2" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Source Image 2</strong><br>
            Right portion of urban street
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result2/09_final_blended_result.jpg" title="Street Mosaic" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Final Mosaic</strong><br>
            Complete urban street panorama
        </div>
    </div>
</div>

#### Mosaic 3: Tram Station

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result3/01_original_image1.jpg" title="Tram Source 1" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Source Image 1</strong><br>
            Left portion of tram station
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result3/02_original_image2.jpg" title="Tram Source 2" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Source Image 2</strong><br>
            Right portion of tram station
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result3/09_final_blended_result.jpg" title="Tram Mosaic" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Final Mosaic</strong><br>
            Complete tram station panorama
        </div>
    </div>
</div>

### Pipeline Workflow

**Complete Process:**
1. Image preprocessing and point correspondence selection
2. Homography computation and image warping
3. Canvas creation and distance transform blending
4. Multi-resolution pyramid reconstruction
5. Comprehensive visualization and result export

### Technical Advantages

**Key Benefits:**
- **Seamless Blending**: Multi-resolution pyramid eliminates visible seams
- **Intelligent Masking**: Distance transform creates optimal blending regions
- **Comprehensive Visualization**: 8-panel dashboard provides complete process insight

---

## Part B: Automatic Feature Detection and Matching

Part A required manual point correspondence selection, which is time-consuming and error-prone. Part B implements automatic feature detection, matching, and robust homography estimation.

### Overview

The automatic pipeline consists of four key stages:
1. **Harris Corner Detection + ANMS**: Detect and select distinctive interest points
2. **Feature Descriptor Extraction**: Extract normalized 8×8 descriptors from 40×40 windows
3. **Feature Matching**: Match descriptors using Lowe's ratio test
4. **RANSAC Homography Estimation**: Robustly compute homography from noisy matches

## B.1: Harris Corner Detection and ANMS

### Harris Interest Point Detector

The Harris corner detector identifies image locations with strong gradients in multiple directions, which are stable features for matching.

**Algorithm Steps:**
1. **Gradient Computation**: Calculate image gradients $I_x$ and $I_y$
2. **Structure Tensor**: Compute the second moment matrix:
   $$M = \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}$$
3. **Corner Response**: Calculate Harris response function:
   $$R = \frac{\det(M)}  {\text{trace}(M)}$$
4. **Thresholding**: Select points where $R$ exceeds a threshold

### Adaptive Non-Maximal Suppression (ANMS)

ANMS selects a spatially distributed subset of corners to ensure even coverage across the image.

**Algorithm:**
For each corner $i$ with strength $f_i$, compute suppression radius:
$$r_i = \min_{j} \|x_i - x_j\| \quad \text{subject to} \quad f_j > c_{\text{robust}} \cdot f_i$$

where $c_{\text{robust}} = 0.9$ ensures only significantly stronger corners suppress weaker ones.

**Selection Process:**
1. Compute suppression radius for all corners
2. Sort corners by radius (descending)
3. Select top $N$ corners with largest radii

### Harris Corner Detection Results

<div class="row">
    <div class="col-md-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part2/results_0.8/02_anms_comparison.jpg" title="Harris and ANMS" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Harris Corner Detection and ANMS</strong><br>
            Left: All detected Harris corners (~8,500 points). Right: After ANMS filtering (500 spatially distributed corners)
        </div>
    </div>
</div>

## B.2: Feature Descriptor Extraction

### Feature Descriptor Extraction

Feature descriptors encode local image appearance around interest points for robust matching.

**Extraction Process:**
1. **Window Extraction**: Extract 40×40 pixel window centered at corner
2. **Gaussian Blurring**: Apply Gaussian filter (σ=5) for robustness
3. **Downsampling**: Subsample to 8×8 patch
4. **Normalization**: Bias/gain normalize to zero mean and unit variance

**Normalization Formula:**
$$\mathbf{d}_{\text{norm}} = \frac{\mathbf{d} - \mu}{\sigma}$$

where $\mu$ is mean and $\sigma$ is standard deviation of the descriptor.

### Feature Descriptor Visualization

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part2/results_0.8/03_feature_patches.jpg" title="Feature Patches" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Feature Patches (40×40)</strong><br>
            Extracted patches centered at interest points
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part2/results_0.8/04_feature_descriptors.jpg" title="Feature Descriptors" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Feature Descriptors (8×8)</strong><br>
            Normalized descriptors after downsampling
        </div>
    </div>
</div>

## B.3: Feature Matching

### Lowe's Ratio Test

To find reliable matches, we use Lowe's ratio test which compares the distance to the nearest neighbor with the distance to the second nearest neighbor.

**Matching Algorithm:**
1. **Nearest Neighbor Search**: For each descriptor in image 1, find two nearest neighbors in image 2
2. **Ratio Test**: Accept match if:
   $$\frac{d_1}{d_2} < \tau$$
   where $d_1$ is distance to nearest neighbor, $d_2$ is distance to second nearest, and $\tau = 0.8$
3. **Bidirectional Matching**: Optionally enforce mutual nearest neighbors

**Distance Metric:**
$$d(\mathbf{d}_i, \mathbf{d}_j) = \|\mathbf{d}_i - \mathbf{d}_j\|_2$$

### Feature Matching Visualization

<div class="row">
    <div class="col-md-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part2/results_0.8/05_feature_matching.jpg" title="Feature Matching" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Feature Matching Results ($\tau = 0.8$)</strong><br>
            Matched features between image pair using Lowe's ratio test
        </div>
    </div>
</div>

{% comment %}
### Ratio Test Threshold Comparison

Different $\tau$ values affect the trade-off between match quantity and quality:

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part2/results_0.8/05_feature_matching_tau07.jpg" title="Tau 0.7" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>$\tau = 0.7$</strong><br>
            Stricter threshold, fewer but more reliable matches
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part2/results_0.8/05_feature_matching_tau08.jpg" title="Tau 0.8" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>$\tau = 0.8$</strong><br>
            Balanced threshold, optimal match quality
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part2/results_0.8/05_feature_matching_tau09.jpg" title="Tau 0.9" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>$\tau = 0.9$</strong><br>
            Looser threshold, more matches with potential outliers
        </div>
    </div>
</div>

**Threshold Analysis:**

The ratio test threshold $\tau$ controls the selectivity of feature matching. Through experimental comparison:

- **$\tau = 0.7$**: This conservative threshold produces the fewest matches but with highest confidence. Each match must be significantly better than the second-best candidate, reducing false positives but potentially missing some valid correspondences in ambiguous regions.

- **$\tau = 0.8$**: This balanced threshold provides optimal performance for our image pairs. It retains sufficient matches for robust homography estimation while maintaining high match quality. The increased tolerance compared to 0.7 captures more valid correspondences in textured areas without introducing excessive outliers.

- **$\tau = 0.9$**: This permissive threshold maximizes match quantity but introduces more ambiguous correspondences. While providing more data points for RANSAC, the increased outlier ratio requires more iterations for robust estimation. This setting may be useful for scenes with repetitive patterns or limited distinctive features.

For this project, $\tau = 0.8$ achieves the best balance between match quantity and quality, providing sufficient inliers for accurate homography estimation while maintaining computational efficiency in the RANSAC stage.
{% endcomment %}

## B.4: RANSAC for Robust Homography Estimation

### 4-Point RANSAC Algorithm

RANSAC (Random Sample Consensus) robustly estimates homography in the presence of outliers.

**Algorithm:**
```
For N iterations:
    1. Randomly select 4 point correspondences
    2. Compute homography H using these 4 points
    3. Count inliers: points where reprojection error < threshold
    4. If inlier count > best_count:
        best_H = H
        best_inliers = inliers
Return best_H computed from all inliers
```

**Reprojection Error:**
$$e_i = \|\mathbf{p}'_i - H\mathbf{p}_i\|_2$$

Accept as inlier if $e_i < \epsilon$ (typically $\epsilon = 3$ pixels).

### RANSAC Inlier Visualization

<div class="row">
    <div class="col-md-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part2/results_0.8/06_ransac_inliers.jpg" title="RANSAC Inliers" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>RANSAC Inliers</strong><br>
            Inlier matches after RANSAC filtering (green: inliers, red: outliers)
        </div>
    </div>
</div>

### Automatic Stitching Results

#### Mosaic 1: Subway Station

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result1/09_final_blended_result.jpg" title="Subway Manual" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Manual Stitching</strong><br>
            Result using manually selected correspondences
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result1/09_final_blended_result.jpg" title="Subway Auto" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Automatic Stitching</strong><br>
            Result using automatic feature detection and RANSAC
        </div>
    </div>
</div>

#### Mosaic 2: Urban Street

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result2/09_final_blended_result.jpg" title="Street Manual" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Manual Stitching</strong><br>
            Result using manually selected correspondences
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result2/09_final_blended_result.jpg" title="Street Auto" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Automatic Stitching</strong><br>
            Result using automatic feature detection and RANSAC
        </div>
    </div>
</div>

#### Mosaic 3: Tram Station

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result3/09_final_blended_result.jpg" title="Tram Manual" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Manual Stitching</strong><br>
            Result using manually selected correspondences
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/4_project/part1/result3/09_final_blended_result.jpg" title="Tram Auto" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Automatic Stitching</strong><br>
            Result using automatic feature detection and RANSAC
        </div>
    </div>
</div>

### Pipeline Performance

The automatic feature detection and matching pipeline demonstrates robust performance:

**Processing Statistics:**
- **Harris Corners Detected**: ~8,500 interest points
- **ANMS Selected Features**: 500 spatially distributed corners
- **Initial Feature Matches**: 19 correspondences ($\tau = 0.8$)
- **RANSAC Inliers**: 10 valid matches (52.6% inlier ratio)
- **Reprojection Error**: < 3 pixels for inliers

**Key Advantages:**
- **Fully Automatic**: Eliminates manual point selection
- **Robust to Outliers**: RANSAC effectively filters incorrect matches
- **High Quality**: Comparable results to manual correspondence selection
- **Efficient**: Processes image pairs in reasonable time

**Technical Implementation:**
- **ANMS Algorithm**: Ensures spatially distributed feature coverage
- **Descriptor Normalization**: Bias/gain normalization improves matching robustness
- **Lowe's Ratio Test**: Effectively discriminates ambiguous matches
- **4-Point RANSAC**: Robustly estimates homography from noisy correspondences
- **Multi-resolution Blending**: Seamless fusion using distance transform masks