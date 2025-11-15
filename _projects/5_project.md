---
layout: page
title: "Project 5"
description: "Neural Radiance Fields (NeRF)"
img: assets/img/5_project/nerf_preview.jpg
importance: 5
category: CS180
related_publications: false
---

# Neural Radiance Fields (NeRF)

This project implements a Neural Radiance Field (NeRF) to represent and render 3D scenes from multi-view 2D images. Starting with camera calibration and 2D neural fields, we progressively build up to full 3D volumetric rendering.

## Project Overview

Neural Radiance Fields represent 3D scenes as continuous volumetric functions that map 3D coordinates and viewing directions to color and density values. The key components include:

- **Camera Calibration**: Recovering intrinsic and extrinsic camera parameters using ArUco markers
- **2D Neural Fields**: Learning to represent images as continuous functions
- **3D NeRF**: Volumetric rendering from multi-view images
- **Custom Dataset**: Training NeRF on self-captured objects

---

## Part 0: Camera Calibration and 3D Scanning

### 0.1: Camera Calibration

Camera calibration recovers the intrinsic parameters (focal length, principal point) and distortion coefficients of the camera using ArUco markers.

#### Calibration Process

1. **Capture Calibration Images**: 30-50 images of ArUco calibration tags from various angles and distances
2. **Detect ArUco Markers**: Use OpenCV's ArUco detector to find tag corners in each image
3. **Collect Correspondences**: Match 2D image points to 3D world coordinates
4. **Compute Intrinsics**: Use `cv2.calibrateCamera()` to recover camera matrix and distortion coefficients

#### Calibration Results

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part0/camera/1.jpg" title="Calibration View 1" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>View 1</strong>
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part0/camera/2.jpg" title="Calibration View 2" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>View 2</strong>
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part0/camera/3.jpg" title="Calibration View 3" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>View 3</strong>
        </div>
    </div>
</div>

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part0/camera/12.jpg" title="Calibration View 4" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>View 4</strong>
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part0/camera/25.jpg" title="Calibration View 5" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>View 5</strong>
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part0/camera/33.jpg" title="Calibration View 6" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>View 6</strong>
        </div>
    </div>
</div>

**Camera Intrinsics:**

$$
K = \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

### 0.2: 3D Object Scanning

Captured 30-50 images of a chosen object with a single ArUco tag for pose estimation.

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part0/object/1.jpg" title="Scan View 1" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>View 1</strong>
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part0/object/3.jpg" title="Scan View 2" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>View 2</strong>
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part0/object/4.jpg" title="Scan View 3" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>View 3</strong>
        </div>
    </div>
</div>

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part0/object/34.jpg" title="Scan View 4" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>View 4</strong>
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part0/object/44.jpg" title="Scan View 5" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>View 5</strong>
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part0/object/50.jpg" title="Scan View 6" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>View 6</strong>
        </div>
    </div>
</div>

### 0.3: Camera Pose Estimation

Using the calibrated intrinsics, we estimate camera pose for each image using Perspective-n-Point (PnP).

#### PnP Algorithm

Given 3D-2D correspondences and camera intrinsics, `cv2.solvePnP()` recovers the rotation and translation. The algorithm converts the axis-angle rotation vector to a rotation matrix using `cv2.Rodrigues()`, then constructs the camera-to-world transformation matrix by inverting the world-to-camera transformation.

#### Camera Frustum Visualization

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part0/camera_frustums_view1.jpg" title="Camera Frustums 1" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Camera Poses - View 1</strong><br>
            Estimated camera frustums showing capture positions
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part0/camera_frustums_view2.jpg" title="Camera Frustums 2" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Camera Poses - View 2</strong><br>
            Multiple viewpoints around the object
        </div>
    </div>
</div>

### 0.4: Image Undistortion and Dataset Creation

Removed lens distortion using `cv2.undistort()` and packaged data for NeRF training. Used `cv2.getOptimalNewCameraMatrix()` to handle black boundaries from undistortion, cropping images to the valid region of interest and updating the principal point accordingly. The final dataset is saved in `.npz` format containing training/validation images, camera poses, and focal length.

---

## Part 1: 2D Neural Field

Before tackling 3D NeRF, we start with a simpler 2D case: representing an image as a continuous neural field \(f(x, y) \rightarrow (r, g, b)\).

### 1.1: Network Architecture

**Multilayer Perceptron (MLP) with Positional Encoding:**

- **Input**: 2D pixel coordinates \((x, y)\) normalized to \([0, 1]\)
- **Positional Encoding**: Expands input dimensionality using sinusoidal functions
- **Output**: RGB color values in \([0, 1]\)

$$
\text{PE}(p) = [p, \sin(2^0 \pi p), \cos(2^0 \pi p), \ldots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p)]
$$

where \(L\) is the maximum frequency level.

**Network Structure:**

<div class="row">
    <div class="col-md-12 mt-3">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part1/mlp_img.jpg" title="2D MLP Architecture" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>2D Neural Field MLP Architecture</strong><br>
            Input (x, y) → PE(42-dim) → FC(256) → ReLU → FC(256) → ReLU → FC(256) → ReLU → FC(3) → Sigmoid → RGB
        </div>
    </div>
</div>

The network uses three fully connected layers with ReLU activations, followed by a Sigmoid activation to constrain outputs to valid color range [0, 1].

### 1.2: Training Process

**Hyperparameters:**
- Learning Rate: 1e-2
- Optimizer: Adam
- Batch Size: 10,000 pixels per iteration
- Iterations: 1000-3000
- Loss: MSE between predicted and ground truth colors

**PSNR Metric:**

$$
\text{PSNR} = -10 \cdot \log_{10}(\text{MSE})
$$

### 1.3: Training Results

<div class="row">
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part1/fox.jpg" title="Ground Truth" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Ground Truth</strong>
        </div>
    </div>
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part1/neural_image_iter_50.png" title="Iteration 50" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Iteration 50</strong>
        </div>
    </div>
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part1/neural_image_iter_100.png" title="Iteration 100" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Iteration 100</strong>
        </div>
    </div>
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part1/neural_image_iter_500.png" title="Iteration 500" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Iteration 500</strong>
        </div>
    </div>
</div>

<div class="row">
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part1/neural_image_iter_1000.png" title="Iteration 1000" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Iteration 1000</strong>
        </div>
    </div>
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part1/neural_image_iter_3000.png" title="Iteration 3000" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Iteration 3000</strong>
        </div>
    </div>
</div>

<div class="row mt-3">
    <div class="col-md-12">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part1/neural_image_training_curves.png" title="PSNR Curve" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Training PSNR Curve</strong><br>
            PSNR improves steadily as the network learns the image
        </div>
    </div>
</div>

### 1.4: Hyperparameter Analysis

**Effect of Positional Encoding Frequency (L):**

<div class="row">
    <div class="col-md-12 mt-3">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part1/ablation_L_comparison.png" title="Frequency Comparison" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Positional Encoding Frequency Comparison</strong><br>
            Different L values affect the network's ability to capture high-frequency details. Lower L values produce blurry results, while higher L values enable sharp reconstruction.
        </div>
    </div>
</div>

**Effect of Network Width:**

<div class="row">
    <div class="col-md-12 mt-3">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part1/ablation_width_comparison.png" title="Width Comparison" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Network Width Comparison</strong><br>
            Network capacity affects reconstruction quality. Narrow networks struggle with complex patterns, while wider networks provide better representation capacity.
        </div>
    </div>
</div>

---

## Part 2: 3D Neural Radiance Field

### 2.1: Ray Generation

#### Coordinate Transformations

**Camera to World Transformation:**

The transformation between camera space and world space is:

$$
\begin{bmatrix}
\mathbf{x}_w \\
1
\end{bmatrix} = \begin{bmatrix}
R & \mathbf{t} \\
\mathbf{0}^T & 1
\end{bmatrix} \begin{bmatrix}
\mathbf{x}_c \\
1
\end{bmatrix}
$$

where the camera-to-world (c2w) transformation matrix is a 4×4 matrix containing rotation \(R\) (3×3) and translation \(\mathbf{t}\) (3×1).

**Pixel to Camera Coordinates:**

Given intrinsic matrix \(K\) and pixel coordinates \((u, v)\):

$$
\mathbf{x}_c = s \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

**Ray Generation:**

For each pixel, we compute:
- **Ray Origin**: \(\mathbf{o} = \mathbf{t}\) (camera position in world space)
- **Ray Direction**: \(\mathbf{d} = \frac{R \cdot K^{-1} [u, v, 1]^T}{\|R \cdot K^{-1} [u, v, 1]^T\|}\)

The implementation adds 0.5 to pixel coordinates to account for pixel centers, transforms from pixel to camera coordinates using the inverse intrinsic matrix, then to world coordinates using the camera-to-world transformation.

### 2.2: Sampling Strategy

**Sampling Rays:** Randomly sample 10,000 rays per iteration from all training images.

**Sampling Points Along Rays:**

For each ray, sample \(N\) points between near and far planes:

$$
\mathbf{p}_i = \mathbf{o} + t_i \mathbf{d}, \quad t_i \in [t_{\text{near}}, t_{\text{far}}]
$$

Stratified sampling with random perturbation during training ensures every location along the ray is explored, preventing overfitting to discrete sample locations.

### 2.3: Rays and Samples Visualization

<div class="row">
    <div class="col-md-12 mt-3">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part2/rays_cameras.jpg" title="Rays and Cameras" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Camera Frustums with Sampled Rays</strong><br>
            100 rays sampled from training cameras
        </div>
    </div>
</div>

<div class="row">
    <div class="col-md-12 mt-3">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part2/render.png" title="3D Samples" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Sample Points in 3D Space</strong><br>
            Points sampled along rays for volume rendering
        </div>
    </div>
</div>

### 2.4: NeRF Network Architecture

**Input:**
- 3D position \(\mathbf{x}\) with positional encoding (\(L=10\))
- View direction \(\mathbf{d}\) with positional encoding (\(L=4\))

**Output:**
- Density \(\sigma \geq 0\) (ReLU activation)
- RGB color \(\mathbf{c} \in [0, 1]^3\) (Sigmoid activation)

**Network Structure:**

<div class="row">
    <div class="col-md-12 mt-3">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part2/mlp_nerf.png" title="NeRF MLP Architecture" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>NeRF Network Architecture</strong><br>
            Position PE(63-dim) → FC(256) → ReLU → FC(256) → ReLU → [Concat Position PE] → 
            FC(256) → ReLU → FC(256) → ReLU → FC(256+1) → Density (1) + Feature (256) → 
            [Concat Direction PE(27-dim)] → FC(128) → ReLU → FC(3) → Sigmoid → RGB
        </div>
    </div>
</div>

The network processes position encoding through multiple layers with a skip connection to inject input features in the middle. The density branch uses ReLU activation for non-negative values, while the color branch conditions on view direction and uses Sigmoid for valid RGB range.

### 2.5: Volume Rendering

**Continuous Volume Rendering Equation:**

$$
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) dt
$$

where transmittance \(T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) ds\right)\)

**Discrete Approximation:**

$$
\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i (1 - \exp(-\sigma_i \delta_i)) \mathbf{c}_i
$$

where:
- \(T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)\) is the accumulated transmittance
- \(\delta_i = t_{i+1} - t_i\) is the distance between samples
- \(\alpha_i = 1 - \exp(-\sigma_i \delta_i)\) is the opacity

The implementation computes opacity from density, calculates accumulated transmittance using cumulative product, and performs weighted sum of colors to produce the final rendered pixel color.

### 2.6: Training Results on Lego Dataset

**Training Configuration:**
- Dataset: Lego (200×200 resolution, 100 training views)
- Batch Size: 10,000 rays per iteration
- Learning Rate: 5e-4 (Adam optimizer)
- Iterations: 1000-5000
- Near/Far Planes: 2.0 / 6.0
- Samples per Ray: 64

<div class="row">
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part2/lego_iter_0100.jpg" title="Lego 100" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Iteration 100</strong>
        </div>
    </div>
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part2/lego_iter_0500.jpg" title="Lego 500" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Iteration 500</strong>
        </div>
    </div>
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part2/lego_iter_1000.jpg" title="Lego 1000" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Iteration 1000</strong>
        </div>
    </div>
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part2/lego_iter_5000.jpg" title="Lego 5000" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Iteration 5000</strong>
        </div>
    </div>
</div>

<div class="row mt-3">
    <div class="col-md-12">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part2/lego_psnr_curve.jpg" title="Lego PSNR" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Validation PSNR Curve</strong><br>
            Achieved >23 PSNR on validation set after 1000 iterations
        </div>
    </div>
</div>

#### Novel View Synthesis

<div class="row">
    <div class="col-md-12 mt-3">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part2/lego_novel_views.gif" title="Lego Novel Views" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Novel View Rendering</strong><br>
            Spherical camera trajectory around the lego bulldozer
        </div>
    </div>
</div>

---

## Part 2.6: Custom Object NeRF

### Dataset: Custom Captured Object

Trained NeRF on a custom object captured using the calibration pipeline from Part 0.

**Dataset Statistics:**
- Training Images: 35
- Validation Images: 5
- Image Resolution: 400×300 (resized from original)
- Near/Far Planes: 0.02 / 0.5 (adjusted for object scale)
- Samples per Ray: 64

**Training Modifications:**
- Adjusted near/far bounds based on object distance
- Increased samples per ray from 32 to 64 for better quality
- Resized images to manage training time
- Updated intrinsics matrix after resizing

### Training Progression

<div class="row">
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part2/custom_iter_0500.jpg" title="Custom 500" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Iteration 500</strong><br>
            Basic shape emerging
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part2/custom_iter_2000.jpg" title="Custom 2000" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Iteration 2000</strong><br>
            Details becoming clearer
        </div>
    </div>
    <div class="col-md-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part2/custom_iter_6000.jpg" title="Custom 6000" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Iteration 6000</strong><br>
            High-quality reconstruction
        </div>
    </div>
</div>

### Training Loss Curve

<div class="row">
    <div class="col-md-12">
        {% include figure.liquid loading="eager" path="assets/img/5_project/part2/training_curves.png" title="Custom Loss" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Training Loss Over Iterations</strong><br>
            Steady convergence with adjusted hyperparameters
        </div>
    </div>
</div>

### Novel View Synthesis

<div class="row">
    <div class="col-md-12 mt-3">
        <div style="transform: rotate(-90deg); transform-origin: center; width: 100%; display: flex; justify-content: center; align-items: center; min-height: 600px;">
            {% include figure.liquid loading="eager" path="assets/img/5_project/part2/chips.gif" title="Custom Novel Views" class="img-fluid rounded z-depth-1" %}
        </div>
        <div class="caption">
            <strong>360° Novel View Rendering</strong><br>
            Camera circling around the captured object
        </div>
    </div>
</div>

**Circular Camera Path Generation:**

The novel view synthesis uses a circular camera trajectory that looks at the object origin. For each angle, a camera-to-world matrix is computed using the look-at transformation, then rotated around the object. The trained NeRF model renders each view by sampling rays through the scene and applying volume rendering to produce the final images.

---

## Key Learnings

### Practical Considerations

- **Near/Far Bounds**: Must be carefully tuned for each scene. Too wide wastes samples in empty space; too narrow clips the object.

- **Image Resolution**: Higher resolution requires more memory and training time. Resizing images is often necessary for practical training.

- **Training Time**: NeRF training is computationally intensive. GPU acceleration is essential, and training can take hours even for small scenes.

- **View-Dependent Effects**: The direction-conditioned color prediction allows NeRF to model specular reflections and view-dependent appearance.
