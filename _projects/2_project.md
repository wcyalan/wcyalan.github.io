---
layout: page
title: "Project 1"
description: "Images of the Russian Empire: Colorizing the Prokudin-Gorskii Photo Collection"
img: assets/img/2_project/addition/emir_grad.jpg
importance: 2
category: CS180
related_publications: false
---

## Project Overview

This project reproduces and implements the automatic alignment and colorization process for the Prokudin‑Gorskii glass plate negatives. The goal is to take a grayscale scan containing three stacked channels, split them into B/G/R channels, align them properly, and merge them into a single color image. Several enhancement steps (cropping, contrast adjustment, white balance) are also included to improve the visual quality.

---

## Methodology

---

### 1. Channel Separation

* The input grayscale image is split into three equal vertical sections: Blue (top), Green (middle), and Red (bottom).
* Each channel is converted to floating-point precision for further processing.

---

### 2. Alignment

Two approaches are available depending on file type:

---

1. **Single-Level Alignment (default for non-TIFF images)**

   * Uses brute-force search within a range of ±15 pixels.
   * Alignment metrics supported:

     * **MSE (Mean Squared Error)**:
       $$\text{MSE} = \frac{1}{N} \sum (I_1 - I_2)^2$$

     * **NCC (Normalized Cross-Correlation)**: 
       $$\text{NCC} = \frac{\sum (I_1 - \bar{I_1})(I_2 - \bar{I_2})}{\sqrt{\sum (I_1 - \bar{I_1})^2 \sum (I_2 - \bar{I_2})^2}}$$

     * **Gradient Correlation**: 
       $$\text{Gradient NCC} = \text{NCC}(|\nabla I_1|, |\nabla I_2|)$$
       where $|\nabla I| = \sqrt{G_x^2 + G_y^2}$ using Sobel operators
       
       **Purpose and Advantages:**
       - **Robustness to Illumination Changes**: Edge features are less sensitive to brightness variations between color channels
       - **Better Performance on Challenging Images**: Particularly effective for images like Emir where channels have different brightness levels
       - **Enhanced Feature Detection**: Focuses on structural content rather than absolute pixel intensities
       - **Reduced Noise Sensitivity**: Edge-based matching is more robust to image noise and artifacts

       **Emir Alignment Method Comparison:**

---
       
<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2_project/addition/emir_ncc.jpg" title="NCC Alignment" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>NCC Alignment</strong><br>
            Standard cross-correlation
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2_project/addition/emir_grad.jpg" title="Gradient Correlation" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Gradient Correlation</strong><br>
            Edge-based alignment
        </div>
    </div>
</div>

---

   **Results - Single-Level Alignment:**

   **Visual Results:**

---

   <div class="row">
       <div class="col-md-4 mt-3 mt-md-0">
           {% include figure.liquid loading="eager" path="assets/img/2_project/result/cathedral_g(2,5)_r(3,12).jpg" title="Cathedral" class="img-fluid rounded z-depth-1" %}
           <div class="caption">
               <strong>Cathedral</strong><br>
               G offset: (2, 5)<br>
               R offset: (3, 12)
           </div>
       </div>
       <div class="col-md-4 mt-3 mt-md-0">
           {% include figure.liquid loading="eager" path="assets/img/2_project/result/monastery_g(2,-3)_r(2,3).jpg" title="Monastery" class="img-fluid rounded z-depth-1" %}
           <div class="caption">
               <strong>Monastery</strong><br>
               G offset: (2, -3)<br>
               R offset: (2, 3)
           </div>
       </div>
       <div class="col-md-4 mt-3 mt-md-0">
           {% include figure.liquid loading="eager" path="assets/img/2_project/result/tobolsk_g(3,3)_r(3,6).jpg" title="Tobolsk" class="img-fluid rounded z-depth-1" %}
           <div class="caption">
               <strong>Tobolsk</strong><br>
               G offset: (3, 3)<br>
               R offset: (3, 6)
           </div>
       </div>
   </div>

---

2. **Multi-Scale Pyramid Alignment (default for `.tif/.tiff`)**

   * Builds an image pyramid by downsampling channels.
   * Alignment starts at the coarsest level with a small search range.
   * Estimated shifts are propagated to finer levels, doubling at each step.
   * Dynamic search range increases with pyramid depth (from 3 up to 15 pixels).
   * Robust to large displacements and faster than brute-force on high-resolution images.

   **Results - Multi-Scale Pyramid Alignment:**

   **Visual Results:**

---

   <div class="row">
       <div class="col-sm-4 mt-3 mt-md-0">
           {% include figure.liquid loading="eager" path="assets/img/2_project/result/emir_g(23,49)_r(40,107).jpg" title="Emir" class="img-fluid rounded z-depth-1" %}
            <div class="caption">
               <strong>Emir</strong><br>
               G offset: (23, 49)<br>
               R offset: (40, 107)
           </div>
       </div>
       <div class="col-sm-4 mt-3 mt-md-0">
           {% include figure.liquid loading="eager" path="assets/img/2_project/result/harvesters_g(17,60)_r(13,123).jpg" title="Harvesters" class="img-fluid rounded z-depth-1" %}
            <div class="caption">
               <strong>Harvesters</strong><br>
               G offset: (17, 60)<br>
               R offset: (13, 123)
            </div>
       </div>
       <div class="col-sm-4 mt-3 mt-md-0">
           {% include figure.liquid loading="eager" path="assets/img/2_project/result/icon_g(17,42)_r(23,90).jpg" title="Icon" class="img-fluid rounded z-depth-1" %}
           <div class="caption">
               <strong>Icon</strong><br>
               G offset: (17, 42)<br>
               R offset: (23, 90)
           </div>
       </div>
   </div>
   <div class="row">
       <div class="col-sm-4 mt-3 mt-md-0">
           {% include figure.liquid loading="eager" path="assets/img/2_project/result/italil_g(22,38)_r(36,77).jpg" title="Italil" class="img-fluid rounded z-depth-1" %}
           <div class="caption">
               <strong>Italil</strong><br>
               G offset: (22, 38)<br>
               R offset: (36, 77)
           </div>
       </div>
       <div class="col-sm-4 mt-3 mt-md-0">
           {% include figure.liquid loading="eager" path="assets/img/2_project/result/lastochikino_g(-2,-2)_r(-8,75).jpg" title="Lastochikino" class="img-fluid rounded z-depth-1" %}
           <div class="caption">
               <strong>Lastochikino</strong><br>
               G offset: (-2, -2)<br>
               R offset: (-8, 75)
           </div>
       </div>
       <div class="col-sm-4 mt-3 mt-md-0">
           {% include figure.liquid loading="eager" path="assets/img/2_project/result/lugano_g(-16,41)_r(-29,92).jpg" title="Lugano" class="img-fluid rounded z-depth-1" %}
           <div class="caption">
               <strong>Lugano</strong><br>
               G offset: (-16, 41)<br>
               R offset: (-29, 92)
           </div>
       </div>
   </div>
   <div class="row">
       <div class="col-sm-4 mt-3 mt-md-0">
           {% include figure.liquid loading="eager" path="assets/img/2_project/result/melons_g(9,82)_r(12,178).jpg" title="Melons" class="img-fluid rounded z-depth-1" %}
           <div class="caption">
               <strong>Melons</strong><br>
               G offset: (9, 82)<br>
               R offset: (12, 178)
           </div>
       </div>
       <div class="col-sm-4 mt-3 mt-md-0">
           {% include figure.liquid loading="eager" path="assets/img/2_project/result/self_portrait_g(28,78)_r(36,176).jpg" title="Self Portrait" class="img-fluid rounded z-depth-1" %}
           <div class="caption">
               <strong>Self Portrait</strong><br>
               G offset: (28, 78)<br>
               R offset: (36, 176)
           </div>
       </div>
       <div class="col-sm-4 mt-3 mt-md-0">
           {% include figure.liquid loading="eager" path="assets/img/2_project/result/siren_g(-7,49)_r(-25,95).jpg" title="Siren" class="img-fluid rounded z-depth-1" %}
           <div class="caption">
               <strong>Siren</strong><br>
               G offset: (-7, 49)<br>
               R offset: (-25, 95)
           </div>
       </div>
   </div>
   <div class="row justify-content-center">
       <div class="col-md-4 mt-3 mt-md-0">
           {% include figure.liquid loading="eager" path="assets/img/2_project/result/three_generations_g(14,53)_r(10,111).jpg" title="Three Generations" class="img-fluid rounded z-depth-1" %}s
           <div class="caption">
               <strong>Three Generations</strong><br>
               G offset: (14, 53)<br>
               R offset: (10, 111)
           </div>
       </div>
       <div class="col-md-4 mt-3 mt-md-0">
           {% include figure.liquid loading="eager" path="assets/img/2_project/result/church_g(4,25)_r(-4,58).jpg" title="Church" class="img-fluid rounded z-depth-1" %}
           <div class="caption">
               <strong>Church</strong><br>
               G offset: (4, 25)<br>
               R offset: (-4, 58)
           </div>
       </div>
   </div>

---

### 3. Border Cropping

* **Option 1: Automatic Cropping** (`--auto-crop` flag)

  * An algorithm scans rows/columns for brightness transitions (black borders, white patches, content regions) to detect borders dynamically.
  * This ensures that noisy scan margins are removed adaptively.

* **Option 2: Fixed Margin Cropping** (default)

  * For `.tif/.tiff` files, a margin of 200 pixels is cropped.
  * For smaller files (e.g., `.jpg`), a margin of 30 pixels is applied.

**Results - Border Cropping:**

**Emir Automatic Cropping Comparison:**

---

<div class="row">
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2_project/addition/emir_uncropped.jpg" title="Before Cropping" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Before Cropping</strong><br>
            With original borders
        </div>
    </div>
    <div class="col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2_project/result/emir_g(23,49)_r(40,107).jpg" title="After Cropping" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>After Automatic Cropping</strong><br>
            Borders removed
        </div>
    </div>
</div>

---

### 4. White Balance Adjustments

(Optional, enabled with `--white-balance <method>`)

* **gray\_world**: Forces average R/G/B channel intensities to be equal.
* **white\_patch**: Scales channels based on the brightest 0.1% pixels.
* **histogram**: Performs percentile-based histogram stretching (2–98%).
* **none**: No adjustment.

**Results - White Balance:**

**White Balance Method Comparisons:**

#### Lastochikino White Balance Results:

<div class="row">
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2_project/result/lastochikino_g(-2,-2)_r(-8,75).jpg" title="Original" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Original</strong><br>
            No white balance
        </div>
    </div>
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2_project/addition/lastochikino_gray_world.jpg" title="Gray World" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Gray World</strong><br>
            Average channel equalization
        </div>
    </div>
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2_project/addition/lastochikino_white.jpg" title="White Patch" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>White Patch</strong><br>
            Brightest pixel scaling
        </div>
    </div>
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2_project/addition/lastochiko_histogram.jpg" title="Histogram" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Histogram</strong><br>
            Percentile stretching (2-98%)
        </div>
    </div>
</div>

#### Siren White Balance Results:

<div class="row">
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2_project/result/siren_g(-7,49)_r(-25,95).jpg" title="Original" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Original</strong><br>
            No white balance
        </div>
    </div>
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2_project/addition/siren_gray_world.jpg" title="Gray World" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Gray World</strong><br>
            Average channel equalization
        </div>
    </div>
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2_project/addition/siren_white_patch.jpg" title="White Patch" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>White Patch</strong><br>
            Brightest pixel scaling
        </div>
    </div>
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2_project/addition/siren_histogram.jpg" title="Histogram" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Histogram</strong><br>
            Percentile stretching (2-98%)
        </div>
    </div>
</div>

#### Church White Balance Results:

<div class="row">
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2_project/result/church_g(4,25)_r(-4,58).jpg" title="Original" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Original</strong><br>
            No white balance
        </div>
    </div>
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2_project/addition/church_gray_world.jpg" title="Gray World" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Gray World</strong><br>
            Average channel equalization
        </div>
    </div>
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2_project/addition/church_white_patch.jpg" title="White Patch" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>White Patch</strong><br>
            Brightest pixel scaling
        </div>
    </div>
    <div class="col-md-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2_project/addition/church_histogram.jpg" title="Histogram" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Histogram</strong><br>
            Percentile stretching (2-98%)
        </div>
    </div>
</div>

---

### 5. Output Generation

* After alignment and optional enhancement, the channels are stacked into an `[H, W, 3]` color image.
* Converted to `uint8` for saving.
* Output file names include alignment shifts, e.g.:
  `out_path/church_g(4,25)_r(-4,58).jpg`
* The program also displays the final aligned image.

---

## Usage Example

```bash
python colorize.py --input church.tif --white-balance gray_world --auto-crop
```

This will:

* Load `church.tif`
* Automatically crop noisy borders
* Align channels using pyramid + gradient correlation

---

## Challenges & Solutions

* **Large displacements on high‑res images slow brute force search**: solved with image pyramids. Runtime reduced from minutes to under a minute.
* **Color fringes / borders**: automatic cropping + white balance correction mitigated colored borders.
* **Channel brightness mismatch**: gradient matching and local normalization improved robustness.
