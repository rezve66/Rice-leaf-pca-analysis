![Python](https://img.shields.io/badge/Python-3.9-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-ImageProcessing-green)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-PCA-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

# 🌾 Rice Leaf Infection Analysis using PCA

A Python-based image analysis pipeline that applies **Principal Component Analysis (PCA)** to rice leaf RGB images to highlight **infection-related patterns** using color features and vegetation indices.

This project processes a rice leaf image and generates:

* PCA feature loadings
* Spatial score maps for principal components
* False-color composite visualization

The goal is to **enhance subtle disease patterns in rice leaves** that are difficult to observe in standard RGB images.

---

# 📌 Project Overview

Plant diseases significantly impact crop production. Early detection through image analysis can help improve crop management.

This project explores how **unsupervised dimensionality reduction (PCA)** can extract meaningful patterns from leaf images by combining:

* RGB color channels
* Vegetation indices
* Normalized color ratios
* Color deconvolution features
* Gaussian blurred features

These features are analyzed using PCA to reveal hidden structures associated with **leaf health and infection patterns**.

---

# 🧠 Methodology

The pipeline consists of several key stages:

### 1️⃣ Image Loading

The input rice leaf image is loaded using **OpenCV** and converted from BGR to RGB format.

### 2️⃣ White Balance Correction

A **Gray-World assumption** is applied to normalize lighting conditions across the image.

### 3️⃣ Feature Extraction

Multiple features are generated from the image:

#### Color Channels

* R
* G
* B

#### Vegetation Indices

* **ExG (Excess Green)**
  Detects vegetation intensity.

* **ExR (Excess Red)**
  Helps differentiate disease spots.

* **NDI (Normalized Difference Index)**
  Highlights chlorophyll differences.

#### Normalized RGB

* r_norm
* g_norm
* b_norm

#### Color Deconvolution (HED)

Separates stain-like components:

* Hematoxylin
* Eosin
* DAB

*(Optional depending on image compatibility)*

#### Gaussian Blurred Features

Two blurred versions are generated:

* σ = 1
* σ = 2

These help capture **texture and spatial information**.

---

# 📊 Principal Component Analysis (PCA)

All extracted features are:

1. **Flattened**
2. **Z-score normalized**
3. Processed using **PCA**

The model computes:

* **PC1**
* **PC2**
* **PC3**

These components capture the **most significant variations** within the leaf image.

---

# 🖼 Generated Outputs

The script produces several visualization results.

### 1️⃣ PCA Loadings Plot

Shows how each feature contributes to the principal components.

```
pca_loadings.png
```

Helps identify which features influence disease detection.

---

### 2️⃣ PCA Score Maps

Spatial maps showing how each principal component varies across the leaf.

```
pc1_scoremap.png
pc2_scoremap.png
pc3_scoremap.png
```

These maps often reveal:

* Infection regions
* Chlorosis
* Texture changes

---

### 3️⃣ False-Color PCA Composite

Combines the three principal components into an RGB visualization.

```
pc_rgb_composite.png
```

Mapping:

* Red → PC1
* Green → PC2
* Blue → PC3

This produces a **feature-enhanced leaf visualization**.

---

# ⚙️ Installation

Install required dependencies:

```bash
pip install scikit-image scikit-learn opencv-python matplotlib
```

Or run in Google Colab:

```python
!pip install scikit-image scikit-learn opencv-python matplotlib
```

---

# 🚀 Usage

### Step 1 — Run the script

Execute the notebook or Python script.

### Step 2 — Upload a rice leaf image

The script will prompt:

```
Upload image interactively
```

Select a rice leaf image.

### Step 3 — Automatic Processing

The pipeline will:

1. Extract features
2. Run PCA
3. Generate visualizations
4. Save outputs

Results are stored in:

```
pca_results/
```

---

# 📁 Project Structure

```
rice-leaf-pca-analysis/
│
├── rice_leaf_pca.py
├── README.md
│
├── pca_results/
│   ├── pca_loadings.png
│   ├── pc1_scoremap.png
│   ├── pc2_scoremap.png
│   ├── pc3_scoremap.png
│   └── pc_rgb_composite.png
│
└── sample_images/
    └── rice_leaf_example.jpg
```

---

# 🔬 Applications

This approach can be used for:

* 🌾 Rice disease detection
* 🌱 Plant stress analysis
* 🧪 Agricultural research
* 📷 Image-based crop monitoring
* 🤖 Feature engineering for ML models

---

# 🧩 Future Improvements

Possible enhancements:

* Integrate **disease classification models**
* Add **automatic lesion segmentation**
* Use **hyperspectral data**
* Apply **deep learning feature extraction**
* Build a **web interface for farmers**

---

# 🛠 Technologies Used

* Python
* OpenCV
* Scikit-image
* Scikit-learn
* Matplotlib
* Google Colab

---

# 📜 License

This project is open-source and available under the **MIT License**.

---

# 👨‍💻 Author

Developed for research in **plant image analysis and crop disease detection**.

If you found this useful, consider ⭐ starring the repository.
