![Python](https://img.shields.io/badge/Python-3.9-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-ImageProcessing-green)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-PCA-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

# 🌾 Rice Leaf Disease Analysis Framework

A comprehensive **computer vision framework** for analyzing rice leaf diseases using multiple approaches:

- 📊 **Principal Component Analysis (PCA)** for infection pattern discovery  
- 🌿 **Classical computer vision** for disease segmentation and severity estimation  
- 🤖 **Deep learning models** for advanced disease detection  

The system processes rice leaf images and produces:

- PCA feature loadings  
- spatial infection maps  
- automatic leaf segmentation  
- disease spot detection  
- infection severity estimation  
- lesion segmentation using deep learning  
- disease classification  
- severity heatmaps  

The goal is to **enhance subtle disease patterns in rice leaves** that are difficult to observe in standard RGB images.

---

# 📌 Project Overview

Plant diseases significantly impact **global agricultural productivity**, especially in rice farming.

Early disease detection helps:

- reduce crop loss  
- improve yield management  
- support precision agriculture  

This project combines **three different computer vision approaches** for analyzing rice leaf infections.

### 1️⃣ PCA Feature Analysis
Unsupervised feature analysis used to detect hidden infection patterns.

### 2️⃣ Classical Computer Vision Pipeline
Traditional image processing methods for disease detection.

### 3️⃣ Deep Learning Pipeline
Modern deep learning models for segmentation and classification.

Together these pipelines create a **complete research workflow for plant disease detection**.

---

# 🧠 Methodology

The framework contains **three main analysis pipelines**.

---

# 📊 Pipeline 1 — PCA Infection Pattern Analysis

This pipeline reveals hidden infection patterns using **Principal Component Analysis**.

### 1️⃣ Image Loading

The rice leaf image is loaded using **OpenCV** and converted from BGR to RGB format.

### 2️⃣ White Balance Correction

A **Gray-World assumption** is applied to normalize lighting conditions.

### 3️⃣ Feature Extraction

Multiple image features are extracted.

#### Color Channels
- R  
- G  
- B  

#### Vegetation Indices
- **ExG (Excess Green)** – vegetation intensity  
- **ExR (Excess Red)** – highlights disease regions  
- **NDI (Normalized Difference Index)** – chlorophyll variation  

#### Normalized RGB
- r_norm  
- g_norm  
- b_norm  

#### Color Deconvolution (HED)
Separates stain-like components:

- Hematoxylin  
- Eosin  
- DAB  

*(optional depending on image compatibility)*

#### Gaussian Blur Features

Two blurred images are generated:

- σ = 1  
- σ = 2  

These capture **texture and spatial information**.

---

# 📊 Principal Component Analysis (PCA)

All extracted features are:

1. Flattened  
2. Z-score normalized  
3. Processed using PCA  

The model extracts:

- **PC1**
- **PC2**
- **PC3**

These components represent the **dominant patterns in the leaf image**.

---

# 🌿 Pipeline 2 — Classical Computer Vision Disease Detection

This pipeline performs **automatic disease detection using image processing**.

### Leaf Segmentation
The leaf is isolated using **HSV color thresholding**.

### Disease Spot Detection
Disease regions are detected using vegetation indices such as **ExG**.

### Infection Severity Estimation

Disease severity is calculated as:


severity = diseased_area / leaf_area × 100


### Machine Learning Classification

A **Random Forest classifier** predicts disease type.

Example classes:

- Healthy  
- Brown Spot  
- Leaf Blast  

---

# 🤖 Pipeline 3 — Deep Learning Disease Detection

A deep learning pipeline is implemented for **advanced disease analysis**.

### Leaf Isolation
Uses **DeepLabV3** semantic segmentation to isolate the leaf.

### Lesion Segmentation
Uses a **U-Net CNN** to detect infected regions.

### Disease Classification
Uses **EfficientNet** to classify leaf diseases.

Example categories:

- Healthy  
- Brown Spot  
- Leaf Blast  

### Severity Heatmap

A color heatmap visualizes the **infection intensity across the leaf**.

---

# 🖼 Generated Outputs

The framework generates several visual outputs.

### 1️⃣ PCA Loadings Plot


pca_loadings.png


Shows how each feature contributes to principal components.

---

### 2️⃣ PCA Score Maps


pc1_scoremap.png
pc2_scoremap.png
pc3_scoremap.png


These maps highlight:

- infection regions  
- chlorosis  
- texture variation  

---

### 3️⃣ False Color PCA Composite


pc_rgb_composite.png


Mapping:

- Red → PC1  
- Green → PC2  
- Blue → PC3  

This creates a **feature-enhanced visualization of the leaf**.

---

### 4️⃣ Leaf Segmentation Map

Binary mask representing detected leaf area.

---

### 5️⃣ Disease Detection Map

Detected infection regions on the leaf.

---

### 6️⃣ Severity Heatmap

Color heatmap highlighting disease intensity.

---

# ⚙️ Installation

Install required dependencies:


pip install scikit-image scikit-learn opencv-python matplotlib torch torchvision segmentation-models-pytorch


Or in Google Colab:


!pip install scikit-image scikit-learn opencv-python matplotlib torch torchvision segmentation-models-pytorch


---

# 🚀 Usage

### Step 1 — Run the script

Run one of the pipelines.

### PCA Analysis


python pca_rice_leaf_analysis.py --image rice_leaf.jpg


### Classical Detection


python classical_rice_disease_detection.py


### Deep Learning Detection


python deep_learning_rice_disease_detection.py


---

# 📁 Project Structure


rice-leaf-disease-analysis/
│
├── pca_rice_leaf_analysis.py
├── classical_rice_disease_detection.py
├── deep_learning_rice_disease_detection.py
│
├── README.md
│
├── results/
│ ├── pca_loadings.png
│ ├── pc1_scoremap.png
│ ├── pc2_scoremap.png
│ ├── pc3_scoremap.png
│ └── pc_rgb_composite.png
│
└── sample_images/
└── rice_leaf_example.jpg


---

# 🔬 Applications

This framework can be used for:

- 🌾 Rice disease detection  
- 🌱 Plant stress analysis  
- 🧪 Agricultural research  
- 📷 Image-based crop monitoring  
- 🤖 Feature engineering for ML models  
- 🚜 Precision agriculture  

---

# 🧩 Future Improvements

Possible enhancements:

- real-time disease detection systems  
- hyperspectral imaging analysis  
- Vision Transformer models  
- mobile applications for farmers  
- cloud-based crop monitoring platforms  

---

# 🛠 Technologies Used

- Python  
- OpenCV  
- Scikit-image  
- Scikit-learn  
- PyTorch  
- segmentation-models-pytorch  
- Matplotlib  
- Google Colab  

---

# 📜 License

This project is released under the **MIT License**.

---

# 👨‍💻 Author

Developed for research in **plant disease detection using computer vision and deep learning**.

If you found this project useful, consider ⭐ starring the repository.
