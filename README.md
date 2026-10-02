# Rice Leaf Image Analysis

A simple computer-vision workflow for exploring colour, texture, and symptom-related patterns in rice leaf images. It combines PCA-based visualization with image analysis to support preliminary comparison of control and diseased leaves.

## What’s Included

### 1. PCA-Based Image Analysis

Analyzes one uploaded leaf image through:

* Basic image correction and feature extraction
* RGB, vegetation-related, normalized-colour, HED, and Gaussian features
* PCA loadings and PC1–PC3 maps
* A false-colour composite of the first three components

### 2. Control vs. Diseased Leaf Comparison

Compares two images using:

* Automatic leaf segmentation and mask visualization
* Green, yellow, brown/dark, and other colour estimates
* Candidate symptom-region detection and measurements
* Lesion shape and size features
* Normalized 4 × 4 spatial distribution maps
* Colour and texture analysis using GLCM, LBP, and Gabor filters

The workflow also generates comparison plots, CSV tables, a text report, and a ZIP archive of the results.

## How to Run

1. Open the notebook in Google Colab.
2. Run the analysis cell and upload your image.
3. For PCA, upload one leaf image.
4. For comparison, upload the control image first, followed by the diseased image.
5. Review the generated masks and results, then download the outputs.

## Requirements

Python libraries: OpenCV, NumPy, Matplotlib, scikit-image, scikit-learn, SciPy, pandas, and seaborn. Required packages are installed in Colab when needed.


## License

MIT License
