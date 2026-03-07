"""
Rice Leaf PCA Analysis
----------------------

This script performs PCA analysis on a rice leaf RGB image
to visualize infection-related patterns.

Outputs:
- PCA loadings plot
- PC1, PC2, PC3 score maps
- False-color PCA composite

Author: Your Name
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.color import rgb2hed
from skimage.filters import gaussian
from pathlib import Path
import argparse


# ---------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------

def load_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    return img


def white_balance(img):
    means = img.reshape(-1, 3).mean(axis=0)
    scale = means.mean() / np.maximum(means, 1e-6)
    img_wb = np.clip(img * scale, 0, 1)

    return img_wb


# ---------------------------------------------------
# Feature Extraction
# ---------------------------------------------------

def extract_features(img):

    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    eps = 1e-6

    # Vegetation indices
    exg = 2 * G - R - B
    exr = 1.4 * R - G
    ndi = (G - R) / (G + R + eps)

    # Normalized RGB
    rgb_sum = R + G + B + eps
    r_norm = R / rgb_sum
    g_norm = G / rgb_sum
    b_norm = B / rgb_sum

    features = [
        img,
        exg[..., None],
        exr[..., None],
        ndi[..., None],
        r_norm[..., None],
        g_norm[..., None],
        b_norm[..., None]
    ]

    # HED color space
    try:
        hed = rgb2hed(img)
        features.append(hed[..., 0, None])
        features.append(hed[..., 1, None])
        features.append(hed[..., 2, None])
    except Exception:
        pass

    # Gaussian blur features
    blur1 = gaussian(img, sigma=1, channel_axis=-1)
    blur2 = gaussian(img, sigma=2, channel_axis=-1)

    features.append(blur1)
    features.append(blur2)

    F = np.concatenate(features, axis=-1)

    return F


# ---------------------------------------------------
# PCA Analysis
# ---------------------------------------------------

def run_pca(features):

    H, W, C = features.shape

    X = features.reshape(-1, C)

    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6

    Xz = (X - mu) / sd

    pca = PCA(n_components=3)
    pca.fit(Xz)

    scores = pca.transform(Xz)
    scores = scores.reshape(H, W, 3)

    loadings = pca.components_
    explained = pca.explained_variance_ratio_

    return scores, loadings, explained


# ---------------------------------------------------
# Visualization
# ---------------------------------------------------

def save_loadings(loadings, explained, output_dir):

    feature_names = [
        "R", "G", "B",
        "ExG", "ExR", "NDI",
        "r_norm", "g_norm", "b_norm",
        "HED-H", "HED-E", "HED-D",
        "Blur1_R", "Blur1_G", "Blur1_B",
        "Blur2_R", "Blur2_G", "Blur2_B"
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7))

    for i, ax in enumerate(axes):

        ax.bar(range(len(loadings[i])), loadings[i])

        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=70)

        ax.set_title(
            f"PC{i+1} Loadings (Explained Var {explained[i]:.2%})")

        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(output_dir / "pca_loadings.png", dpi=600)
    plt.close()


def save_scoremaps(scores, output_dir):

    for i in range(3):

        sm = scores[..., i]
        sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-6)

        plt.figure(figsize=(5, 5))
        plt.imshow(sm, cmap="viridis")
        plt.axis("off")

        plt.title(f"PC{i+1} Score Map")

        plt.savefig(output_dir / f"pc{i+1}_scoremap.png",
                    dpi=600,
                    bbox_inches="tight")

        plt.close()


def save_rgb_composite(scores, output_dir):

    rgb = np.zeros_like(scores)

    for i in range(3):

        sm = scores[..., i]
        rgb[..., i] = (sm - sm.min()) / (sm.max() - sm.min() + 1e-6)

    plt.figure(figsize=(6, 6))

    plt.imshow(rgb)
    plt.axis("off")

    plt.title("False Color Composite (PC1, PC2, PC3)")

    plt.savefig(output_dir / "pc_rgb_composite.png",
                dpi=600,
                bbox_inches="tight")

    plt.close()


# ---------------------------------------------------
# Main Pipeline
# ---------------------------------------------------

def analyze_rice_leaf(image_path, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading image...")
    img = load_image(image_path)

    print("Applying white balance...")
    img = white_balance(img)

    print("Extracting features...")
    features = extract_features(img)

    print("Running PCA...")
    scores, loadings, explained = run_pca(features)

    print("Saving results...")
    save_loadings(loadings, explained, output_dir)
    save_scoremaps(scores, output_dir)
    save_rgb_composite(scores, output_dir)

    print(f"Analysis complete! Results saved to {output_dir}")


# ---------------------------------------------------
# CLI
# ---------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Rice Leaf PCA Infection Analysis")

    parser.add_argument(
        "--image",
        required=True,
        help="Path to rice leaf image")

    parser.add_argument(
        "--output",
        default="results",
        help="Output folder")

    args = parser.parse_args()

    analyze_rice_leaf(args.image, args.output)


if __name__ == "__main__":
    main()
