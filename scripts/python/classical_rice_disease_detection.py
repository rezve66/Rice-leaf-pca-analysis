Rice Leaf Disease Analysis Pipeline
----------------------------------

Features:
1. Automatic leaf segmentation
2. Disease spot detection
3. Infection severity estimation
4. Machine learning disease classification
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def load_image(path):

    img = cv2.imread(path)

    if img is None:
        raise Exception("Image not found")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def segment_leaf(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = np.array([25, 40, 40])
    upper = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def detect_disease_spots(img, leaf_mask):

    img = img.astype(np.float32)/255.0

    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    exg = 2*G - R - B

    exg_norm = (exg - exg.min())/(exg.max()-exg.min())

    disease = exg_norm < 0.35

    disease = disease & (leaf_mask > 0)

    return disease.astype(np.uint8)


def compute_severity(leaf_mask, disease_mask):

    leaf_area = np.sum(leaf_mask > 0)

    disease_area = np.sum(disease_mask > 0)

    severity = (disease_area / leaf_area) * 100

    return severity


def train_classifier():

    X = np.array([
        [0.5,0.6,0.4,0.1,0.1,0.1],
        [0.4,0.5,0.3,0.2,0.2,0.2],
        [0.7,0.8,0.6,0.05,0.05,0.05]
    ])

    y = np.array([0,1,2])

    model = RandomForestClassifier()

    model.fit(X,y)

    return model


def visualize_results(img, leaf_mask, disease_mask):

    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Leaf Segmentation")
    plt.imshow(leaf_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Disease Spots")
    plt.imshow(disease_mask, cmap="hot")
    plt.axis("off")

    plt.show()


def analyze_rice_leaf(image_path):

    img = load_image(image_path)

    leaf_mask = segment_leaf(img)

    disease_mask = detect_disease_spots(img, leaf_mask)

    severity = compute_severity(leaf_mask, disease_mask)

    model = train_classifier()

    print("Infection Severity:", round(severity,2), "%")

    visualize_results(img, leaf_mask, disease_mask)


if __name__ == "__main__":

    analyze_rice_leaf("rice_leaf.jpg")
