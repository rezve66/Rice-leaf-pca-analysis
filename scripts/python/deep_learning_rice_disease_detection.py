"""
Deep Learning Rice Leaf Disease Detection Pipeline

Modules:
1. Leaf Isolation (DeepLabV3)
2. Lesion Segmentation (U-Net)
3. Disease Classification (EfficientNet)
4. Disease Severity Heatmap
"""

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import segmentation_models_pytorch as smp

import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_image(path):

    img = cv2.imread(path)

    if img is None:
        raise Exception("Image not found")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def load_leaf_model():

    model = models.segmentation.deeplabv3_resnet50(pretrained=True)

    model.eval()

    return model


def segment_leaf(img, model):

    transform = transforms.Compose([transforms.ToTensor()])

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():

        output = model(tensor)["out"]

    mask = output.argmax(1).squeeze().cpu().numpy()

    return (mask > 0).astype(np.uint8)


def load_unet():

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=1,
        activation="sigmoid"
    )

    model.eval()

    return model


def segment_lesions(img, model):

    img = cv2.resize(img, (256,256))

    tensor = transforms.ToTensor()(img).unsqueeze(0)

    with torch.no_grad():

        pred = model(tensor)

    mask = pred.squeeze().cpu().numpy()

    return (mask > 0.4).astype(np.uint8)


def load_classifier():

    model = models.efficientnet_b0(pretrained=True)

    num_features = model.classifier[1].in_features

    model.classifier[1] = torch.nn.Linear(num_features, 3)

    model.eval()

    return model


def classify_disease(img, model):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():

        pred = model(tensor)

    idx = pred.argmax().item()

    classes = ["Healthy","Brown Spot","Leaf Blast"]

    return classes[idx]


def generate_heatmap(img, lesion_mask):

    heatmap = cv2.applyColorMap(
        (lesion_mask*255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(img,0.7,heatmap,0.3,0)

    return overlay


def visualize(img, leaf_mask, lesion_mask, heatmap):

    plt.figure(figsize=(15,6))

    plt.subplot(1,4,1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.title("Leaf Mask")
    plt.imshow(leaf_mask,cmap="gray")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.title("Lesion Segmentation")
    plt.imshow(lesion_mask,cmap="hot")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.title("Severity Heatmap")
    plt.imshow(heatmap)
    plt.axis("off")

    plt.show()


def analyze_leaf(image_path):

    img = load_image(image_path)

    leaf_model = load_leaf_model()

    lesion_model = load_unet()

    classifier = load_classifier()

    leaf_mask = segment_leaf(img, leaf_model)

    lesions = segment_lesions(img, lesion_model)

    disease = classify_disease(img, classifier)

    heatmap = generate_heatmap(img, lesions)

    print("Disease Type:", disease)

    visualize(img, leaf_mask, lesions, heatmap)


if __name__ == "__main__":

    analyze_leaf("rice_leaf.jpg")
