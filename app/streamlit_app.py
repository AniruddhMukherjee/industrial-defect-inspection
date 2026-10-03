import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.models.backbone import PatchFeatureExtractor
from src.models.memory_bank import make_heatmap, score_batch

st.set_page_config(page_title="Industrial Defect Inspection", layout="centered")


@st.cache_data
def load_config():
    with open("outputs/category_config.json") as f:
        return json.load(f)


@st.cache_resource
def load_backbone():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return PatchFeatureExtractor(device=device), device


@st.cache_resource
def load_memory_bank(bank_path, _device):
    return torch.load(bank_path, map_location="cpu").to(_device)


config = load_config()

st.title("Automated Visual Defect Inspection")
st.caption("PatchCore-style anomaly detection on MVTec AD")

category = st.selectbox("Product category", list(config.keys()))
cat_info = config[category]

with st.sidebar:
    st.header(f"{category.capitalize()} — model stats")

    st.subheader("Performance")
    st.metric("Image AUROC", cat_info["image_auroc"])
    st.metric("Pixel AUROC", cat_info["pixel_auroc"])
    st.metric("Best IoU", cat_info["best_iou"])

    st.subheader("Dataset")
    st.write(f"Train images: {cat_info['num_train_images']}")
    st.write(f"Test images: {cat_info['num_test_images']}")

    st.subheader("Decision threshold")
    st.write(f"{cat_info['threshold']}")

extractor, device = load_backbone()
memory_bank = load_memory_bank(cat_info["memory_bank_path"], device)
threshold = cat_info["threshold"]

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    img_tensor = TF.resize(image, 256)
    img_tensor = TF.center_crop(img_tensor, 224)
    img_tensor = TF.to_tensor(img_tensor).unsqueeze(0)

    feat = extractor(img_tensor)
    B, C, H, W = feat.shape
    patch_features = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

    scores, min_dists = score_batch(patch_features, memory_bank)
    heatmap = make_heatmap(min_dists[0], grid_size=H, image_size=224).cpu().numpy()

    display_img = TF.resize(image, 256)
    display_img = TF.center_crop(display_img, 224)
    display_img = np.array(display_img) / 255.0

    col1, col2 = st.columns(2)
    with col1:
        st.image(display_img, caption="Input (resized + cropped)", use_container_width=True)
    with col2:
        fig, ax = plt.subplots()
        ax.imshow(display_img)
        ax.imshow(heatmap, cmap="jet", alpha=0.5)
        ax.axis("off")
        st.pyplot(fig)
        st.caption("Model-generated anomaly heatmap — red/warm areas indicate regions least similar to normal training examples.")

    score = scores.item()
    st.metric("Anomaly score", f"{score:.4f}")
    st.caption(
        "Distance from this image's patches to the nearest match in the category's "
        "reference memory bank (built from defect-free training images). Higher = less "
        "similar to anything seen as normal."
    )

    if score > threshold:
        st.error(f"⚠️ Flagged as DEFECTIVE (threshold: {threshold})")
    else:
        st.success(f"✅ Passed as NORMAL (threshold: {threshold})")
    st.caption(
        f"Threshold ({threshold}) is calibrated per category as the midpoint between the "
        "highest score seen on a known-normal test image and the lowest score seen on a "
        "known-defective one."
    )