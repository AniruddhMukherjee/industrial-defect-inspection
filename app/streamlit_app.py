import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image

ANOMALY_THRESHOLDS = {
    "carpet": 3.806,
    "bottle": 4.3088,
} 

from src.models.backbone import PatchFeatureExtractor
from src.models.memory_bank import score_batch, make_heatmap

st.set_page_config(page_title="Industrial Defect Inspection", layout="centered")
st.title("Automated Visual Defect Inspection")
st.caption("PatchCore-style anomaly detection on MVTec AD (carpet, bottle)")

category = st.selectbox("Product category", ["carpet", "bottle"])
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


@st.cache_resource
def load_backbone():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return PatchFeatureExtractor(device=device), device


@st.cache_resource
def load_memory_bank(category, _device):
    return torch.load(f"outputs/memory_bank_{category}.pt").to(_device)


extractor, device = load_backbone()
memory_bank = load_memory_bank(category, device)

st.write(f"Loaded memory bank for **{category}**: {memory_bank.shape[0]} reference patches on `{device}`")

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

    score = scores.item()
    st.metric("Anomaly score", f"{score:.4f}")

    if score > ANOMALY_THRESHOLDS[category]:
        st.error(f"⚠️ Flagged as DEFECTIVE (threshold: {ANOMALY_THRESHOLDS[category]})")
    else:
        st.success(f"✅ Passed as NORMAL (threshold: {ANOMALY_THRESHOLDS[category]})")