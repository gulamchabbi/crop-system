# Crop Disease Detection using AI
# Author: Gulam N Chabbi

import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ---------------- DISEASE DATABASE ----------------
CROP_DB = {
    "Healthy": {
        "description": "The leaf is healthy with no visible disease.",
        "treatment": "No treatment required. Maintain regular care."
    },
    "Angular Leaf Spot": {
        "description": "Fungal disease causing brown angular lesions.",
        "treatment": "Apply copper-based fungicide and remove infected leaves."
    },
    "Bean Rust": {
        "description": "Rust-colored spots caused by fungal infection.",
        "treatment": "Use sulfur-based fungicides and monitor nearby plants."
    }
}

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="nateraw/vit-base-beans",
        device=-1
    )

# ---------------- UI ----------------
st.title("🌿 AI Crop Disease Detection")

st.write("Upload a leaf image and the AI will predict the disease.")

uploaded_file = st.file_uploader(
    "Upload leaf image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Run Analysis"):
        with st.spinner("Analyzing image..."):
            model = load_model()
            results = model(image)

            # Top prediction
            top = results[0]
            label_raw = top["label"]
            score = top["score"] * 100

            # Clean label formatting
            label = label_raw.replace("_", " ").title()

            st.success(f"Detected Condition: **{label}**")
            st.metric("Confidence", f"{score:.2f}%")

            # Show disease info
            info = CROP_DB.get(label, {
                "description": "Disease not found in local database.",
                "treatment": "Consult agriculture expert."
            })

            st.subheader("Disease Description")
            st.write(info["description"])

            st.subheader("Recommended Treatment")
            st.info(info["treatment"])

            # Show all predictions
            st.subheader("All Model Predictions")

            chart_data = pd.DataFrame([
                {
                    "Disease": r["label"].replace("_", " "),
                    "Probability": r["score"] * 100
                }
                for r in results
            ])

            st.bar_chart(chart_data.set_index("Disease"))
