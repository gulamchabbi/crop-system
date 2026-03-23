# 🌿 Crop Disease Detection AI (PRO VERSION)
# Author: Gulam N Chabbi

import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Crop Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ---------------- BACKGROUND + GLASS UI ----------------
def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6?q=80&w=1600");
            background-size: cover;
            background-position: center;
        }

        /* Glass card effect */
        .glass {
            background: rgba(255, 255, 255, 0.15);
            padding: 25px;
            border-radius: 20px;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }

        h1, h2, h3, h4 {
            color: white !important;
        }

        p, label {
            color: #f1f1f1 !important;
        }

        .stButton>button {
            background: linear-gradient(45deg, #00c853, #64dd17);
            color: white;
            border-radius: 10px;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

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
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline(
        "image-classification",
        model="nateraw/vit-base-beans"
    )

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🌿 AI Crop Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a leaf image and detect diseases instantly using AI</p>", unsafe_allow_html=True)

# ---------------- UPLOAD SECTION ----------------
with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MAIN LOGIC ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing... Please wait"):
                model = load_model()
                results = model(image)

                top = results[0]
                label_raw = top["label"]
                score = top["score"] * 100
                label = label_raw.replace("_", " ").title()

                st.success(f"🌱 Detected: {label}")
                st.metric("Confidence", f"{score:.2f}%")

                info = CROP_DB.get(label, {
                    "description": "Not found in database.",
                    "treatment": "Consult an agriculture expert."
                })

                st.subheader("📖 Description")
                st.write(info["description"])

                st.subheader("💊 Treatment")
                st.info(info["treatment"])

                # ---------------- REPORT DOWNLOAD ----------------
                report = f"""
Crop Disease Detection Report

Detected Disease: {label}
Confidence: {score:.2f}%

Description:
{info['description']}

Treatment:
{info['treatment']}

Developed by Gulam N Chabbi
"""
                st.download_button("📄 Download Report", report)

                # ---------------- CHART ----------------
                st.subheader("📊 All Predictions")

                chart_data = pd.DataFrame([
                    {
                        "Disease": r["label"].replace("_", " "),
                        "Probability": r["score"] * 100
                    }
                    for r in results
                ])

                st.bar_chart(chart_data.set_index("Disease"))

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>🚀 Developed by <b>Gulam N Chabbi</b></p>",
    unsafe_allow_html=True
)
