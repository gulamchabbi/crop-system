# 🌿 AI Crop Disease Detection (ULTRA PRO VERSION)
# Author: Gulam N Chabbi

import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Crop Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ---------------- PREMIUM UI ----------------
def load_css():
    st.markdown("""
    <style>

    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }

    /* Glass Cards */
    .card {
        background: rgba(255, 255, 255, 0.08);
        padding: 25px;
        border-radius: 18px;
        backdrop-filter: blur(18px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: 0.3s;
    }

    .card:hover {
        transform: scale(1.01);
    }

    /* Title */
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: white;
    }

    .subtitle {
        text-align: center;
        color: #dcdcdc;
        margin-bottom: 20px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #00c853, #b2ff59);
        color: black;
        border-radius: 12px;
        font-weight: bold;
        padding: 10px 20px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(0,0,0,0.6);
    }

    /* Text color */
    h1, h2, h3, h4, p, label {
        color: white !important;
    }

    </style>
    """, unsafe_allow_html=True)

load_css()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")
st.sidebar.write("Customize your experience")

show_all_predictions = st.sidebar.toggle("Show all predictions", True)
show_confidence = st.sidebar.toggle("Show confidence score", True)

st.sidebar.markdown("---")
st.sidebar.info("🌿 AI detects crop diseases using deep learning")

# ---------------- DATABASE ----------------
CROP_DB = {
    "Healthy": {
        "description": "The leaf is healthy with no visible disease.",
        "treatment": "No treatment required."
    },
    "Angular Leaf Spot": {
        "description": "Fungal disease causing angular brown spots.",
        "treatment": "Apply copper fungicide."
    },
    "Bean Rust": {
        "description": "Rust-colored fungal infection.",
        "treatment": "Use sulfur fungicide."
    }
}

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="nateraw/vit-base-beans")

# ---------------- HEADER ----------------
st.markdown('<div class="title">🌿 AI Crop Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a leaf image and get instant AI diagnosis</div>', unsafe_allow_html=True)

# ---------------- UPLOAD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MAIN ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1.2])

    # LEFT SIDE (IMAGE)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📸 Uploaded Image")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT SIDE (RESULT)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if st.button("🚀 Run AI Analysis"):
            with st.spinner("Analyzing image..."):
                model = load_model()
                results = model(image)

                top = results[0]
                label = top["label"].replace("_", " ").title()
                score = top["score"] * 100

                st.success(f"🌱 {label}")

                if show_confidence:
                    st.progress(int(score))
                    st.write(f"Confidence: {score:.2f}%")

                info = CROP_DB.get(label, {
                    "description": "Not available.",
                    "treatment": "Consult expert."
                })

                st.markdown("### 📖 Description")
                st.write(info["description"])

                st.markdown("### 💊 Treatment")
                st.info(info["treatment"])

                # REPORT
                report = f"""
Disease: {label}
Confidence: {score:.2f}%

Description:
{info['description']}

Treatment:
{info['treatment']}
"""
                st.download_button("📄 Download Report", report)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- PREDICTIONS ----------------
    if show_all_predictions:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Detailed Predictions")

        chart_data = pd.DataFrame([
            {
                "Disease": r["label"].replace("_", " "),
                "Probability": r["score"] * 100
            }
            for r in results
        ])

        st.dataframe(chart_data, use_container_width=True)
        st.bar_chart(chart_data.set_index("Disease"))

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>🚀 Developed by <b>Gulam N Chabbi</b> | AI Project</p>",
    unsafe_allow_html=True
)
