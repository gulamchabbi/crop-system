# Author: Gulam N Chabbi
# Project: Crop Disease Detection using AI

import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import webbrowser

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crop Disease Detection | AgriScan",
    page_icon="🌿",
    layout="wide"
)

# ---------------- DISEASE DATABASE ----------------
CROP_DB = {
    "Healthy": {
        "severity": "low",
        "description": "Leaf appears healthy with no visible disease.",
        "treatment": "No treatment required. Maintain current farming practices.",
        "action": "Continue regular irrigation and monitoring."
    },
    "Angular Leaf Spot": {
        "severity": "high",
        "description": "Fungal disease causing angular brown lesions.",
        "treatment": "Apply copper-based fungicides and remove infected leaves.",
        "action": "Immediate treatment recommended."
    },
    "Bean Rust": {
        "severity": "high",
        "description": "Rust-colored pustules on leaf surface caused by fungus.",
        "treatment": "Use sulfur or systemic fungicides.",
        "action": "Monitor nearby plants for spread."
    }
}

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="nateraw/vit-base-beans",
        device=-1
    )

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Controls")
    confidence_threshold = st.slider(
        "Confidence Threshold (%)", 0, 100, 50
    )
    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

# ---------------- MAIN UI ----------------
st.title("🌿 Crop Disease Detection System")

tab_scan, tab_info, tab_help = st.tabs(
    ["🔍 Leaf Scanner", "📚 Disease Info", "🧑‍🌾 Expert Locator"]
)

# ---------------- TAB 1: SCANNER ----------------
with tab_scan:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Upload Leaf Image")
        img_file = st.file_uploader(
            "Choose a leaf image", type=["jpg", "png", "jpeg"]
        )

        if img_file:
            img = Image.open(img_file).convert("RGB")
            st.image(img, caption="Uploaded Leaf")

            if st.button("🚀 Run Analysis"):
                with st.spinner("Analyzing leaf..."):
                    model = load_model()
                    results = model(img)
                    st.session_state['results'] = results

    with col2:
        st.subheader("Diagnosis Result")

        if 'results' in st.session_state:
            try:
                top = st.session_state['results'][0]
                score = top['score'] * 100
                label_raw = top['label']

                # Clean label text
                label = label_raw.replace("_", " ").title()

                if score < confidence_threshold:
                    st.error("Low confidence. Try a clearer image.")
                else:
                    st.success(f"Detected: {label}")
                    st.metric("Confidence", f"{score:.2f}%")

                    info = CROP_DB.get(label, {
                        "description": "Unknown disease.",
                        "treatment": "Consult agriculture expert.",
                        "action": "Further lab analysis recommended."
                    })

                    st.markdown("### Description")
                    st.write(info['description'])

                    st.markdown("### Treatment")
                    st.info(info['treatment'])

                    st.markdown("### Recommended Action")
                    st.warning(info['action'])

                    # Top predictions chart
                    chart_data = pd.DataFrame([
                        {
                            "Condition": r['label'].replace("_", " "),
                            "Probability": r['score'] * 100
                        }
                        for r in st.session_state['results']
                    ])
                    st.bar_chart(chart_data.set_index("Condition"))

            except Exception as e:
                st.error("Model failed to analyze the image.")
                st.exception(e)

# ---------------- TAB 2: DISEASE INFO ----------------
with tab_info:
    st.header("Crop Disease Encyclopedia")
    disease = st.selectbox("Select Disease", list(CROP_DB.keys()))
    data = CROP_DB[disease]

    st.subheader(disease)
    st.write(data['description'])
    st.write("Treatment:", data['treatment'])
    st.write("Action:", data['action'])

# ---------------- TAB 3: EXPERT LOCATOR ----------------
with tab_help:
    st.header("Find Agricultural Experts")

    if st.button("Search Experts Near Me"):
        webbrowser.open_new_tab(
            "https://www.google.com/maps/search/agriculture+officer+near+me"
        )
