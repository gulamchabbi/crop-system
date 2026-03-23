# Author: Gulam N Chabbi (Modified for Crop Disease Detection)

import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import webbrowser

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Crop Disease Detection | AI AgriScan",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CROP DISEASE DATABASE ---
CROP_DB = {

    "Tomato Early Blight": {
        "severity": "high",
        "risk_label": "HIGH RISK – YIELD LOSS POSSIBLE",
        "description": "Fungal disease causing dark spots with concentric rings on leaves.",
        "features": "• Brown circular spots\n• Yellowing leaves\n• Leaf drop",
        "causes": "Caused by Alternaria fungus thriving in warm, humid conditions.",
        "treatment": "Apply fungicides such as chlorothalonil. Remove infected leaves.",
        "action": "Immediate treatment recommended to prevent spread."
    },

    "Tomato Late Blight": {
        "severity": "critical",
        "risk_label": "CRITICAL – CAN DESTROY ENTIRE CROP",
        "description": "A fast-spreading fungal disease affecting leaves and fruits.",
        "features": "• Dark, water-soaked patches\n• White fungal growth under leaves",
        "causes": "Spread by airborne spores in cool, wet environments.",
        "treatment": "Use copper-based fungicides and destroy infected plants.",
        "action": "Urgent action required to save nearby plants."
    },

    "Healthy Leaf": {
        "severity": "low",
        "risk_label": "HEALTHY CROP",
        "description": "No disease detected. Leaf appears healthy.",
        "features": "• Uniform green color\n• No visible spots or discoloration",
        "causes": "Proper plant nutrition and disease-free growth.",
        "treatment": "No treatment required.",
        "action": "Maintain current farming practices."
    }
}

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="Sanchit2810/plant-disease-classification-resnet50"
    )


# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ AgriScan Controls")
    confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 50)

    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

# --- 5. MAIN INTERFACE ---
st.title("🌿 Crop Disease Detection System")
tab_scan, tab_dict, tab_help = st.tabs(
    ["🔍 Leaf Scanner", "📚 Disease Library", "🧑‍🌾 Expert Locator"]
)

# --- TAB 1: SCANNER ---
with tab_scan:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        img_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

        if img_file:
            img = Image.open(img_file)
            st.image(img, caption="Uploaded Leaf")

            if st.button("Run Analysis"):
                with st.spinner("Analyzing plant tissue..."):
                    model = load_model()
                    results = model(img)
                    st.session_state['results'] = results

    with col2:
        if 'results' in st.session_state:
            top = st.session_state['results'][0]
            score = top['score'] * 100
            label = top['label']

            if score < confidence_threshold:
                st.error("Low confidence. Please upload clearer image.")
            else:
                info = CROP_DB.get(label, {
                    "severity": "low",
                    "risk_label": "Unknown",
                    "description": "Not in database",
                    "features": "",
                    "causes": "",
                    "treatment": "",
                    "action": "Consult agriculture expert"
                })

                st.subheader(f"Detection: {label}")
                st.metric("Confidence", f"{score:.2f}%")
                st.write(info['description'])

                st.markdown("### Symptoms")
                st.write(info['features'])

                st.markdown("### Causes")
                st.write(info['causes'])

                st.markdown("### Treatment")
                st.write(info['treatment'])

                st.warning(info['action'])

                chart_data = pd.DataFrame([
                    {"Disease": r['label'], "Probability": r['score']*100}
                    for r in st.session_state['results'][:3]
                ])
                st.bar_chart(chart_data.set_index("Disease"))

# --- TAB 2: DISEASE LIBRARY ---
with tab_dict:
    disease = st.selectbox("Select Disease", list(CROP_DB.keys()))
    data = CROP_DB[disease]

    st.subheader(disease)
    st.write(data['description'])
    st.write("Symptoms:", data['features'])
    st.write("Treatment:", data['treatment'])

# --- TAB 3: EXPERT LOCATOR ---
with tab_help:
    st.header("Find Agriculture Experts Nearby")

    if st.button("Search Experts"):
        webbrowser.open_new_tab(
            "https://www.google.com/maps/search/agriculture+officer+near+me"
        )
