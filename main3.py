import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import time

st.set_page_config(
    page_title=" Skin Cancer Detector using Deep Learning 🤖",
    page_icon="🧬",
    layout="centered",
)

st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #fafafa;
        }
        .title {
            font-size: 48px !important;
            text-align: center;
            color: #00FFFF;
            text-shadow: 0 0 20px #00FFFF;
            font-weight: 900 !important;
            margin-bottom: -10px;
        }
        .subtitle {
            text-align: center;
            color: #b3b3b3;
            font-size: 18px;
            margin-top: -10px;
        }
        .prediction {
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            margin-top: 25px;
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
        }
        .benign {
            color: #00FF7F;
            border: 2px solid #00FF7F;
            background: rgba(0,255,127,0.1);
        }
        .malignant {
            color: #FF4C4C;
            border: 2px solid #FF4C4C;
            background: rgba(255,76,76,0.1);
        }
        .confidence-bar {
            height: 18px;
            border-radius: 10px;
            background: linear-gradient(90deg, #00FFFF, #0088FF);
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #7f8c8d;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cnn_model():
    return load_model('skin_cancer_cnn.h5')

model = load_cnn_model()

def predict_skin_cancer(image_file, model):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)
    class_label = "Malignant" if prediction > 0.5 else "Benign"
    return class_label, confidence, img

st.markdown("<h1 class='title'> Skin Cancer Detection using Deep Learning 🤖</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Deep Learning–powered dermatology assistant trained to detect melanoma and benign skin lesions.</p>", unsafe_allow_html=True)
st.write("---")

uploaded_image = st.file_uploader("📸 Upload a Skin Lesion Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        st.caption("🧩 Image Preprocessing in progress...")
    with col2:
        with st.spinner("🤖 Analyzing image with deep learning model..."):
            time.sleep(1.2)
            class_label, confidence, img = predict_skin_cancer(uploaded_image, model)
        st.markdown("<h3 style='text-align:center;'>🔍 AI Diagnostic Report</h3>", unsafe_allow_html=True)
        color_class = "malignant" if class_label == "Malignant" else "benign"
        st.markdown(f"<div class='prediction {color_class}'>Prediction: <b>{class_label}</b></div>", unsafe_allow_html=True)
        st.write("")
        st.progress(confidence)
        st.write(f"**Model Confidence:** `{confidence*100:.2f}%`")
        if class_label == "Malignant":
            st.warning("⚠️ This lesion appears malignant. Please consult a dermatologist for professional analysis.")
        else:
            st.success("✅ This lesion appears benign. No immediate concern detected.")
    st.write("---")
    st.subheader("🩻 Model Input Preview")
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"Model View — {class_label}")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("⬆️ Upload an image of a skin lesion to begin the AI diagnosis.")

with st.expander("📘 About this Project"):
    st.markdown("""
    **Skin Cancer Detection Model** uses a Convolutional Neural Network (CNN) 
    trained on dermatoscopic images to classify lesions as:
    - 🟢 **Benign** (non-cancerous)
    - 🔴 **Malignant** (cancerous)

    **Model Details:**
    - Input Image Size: `224x224`
    - Framework: `TensorFlow / Keras`
    - Architecture: Custom CNN with Conv2D, MaxPooling2D, Dropout layers
    - Optimizer: `Adam`
    - Loss: `Binary Crossentropy`
    
    **Note:**  
    This tool is intended for **educational and research purposes only** and should not replace professional medical diagnosis.
    """)

# st.markdown("<div class='footer'>Developed with ❤️ by <b>Ankit Sharma</b> | Powered by TensorFlow & Streamlit</div>", unsafe_allow_html=True)


