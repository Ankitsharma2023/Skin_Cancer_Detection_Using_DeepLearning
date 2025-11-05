import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image


st.set_page_config(
    page_title="Skin Cancer Detector 🩺",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 38px !important;
            font-weight: 800 !important;
            color: #2E8B57;
        }
        .sub {
            text-align: center;
            color: #555;
            font-size: 18px;
            margin-top: -10px;
        }
        .prediction {
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
        }
        .benign {
            background-color: #D4EDDA;
            color: #155724;
            border: 1px solid #C3E6CB;
        }
        .malignant {
            background-color: #F8D7DA;
            color: #721C24;
            border: 1px solid #F5C6CB;
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


st.markdown("<h1 class='title'>🧬 Skin Cancer Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Upload a skin lesion image to predict whether it's <b>Benign</b> or <b>Malignant</b>.</p>", unsafe_allow_html=True)
st.write("---")

uploaded_image = st.file_uploader("📁 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
   
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.info("🔍 Processing image... Please wait.")
        class_label, confidence, img = predict_skin_cancer(uploaded_image, model)

       
        color_class = "malignant" if class_label == "Malignant" else "benign"
        st.markdown(f"<div class='prediction {color_class}'>Prediction: <b>{class_label}</b></div>", unsafe_allow_html=True)

        st.progress(confidence)
        st.write(f"**Confidence:** {confidence*100:.2f}%")

    st.write("---")

  
    st.subheader("🩻 Model Input (Resized Image)")
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"Processed for model: {class_label}")
    ax.axis("off")
    st.pyplot(fig)

else:
    st.warning("⬆️ Please upload an image to begin prediction.")


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

# Footer
st.write("---")
# st.markdown("<p style='text-align:center; color:gray;'>Developed with ❤️ by Ankit Sharma</p>", unsafe_allow_html=True)

