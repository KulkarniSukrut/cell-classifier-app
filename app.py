import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load your trained model
model = load_model("cell_classifier_cnn.h5")

# Streamlit UI
st.set_page_config(page_title="Cell Classifier", layout="centered")
st.title("🔬 Cell Classification using CNN")
st.write("Upload a microscopic cell image to classify it as **Prokaryotic** or **Eukaryotic**.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    class_name = "Prokaryotic Cell" if prediction > 0.5 else "Eukaryotic Cell"
    confidence = f"{(prediction if prediction > 0.5 else 1 - prediction) * 100:.2f}%"

    # Output
    st.markdown(f"### 🧬 Prediction: `{class_name}`")
    st.markdown(f"### ✅ Confidence: `{confidence}`")
