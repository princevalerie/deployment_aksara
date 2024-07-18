import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Fungsi untuk memuat model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'cnn_model.h5'  # saya menaruh ini digithub yg terhubung dengan streamlit
    return tf.keras.models.load_model(model_path)

# Fungsi untuk memproses gambar
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32) / 255.0
    threshold = 0.87
    adjusted_image_array = np.where(image > threshold, 1.0, 0.0)
    negative_adjusted_image = 1.0 - adjusted_image_array
    negative_adjusted_image = np.clip(negative_adjusted_image, 0.0, 1.0)
    negative_adjusted_image = (negative_adjusted_image * 255).astype(np.uint8)
    return negative_adjusted_image

# Fungsi untuk melakukan prediksi
def predict(image, model, label_mapping):
    preprocessed_image = preprocess_image(image)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    prediction = model.predict(preprocessed_image)
    predicted_label = np.argmax(prediction, axis=1)[0]
    return label_mapping[predicted_label]

# Label mapping
label_mapping = {
    0: "ha",
    1: "na",
    2: "ca",
    3: "ra",
    4: "ka",
    5: "da",
    6: "ta",
    7: "sa",
    8: "wa",
    9: "la",
    10: "pa",
    11: "dha",
    12: "ja",
    13: "ya",
    14: "nya",
    15: "ma",
    16: "ga",
    17: "ba",
    18: "tha",
    19: "nga"
}

# Streamlit App
st.title('Aksara Jawa Prediction')
st.write("Unggah gambar aksara Jawa untuk diprediksi")

uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah.', use_column_width=True)
    

    model = load_model()  # Memuat model dari direktori lokal
    prediction = predict(image, model, label_mapping)
    st.write(f"Hasil Prediksi: {prediction}")
