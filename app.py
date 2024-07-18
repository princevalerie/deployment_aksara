import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import requests
import os

# Fungsi untuk mengunduh model dari GitHub
def download_model(url, model_path):
    if not os.path.exists(model_path):
        response = requests.get(url)
        with open(model_path, 'wb') as file:
            file.write(response.content)

# Fungsi untuk memuat model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Fungsi untuk memproses citra
def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))  # Sesuaikan dengan ukuran input model Anda
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# URL dan path model
model_url = 'https://github.com/princevalerie/deployment_aksara/blob/main/cnn_model.h5'
model_path = 'cnn_model.h5'

# Mengunduh model jika belum ada
download_model(model_url, model_path)

# Memuat model
model = load_model(model_path)

# Judul aplikasi
st.title('Aplikasi Prediksi Aksara Jawa')

# Mengunggah gambar
uploaded_file = st.file_uploader("Pilih gambar aksara Jawa", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    # Pra-pemrosesan gambar
    processed_image = preprocess_image(image)
    
    # Melakukan prediksi
    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction, axis=1)[0]
    
    # Menampilkan hasil prediksi
    st.write(f'Hasil Prediksi: {predicted_label}')
