import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Definision jalur model
model_path = r'C:\Users\Asus\Documents\PBP\Tugas_Modul5_A_Dion_12025\best_model_tf.h5'

# Muat model
if os.path.exists(model_path):
    try:
        # Mengurangi verbosity dari TensorFlow
        tf.get_logger().setLevel('ERROR')
        model = tf.keras.models.load_model(model_path, compile=False)

        # Nama kelas untuk Fashion MNIST
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        # Fungsi untuk memproses gambar
        def preprocess_image(image):
            image = image.resize((28, 28))  # Ubah ukuran menjadi 28x28 piksel
            image = image.convert('L')  # Ubah menjadi grayscale
            image_array = np.array(image) / 255.0  # Normalisasi
            image_array = np.reshape(image_array, (28, 28, 1))  # Ubah bentuk menjadi 4D array
            return image_array

        # UI Streamlit
        st.title('Fashion MNIST Image Classifier 2025') # Ganti xxxx dengan 4 digit NPM
        st.write('Unggah satu atau lebih gambar item fashion (misalnya sepatu, tas, baju), dan model akan memprediksi kelasnya.')

        # File uploader untuk input beberapa gambar
        uploaded_files = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        # Sidebar untuk tombol prediksi
        with st.sidebar:
            predict_button = st.button("Predict")

        # Tampilkan gambar yang diunggah
        if uploaded_files:
            images = [Image.open(file) for file in uploaded_files]
            st.image(images, caption=[file.name for file in uploaded_files], use_column_width=True)

        # Jika tombol predict di sidebar ditekan
        if predict_button:
            st.write("### Hasil Prediksi")

            # Proses setiap gambar yang diunggah
            for file, image in zip(uploaded_files, images):
                # Proses gambar dan prediksi
                processed_image = preprocess_image(image)
                predictions = model.predict(np.expand_dims(processed_image, axis=0))
                predicted_class = np.argmax(predictions)

                # Tampilkan hasil
                st.sidebar.write(f"### {file.name}:")
                st.sidebar.write(f"Class: {class_names[predicted_class]}")
                st.sidebar.write(f"Confidence: {predictions[0][predicted_class]:.2f}")

    except Exception as e:
        st.error(f"Terjadi error: {e}")
else:
    st.error("File model tidak ditemukan")