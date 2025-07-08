import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

file_id = "184n7vuunCVccxz-iGiBgHX-hXyfWKzvk"
url = 'https://drive.google.com/uc?id=184n7vuunCVccxz-iGiBgHX-hXyfWKzvk'
model_path = "trained_plant_disease_model.keras"

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

def model_prediction(test_image):
    model = load_model()
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

try:
    header_img = Image.open("Diseases.png")
    st.image(header_img, use_column_width=True)
except FileNotFoundError:
    st.warning("Diseases.png not found. Please place it in the app directory.")

if app_mode == "HOME":
    st.markdown(
        "<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>",
        unsafe_allow_html=True
    )

elif app_mode == "DISEASE RECOGNITION":
    st.header("Upload a Plant Leaf Image")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        image = Image.open(test_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction:")
            result_index = model_prediction(test_image)
            class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
            st.success(f"Model is predicting it's **{class_name[result_index]}**")
