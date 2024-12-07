import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('stroke_prediction_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert('RGB')  # Convert to RGB
    image = image.resize((64, 64))  # Resize to model input size
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit interface
st.set_page_config(page_title="Stroke Prediction", page_icon="🧠", layout="centered")
image_base64 = get_base64_image("brain.jpeg")
# Background styling
st.markdown(
    """
    <style>
    body {
        background-image: url('brain.jpeg);
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title('🧠 Stroke Prediction from Brain Image')
st.subheader('Upload an image of a brain scan to predict if it shows a stroke or not.')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image_array = preprocess_image(image)

    # Prediction button
    if st.button("Predict Stroke or Normal"):
        st.write("Processing... Please wait.")
        prediction = model.predict(image_array)

        # Display prediction result
        if prediction > 0.5:
            st.markdown('<h2 style="color: green;">Prediction: Stroke</h2>', unsafe_allow_html=True)
        else:
            st.markdown('<h2 style="color: blue;">Prediction: Normal</h2>', unsafe_allow_html=True)
