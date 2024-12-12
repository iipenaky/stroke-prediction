import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('stroke_prediction_model.keras')

# Function to preprocess the image
def preprocess_image(image_path):
    """
    Loads an image from the specified path, resizes it to the required dimensions (256x256x3),
    and normalizes pixel values.
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    # Resize the image to (256, 256)
    resized_image = cv2.resize(image, (256, 256))
    
    # Normalize the pixel values to [0, 1]
    normalized_image = resized_image / 255.0
    
    # Add a batch dimension (1, 256, 256, 3)
    input_array = np.expand_dims(normalized_image, axis=0)
    
    return input_array

# Streamlit interface
st.set_page_config(page_title="Stroke Prediction", page_icon="🧠", layout="centered")

# Custom CSS for green and blue aesthetics
st.markdown(
    """
    <style>
    body {
        background-color: white;  /* White background */
    }
    h1, h2 {
        text-align: center;
    }
    h1 {
        color: green;  /* Green for titles */
    }
    h2 {
        color: blue;  /* Blue for subtitles */
    }
    .stButton button {
        background-color: #28a745;  /* Green button */
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
    }
    .stFileUploader {
        border: 2px solid #28a745;
        padding: 10px;
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
