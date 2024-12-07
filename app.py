import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import base64

# Load the trained model (make sure to save your model as 'stroke_prediction_model.h5')
model = load_model('stroke_prediction_model.h5')
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded
# Function to preprocess the image
def preprocess_image(image):
    # Convert the uploaded image to RGB format (it might be in RGBA, etc.)
    image = image.convert('RGB')
    # Resize the image to the required input size for the model
    image = image.resize((64, 64))
    # Convert the image to a NumPy array and normalize it
    image_array = np.array(image) / 255.0
    # Add batch dimension (required by the model)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Streamlit interface
st.set_page_config(page_title="Stroke Prediction", page_icon="🧠", layout="centered")
# Background styling using Base64-encoded image
image_base64 = get_base64_image("brain.jpeg")

st.markdown(
    f"""
    <style>
    body {{
        background-image: url('data:image/jpeg;base64,{image_base64}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    h1 {{
        text-align: center;
        color: green;
    }}
    h2 {{
        text-align: center;
        color: blue;
    }}
    .stButton button {{
        background-color: #28a745;
        color: white;
        font-size: 16px;
        font-weight: bold;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.title('🧠 Stroke Prediction from Brain Image')
st.subheader('Upload an image of a brain scan to predict if it shows a stroke or not.')

# File uploader widget for the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display the uploaded image with some styling
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True, clamp=True)

    # Preprocess the image
    image_array = preprocess_image(image)

    # Prediction
    if st.button("Predict Stroke or Normal"):
        st.write("Processing... Please wait.")
        prediction = model.predict(image_array)

        # Show the result of the prediction
        if prediction > 0.5:
            st.markdown('<h2 style="color: green;">Prediction: Stroke</h2>', unsafe_allow_html=True)
        else:
            st.markdown('<h2 style="color: blue;">Prediction: Normal</h2>', unsafe_allow_html=True)

# # Styling for the page
# st.markdown("""
#     <style>
#         .css-ffhzg2 {  /* Title styling */
#             text-align: center;
#             color: green;
#         }
#         .css-1b9g2ys {  /* Subheader styling */
#             text-align: center;
#             color: blue;
#         }
#         .css-16mhj2r {  /* Button styling */
#             background-color: green;
#             color: white;
#         }
#         .stButton button {  /* Styling the button to make it more prominent */
#             background-color: #28a745;
#             color: white;
#             font-size: 16px;
#             font-weight: bold;
#         }
#         .stFileUploader {  /* Custom file uploader style */
#             border: 2px solid #28a745;
#             padding: 10px;
#         }
#     </style>
# """, unsafe_allow_html=True)
