
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import os

# Define the path to the saved model in Google Drive
# IMPORTANT: For Streamlit Cloud deployment, the model needs to be in your GitHub repository
# This path is relative to the Streamlit app's root directory in GitHub
MODEL_PATH = 'sun_moon_model.h5'

# Load the trained model
@st.cache_resource # Cache the model loading for performance
def load_model():
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}") # Fixed: escaped curly braces around e
        st.error("Please ensure 'sun_moon_model.h5' is in the same directory as app.py in your GitHub repo.")
        return None

model = load_model()

# Define class names (must be in the same order as during training)
# The class_names from ImageDataGenerator were: ['Sun', 'moon']
CLASS_NAMES = ['Sun', 'moon']

st.title("🌕 Sun vs. Moon Image Classifier ☀️")
st.write("Upload an image to classify if it's a Sun or a Moon!")

if model is None:
    st.stop()

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("") # Add a little space

    st.write("Classifying...")

    # Preprocess the image for model prediction
    # Resize image to 224x224 (as trained)
    img_resized = image.resize((224, 224))
    img_array = np.asarray(img_resized)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Normalize pixel values

    # Make prediction
    predictions = model.predict(img_array)
    # Get the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100

    st.success(f"Prediction: **{predicted_class_name}**")
    st.info(f"Confidence: **{confidence:.2f}%**")

    # Display interpretation if needed
    if 'sun' in predicted_class_name.lower():
        st.markdown("Looks like a radiant Sun!")
    elif 'moon' in predicted_class_name.lower():
        st.markdown("Looks like a serene Moon!")

