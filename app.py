import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('pneumonia_detection_model.keras')

model = load_my_model()

st.title("🫁 Pneumonia Detection AI")
st.write("Upload a Chest X-ray image to check for signs of Pneumonia.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-ray.', width="content")
    
    # Preprocess the image for the model
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make Prediction
    if st.button('Analyze Image'):
        prediction = model.predict(img_array)
        
        # Display Results
        if prediction[0][0] > 0.5:
            st.error(f"Prediction: **Pneumonia Detected** (Confidence: {prediction[0][0]*100:.2f}%)")
        else:
            st.success(f"Prediction: **Normal** (Confidence: {(1-prediction[0][0])*100:.2f}%)")
            
    st.warning("Disclaimer: This is an only prototype")