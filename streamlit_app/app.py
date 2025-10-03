import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import os
import requests
import gdown

# Configure page
st.set_page_config(
    page_title="Coriander vs Parsley Classifier",
    page_icon="🌿",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #2E8B57;
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 2rem;
}
.result-box {
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    font-size: 1.2rem;
    font-weight: bold;
    margin: 1rem 0;
}
.coriander {
    background-color: #90EE90;
    color: #2E7D32;
}
.parsley {
    background-color: #87CEEB;
    color: #1565C0;
}
.info-box {
    background-color: #F0F8FF;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #4CAF50;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Coriander vs Parsley Classifier</h1>', unsafe_allow_html=True)

# Model download function
@st.cache_resource
def download_model():
    """Download model from Google Drive if not exists"""
    model_filename = "coriander_vs_parsely_vgg16_finetuned.h5"
    
    # Check if model already exists
    if os.path.exists(model_filename):
        return model_filename
    
    # Google Drive file ID (replace with your actual file ID)
    file_id = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
    
    if file_id == "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
        st.error("Please configure the Google Drive file ID in the code!")
        st.info("""
        **Setup Instructions:**
        1. Upload 'coriander_vs_parsely_vgg16_finetuned.h5' to Google Drive
        2. Right-click → Share → Anyone with the link can view
        3. Copy the file ID from the shareable link
        4. Replace 'YOUR_GOOGLE_DRIVE_FILE_ID_HERE' in the code
        """)
        return None
    
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        with st.spinner("Downloading model file (first time only, ~115MB)..."):
            progress_placeholder = st.empty()
            progress_placeholder.info("Starting download...")
            
            # Download with gdown
            gdown.download(url, model_filename, quiet=False)
            progress_placeholder.success("Model downloaded successfully!")
        
        return model_filename
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        st.error("Please check that the Google Drive file is publicly accessible and the file ID is correct.")
        return None

# Model loading function
@st.cache_resource
def load_model():
    """Load the trained VGG16 model"""
    # First try to download model if needed
    model_path = download_model()
    
    if model_path and os.path.exists(model_path):
        try:
            with st.spinner("Loading model..."):
                model = keras.models.load_model(model_path)
            return model, model_path
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None
    
    return None, None

# Image preprocessing function
def preprocess_image(image):
    """Preprocess image for VGG16 model"""
    # Resize to VGG16 input size
    image = image.resize((224, 224))
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    
    return img_array

# Prediction function
def predict_herb(model, image):
    """Make prediction on the uploaded image"""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    # Get confidence score
    confidence = float(prediction[0][0])
    
    # Determine class (assuming 0 = coriander, 1 = parsley)
    if confidence > 0.5:
        herb = "Parsley"
        confidence_percent = confidence * 100
        css_class = "parsley"
    else:
        herb = "Coriander"
        confidence_percent = (1 - confidence) * 100
        css_class = "coriander"
    
    return herb, confidence_percent, css_class

# Main app
def main():
    # Load model
    model, model_path = load_model()
    
    if model is None:
        st.error("Model not found! Please ensure 'coriander_vs_parsely_vgg16_finetuned.h5' is in the project directory.")
        st.info("Expected model locations:")
        st.code("""
        - ../coriander_vs_parsely_vgg16_finetuned.h5
        - coriander_vs_parsely_vgg16_finetuned.h5
        - (parent directory)/coriander_vs_parsely_vgg16_finetuned.h5
        """)
        return
    
    # Show model info
    st.success(f"Model loaded successfully from: {model_path}")
    
    # Information section
    st.markdown("""
    <div class="info-box">
    <h3>About This Classifier</h3>
    <p>This AI model uses a fine-tuned VGG16 architecture to distinguish between coriander and parsley leaves. 
    Upload an image of either herb and get an instant classification with confidence score.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image of coriander or parsley...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image showing the leaves of the herb"
    )
    
    if uploaded_file is not None:
        # Display the image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Your uploaded image", use_column_width=True)
        
        with col2:
            st.subheader("Classification Result")
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                herb, confidence, css_class = predict_herb(model, image)
            
            # Display result
            st.markdown(f"""
            <div class="result-box {css_class}">
                This is <strong>{herb}</strong><br>
                Confidence: {confidence:.1f}%
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar for confidence
            st.progress(confidence / 100)
            
            # Additional info based on prediction
            if herb == "Coriander":
                st.info("""
                **Coriander (Cilantro) Facts:**
                - Also known as cilantro or Chinese parsley
                - Has a distinctive citrusy, slightly spicy flavor
                - Commonly used in Asian, Mexican, and Middle Eastern cuisine
                - Rich in vitamins A, C, and K
                """)
            else:
                st.info("""
                **Parsley Facts:**
                - Two main varieties: flat-leaf (Italian) and curly-leaf
                - Has a fresh, slightly peppery flavor
                - Commonly used as garnish and in European cuisine
                - Rich in vitamins C, K, and folate
                """)
    
    # Tips section
    st.markdown("---")
    st.subheader("Tips for Best Results")
    st.markdown("""
    - Use well-lit, clear images
    - Show the leaves clearly without too much background
    - Ensure the herb takes up most of the image frame
    - Both fresh and dried herbs can be classified
    """)

if __name__ == "__main__":
    main()
