import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image
import os
import sys

# Add parent directory to path to access model
sys.path.append('..')

# Configure page
st.set_page_config(
    page_title="Coriander vs Parsley AI Classifier",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .prediction-container {
        background: linear-gradient(135deg, #f0f8f0, #e8f5e8);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #2E8B57;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .coriander-result {
        color: #228B22;
    }
    .parsley-result {
        color: #32CD32;
    }
    .confidence-container {
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 25px;
        overflow: hidden;
        height: 25px;
        margin: 0.5rem 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff6b6b 0%, #ffa500 25%, #32CD32 50%, #228B22 75%, #006400 100%);
        transition: width 0.8s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .upload-box {
        border: 3px dashed #2E8B57;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background-color: #f8fff8;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4169E1;
        margin: 1rem 0;
    }
    .tip-box {
        background-color: #fffef0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FFD700;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    """Load the trained VGG16 model"""
    # Try different possible model paths
    possible_paths = [
        "../coriander_vs_parsely_vgg16_finetuned.h5",  # Parent directory
        "coriander_vs_parsley_vgg16.h5",     # Current directory
        "../models/coriander_vs_parsley_vgg16.h5",  # Models folder
        "models/coriander_vs_parsley_vgg16.h5"      # Local models folder
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        st.error("Model file not found!")
        st.info("""
        **Please ensure you have a trained model file named:**
        - `coriander_vs_parsley_vgg16.h5`
        
        **Expected locations:**
        - Parent directory: `../coriander_vs_parsley_vgg16.h5`
        - Current directory: `./coriander_vs_parsley_vgg16.h5`
        - Models folder: `../models/coriander_vs_parsley_vgg16.h5`
        
        **To train the model, run:**
        ```bash
        cd ..
        python vgg16_training.py
        ```
        """)
        st.stop()
    
    try:
        with st.spinner("Loading AI model..."):
            model = load_model(model_path)
        st.success(f"Model loaded successfully from: `{model_path}`")
        return model, model_path
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def preprocess_image(uploaded_file):
    """Preprocess uploaded image for prediction"""
    try:
        # Open image
        image = Image.open(uploaded_file)
        original_size = image.size
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            st.info(f"Converted image from {uploaded_file.type} to RGB format")
        
        # Resize to model input size (224x224 for VGG16)
        image_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = img_to_array(image_resized)
        img_array = img_array / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        return img_array, image, image_resized, original_size
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None, None

def predict_image(model, img_array):
    """Make prediction on preprocessed image"""
    try:
        # Get prediction
        with st.spinner("AI is analyzing your image..."):
            prediction = model.predict(img_array, verbose=0)
        
        confidence_raw = prediction[0][0]
        
        # Class mapping (assuming binary classification: 0=Coriander, 1=Parsley)
        if confidence_raw > 0.5:
            predicted_class = "Parsley"
            confidence_score = confidence_raw
            css_class = "parsley-result"
        else:
            predicted_class = "Coriander"
            confidence_score = 1 - confidence_raw
            css_class = "coriander-result"
            
        return predicted_class, confidence_score, css_class
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def get_confidence_interpretation(confidence_percent):
    """Get interpretation and color for confidence score"""
    if confidence_percent >= 90:
        return "**Extremely Confident** - Almost certainly correct!", "#006400"
    elif confidence_percent >= 80:
        return "**Very Confident** - Very likely correct!", "#228B22"
    elif confidence_percent >= 70:
        return "**Confident** - Likely correct", "#32CD32"
    elif confidence_percent >= 60:
        return "**Moderately Confident** - Probably correct", "#FFD700"
    else:
        return "**Low Confidence** - Please try a clearer image", "#FF6B6B"

def main():
    # Header
    st.markdown('<h1 class="main-header">Coriander vs Parsley AI Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image and let our AI identify whether it\'s coriander or parsley!</p>', unsafe_allow_html=True)
    
    # Load model
    model, model_path = load_trained_model()
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("## Model Information")
        st.markdown("**Architecture:** VGG16 Transfer Learning")
        st.markdown("**Input Size:** 224×224 pixels")
        st.markdown("**Classes:** Coriander, Parsley")
        st.markdown(f"**Model File:** `{os.path.basename(model_path)}`")
        
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path) / (1024*1024)
            st.markdown(f"**Model Size:** {model_size:.1f} MB")
        
        st.markdown("---")
        st.markdown("## Plant Information")
        
        st.markdown("### Coriander (Cilantro)")
        st.markdown("- Also known as cilantro")
        st.markdown("- Delicate, lacy leaves")
        st.markdown("- Strong, distinctive aroma")
        st.markdown("- Used in Asian, Mexican cuisine")
        
        st.markdown("### Parsley")
        st.markdown("- Flat-leaf or curly varieties")
        st.markdown("- Broader leaf segments")
        st.markdown("- Milder, fresh taste")
        st.markdown("- Common in European cuisine")
        
        st.markdown("---")
        st.markdown("## Pro Tips")
        st.markdown("- Use **well-lit, clear images**")
        st.markdown("- **Close-up shots** work best")
        st.markdown("- Show **multiple leaves** if possible")
        st.markdown("- Avoid **blurry or dark images**")
    
    # Main upload area
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.markdown("### Upload Your Plant Image")
    uploaded_file = st.file_uploader(
        "",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a clear image of coriander or parsley leaves for AI analysis"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # File information
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB",
            "File type": uploaded_file.type
        }
        
        # Preprocess image
        img_array, original_image, processed_image, original_size = preprocess_image(uploaded_file)
        
        if img_array is not None:
            # Layout: Original image and prediction side by side
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### Your Image")
                st.image(original_image, caption=f"Original Image ({original_size[0]}×{original_size[1]})", use_container_width=True)
                
                # File details
                st.markdown("**File Details:**")
                for key, value in file_details.items():
                    st.markdown(f"- **{key}:** {value}")
            
            with col2:
                st.markdown("#### AI Analysis")
                
                # Make prediction
                predicted_class, confidence, css_class = predict_image(model, img_array)
                
                if predicted_class is not None:
                    confidence_percent = confidence * 100
                    interpretation, color = get_confidence_interpretation(confidence_percent)
                    
                    # Prediction results container
                    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
                    
                    # Main result
                    st.markdown(f'<div class="prediction-result {css_class}">{predicted_class}</div>', unsafe_allow_html=True)
                    
                    # Confidence display
                    st.markdown('<div class="confidence-container">', unsafe_allow_html=True)
                    st.markdown(f"**Confidence Score:** {confidence_percent:.1f}%")
                    
                    # Animated confidence bar
                    st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence_percent}%; background-color: {color};">
                            {confidence_percent:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(interpretation)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show processed image
                    st.markdown("#### Processed for AI")
                    st.image(processed_image, caption="Resized to 224×224 for VGG16", use_container_width=True)
        
        # Try another image button
        if st.button("Try Another Image", type="secondary"):
            st.rerun()
    
    else:
        # Instructions when no image is uploaded
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### How to Use This AI Classifier")
        st.markdown("""
        1. **Click the upload button** above to select an image
        2. **Choose a clear photo** of coriander or parsley leaves
        3. **Wait for AI analysis** - usually takes just a few seconds
        4. **View the results** with confidence score and interpretation
        5. **Try multiple images** to test different scenarios!
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="tip-box">', unsafe_allow_html=True)
        st.markdown("### Best Practices for Accurate Results")
        st.markdown("""
        - **Lighting**: Use natural light or good indoor lighting
        - **Focus**: Ensure leaves are sharp and in focus  
        - **Angle**: Take photos from above or at a slight angle
        - **Background**: Plain backgrounds work better than cluttered ones
        - **Distance**: Close enough to see leaf details clearly
        - **Quality**: Higher resolution images generally work better
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Made with love using <strong>Streamlit</strong> and <strong>TensorFlow</strong><br>
        Powered by VGG16 Transfer Learning | Helping solve culinary confusion since 2025
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()