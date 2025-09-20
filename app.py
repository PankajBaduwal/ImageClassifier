import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="🍎 Smart Produce Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #4CAF50, #45a049, #2E7D32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #4CAF50;
        text-align: center;
        margin: 2rem 0;
    }
    
    .result-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .prediction-text {
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .accuracy-text {
        font-size: 1.3rem;
        margin: 1rem 0;
    }
    
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .image-preview {
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 3px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🍎 Smart Produce Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image of fruits or vegetables to get instant AI-powered classification</p>', unsafe_allow_html=True)

# Load the model once (with caching to improve performance)
@st.cache_resource
def load_classification_model():
    try:
        # Try different possible paths
        possible_paths = [
            'Image_classify.keras',  # Same directory
            'model/Image_classify.keras',  # In model folder
            './Image_classify.keras'  # Current directory explicitly
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return load_model(path)
        
        # If no local model found, show download instructions
        st.error("⚠️ Model file not found! Please upload 'Image_classify.keras' to your repository.")
        st.info("💡 Make sure your model file is in the same directory as app.py")
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

model = load_classification_model()

# Categories
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
           'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 
           'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 
           'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 
           'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

# Image dimensions
img_height = 180
img_width = 180

# Sidebar with information
with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.info(f"**Categories Supported:** {len(data_cat)}")
    st.info(f"**Input Size:** {img_width}x{img_height}")
    
    st.markdown("### 🥬 Supported Categories")
    # Display categories in a nice format
    fruits = ['apple', 'banana', 'grapes', 'kiwi', 'lemon', 'mango', 'orange', 'pear', 'pineapple', 'pomegranate', 'watermelon']
    vegetables = [cat for cat in data_cat if cat not in fruits]
    
    with st.expander("🍎 Fruits"):
        for fruit in fruits:
            if fruit in data_cat:
                st.write(f"• {fruit.title()}")
    
    with st.expander("🥕 Vegetables"):
        for veg in vegetables:
            st.write(f"• {veg.title()}")

# Only show the app if model is loaded
if model is not None:
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📁 Upload Your Image")
        
        # File uploader with custom styling
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a fruit or vegetable for best results"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image with custom styling
            st.markdown("### 🖼️ Image Preview")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Display image info
            st.markdown("#### 📋 Image Details")
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Width", f"{image.size[0]}px")
                st.metric("Height", f"{image.size[1]}px")
            with col_info2:
                st.metric("Format", image.format)
                st.metric("Mode", image.mode)

    with col2:
        if uploaded_file is not None:
            st.markdown("### 🤖 AI Analysis")
            
            # Add a loading spinner
            with st.spinner('🔍 Analyzing image...'):
                # Preprocess the image
                image = Image.open(uploaded_file)
                image_resized = image.resize((img_height, img_width))
                img_arr = tf.keras.preprocessing.image.img_to_array(image_resized)
                img_arr = tf.expand_dims(img_arr, 0)  # Create a batch
                
                # Make prediction
                predict = model.predict(img_arr, verbose=0)
                score = tf.nn.softmax(predict[0])
                
                # Get top prediction
                predicted_class = data_cat[np.argmax(score)]
                confidence = float(np.max(score) * 100)  # Convert to Python float
            
            # Display results in a beautiful container
            st.markdown(f"""
            <div class="result-container">
                <h2>🎯 Classification Result</h2>
                <div class="prediction-text">
                    {predicted_class.title()}
                </div>
                <div class="accuracy-text">
                    Confidence: {confidence:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence meter
            st.markdown("#### 📊 Confidence Score")
            progress_color = "🟢" if confidence > 80 else "🟡" if confidence > 60 else "🔴"
            st.progress(confidence/100.0)  # Ensure it's a Python float
            st.write(f"{progress_color} **{confidence:.1f}%** confidence")
            
            # Show top 5 predictions
            st.markdown("#### 🏆 Top 5 Predictions")
            top_5_indices = np.argsort(score)[-5:][::-1]
            top_5_scores = [float(score[i] * 100) for i in top_5_indices]  # Convert to Python floats
            top_5_labels = [data_cat[i] for i in top_5_indices]
            
            # Create a DataFrame for the chart
            df = pd.DataFrame({
                'Category': [label.title() for label in top_5_labels],
                'Confidence': top_5_scores
            })
            
            # Use Streamlit's built-in bar chart
            st.bar_chart(df.set_index('Category'), height=300)
            
        else:
            # Placeholder content when no image is uploaded
            st.markdown("### 🌟 Ready to Classify!")
            st.markdown("""
            <div class="upload-section">
                <h3>👆 Upload an image to get started</h3>
                <p>Our AI model can identify 36 different types of fruits and vegetables with high accuracy!</p>
                <p><strong>Supported formats:</strong> JPG, JPEG, PNG</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show some example images or tips
            st.markdown("#### 💡 Tips for Best Results")
            tips_col1, tips_col2 = st.columns(2)
            with tips_col1:
                st.write("✅ Use clear, well-lit images")
                st.write("✅ Center the item in the frame")
            with tips_col2:
                st.write("✅ Avoid cluttered backgrounds")
                st.write("✅ Single item per image works best")

    # Footer
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    with col_footer1:
        st.metric("🎯 Categories", len(data_cat))
    with col_footer2:
        if uploaded_file is not None:
            st.metric("📁 Status", "Image Loaded")
        else:
            st.metric("📁 Status", "Ready")
    with col_footer3:
        st.metric("🤖 Model", "Loaded")

else:
    # Show instructions if model is not loaded
    st.error("🚨 **Model Not Found**")
    st.markdown("""
    ### 📥 To deploy this app, you need to:
    
    1. **Upload your model file** (`Image_classify.keras`) to the same directory as `app.py`
    2. **Push to GitHub** and deploy
    
    ### 📂 Expected file structure:
    ```
    your-repo/
    ├── app.py
    ├── Image_classify.keras  ← Your model file
    └── requirements.txt
    ```
    """)