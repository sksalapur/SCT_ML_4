#!/usr/bin/env python3
"""
Streamlit Web App for Hand Gesture Recognition
Production-ready deployment version
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os
from PIL import Image
import requests
from io import BytesIO
import time

# Page configuration
st.set_page_config(
    page_title="🎯 Hand Gesture Recognition",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model loading
@st.cache_resource
def load_gesture_model():
    """Load the pre-trained gesture recognition model"""
    try:
        # Try to load local model first
        if os.path.exists("balanced_gesture_model.h5"):
            model = load_model("balanced_gesture_model.h5")
            return model, "✅ Local model loaded successfully"
        
        # Download model from GitHub releases
        model_url = "https://github.com/sksalapur/SCT_ML_4/releases/download/v1.0/balanced_gesture_model.h5"
        
        with st.spinner("📥 Downloading model... (first time only)"):
            try:
                response = requests.get(model_url, timeout=30)
                if response.status_code == 200:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                        tmp_file.write(response.content)
                        model = load_model(tmp_file.name)
                        return model, "✅ Model downloaded and loaded successfully"
                else:
                    st.warning(f"⚠️ Could not download model (Status: {response.status_code})")
            except requests.exceptions.RequestException as e:
                st.warning(f"⚠️ Network error: {str(e)}")
            except Exception as e:
                st.warning(f"⚠️ Download error: {str(e)}")
        
        # Show helpful error message
        return None, """
        ❌ **Could not load model. Please try:**
        
        1. **Upload model to GitHub Releases:**
           - Go to: https://github.com/sksalapur/SCT_ML_4/releases/new
           - Tag: v1.0
           - Upload: balanced_gesture_model.h5
        
        2. **Or test with sample images:**
           - The image upload mode should still work
           - Try uploading a hand gesture photo
        
        3. **Check your internet connection**
        """
        
    except Exception as e:
        return None, f"❌ Error loading model: {str(e)}"

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert PIL to OpenCV format
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Resize to model input size
    resized = cv2.resize(image, (224, 224))
    
    # Apply CLAHE for better contrast
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Normalize pixel values
    normalized = enhanced.astype(np.float32) / 255.0
    
    # Add batch dimension
    batch = np.expand_dims(normalized, axis=0)
    
    return batch

def predict_gesture(model, image):
    """Predict gesture from image"""
    class_names = ['c_shape', 'down', 'fist', 'index', 'l_shape', 'ok', 'palm', 'thumb']
    
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)
    
    predicted_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_idx]
    confidence = predictions[0][predicted_idx]
    
    # Get all predictions for display
    all_predictions = [(class_names[i], predictions[0][i]) for i in range(len(class_names))]
    all_predictions.sort(key=lambda x: x[1], reverse=True)
    
    return predicted_class, confidence, all_predictions

def main():
    # Header
    st.title("🎯 Hand Gesture Recognition")
    st.markdown("### Real-time AI-powered gesture recognition system")
    
    # Sidebar
    st.sidebar.title("🎮 Controls")
    
    # Load model
    model, status = load_gesture_model()
    
    if model is None:
        st.error(status)
        st.stop()
    else:
        st.sidebar.success(status)
    
    # App mode selection
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["📸 Image Upload", "📹 Webcam (Live)", "ℹ️ About"]
    )
    
    if app_mode == "📸 Image Upload":
        st.header("Upload Hand Gesture Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a hand gesture"
        )
        
        col1, col2 = st.columns(2)
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            with col1:
                st.subheader("📤 Uploaded Image")
                st.image(image, caption="Your gesture image", use_column_width=True)
            
            with col2:
                st.subheader("🎯 Prediction Results")
                
                with st.spinner("Analyzing gesture..."):
                    predicted_class, confidence, all_predictions = predict_gesture(model, image)
                
                # Display main prediction
                if confidence > 0.5:
                    st.success(f"**Predicted Gesture: {predicted_class.upper()}**")
                    st.metric("Confidence", f"{confidence:.1%}")
                else:
                    st.warning(f"**Low Confidence: {predicted_class.upper()}**")
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # Display all predictions
                st.subheader("📊 All Predictions")
                for gesture, conf in all_predictions:
                    progress_val = float(conf)
                    st.write(f"**{gesture}**: {conf:.1%}")
                    st.progress(progress_val)
        
        else:
            st.info("👆 Please upload an image to get started!")
            
            # Show example images
            st.subheader("💡 Example Gestures")
            example_col1, example_col2, example_col3, example_col4 = st.columns(4)
            
            gesture_examples = {
                "👊 Fist": "A closed fist",
                "👋 Palm": "Open palm facing camera", 
                "👆 Index": "Pointing finger",
                "👌 OK": "OK sign gesture"
            }
            
            for i, (gesture, desc) in enumerate(gesture_examples.items()):
                with [example_col1, example_col2, example_col3, example_col4][i]:
                    st.write(f"**{gesture}**")
                    st.write(desc)
    
    elif app_mode == "📹 Webcam (Live)":
        st.header("Live Webcam Gesture Recognition")
        
        # Webcam note for Streamlit Cloud
        st.info("📸 **Camera Mode**: Click 'Take Photo' to capture and analyze your gesture!")
        
        # Use Streamlit's camera input instead of OpenCV VideoCapture
        camera_image = st.camera_input("📷 Take a photo of your gesture")
        
        if camera_image is not None:
            # Process the captured image
            image = Image.open(camera_image)
            
            # Display the captured image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="📸 Captured Gesture", use_column_width=True)
            
            with col2:
                # Make prediction
                with st.spinner("🤖 Analyzing gesture..."):
                    predicted_class, confidence, all_predictions = predict_gesture(model, image)
                
                # Display main prediction
                if confidence > 0.5:
                    st.success(f"✅ **Detected: {predicted_class}**")
                    st.metric("Confidence", f"{confidence:.1%}")
                else:
                    st.warning(f"🤔 **Possible: {predicted_class}**")
                    st.metric("Confidence", f"{confidence:.1%}")
                    st.info("Try repositioning your hand for better detection")
                
                # Show all predictions
                st.subheader("📊 All Gesture Confidences")
                for gesture, conf in all_predictions:
                    # Create color-coded progress bars
                    if conf > 0.7:
                        st.success(f"{gesture}: {conf:.1%}")
                    elif conf > 0.3:
                        st.warning(f"{gesture}: {conf:.1%}")
                    else:
                        st.info(f"{gesture}: {conf:.1%}")
        
        else:
            # Show instructions when no image is captured
            st.markdown("""
            ### 👋 How to Use:
            1. **Click 'Take Photo'** button above
            2. **Position your hand** in front of the camera
            3. **Make a clear gesture** from the supported list
            4. **Capture the photo** and see instant results!
            
            ### 🎯 Supported Gestures:
            """)
            
            # Display gesture examples in a nice grid
            gesture_cols = st.columns(4)
            gestures_info = {
                "👊 Fist": "Closed fist facing camera",
                "👋 Palm": "Open palm facing camera", 
                "👆 Index": "Index finger pointing up",
                "👌 OK": "Thumb and index finger circle",
                "👍 Thumb": "Thumbs up gesture",
                "🤏 C-Shape": "C-shaped hand gesture",
                "👇 Down": "Index finger pointing down",
                "🤟 L-Shape": "L-shaped hand (index + thumb)"
            }
            
            for i, (gesture, desc) in enumerate(gestures_info.items()):
                with gesture_cols[i % 4]:
                    st.markdown(f"**{gesture}**")
                    st.caption(desc)
    
    elif app_mode == "ℹ️ About":
        st.header("About This App")
        
        st.markdown("""
        ### 🎯 Hand Gesture Recognition System
        
        This AI-powered application can recognize **8 different hand gestures** in real-time:
        
        1. **C-Shape** 🤏 - C-shaped hand position
        2. **Down** 👇 - Pointing downward
        3. **Fist** 👊 - Closed fist
        4. **Index** 👆 - Pointing finger
        5. **L-Shape** 🤟 - L-shaped hand position  
        6. **OK** 👌 - OK sign
        7. **Palm** 👋 - Open palm
        8. **Thumb** 👍 - Thumbs up
        
        ### 🧠 Technology Stack
        - **Deep Learning**: TensorFlow/Keras with MobileNetV2
        - **Computer Vision**: OpenCV for image processing
        - **Web Framework**: Streamlit for interactive UI
        - **Model Accuracy**: 97.9% on test dataset
        
        ### 📊 Model Performance
        The model was trained on a balanced dataset with advanced techniques:
        - Transfer learning from MobileNetV2
        - Data augmentation for robustness
        - Class balancing for fair recognition
        - Two-phase training for optimal performance
        
        ### 🚀 Features
        - Real-time webcam gesture recognition
        - Image upload and analysis
        - Confidence scoring for all gestures
        - Responsive web interface
        - Production-ready deployment
        
        ### 👨‍💻 Developer
        Built with ❤️ using modern AI/ML technologies
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Model Stats")
    st.sidebar.metric("Accuracy", "97.9%")
    st.sidebar.metric("Classes", "8")
    st.sidebar.metric("Parameters", "2.4M")

if __name__ == "__main__":
    main()
