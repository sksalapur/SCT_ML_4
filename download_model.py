#!/usr/bin/env python3
"""
Model download script for deployment
Downloads the trained model from cloud storage
"""

import os
import requests
from pathlib import Path
import hashlib

def download_model():
    """Download the gesture recognition model"""
    
    print("📥 DOWNLOADING GESTURE RECOGNITION MODEL")
    print("=" * 50)
    
    # Model file info
    model_filename = "balanced_gesture_model.h5"
    
    # Check if model already exists
    if os.path.exists(model_filename):
        print(f"✅ Model already exists: {model_filename}")
        return True
    
    # Model URLs (you'll need to upload to one of these)
    model_urls = [
        # GitHub Releases (recommended for <100MB files)
        "https://github.com/YOUR_USERNAME/gesture-recognition/releases/download/v1.0/balanced_gesture_model.h5",
        
        # Google Drive (for larger files)
        "https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_FILE_ID",
        
        # Hugging Face (recommended for ML models)
        "https://huggingface.co/YOUR_USERNAME/gesture-recognition/resolve/main/balanced_gesture_model.h5",
        
        # Dropbox
        "https://www.dropbox.com/s/YOUR_DROPBOX_LINK/balanced_gesture_model.h5?dl=1",
    ]
    
    for i, url in enumerate(model_urls):
        try:
            print(f"🌐 Trying download source {i+1}...")
            
            # Download with progress
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                
                print(f"📦 Downloading {model_filename} ({total_size/1024/1024:.1f} MB)...")
                
                with open(model_filename, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Progress bar
                            if total_size > 0:
                                progress = downloaded / total_size * 100
                                print(f"\r⏳ Progress: {progress:.1f}%", end="", flush=True)
                
                print(f"\n✅ Model downloaded successfully: {model_filename}")
                return True
            
        except Exception as e:
            print(f"❌ Failed to download from source {i+1}: {e}")
            continue
    
    print("❌ All download sources failed!")
    print("\n💡 MANUAL SETUP REQUIRED:")
    print("1. Upload your model to GitHub Releases, Google Drive, or Hugging Face")
    print("2. Update the URLs in this script")
    print("3. Or place 'balanced_gesture_model.h5' in the project root")
    
    return False

def verify_model():
    """Verify the downloaded model"""
    model_file = "balanced_gesture_model.h5"
    
    if not os.path.exists(model_file):
        return False
    
    # Check file size (should be around 9-15MB)
    file_size = os.path.getsize(model_file) / 1024 / 1024
    
    if file_size < 5:
        print(f"⚠️ Warning: Model file seems too small ({file_size:.1f} MB)")
        return False
    
    print(f"✅ Model verification passed ({file_size:.1f} MB)")
    return True

if __name__ == "__main__":
    success = download_model()
    
    if success:
        verify_model()
        print("\n🎉 Setup complete! You can now run the Streamlit app.")
    else:
        print("\n❌ Setup failed. Please check the instructions above.")
