# 🎯 Hand Gesture Recognition

A real-time AI-powered hand gesture recognition system built with TensorFlow and Streamlit.

## 🚀 [Live Demo](https://your-app-name.streamlit.app) 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## ✨ Features

- **Real-time Recognition**: Live webcam gesture detection
- **8 Gesture Classes**: Supports fist, palm, index, ok, thumb, c_shape, down, l_shape
- **97.9% Accuracy**: High-performance deep learning model
- **Web Interface**: Easy-to-use Streamlit web app
- **Image Upload**: Test with your own gesture images

## 🎯 Supported Gestures

| Gesture | Description | Use Case |
|---------|-------------|----------|
| 👊 Fist | Closed fist | Stop/Select |
| 👋 Palm | Open palm | Hello/Stop |
| 👆 Index | Pointing finger | Direction/Select |
| 👌 OK | OK sign | Confirm/Good |
| 👍 Thumb | Thumbs up | Like/Approve |
| 🤏 C-Shape | C-shaped hand | Grab/Pinch |
| 👇 Down | Pointing down | Navigate down |
| 🤟 L-Shape | L-shaped hand | Custom action |

## 🔧 Technology Stack

- **Deep Learning**: TensorFlow 2.x + Keras
- **Model**: MobileNetV2 (Transfer Learning)
- **Computer Vision**: OpenCV
- **Web Framework**: Streamlit
- **Deployment**: Streamlit Cloud

## 📊 Model Performance

- **Training Accuracy**: 97.2%
- **Validation Accuracy**: 97.9%
- **Model Size**: 2.4M parameters
- **Inference Speed**: ~50ms per prediction
- **Dataset**: 5,120 training images (balanced)

## 🚀 Quick Start

### Option 1: Use the Web App (Recommended)
Visit the [live demo](https://your-app-name.streamlit.app) and start using immediately!

### Option 2: Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/gesture-recognition.git
cd gesture-recognition
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the model** (auto-downloaded on first run)

4. **Run the app**
```bash
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
gesture-recognition/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── .gitignore               # Git ignore file
├── setup.py                 # Package setup
└── models/
    └── download_model.py    # Model download script
```

## 🎮 Usage

### Web Interface
1. **Image Upload Mode**: Upload a gesture image and get instant predictions
2. **Webcam Mode**: Enable live gesture recognition through your camera
3. **Confidence Scores**: View prediction confidence for all gesture classes

### API Usage (Advanced)
```python
from streamlit_app import load_gesture_model, predict_gesture
import cv2

# Load model
model, status = load_gesture_model()

# Make prediction
image = cv2.imread("gesture_image.jpg")
predicted_class, confidence, all_predictions = predict_gesture(model, image)

print(f"Predicted: {predicted_class} ({confidence:.2f})")
```

## 🔬 Model Training

The model was trained using:
- **Dataset**: LeapGestRecog dataset (balanced to 640 images per class)
- **Architecture**: MobileNetV2 with custom head
- **Training**: Two-phase training (frozen → fine-tuning)
- **Augmentation**: Rotation, zoom, shift, brightness adjustment
- **Optimization**: Adam optimizer with learning rate scheduling

## 📈 Performance Metrics

| Gesture | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| c_shape | 0.952 | 1.000 | 0.976 |
| down | 1.000 | 0.975 | 0.987 |
| fist | 0.981 | 0.975 | 0.978 |
| index | 1.000 | 1.000 | 1.000 |
| l_shape | 0.889 | 1.000 | 0.941 |
| ok | 1.000 | 0.912 | 0.954 |
| palm | 1.000 | 0.981 | 0.991 |
| thumb | 1.000 | 0.988 | 0.994 |

## 🛠️ Development

### Setup Development Environment
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/gesture-recognition.git
cd gesture-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```

### Testing
```bash
python -m pytest tests/
```

## 🚀 Deployment

### Streamlit Cloud (Recommended)
1. Fork this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy directly from GitHub
4. Your app will be live at `https://your-app-name.streamlit.app`

### Docker
```bash
docker build -t gesture-recognition .
docker run -p 8501:8501 gesture-recognition
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: LeapGestRecog dataset from Kaggle
- **Inspiration**: Advances in computer vision and HCI

## 📞 Contact

- **GitHub**: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- **Email**: your.email@example.com
- **Live Demo**: [gesture-recognition.streamlit.app](https://your-app-name.streamlit.app)

---

**⭐ Star this repository if you found it helpful!**
