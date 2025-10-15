# 🧠 BrainWave AI: Advanced MRI Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11+-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.2+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

BrainWave AI is a state-of-the-art web application that combines deep learning models for brain MRI analysis, providing both **tumor classification** and **tumor segmentation** capabilities with an intuitive, modern web interface.

## 🎯 Key Features

- **🔍 Tumor Detection**: Multi-class classification (Glioma, Meningioma, Pituitary, No Tumor)
- **🎨 Tumor Segmentation**: Precise U-Net based tumor boundary detection
- **📱 Modern UI**: Glassmorphism design with drag-and-drop functionality
- **📊 Scan History**: Persistent storage and visualization of previous analyses
- **⚡ Real-time Processing**: Instant results with confidence scores
- **🌐 Web-based**: No installation required for end users

## 🏗️ Architecture Overview

### Model Pipeline
```
Input MRI Image → Classification Model (VGG16) → Tumor Type + Confidence
                ↓
                Segmentation Model (U-Net) → Tumor Mask + Overlay
```

### Technology Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **ML Framework**: TensorFlow/Keras
- **Image Processing**: OpenCV, PIL
- **UI Framework**: Bootstrap 5
- **Styling**: Glassmorphism, CSS Grid/Flexbox

## 📚 Dataset Information

### Primary Dataset: Brain Tumor MRI Dataset
- **Source**: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)
- **Size**: 7,023 MRI images
- **Classes**: 4 categories
  - **Glioma**: 1,321 images
  - **Meningioma**: 1,339 images  
  - **Pituitary**: 1,457 images
  - **No Tumor**: 2,000 images
- **Format**: JPG images, varying sizes
- **Split**: 80% Training, 20% Testing

### Segmentation Dataset: LGG MRI Segmentation
- **Source**: [Kaggle - LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- **Size**: 3,929 brain MRI images
- **Type**: FLAIR (Fluid Attenuated Inversion Recovery) sequences
- **Annotations**: Pixel-level tumor masks
- **Format**: TIFF images with corresponding masks

## 🤖 Model Architecture

### 1. Classification Model (VGG16-based)

```python
# Base Architecture
Input(128, 128, 3) → VGG16(pretrained) → Flatten → Dense(128) → Dense(4)

# Training Details
- Base Model: VGG16 (ImageNet pretrained)
- Fine-tuning: Last 3 layers unfrozen
- Optimizer: Adam (lr=0.0001)
- Loss: Sparse Categorical Crossentropy
- Augmentation: Brightness, Contrast, Rotation
- Epochs: 15
- Batch Size: 20
```

**Performance Metrics:**
- Training Accuracy: ~94%
- Validation Accuracy: ~91%
- Model Size: ~58MB (.h5 format)

### 2. Segmentation Model (U-Net)

```python
# Architecture
Input(256, 256, 3) → Encoder → Bottleneck → Decoder → Output(256, 256, 1)

# Training Details
- Architecture: Custom U-Net with skip connections
- Optimizer: Adam (lr=0.0001)
- Loss: Dice Coefficient Loss
- Metrics: IoU, Dice Coefficient, Binary Accuracy
- Epochs: 200
- Batch Size: 32
```

**Performance Metrics:**
- Dice Coefficient: ~0.89
- IoU Score: ~0.84
- Binary Accuracy: ~96%
- Model Size: ~31MB (.keras format)

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM recommended
- GPU optional (CPU inference supported)

### Quick Start (Windows PowerShell)

```powershell
# Clone repository
git clone https://github.com/yourusername/brainwave-ai.git
cd brainwave-ai

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Place your trained models
# - model.h5 (classification model)
# - unet.keras (segmentation model)

# Run application
python main.py
```

### Alternative Setup (Linux/Mac)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

### Docker Deployment (Optional)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "main.py"]
```

## 📁 Project Structure

```
BrainWave/
├── 📄 main.py                     # Flask application
├── 📁 templates/
│   └── 📄 index.html             # Main UI template
├── 📁 static/
│   ├── 📁 css/
│   │   └── 📄 style.css          # Modern styling
│   ├── 📁 uploads/               # User uploaded images
│   └── 📁 results/               # Generated masks/overlays
├── 📁 notebooks/
│   ├── 📄 mri_notebook.ipynb     # Classification training
│   └── 📄 Untitled (3).ipynb    # Segmentation training
├── 📄 model.h5                   # Trained classification model
├── 📄 unet.keras                 # Trained segmentation model
├── 📄 requirements.txt           # Python dependencies
├── 📄 scan_history.json         # Scan history storage
└── 📄 README.md                  # This documentation
```

## 🔬 Model Training Process

### Classification Model Training

The classification model was trained using transfer learning with VGG16:

1. **Data Preprocessing**:
   ```python
   # Image augmentation
   - Brightness adjustment (0.8-1.2x)
   - Contrast enhancement (0.8-1.2x)
   - Normalization (0-1 range)
   - Resize to 128x128
   ```

2. **Model Architecture**:
   ```python
   # VGG16 base + custom classifier
   base_model = VGG16(weights='imagenet', include_top=False)
   # Freeze most layers, fine-tune last 3
   model.add(Flatten())
   model.add(Dense(128, activation='relu'))
   model.add(Dropout(0.2))
   model.add(Dense(4, activation='softmax'))
   ```

3. **Training Configuration**:
   - Optimizer: Adam (lr=0.0001)
   - Loss: Sparse Categorical Crossentropy
   - Metrics: Sparse Categorical Accuracy
   - Callbacks: Early stopping, model checkpointing

### Segmentation Model Training

The U-Net model was trained for precise tumor segmentation:

1. **Data Preprocessing**:
   ```python
   # Image and mask processing
   - Resize to 256x256
   - Grayscale to RGB conversion
   - Normalization (0-1 range)
   - Mask binarization (threshold=0.5)
   ```

2. **U-Net Architecture**:
   ```python
   # Encoder-Decoder with skip connections
   - Encoder: 4 convolutional blocks with max pooling
   - Bottleneck: Dense convolutional layer
   - Decoder: 4 upsampling blocks with skip connections
   - Output: Single channel sigmoid activation
   ```

3. **Custom Loss Functions**:
   ```python
   # Dice coefficient loss for better segmentation
   def dice_coefficient_loss(y_true, y_pred):
       return 1 - dice_coefficient(y_true, y_pred)
   ```

## 🎨 User Interface Features

### Modern Design Elements
- **Glassmorphism Effects**: Frosted glass appearance with backdrop blur
- **Gradient Backgrounds**: Dynamic color schemes with floating animations
- **Responsive Design**: Mobile-first approach with breakpoints
- **Micro-interactions**: Smooth hover effects and transitions

### Key UI Components

1. **Upload Interface**:
   - Drag-and-drop functionality
   - Live image preview
   - File type validation
   - Progress indicators

2. **Results Display**:
   - Classification results with confidence meters
   - Side-by-side image comparison
   - Full-screen modal views
   - Interactive overlay visualization

3. **Scan History**:
   - Persistent storage (JSON-based)
   - Thumbnail grid layout
   - Timestamp tracking
   - Batch management (clear history)

## 📊 API Endpoints

### Main Routes

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/` | GET/POST | Main application interface |
| `/clear_history` | POST | Clear scan history |
| `/static/<path>` | GET | Serve static files |

### Response Format

```json
{
  "result": "Tumor: glioma",
  "confidence": "94.56",
  "uploaded_image": "/static/uploads/image.jpg",
  "mask_image": "/static/results/mask.png",
  "overlay_image": "/static/results/overlay.png",
  "scan_history": [...]
}
```

## 🔧 Configuration Options

### Environment Variables

```bash
FLASK_SECRET=your-secret-key        # Flask session secret
FLASK_ENV=development               # Environment mode
PORT=5000                          # Application port
```

### Model Configuration

```python
# Adjust in main.py
IMAGE_SIZE_CLASSIFICATION = 128     # Classification input size
IMAGE_SIZE_SEGMENTATION = 256       # Segmentation input size
CONFIDENCE_THRESHOLD = 0.5          # Segmentation threshold
HISTORY_LIMIT = 20                  # Max stored scans
```

## 🔍 Performance Optimization

### Backend Optimizations
- **Model Caching**: Models loaded once at startup
- **Image Processing**: Efficient PIL/OpenCV operations
- **Memory Management**: Proper cleanup of large arrays
- **File Handling**: Secure filename generation and validation

### Frontend Optimizations
- **CSS Grid/Flexbox**: Hardware-accelerated layouts
- **Image Lazy Loading**: Reduced initial load times
- **Animation Optimization**: GPU-accelerated transforms
- **Responsive Images**: Optimized for different screen sizes

## 🧪 Testing & Validation

### Model Validation

1. **Classification Metrics**:
   ```python
   # Confusion matrix analysis
   # ROC curve and AUC scores
   # Per-class precision/recall
   ```

2. **Segmentation Metrics**:
   ```python
   # Dice coefficient: 0.89
   # IoU score: 0.84
   # Pixel accuracy: 96%
   ```

### Testing Checklist
- [ ] Upload functionality (drag-drop + click)
- [ ] Model inference pipeline
- [ ] History storage and retrieval
- [ ] Mobile responsiveness
- [ ] Error handling and validation
- [ ] Performance under load

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   ```bash
   # Solution: Install compatible TensorFlow version
   pip install tensorflow==2.11.0
   ```

2. **Memory Issues**:
   ```python
   # Solution: Reduce batch size or image dimensions
   # Monitor memory usage during inference
   ```

3. **Segmentation Model Loading**:
   ```bash
   # Solution: Install Keras 3+ for .keras format
   pip install keras>=3.0
   ```

4. **File Upload Issues**:
   ```python
   # Check file permissions and upload directory
   # Verify allowed file extensions
   ```

## 🔮 Future Enhancements

### Planned Features
- [ ] **3D Volume Analysis**: Support for 3D MRI sequences
- [ ] **Multi-modal Fusion**: Combine different MRI sequences
- [ ] **Advanced Metrics**: Tumor volume calculation
- [ ] **Report Generation**: PDF reports with analysis
- [ ] **User Authentication**: Multi-user support
- [ ] **Cloud Deployment**: AWS/Azure integration
- [ ] **API Documentation**: REST API with Swagger
- [ ] **Real-time Collaboration**: Multi-user sessions

### Model Improvements
- [ ] **Ensemble Methods**: Combine multiple architectures
- [ ] **Attention Mechanisms**: Focus on relevant regions
- [ ] **Data Augmentation**: Advanced synthetic data generation
- [ ] **Transfer Learning**: Domain adaptation techniques
- [ ] **Uncertainty Quantification**: Confidence intervals

## 📄 License & Citation

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation
If you use this project in your research, please cite:

```bibtex
@software{brainwave_ai_2025,
  title={BrainWave AI: Advanced MRI Analysis Platform},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/brainwave-ai}
}
```

### Dataset Citations

```bibtex
@dataset{nickparvar2021brain,
  title={Brain Tumor MRI Dataset},
  author={Nickparvar, Masoud},
  year={2021},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset}
}

@dataset{buda2019lgg,
  title={LGG Segmentation Dataset},
  author={Buda, Mateusz and Saha, Ashirbani and Mazurowski, Maciej A},
  year={2019},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation}
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for API changes

## 📧 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/brainwave-ai/issues)
- **Email**: your.email@example.com
- **Documentation**: [Wiki](https://github.com/yourusername/brainwave-ai/wiki)

---

**⚠️ Disclaimer**: This tool is for research and educational purposes only. It is not intended for clinical diagnosis or medical decision-making. Always consult qualified healthcare professionals for medical advice.
#   B r a i n W a v e - A I - A p p - f o r - B r a i n - T u m o r - D e t e c t i o n - a n d - S e g m e n t a t i o n  
 