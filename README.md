# 🛡️ Telugu Fake News Classifier

AI-powered misinformation detection system for Telugu language news with English translation support.

## 📋 Overview

This project uses machine learning to classify Telugu news articles as **Real**, **Fake**, or **Unverifiable**. It includes a beautiful web interface with English translation capabilities.

### ✨ Features

- **ML Classification**: Trained model with F1 score of 0.78
- **English Translation**: Automatic Telugu to English translation using Google Translate
- **Keyword Detection**: Identifies suspicious keywords in text
- **Modern UI**: Beautiful, responsive web interface with animations
- **Real-time Analysis**: Instant classification results
- **Sample Texts**: 5 pre-loaded examples for testing

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Web App

```bash
python app_final.py
```

### 3. Open in Browser

Navigate to: **http://localhost:5000**

## 📁 Project Structure

```
telugu_fake_news_det/
├── app_final.py              # Main Flask web application (ENHANCED UI)
├── interactive_classifier.py # Command-line interface
├── test_model_directly.py    # Direct model testing script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── src/                      # Source code
│   ├── train.py             # Model training
│   ├── predict.py           # Prediction logic
│   ├── preprocess.py        # Text preprocessing & keyword detection
│   ├── evaluate.py          # Model evaluation
│   ├── explain.py           # LIME explanations
│   ├── dataset_manager.py   # Dataset handling
│   └── config.py            # Configuration
│
├── data/                     # Datasets
│   ├── train.csv            # Training data (41,318 samples)
│   ├── val.csv              # Validation data
│   ├── factify_telugu.csv   # FACTIFY dataset
│   └── indicglue_telugu.csv # IndicGLUE dataset
│
├── telugu-fake-news-model/  # Trained model files
│   ├── model.safetensors    # Model weights
│   ├── config.json          # Model config
│   ├── tokenizer.json       # Tokenizer
│   └── tokenizer_config.json
│
├── results/                  # Training results
│   ├── training_report.json # Performance metrics
│   └── dataset_stats.json   # Dataset statistics
│
└── notebooks/               # Jupyter notebooks
    └── Telugu_Fake_News_Training_Colab.ipynb
```

## 🎯 Usage

### Web Interface (Recommended)

1. Run `python app_final.py`
2. Open http://localhost:5000
3. Enter Telugu text or click a sample
4. Check "Show English Translation" (enabled by default)
5. Click "Analyze"
6. View results with confidence scores and keywords

### Command Line

```bash
python interactive_classifier.py
```

### Direct Testing

```bash
python test_model_directly.py
```

## 📊 Model Performance

- **F1 Score**: 0.78
- **Training Samples**: 41,318
- **Improvement over GPT-4**: 27.87%
- **Base Model**: MuRIL (Multilingual Representations for Indian Languages)

### Classification Categories

1. **Real** ✅ - Verified, factual news
2. **Fake** 🚨 - Misinformation, scams, false claims
3. **Unverifiable** ⚠️ - Cannot be confirmed or denied

## 🔧 Technical Details

### Dependencies

- **PyTorch** 2.3.1+ - Deep learning framework
- **Transformers** 4.44.2 - Hugging Face models
- **Flask** 3.0+ - Web framework
- **googletrans** 4.0.0rc1 - Translation API
- **LIME** 0.2.0.1 - Model explanations
- **scikit-learn** 1.3.0+ - ML utilities

### Model Architecture

- **Base**: google/muril-base-cased
- **Task**: Sequence Classification (3 classes)
- **Max Length**: 512 tokens
- **Training**: Fine-tuned on Telugu fake news dataset

## 🌐 Web Interface Features

### Modern UI Design

- **Gradient backgrounds** with depth effects
- **Smooth animations** (slide-up, bounce, shimmer)
- **Glassmorphism** effects
- **Responsive design** for mobile/tablet/desktop
- **Interactive elements** with hover effects
- **Color-coded results** for easy interpretation

### Translation

- Automatic Telugu to English translation
- Uses Google Translate API
- Fallback to word-by-word dictionary
- Toggle on/off with checkbox

### Keyword Detection

Identifies suspicious patterns:
- Scam indicators (ఉచితంగా, గ్యారంటీ)
- Urgency triggers (వెంటనే, రేపు)
- Action prompts (క్లిక్, షేర్, ఫార్వర్డ్)

## 📝 Sample Texts

The app includes 5 pre-loaded samples:

1. **Fake** - Government scam (free money)
2. **Real** - ISRO satellite launch
3. **Fake** - Health miracle cure
4. **Real** - Weather/flood news
5. **Unverifiable** - WHO warning (forward message)

## 🛠️ Development

### Training a New Model

```bash
python src/train.py
```

### Evaluating Model

```bash
python src/evaluate.py
```

### Running Tests

```bash
pytest tests/
```

## 📦 Batch Files

- `start_app.bat` - Quick start the web app
- `RUN_MODEL_TEST.bat` - Run model tests

## 🔍 How It Works

1. **Input**: User enters Telugu text
2. **Preprocessing**: Text cleaning and normalization
3. **Tokenization**: Convert to model input format
4. **Classification**: MuRIL model predicts category
5. **Keyword Detection**: Scan for suspicious patterns
6. **Translation**: Convert to English (if enabled)
7. **Output**: Display results with confidence scores

## 🎨 UI Enhancements

### Version 2.0 Features

- Custom Inter font from Google Fonts
- Advanced CSS animations (0.3s-1s transitions)
- Shimmer effects on progress bars
- Pulse animations on result cards
- Bounce effects on icons
- Hover transformations on all buttons
- Glassmorphism and depth effects
- Mobile-responsive breakpoints

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- Tests pass
- Documentation is updated

## 📧 Support

For issues or questions, please check the code comments or create an issue.

---

**Last Updated**: May 16, 2026  
**Version**: 2.0 (Enhanced UI with Translation)  
**Status**: ✅ Production Ready
