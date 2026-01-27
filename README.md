# AspireNex - Machine Learning Models Collection 🚀

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive collection of machine learning models for real-world applications including fraud detection, customer analytics, and spam filtering.

## 📋 Table of Contents
- [Models Overview](#models-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Dataset Information](#dataset-information)
- [Contributing](#contributing)
- [License](#license)

## 🤖 Models Overview

### 1. Credit Card Fraud Detection
**File:** `creditcardfrauddetection.py`

Detects fraudulent credit card transactions using machine learning algorithms.

**Features:**
- **Algorithms:** Logistic Regression & Decision Tree
- **Preprocessing:** StandardScaler for feature normalization
- **Evaluation:** Accuracy, Confusion Matrix, Classification Report

**Key Highlights:**
- Handles imbalanced datasets common in fraud detection
- Compares multiple algorithms for optimal performance
- Comprehensive evaluation metrics

### 2. Customer Churn Prediction
**File:** `customerchurnprediction.py`

Predicts customer churn probability to help businesses retain customers.

**Features:**
- **Algorithm:** Random Forest Classifier
- **Preprocessing:** Label encoding for categorical variables
- **Metrics:** Accuracy, Precision, Recall, F1-Score

**Key Highlights:**
- Automated categorical variable encoding
- Robust ensemble method for better predictions
- Multiple evaluation metrics for comprehensive analysis

### 3. SMS Spam Detection
**File:** `smsspamdetection.py`

Classifies SMS messages as spam or legitimate using natural language processing.

**Features:**
- **Algorithm:** Multinomial Naive Bayes
- **Text Processing:** TF-IDF Vectorization
- **Preprocessing:** Text cleaning and normalization

**Key Highlights:**
- Advanced text preprocessing pipeline
- TF-IDF feature extraction for better text representation
- High accuracy spam detection

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/patelritiq/AspireNex.git
   cd AspireNex
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your datasets:**
   - Download the required datasets (links provided below)
   - Rename your datasets to match the expected filenames:
     - Credit card dataset → `creditcard.csv`
     - Customer churn dataset → `Churn.csv`
     - SMS spam dataset → `spam.csv`
   - Place all CSV files in the same directory as the Python scripts

### Required Dependencies
```
pandas>=1.3.0
scikit-learn>=1.0.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 🚀 Usage

### Credit Card Fraud Detection
```bash
python creditcardfrauddetection.py
```
**Required Dataset:** `creditcard.csv`

### Customer Churn Prediction
```bash
python customerchurnprediction.py
```
**Required Dataset:** `Churn.csv`

### SMS Spam Detection
```bash
python smsspamdetection.py
```
**Required Dataset:** `spam.csv`

## 📊 Model Performance

| Model | Algorithm | Accuracy | Key Metrics |
|-------|-----------|----------|-------------|
| **Fraud Detection** | Logistic Regression | ~99.9% | High Precision for Fraud Class |
| **Fraud Detection** | Decision Tree | ~99.8% | Good Recall for Fraud Detection |
| **Churn Prediction** | Random Forest | ~85-90% | Balanced Precision & Recall |
| **Spam Detection** | Naive Bayes | ~95-98% | Fast Training & Prediction |

## 📁 Dataset Information

### Credit Card Fraud Detection
- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Required filename:** `creditcard.csv`
- **Features:** 30 numerical features (V1-V28 + Time + Amount)
- **Target:** Class (0: Normal, 1: Fraud)
- **Size:** ~284,807 transactions

### Customer Churn Prediction
- **Source:** Any customer churn dataset with similar structure
- **Required filename:** `Churn.csv`
- **Features:** Customer demographics, account info, usage patterns
- **Target:** Churn (0: Retained, 1: Churned)
- **Note:** The model automatically handles categorical variables

### SMS Spam Detection
- **Source:** [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
- **Required filename:** `spam.csv`
- **Features:** Text messages
- **Target:** Label (ham/spam)
- **Encoding:** Use ISO-8859-1 encoding when saving the file

## 🔧 Project Structure

```
AspireNex/
├── README.md
├── requirements.txt
├── creditcardfrauddetection.py
├── customerchurnprediction.py
├── smsspamdetection.py
├── creditcard.csv (your dataset)
├── Churn.csv (your dataset)
└── spam.csv (your dataset)
```

## 🐛 Troubleshooting

### Common Issues
- **FileNotFoundError:** Ensure your CSV files are named exactly as specified and placed in the same directory
- **Encoding Issues:** For spam.csv, make sure it's saved with ISO-8859-1 encoding
- **Memory Issues:** Large datasets may require more RAM; consider using a machine with sufficient memory

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/AmazingFeature`
3. **Commit changes:** `git commit -m 'Add AmazingFeature'`
4. **Push to branch:** `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include tests for new features
- Update documentation as needed

## 📈 Model Metrics Explained

- **Accuracy:** Overall correctness of the model
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean of precision and recall

## 🐛 Known Issues & Solutions

- **Dataset Path Issues:** Ensure CSV files are in the same directory as Python scripts
- **Memory Issues:** For large datasets, consider using a machine with sufficient RAM
- **Encoding Issues:** Use `encoding='ISO-8859-1'` when saving the spam dataset

## 📞 Support

If you encounter any issues or have questions:
- **Create an Issue:** [GitHub Issues](https://github.com/patelritiq/AspireNex/issues)
- **Email:** [Your Email]
- **LinkedIn:** [Your LinkedIn Profile]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle** for providing high-quality datasets
- **scikit-learn** community for excellent ML tools
- **Open source contributors** who make projects like this possible

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/patelritiq/AspireNex?style=social)
![GitHub forks](https://img.shields.io/github/forks/patelritiq/AspireNex?style=social)
![GitHub issues](https://img.shields.io/github/issues/patelritiq/AspireNex)

---

**Made with ❤️ by [Ritik Pratap Singh Patel](https://github.com/patelritiq)**

*Empowering businesses with intelligent machine learning solutions*
