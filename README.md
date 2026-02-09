# AspireNex Machine Learning Online Internship Projects

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A collection of **3 machine learning classification systems** developed during a project-based virtual internship with AspireNex (July 2024 – September 2024). These models demonstrate practical applications of supervised learning algorithms on real-world datasets spanning fraud detection, customer analytics, and spam filtering.

## � Project Overview

### Internship Context
**Organization**: AspireNex  
**Duration**: July 2024 – September 2024  
**Role**: Machine Learning Virtual Intern  
**Type**: Project-based remote internship

### Project Scope
- **3 ML Classification Systems**: Fraud detection, churn prediction, spam filtering
- **4 Algorithm Implementations**: Logistic Regression, Decision Tree, Random Forest, Naive Bayes
- **Dataset Scale**: 20,000 – 280,000 records across projects
- **Feature Types**: Numerical, categorical, and text data processing
- **Focus**: Handling class imbalance, algorithm comparison, documented pipelines

---

## 🎯 Project Impact & Applications

### For AspireNex
These models contribute to AspireNex's internal resources by:
- **Training Repository**: Serving as reference implementations for future interns and training programs
- **Model Library**: Adding documented, reusable ML pipelines to the organization's codebase
- **Capability Showcase**: Demonstrating practical ML applications across different domains
- **Benchmarking**: Providing algorithm comparison frameworks for educational purposes

### Real-World Applications
While developed as foundational learning projects, these models demonstrate approaches to critical business problems:

**Fraud Detection**
- Protect financial institutions from fraudulent transactions
- Reduce financial losses and customer impact
- Enable real-time transaction monitoring systems

**Customer Churn Prediction**
- Help businesses identify at-risk customers proactively
- Enable targeted retention strategies and personalized interventions
- Reduce revenue loss and improve customer lifetime value

**Spam Detection**
- Improve user experience by filtering unwanted messages
- Reduce security risks from phishing and malicious content
- Enhance communication platform reliability

---

## 🤖 Classification Systems

### 1. Credit Card Fraud Detection
**File**: `models/creditcardfrauddetection.py`  
**Dataset Size**: ~280,000 transaction records  
**Problem Type**: Binary classification with severe class imbalance

**Technical Implementation**:
- **Algorithms Compared**: Logistic Regression vs Decision Tree Classifier
- **Feature Engineering**: StandardScaler normalization for 30 numerical features
- **Evaluation Metrics**: Accuracy, confusion matrix, precision/recall for minority class
- **Challenge**: Handling highly imbalanced dataset (fraud cases < 0.2% of transactions)

**Key Features**:
- Comparative analysis of linear vs tree-based approaches
- Focus on minority-class (fraud) detection performance
- Comprehensive evaluation framework for imbalanced classification

**Business Value**: Demonstrates transaction monitoring systems that can identify fraudulent patterns while minimizing false positives that disrupt legitimate customer transactions.

---

### 2. Customer Churn Prediction
**File**: `models/customerchurnprediction.py`  
**Dataset Size**: ~20,000 customer records  
**Problem Type**: Binary classification with mixed feature types

**Technical Implementation**:
- **Algorithm**: Random Forest Classifier (ensemble method)
- **Feature Engineering**: Automated label encoding for categorical variables
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score
- **Data Handling**: Mixed numerical and categorical customer attributes

**Key Features**:
- Robust ensemble approach for handling complex feature interactions
- Automated preprocessing pipeline for categorical data
- Multi-metric evaluation for balanced performance assessment

**Business Value**: Enables proactive customer retention strategies by identifying churn risk factors, allowing businesses to intervene before customers leave.

---

### 3. SMS Spam Detection
**File**: `models/smsspamdetection.py`  
**Dataset Size**: Several thousand SMS messages  
**Problem Type**: Text classification (binary)

**Technical Implementation**:
- **Algorithm**: Multinomial Naive Bayes (probabilistic classifier)
- **Text Processing**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- **Preprocessing**: Text normalization, lowercasing, special character removal
- **Feature Extraction**: Converting text to numerical feature vectors

**Key Features**:
- Complete NLP preprocessing pipeline
- Efficient text-to-feature transformation
- Fast training and prediction suitable for real-time filtering

**Business Value**: Improves communication platform quality by automatically filtering spam, reducing user frustration and security risks from malicious messages.

---

## � Technical Approach

### Algorithm Selection & Comparison
The project implements **4 different machine learning algorithms** across 3 use cases:

| Algorithm | Use Case | Rationale |
|-----------|----------|-----------|
| **Logistic Regression** | Fraud Detection | Fast, interpretable baseline for binary classification |
| **Decision Tree** | Fraud Detection | Captures non-linear patterns, handles imbalance |
| **Random Forest** | Churn Prediction | Ensemble method robust to mixed feature types |
| **Naive Bayes** | Spam Detection | Efficient for text classification, probabilistic approach |

### Handling Class Imbalance
Class imbalance is a critical challenge in fraud detection and spam filtering where minority classes are rare but important:

**Strategies Employed**:
- **Algorithm Selection**: Choosing models that handle imbalance well (Random Forest, Decision Trees)
- **Evaluation Metrics**: Focusing on precision, recall, and F1-score rather than just accuracy
- **Confusion Matrix Analysis**: Detailed examination of true/false positives and negatives
- **Minority Class Focus**: Prioritizing performance on the rare but critical class (fraud, spam, churn)

### Feature Engineering
- **Numerical Features**: StandardScaler normalization for consistent feature scales
- **Categorical Features**: Label encoding for converting categorical variables to numerical
- **Text Features**: TF-IDF vectorization for converting text to meaningful numerical representations
- **Data Cleaning**: Handling missing values, removing noise, text preprocessing

---

## 🛠️ Installation & Setup

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

## 🚀 Usage & Execution

Each model can be run independently. Ensure datasets are properly named and placed in the project root directory.

### Credit Card Fraud Detection
```bash
python models/creditcardfrauddetection.py
```
**Required Dataset**: `creditcard.csv`
**Output**: Comparative results for Logistic Regression and Decision Tree models

### Customer Churn Prediction
```bash
python models/customerchurnprediction.py
```
**Required Dataset**: `Churn.csv`
**Output**: Random Forest model evaluation with precision, recall, F1-score

### SMS Spam Detection
```bash
python models/smsspamdetection.py
```
**Required Dataset**: `spam.csv`
**Output**: Naive Bayes classification results with accuracy metrics

---

## 📁 Dataset Information & Sources

### Credit Card Fraud Detection Dataset
- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Required Filename**: `creditcard.csv`
- **Size**: ~284,807 transaction records
- **Features**: 30 numerical features (V1-V28 from PCA transformation + Time + Amount)
- **Target**: Class (0: Legitimate, 1: Fraud)
- **Imbalance**: Highly imbalanced (~0.17% fraud cases)

### Customer Churn Prediction Dataset
- **Source**: Standard telecom/subscription churn datasets (various sources available)
- **Required Filename**: `Churn.csv`
- **Size**: ~20,000 customer records
- **Features**: Mixed numerical and categorical (demographics, account info, usage patterns)
- **Target**: Churn (0: Retained, 1: Churned)
- **Note**: Model automatically handles categorical variable encoding

### SMS Spam Detection Dataset
- **Source**: [SMS Spam Collection Dataset (UCI/Kaggle)](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
- **Required Filename**: `spam.csv`
- **Size**: Several thousand SMS messages
- **Features**: Text messages (raw text data)
- **Target**: Label (ham: legitimate, spam: unwanted)
- **Encoding**: Use ISO-8859-1 encoding when saving the file

---

## � Project Structure

```
AspireNex/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── .github/                           # GitHub templates
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
├── models/                            # ML model implementations
│   ├── creditcardfrauddetection.py   # Fraud detection (LR + DT)
│   ├── customerchurnprediction.py    # Churn prediction (RF)
│   └── smsspamdetection.py           # Spam detection (NB)
└── [datasets]                         # Place your CSV files here
    ├── creditcard.csv
    ├── Churn.csv
    └── spam.csv
```

---

## 💡 Key Learnings & Takeaways

### Technical Skills Developed
- **Algorithm Comparison**: Understanding trade-offs between different ML approaches
- **Imbalanced Data**: Strategies for handling skewed class distributions
- **Feature Engineering**: Preprocessing techniques for numerical, categorical, and text data
- **Model Evaluation**: Using appropriate metrics beyond accuracy for real-world problems
- **Pipeline Development**: Building reproducible ML workflows

### Domain Knowledge Gained
- **Financial Fraud**: Understanding transaction patterns and fraud detection challenges
- **Customer Analytics**: Factors influencing customer retention and churn
- **NLP Basics**: Text preprocessing and feature extraction for classification

### Professional Development
- **Documentation**: Creating clear, reusable code for team environments
- **Benchmarking**: Systematic comparison of algorithm performance
- **Real-World Datasets**: Working with large-scale, imbalanced, real-world data

---

## ⚠️ Project Context & Limitations

### Development Context
These models were developed during a **Project-based online internship** as part of a structured learning program. They represent **foundational implementations** created for educational purposes and to contribute to AspireNex's training resources.

### Intended Use
- **Learning & Training**: Reference implementations for ML concepts and workflows
- **Internal Demos**: Showcasing ML capabilities and algorithm comparisons
- **Model Repository**: Reusable code for future projects and interns
- **Benchmarking**: Baseline implementations for algorithm evaluation

### Limitations
- **Production Readiness**: These are foundational models not optimized for production deployment
- **Advanced Techniques**: Does not include state-of-the-art methods like deep learning, advanced ensemble techniques, or hyperparameter optimization
- **Scalability**: Not designed for real-time, large-scale production systems
- **Domain Specificity**: Generic implementations that would require customization for specific business contexts

### Future Enhancements
More advanced implementations could include:
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Advanced imbalance handling (SMOTE, ADASYN, cost-sensitive learning)
- Cross-validation for robust performance estimation
- Feature importance analysis and selection
- Model deployment pipelines
- Real-time prediction APIs

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome! This project serves as a learning resource, and enhancements can benefit future learners.

### How to Contribute
1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/Enhancement`
3. **Commit your changes**: `git commit -m 'Add enhancement'`
4. **Push to branch**: `git push origin feature/Enhancement`
5. **Open a Pull Request**

### Contribution Ideas
- Add hyperparameter tuning examples
- Implement additional algorithms for comparison
- Add visualization of model performance
- Include cross-validation examples
- Improve documentation and code comments
- Add unit tests for preprocessing functions

---

## 🐛 Troubleshooting

### Common Issues

**FileNotFoundError**
- Ensure CSV files are named exactly as specified
- Place datasets in the root directory (same level as `models/` folder)
- Check file paths and working directory

**Encoding Issues (spam.csv)**
- Save the file with ISO-8859-1 encoding
- Use `encoding='ISO-8859-1'` parameter when reading

**Memory Issues**
- Large datasets (especially creditcard.csv) may require sufficient RAM
- Consider using a machine with at least 4GB available memory
- Close other applications to free up memory

**Import Errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version (3.7+)
- Check scikit-learn version compatibility

---

## 📊 Evaluation Metrics Explained

Understanding the metrics used in these projects:

- **Accuracy**: Overall correctness = (TP + TN) / Total
  - *Limitation*: Misleading for imbalanced datasets

- **Precision**: True Positives / (True Positives + False Positives)
  - *Interpretation*: Of all predicted positives, how many are actually positive?
  - *Important for*: Minimizing false alarms (e.g., legitimate transactions flagged as fraud)

- **Recall (Sensitivity)**: True Positives / (True Positives + False Negatives)
  - *Interpretation*: Of all actual positives, how many did we catch?
  - *Important for*: Catching all fraud cases, even if some false positives occur

- **F1-Score**: Harmonic mean of precision and recall = 2 × (Precision × Recall) / (Precision + Recall)
  - *Interpretation*: Balanced measure when you need both precision and recall
  - *Important for*: Imbalanced datasets where both metrics matter

- **Confusion Matrix**: Detailed breakdown of predictions
  ```
  [[TN  FP]
   [FN  TP]]
  ```

---

## 📞 Contact & Support

**Author**: Ritik Pratap Singh Patel  
**Internship**: Machine Learning Virtual Intern @ AspireNex (Jul 2024 – Sep 2024)

For questions, issues, or suggestions:
- **GitHub Issues**: [Create an issue](https://github.com/patelritiq/AspireNex/issues)
- **Email**: patelritiq@gmail.com
- **LinkedIn**: [linkedin.com/in/patelritiq](https://www.linkedin.com/in/patelritiq)

---

## 🙏 Acknowledgments

- **AspireNex**: For providing the internship opportunity and project guidance
- **Kaggle & UCI ML Repository**: For providing high-quality, real-world datasets
- **scikit-learn Community**: For excellent machine learning tools and documentation
- **Open Source Community**: For making learning resources accessible

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📈 Project Stats

![GitHub stars](https://img.shields.io/github/stars/patelritiq/AspireNex?style=social)
![GitHub forks](https://img.shields.io/github/forks/patelritiq/AspireNex?style=social)
![GitHub issues](https://img.shields.io/github/issues/patelritiq/AspireNex)
![GitHub last commit](https://img.shields.io/github/last-commit/patelritiq/AspireNex)

