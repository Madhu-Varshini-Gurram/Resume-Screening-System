# AI-Powered Resume Screening System 🚀

## Overview
This project is an end-to-end Machine Learning pipeline designed to automate the process of resume classification. Built using **Python**, **NLP (Natural Language Processing)**, and **Scikit-Learn**, it classifies resumes into various job categories such as Data Science, HR, Web Designing, Java Development, and many more.

## ✨ Key Features
- **Text Preprocessing**: Robust cleaning pipeline using Regular Expressions to remove URLs, hashtags, mentions, and special characters.
- **NLP Vectorization**: Implements **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert unstructured text into numerical feature vectors.
- **High-Performance Classifier**: Utilizes an optimized **Random Forest Classifier** to achieve high classification accuracy across multiple categories.
- **Live Prediction**: Includes a real-time prediction module to categorize new, unseen resume text instantly.

## 🛠️ Technology Stack
- **Languages**: Python 3
- **Libraries**:
  - `pandas`: Data manipulation and analysis.
  - `scikit-learn`: Machine learning model building and evaluation.
  - `nltk`: Natural language tool kit for text processing.
  - `re`: Regular expressions for advanced text cleaning.

## 📊 Performance
The model is trained on a comprehensive dataset of over 10,000 resumes. 
- **Accuracy**: (Refer to `output_final.txt` for the latest score)
- **Evaluation**: The system is evaluated using Precision, Recall, and F1-Score to ensure reliability across all classes.

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn nltk
   ```
2. Run the main processing script:
   ```bash
   python resume_screening.py
   ```

## 📈 Future Scope
- Integration with PDF/Docx parsers for direct file input.
- Deployment as a Web API using Flask or FastAPI.
- Fine-tuning with Transformer-based models like BERT for deeper semantic understanding.

---
*Developed for Resume Screening Portfolio.*
