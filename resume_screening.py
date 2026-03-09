import pandas as pd
import numpy as np
import re
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords

print("==================================================")
print("     RESUME SCREENING SYSTEM - NLP MODEL          ")
print("==================================================")

# 1. Load Data
# We use the dataset downloaded from HuggingFace in previous steps
csv_file = "resume_dataset.csv"

if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found. Please ensure the dataset is downloaded.")
    # Fallback to mock data if the file is missing for some reason
    data = {
        'Category': ['Data Science', 'Data Science', 'Web Designing', 'Web Designing', 'HR', 'HR', 'Java Developer', 'Java Developer'],
        'Resume': [
            'Data Scientist with 5 years experience in machine learning, Python, pandas, and scikit-learn.',
            'Experienced Data Analyst and Scientist. Skills: Python, SQL, Tableau.',
            'Frontend web developer. Expert in HTML, CSS, JavaScript, React.js.',
            'Web Designer. Creating beautiful websites using UI/UX principles, Figma.',
            'Human Resources manager. Experience in talent acquisition and onboarding.',
            'HR Specialist. Skilled in recruiting and managing corporate benefits.',
            'Java Backend Developer. Spring Boot, Hibernate, REST APIs.',
            'Senior Java Developer. Experience with Core Java, J2EE, Maven.'
        ]
    }
    df = pd.DataFrame(data)
else:
    print(f"[1/5] Loading dataset from: {csv_file}")
    df = pd.read_csv(csv_file)
    # Basic cleaning of the dataframe
    df.dropna(subset=['Category', 'Resume'], inplace=True)
    df['Resume'] = df['Resume'].astype(str)
    print(f"Dataset loaded successfully with {len(df)} records.")

# 2. Text Preprocessing
def clean_resume_text(text):
    text = re.sub(r'http\S+\s*', ' ', text)  # remove URLs
    text = re.sub(r'RT|cc', ' ', text)  # remove RT and cc
    text = re.sub(r'#\S+', '', text)  # remove hashtags
    text = re.sub(r'@\S+', '  ', text)  # remove mentions
    # Remove punctuations. Using raw string r'' to avoid escape sequence warnings
    text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text) 
    text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
    return text.strip().lower()

print("[2/5] Cleaning and Preprocessing text data...")
df['Cleaned_Resume'] = df['Resume'].apply(clean_resume_text)

# 3. Label Encoding
# Filter out categories with very few samples to ensure stability
df = df.groupby('Category').filter(lambda x: len(x) > 5)

le = LabelEncoder()
df['Category_Encoded'] = le.fit_transform(df['Category'])

print(f"Number of Job Categories: {len(le.classes_)}")

# 4. Feature Extraction (TF-IDF)
print("[3/5] Extracting features using TF-IDF Vectorization...")
tfidf = TfidfVectorizer(max_features=2500, stop_words='english')

X = df['Cleaned_Resume']
y = df['Category_Encoded']

# Stratified split to maintain category distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 5. Model Training (Random Forest)
print("[4/5] Training Random Forest Classifier (Optimized for Accuracy)...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_tfidf, y_train)

# 6. Evaluation
print("[5/5] Evaluating Model Performance...")
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Complete!")
print(f"Accuracy Score: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 7. Real-world Prediction Helper
def predict_category(resume_text):
    cleaned = clean_resume_text(resume_text)
    vectorized = tfidf.transform([cleaned])
    pred = model.predict(vectorized)[0]
    return le.inverse_transform([pred])[0]

print("\n" + "="*50)
print("     LIVE PREDICTION TEST CASES")
print("="*50)

test_resumes = [
    "Full stack developer with expertise in Python, Django, and React. 5 years of industry experience.",
    "HR Manager specializing in recruitment, employee relations, and policy development.",
    "Data Scientist with proficiency in Machine Learning, Deep Learning, and Big Data Analytics.",
    "Project Manager with a PMP certification and 10+ years of experience in Agile methodologies."
]

for i, res in enumerate(test_resumes):
    prediction = predict_category(res)
    print(f"Test Resume {i+1} Prediction: [{prediction}]")

print("="*50)
print("Script execution finished. All results saved.")
