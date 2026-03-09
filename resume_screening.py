import pandas as pd
import numpy as np
import re
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import urllib.request
import io
import zipfile

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

print("==================================================")
print("     STARTING RESUME SCREENING SYSTEM             ")
print("==================================================")

# 1. Load Data
print("\n[1/5] Loading and Preprocessing Data...")

# Public Dataset URL - A common Kaggle dataset hosted on GitHub for easy programmatic access
csv_url = "https://raw.githubusercontent.com/laxmimerit/Resume-and-CV-Summarization-and-Parsing-with-Spacy-in-Python/master/resume_dataset.csv"

try:
    print(f"Downloading dataset from: {csv_url}")
    df = pd.read_csv(csv_url)
    print(f"Dataset loaded! Shape: {df.shape}")
except Exception as e:
    print(f"Failed to download from GitHub. Error: {e}")
    print("Falling back to creating a mock dataset for demonstration...")
    # Mock data fallback
    data = {
        'Category': ['Data Science', 'Data Science', 'Web Designing', 'Web Designing', 'HR', 'HR', 'Java Developer', 'Java Developer'],
        'Resume': [
            'Data Scientist with 5 years experience in machine learning, Python, pandas, and scikit-learn. Built predictive marketing models.',
            'Experienced Data Analyst and Scientist. Skills: Python, SQL, Tableau, deep learning (TensorFlow).',
            'Frontend web developer. Expert in HTML, CSS, JavaScript, React.js and building responsive UIs.',
            'Web Designer. Creating beautiful websites using UI/UX principles, Figma, HTML5, CSS3, Bootstrap.',
            'Human Resources manager. Experience in talent acquisition, payroll, employee relations and onboarding.',
            'HR Specialist. Skilled in recruiting, interviewing, resume screening, and managing corporate benefits.',
            'Java Backend Developer. Spring Boot, Hibernate, REST APIs, Microservices architecture, SQL databases.',
            'Senior Java Developer. Experience with Core Java, J2EE, Maven, Jenkins CI/CD, and AWS.'
        ]
    }
    df = pd.DataFrame(data)

# 2. Clean Text
def clean_resume_text(text):
    text = re.sub('http\S+\s*', ' ', text)  # remove URLs
    text = re.sub('RT|cc', ' ', text)  # remove RT and cc
    text = re.sub('#\S+', '', text)  # remove hashtags
    text = re.sub('@\S+', '  ', text)  # remove mentions
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # remove punctuations
    text = re.sub(r'[^\x00-\x7f]',r' ', text) 
    text = re.sub('\s+', ' ', text)  # remove extra whitespace
    return text.lower()

df['Cleaned_Resume'] = df['Resume'].apply(clean_resume_text)
print("Data cleaning complete.")

# 3. Encode Labels
le = LabelEncoder()
df['Category_Encoded'] = le.fit_transform(df['Category'])

# Print category mapping
print("\n[2/5] Categories found:")
category_mapping = dict(zip(le.transform(le.classes_), le.classes_))
for k, v in category_mapping.items():
    print(f"  {k}: {v}")


# 4. Feature Extraction (TF-IDF)
print("\n[3/5] Extracting Features using TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500
)

# Split data FIRST, then fit TF-IDF on training data only to avoid data leakage
X = df['Cleaned_Resume']
y = df['Category_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Fit and transform
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# 5. Model Training
print("\n[4/5] Training K-Nearest Neighbors Classifier...")
# We use OneVsRestClassifier with KNeighborsClassifier for multi-class support
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train_tfidf, y_train)

# 6. Evaluation
print("\n[5/5] Evaluating Model...")
y_pred = clf.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"\n---> Accuracy on Test Set: {acc * 100:.2f}%\n")

try:
    print(classification_report(y_test, y_pred, target_names=le.classes_))
except ValueError:
    # If the mock dataset is too small, some classes might not be in the test set.
    print(classification_report(y_test, y_pred))


# 7. Prediction Function
def predict_resume(text):
    cleaned = clean_resume_text(text)
    vectorized = tfidf_vectorizer.transform([cleaned])
    prediction_encoded = clf.predict(vectorized)[0]
    category = le.inverse_transform([prediction_encoded])[0]
    return category

print("\n==================================================")
print("     TESTING WITH NEW SAMPLE RESUMES              ")
print("==================================================")

sample_1 = """
Proficient in Java, Spring Boot, and Microservices. 
Developed RESTful APIs and maintained SQL databases. 
Experience with Jenkins and Docker for CI/CD pipelines.
"""

sample_2 = """
Data enthusiast with a strong background in Python, machine learning, and statistical analysis.
Proficient with Pandas, NumPy, and Scikit-Learn. Exploring deep learning with TensorFlow.
"""

sample_3 = """
Extensive experience in talent acquisition and payroll management. 
Excellent communication and negotiation skills. Organized team building activities.
"""

print(f"Sample 1 Prediction => **{predict_resume(sample_1)}**")
print(f"Sample 2 Prediction => **{predict_resume(sample_2)}**")
print(f"Sample 3 Prediction => **{predict_resume(sample_3)}**")
print("\nScript execution completed successfully!")
