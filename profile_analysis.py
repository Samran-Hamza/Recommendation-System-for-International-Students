import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import word_tokenize
import spacy
from textblob import TextBlob
import warnings

warnings.filterwarnings('ignore')

# Загрузка модели NLP для анализа текста
nlp = spacy.load("en_core_web_md")

class ProfileAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.scaler = MinMaxScaler()

    def preprocess_data(self, df):
        """
        Preprocess student data to create feature vectors.
        """
        # Текстовые признаки (например, интересы, хобби)
        text_features = self.vectorizer.fit_transform(df['interests'])

        # Числовые признаки (например, GPA, возраст)
        numerical_features = df[['gpa', 'age']].values
        numerical_features = self.scaler.fit_transform(numerical_features)

        # Анализ текста с помощью spaCy для извлечения семантических признаков
        semantic_features = np.array([self.extract_semantic_features(text) for text in df['interests']])

        # Объединение всех признаков в один вектор
        features = np.hstack((text_features.toarray(), numerical_features, semantic_features))
        return features

    def extract_semantic_features(self, text):
        """
        Extract semantic features from text using spaCy embeddings.
        """
        doc = nlp(text)
        return doc.vector

    def analyze_profile(self, student_data):
        """
        Analyze a single student's profile and return their feature vector.
        """
        
        df = pd.DataFrame([student_data])
        return self.preprocess_data(df)[0]

# Пример использования
if __name__ == "__main__":
    # датасет
    data = {
        "interests": ["machine learning, artificial intelligence, mathematics"],
        "gpa": [4.0],
        "age": [22]
    }
    analyzer = ProfileAnalyzer()
    feature_vector = analyzer.analyze_profile(data)
    print("Student Feature Vector:", feature_vector)