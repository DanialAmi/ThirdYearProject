import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords 
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Case folding
    tokens = [word.lower() for word in tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

def load_data_and_vectorizer():
    df = pd.read_csv('data_processing/documents.csv')
    with open('data_processing/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('data_processing/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    return df, vectorizer, tfidf_matrix

def process_query(query, vectorizer, tfidf_matrix):
    preprocessed_query = ' '.join(preprocess_text(query))
    query_vector = vectorizer.transform([preprocessed_query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    return cosine_similarities

df, vectorizer, tfidf_matrix = load_data_and_vectorizer()
query = "We draw the underlying facts from the findings made by the district court, see United States v. Fernandez, 578 F.Supp.2d 243, 244-46 (D.Mass.2008), and the testimony presented at the suppression hearing.   At about 4:30 p.m. on October 20, 2007, Officer Anthony Pistolese was sitting in a parked cruiser across the street from a liquor store in Taunton, Massachusetts, when he observed a red Dodge Magnum pull into the store's parking lot just before three men, two on bicycles and one on foot, arrived there.   The man on foot got into the car and the others pedaled away.The Dodge then pulled out of the parking lot onto Bay Street"
cosine_similarities = process_query(query, vectorizer, tfidf_matrix)

most_similar_doc_index = cosine_similarities.argmax()
most_similar_document = df.iloc[most_similar_doc_index]['original_document']
print(most_similar_doc_index)
# Find indices of the top 5 most similar documents
top_5_indices = cosine_similarities.argsort()[-5:][::-1]
print(top_5_indices)
# Retrieve the top 5 most similar documents from the DataFrame
top_5_documents = df.iloc[top_5_indices]['original_document'].tolist()

