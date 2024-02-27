import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return tokens

def preprocess_text2(text):
    text = re.sub(r'[^\w\s\']', "", text)
    text = re.sub(' +', ' ', text)
    return text

def read_documents(folder_path):
    documents = []
    case_ids = []
    original_paragraphs_list = []
    preprocessed_paragraphs_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            case_id = filename.split('.')[0]
            case_ids.append(case_id)
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                documents.append(text)
                paragraphs = text.split('\n')
                original_paragraphs_list.append(paragraphs)
                preprocessed_paragraphs = [' '.join(preprocess_text(para)) for para in paragraphs]
                preprocessed_paragraphs_list.append(preprocessed_paragraphs)

    return documents, case_ids, original_paragraphs_list, preprocessed_paragraphs_list

folder_path='C:\\Users\\dania\\Documents\\LegalQA\\data_processing\\19'
documents, case_ids, original_paragraphs, preprocessed_paragraphs = read_documents(folder_path)
preprocessed_documents = [' '.join(preprocess_text(doc)) for doc in documents]

def vectorize_documents(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = vectorize_documents(preprocessed_documents)

df = pd.DataFrame({
    'case_id': case_ids,
    'original_document': documents,
    'preprocessed_document': preprocessed_documents,
    'original_paragraphs': original_paragraphs,
    'preprocessed_paragraphs': preprocessed_paragraphs
})

df['processed_training_document'] = df['original_document'].map(preprocess_text2)

df.to_csv('documents.csv', index=False)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)