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
from gensim.corpora import Dictionary
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 500,
    separators= ['\n\n','\n','.',' ','']
)

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
                paragraphs = text_splitter.split_text(text)
                original_paragraphs_list.append(paragraphs)
                preprocessed_paragraphs = [' '.join(preprocess_text(para)) for para in paragraphs]
                preprocessed_paragraphs_list.append(preprocessed_paragraphs)

    return documents, case_ids, original_paragraphs_list, preprocessed_paragraphs_list

if __name__ == "__main__":
    folder_path='data_processing/19'
    documents, case_ids, original_paragraphs, preprocessed_paragraphs = read_documents(folder_path)
    preprocessed_documents1 = [preprocess_text(doc) for doc in documents]
    dictionary = Dictionary(preprocessed_documents1)
    dictionary.save('data_processing/dictionary.dict')

    preprocessed_documents = []

    for i in range(len(preprocessed_documents1)):
        document = ' '.join(preprocessed_documents1[i])
        preprocessed_documents.append(document)
        
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

    df.to_csv('data_processing/documents.csv', index=False)

    with open('data_processing/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('data_processing/tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)