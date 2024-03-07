from gensim.corpora import Dictionary
import re
from nltk.tokenize import word_tokenize
from preproccess import preprocess_text
from gensim.models.fasttext import load_facebook_model
from gensim.models import FastText
from gensim.similarities import SoftCosineSimilarity, WordEmbeddingSimilarityIndex
from gensim.similarities.termsim import SparseTermSimilarityMatrix
import pickle
import time



def text_to_bow(text, dictionary):
    tokens = preprocess_text(text)
    bow_vector = dictionary.doc2bow(tokens)
    return bow_vector

def create_similarity_matrix(fasttext_model, dictionary):
    fasttext_kv = fasttext_model.wv
    similarity_index = WordEmbeddingSimilarityIndex(fasttext_kv)
    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, nonzero_limit=100)
    return similarity_matrix

def calculate_similarity(query, paragraphs, dictionary, similarity_matrix):
    query_bow = text_to_bow(query, dictionary)
    paragraph_bows = [text_to_bow(paragraph, dictionary) for paragraph in paragraphs]
    soft_cosine_similarity = SoftCosineSimilarity(paragraph_bows, similarity_matrix)
    scores = soft_cosine_similarity(query_bow)
    return scores

if __name__ == "__main__":
    initial_time = time.time()
    dictionary_path = 'data_processing/dictionary.dict'   
    dictionary = Dictionary.load(dictionary_path)
    final_time = time.time()
    print("time to load dict:", final_time - initial_time)

    initial_time = time.time()
    fasttext_model_path = 'models/crawl-300d-2M-subword/crawl-300d-2M-subword.bin'
    fasttext_model = load_facebook_model(fasttext_model_path)
    final_time = time.time()
    print("time to load model:", final_time - initial_time )

    similarity_matrix = create_similarity_matrix(fasttext_model, dictionary)

    similarity_matrix_path = 'data_processing/similarity_matrix.pkl'
    with open(similarity_matrix_path, 'wb') as sm_file:
        pickle.dump(similarity_matrix, sm_file)
