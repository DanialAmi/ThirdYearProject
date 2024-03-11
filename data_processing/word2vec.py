from gensim.corpora import Dictionary
from preproccess import preprocess_text
from gensim.similarities import SoftCosineSimilarity, WordEmbeddingSimilarityIndex
from gensim.similarities.termsim import SparseTermSimilarityMatrix
import pickle
import gensim.downloader as api

def text_to_bow(text, dictionary):
    tokens = preprocess_text(text)
    bow_vector = dictionary.doc2bow(tokens)
    return bow_vector

def create_similarity_matrix_word2vec(word2vec_model, dictionary):
    # Initialize the WordEmbeddingSimilarityIndex with Word2Vec model
    similarity_index = WordEmbeddingSimilarityIndex(word2vec_model)
    # Create the SparseTermSimilarityMatrix using the similarity index and dictionary
    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, nonzero_limit=100)
    return similarity_matrix

def calculate_similarity(query, paragraphs, dictionary, similarity_matrix):
    # Convert the query and paragraphs to BoW vectors
    query_bow = text_to_bow(query, dictionary)
    paragraph_bows = [text_to_bow(paragraph, dictionary) for paragraph in paragraphs]
    # Create a SoftCosineSimilarity object with the paragraph BoWs and similarity matrix
    soft_cosine_similarity = SoftCosineSimilarity(paragraph_bows, similarity_matrix)
    # Calculate and return similarity scores
    scores = soft_cosine_similarity[query_bow]
    return scores

if __name__ == "__main__":   
    
    word2vec_model = api.load('word2vec-google-news-300')
    
    dictionary_path = 'data_processing/dictionary.dict'   
    dictionary = Dictionary.load(dictionary_path)
    
    similarity_matrix = create_similarity_matrix_word2vec(word2vec_model, dictionary)

    similarity_matrix_path = 'data_processing/word2vec_similarity_matrix.pkl'
    
    with open(similarity_matrix_path, 'wb') as sm_file:
        pickle.dump(similarity_matrix, sm_file)