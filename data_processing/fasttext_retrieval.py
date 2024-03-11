import pickle
from gensim.corpora import Dictionary
from gensim.similarities import SoftCosineSimilarity, WordEmbeddingSimilarityIndex
from gensim.similarities.termsim import SparseTermSimilarityMatrix
from preproccess import preprocess_text
from fasttext import calculate_similarity

if __name__ == "__main__":
    paragraphs = [
        "Copyright law provides a range of rights to authors of original works, including the right to reproduce the work, to prepare derivative works, to distribute copies, and to perform the work publicly.",
        "In patent law, an inventor is granted exclusive rights to exclude others from making, using, or selling an invention for a limited period of time, in exchange for publishing an enabling public disclosure of the invention.",
        "Contract law governs the legality of agreements made between two or more parties, ensuring that commitments made in contracts are enforceable."
    ]
    query = "What rights does copyright law grant to authors of original works?"

    dictionary_path = 'data_processing/dictionary.dict'   
    dictionary = Dictionary.load(dictionary_path)

    # Load the precomputed similarity matrix
    similarity_matrix_path = 'data_processing/similarity_matrix.pkl'
    with open(similarity_matrix_path, 'rb') as sm_file:
        similarity_matrix = pickle.load(sm_file)
    # Calculate the similarity scores between the query and each paragraph
    scores = calculate_similarity(query, paragraphs, dictionary, similarity_matrix)

    # Display the similarity scores
    for i, score in enumerate(scores):
        print(f"Paragraph {i+1} Similarity Score: {score}")
