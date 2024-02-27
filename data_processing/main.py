from document_retrieval import load_vectorizer_and_matrix, find_top_documents
from machine_comprehension import get_best_answer
import pandas as pd

def load_documents(csv_file_path='documents.csv'):
    return pd.read_csv(csv_file_path)

def main():
    query = "What was Petitioner Williams convicted of"
    vectorizer, tfidf_matrix = load_vectorizer_and_matrix()
    top_indices = find_top_documents(query, vectorizer, tfidf_matrix)
    print(top_indices)
    df = load_documents()
    top_documents = df.iloc[top_indices]['original_document'].tolist()
    # print(top_documents)

    best_answer = get_best_answer(query, top_documents)
    print("Best answer:", best_answer[0])

if __name__ == "__main__":
    main()