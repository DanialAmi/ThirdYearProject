from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pickle
from sentence_transformers import SentenceTransformer

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 500,
    separators= ['\n\n','\n','.',' ','']
)

def create_paragraphs(folder_path):
    documents = []
    case_ids = []
    paragraphs_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            case_id = filename.split('.')[0]
            case_ids.append(case_id)
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                documents.append(text)
                paragraphs = text_splitter.split_text(text)
                paragraphs_list.extend(paragraphs)

    return documents, case_ids, paragraphs_list

folder_path = "data_processing/19"
documents, case_ids, paragraphs = create_paragraphs(folder_path)

bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
bi_encoder.max_seq_length = 512     

corpus_embeddings = bi_encoder.encode(paragraphs, convert_to_tensor=True, show_progress_bar=True)

with open("data_processing/embeddings.pkl", "wb") as fOut:
    pickle.dump({"case_ids": case_ids, "paragraphs": paragraphs, "embeddings": corpus_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)