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
            category = folder_path.split('/')[1]
            case_id = filename.split('.')[0]
            case_id = category + "_" + case_id
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                documents.append(text)
                paragraphs = text_splitter.split_text(text)
                for paragraph in paragraphs:
                    case_ids.append(case_id)  # Append case_id for each paragraph
                    paragraphs_list.append(paragraph)

    return documents, case_ids, paragraphs_list

# folder_path = "legal_case/property_law_and_real_estates"
# documents, case_ids, paragraphs = create_paragraphs(folder_path)

# print(len(paragraphs))
# print(len(case_ids))

def create_paragraph_from_folder(source_folder):
    documents = []
    case_ids = []
    paragraphs_list = []

    for folder_name in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder_name)
        if os.path.isdir(folder_path):  # Check if it's a directory
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    category = folder_name  # Assuming the category is the folder name
                    case_id = filename.split('.')[0]
                    full_case_id = category + "_" + case_id
                    with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                        text = file.read()
                        documents.append(text)
                        paragraphs = text_splitter.split_text(text)
                        for paragraph in paragraphs:
                            case_ids.append(full_case_id)  # Append case_id for each paragraph
                            paragraphs_list.append(paragraph)

    return documents, case_ids, paragraphs_list

documents, case_ids, paragraphs = create_paragraph_from_folder('legal_case')

bi_encoder = SentenceTransformer('msmarco-distilbert-base-v4')
bi_encoder.max_seq_length = 512     

corpus_embeddings = bi_encoder.encode(paragraphs, convert_to_tensor=True, show_progress_bar=True)

# category = folder_path.split('/')[1]

with open("combined.pkl", "wb") as fOut:
    pickle.dump({"case_ids": case_ids, "paragraphs": paragraphs, "embeddings": corpus_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    

# print("writing to pickle file")
# with open(f"pickle_files/{category}.pkl", "wb") as fOut:
#     pickle.dump({"case_ids": case_ids, "paragraphs": paragraphs, "embeddings": corpus_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    
print("done")
